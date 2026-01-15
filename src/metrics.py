import torch
import numpy as np
from scipy.stats import spearmanr
import torch.nn.functional as F


def _to_numpy(data):
    """内部辅助函数：将Tensor或Array统一转为CPU Numpy数组"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def calculate_iou(heatmap, gt_mask, threshold=0.15):
    """
    计算交并比 (IoU) - 衡量解释热力图与真实病灶区域的形状重合度

    Args:
        heatmap: 解释热力图 (0-1)
        gt_mask: 真实病灶掩码 (0/1)
        threshold: 二值化阈值
    """
    # 1. 统一转为 numpy 防止报错
    heatmap = _to_numpy(heatmap)
    gt_mask = _to_numpy(gt_mask)

    # 2. 计算
    binary_map = (heatmap > threshold).astype(int)
    intersection = np.logical_and(binary_map, gt_mask).sum()
    union = np.logical_or(binary_map, gt_mask).sum()

    return intersection / (union + 1e-8)  # 防止除零


def calculate_pointing_game(heatmap, gt_mask):
    """
    计算指向游戏命中率 (Pointing Game Hit Rate)
    逻辑：若热力图最大值所在点落在真实病灶区域内，则为"命中"(1)
    """
    heatmap = _to_numpy(heatmap)
    gt_mask = _to_numpy(gt_mask)

    # 如果没有病灶 (GT全为0)，此指标无意义，返回0
    if gt_mask.sum() == 0:
        return 0.0

    # 找到热力图最大值的坐标 (unravel_index 将平铺索引转为 (y, x))
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    # 检查该坐标是否在真实病灶区域内
    return 1.0 if gt_mask[max_idx] > 0 else 0.0


def calculate_agreement(map1, map2):
    """
    计算 Spearman 相关系数 - 衡量两种解释方法的一致性
    """
    map1 = _to_numpy(map1).flatten()
    map2 = _to_numpy(map2).flatten()

    # 计算相关性
    return spearmanr(map1, map2)[0]


class FidelityAuditor:
    """
    模型解释忠实度审计器 (高性能版)
    通过 Deletion Game (逐步删除重要像素) 评估解释是否忠实于模型决策。
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_auc(self, img_tensor, heatmap, mode='deletion', steps=50, batch_size=32):
        """
        计算 Deletion AUC (支持 GPU 批量加速)

        Args:
            img_tensor: [1, C, H, W] 原始图片
            heatmap: [H, W] 热力图
            mode: 'deletion' (目前只支持删除模式)
            steps: 审计步数 (建议 30-100)
            batch_size: 推理时的批次大小

        Returns:
            auc: 曲线下面积 (越低越好)
            scores: 预测概率变化曲线
        """
        # 1. 设备与数据准备
        if img_tensor.device != self.device:
            img_tensor = img_tensor.to(self.device)

        B, C, H, W = img_tensor.shape
        total_pixels = H * W

        # 处理热力图
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap).to(self.device)
        elif heatmap.device != self.device:
            heatmap = heatmap.to(self.device)

        # 2. 像素重要性排序
        flat_heatmap = heatmap.flatten()
        # Deletion: 从最重要(数值最大)的像素开始删除，所以用 descending=True
        sorted_indices = torch.argsort(flat_heatmap, descending=True)

        # 3. 准备基准图 (Baseline)
        if mode == 'deletion':
            start_img = img_tensor.clone()
            # 目标图：被删除的区域将被替换为全黑 (0)
            target_img = torch.zeros_like(img_tensor)
        else:
            raise NotImplementedError("当前仅支持 'deletion' 模式")

        # 4. 向量化生成扰动图 (Vectorized Generation)
        # 为了速度，我们先在显存中生成所有步骤的图，然后一次性推理
        perturbed_images = [start_img]  # 第0步是原图

        # 展平以便索引操作: [1, C, H*W]
        current_img_flat = start_img.clone().view(1, C, -1)
        target_img_flat = target_img.view(1, C, -1)

        # 计算每一步需要修改的像素数
        pixels_per_step = total_pixels // steps
        step_sizes = [pixels_per_step] * steps
        # 处理余数
        for i in range(total_pixels % steps):
            step_sizes[i] += 1

        current_idx = 0
        for step_size in step_sizes:
            # 获取这一步要修改的像素索引
            indices_to_mod = sorted_indices[current_idx: current_idx + step_size]

            # 核心操作：直接替换像素值
            current_img_flat[:, :, indices_to_mod] = target_img_flat[:, :, indices_to_mod]

            # 恢复形状并保存副本
            perturbed_images.append(current_img_flat.view(1, C, H, W).clone())
            current_idx += step_size

        # 5. 批量推理 (Batch Inference)
        # 堆叠所有图片: [Steps+1, C, H, W]
        batch_input = torch.cat(perturbed_images, dim=0)
        scores = []

        with torch.no_grad():
            # 按 batch_size 切分推理，防止显存溢出
            for i in range(0, len(batch_input), batch_size):
                batch = batch_input[i: i + batch_size]
                logits = self.model(batch)
                # 获取预测概率 (假设是二分类，取 sigmoid)
                probs = torch.sigmoid(logits).view(-1).cpu().numpy().tolist()
                scores.extend(probs)

        # 6. 计算 AUC
        # 归一化 x 轴到 [0, 1] 区间
        auc = np.trapz(scores, dx=1.0 / (len(scores) - 1))

        return auc, scores