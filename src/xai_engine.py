import torch
import numpy as np
import torch.nn.functional as F
from captum.attr import LayerGradCam, IntegratedGradients, NoiseTunnel
from src.base import BaseExplainer
from src.utils import enable_dropout


# Swin Transformer 专用的形状转换函数 (Captum 兼容)
def swin_reshape_transform(tensor):
    """
    将 Swin 的 [B, H, W, C] 输出转换为 [B, C, H, W]
    """
    # 检查维度是否符合 Swin 的特征输出 (通常最后一位是通道数)
    if tensor.ndim == 4 and tensor.shape[-1] != tensor.shape[2]:
        return tensor.permute(0, 3, 1, 2)
    return tensor


class AdvancedXAIEngine(BaseExplainer):
    """
    4.0版 专业级XAI引擎
    - 支持流式贝叶斯方差计算 (极低显存占用)
    - 增强的层自动探测
    - Swin Transformer 完美适配
    """

    def __init__(self, model, device, model_name=None):
        super().__init__(model, device)

        # 属性注入
        if model_name is None:
            self.model_name = getattr(model, 'model_name', 'unknown')
        else:
            self.model_name = model_name

        self.model.eval()
        self.target_layer = self._detect_target_layer()

        # 初始化解释器
        self._init_explainers()

    def _detect_target_layer(self):
        """
        智能探测用于 Grad-CAM 的目标层
        """
        try:
            if 'densenet' in self.model_name:
                # DenseNet: 尝试获取最后一个 DenseBlock 的最后一层
                # 路径: features -> denseblock4 -> denselayer16 -> conv2
                return self.model.features.denseblock4.denselayer16.conv2

            elif 'swin' in self.model_name:
                # Swin: 最后一个 Stage 的最后一个 Block 的 Norm 层
                # 路径: features -> 7 (stage4) -> 1 (block2) -> norm1
                # 注意：不同 torchvision 版本 swin 结构可能略有不同，这里取 features 的最后一个元素
                return self.model.features[-1][-1].norm1

            else:
                # 通用策略：取 features 容器的最后一层，或模型的倒数第二层
                if hasattr(self.model, 'features'):
                    return list(self.model.features.children())[-1]
                else:
                    return list(self.model.children())[-2]
        except Exception as e:
            print(f"⚠️ 警告: 自动层探测失败 ({e})，尝试回退到默认策略。")
            return list(self.model.children())[-1]

    def _init_explainers(self):
        """初始化 Captum 解释器"""
        try:
            if 'swin' in self.model_name:
                self.grad_cam = LayerGradCam(self.model, self.target_layer, reshape_transform=swin_reshape_transform)
            else:
                self.grad_cam = LayerGradCam(self.model, self.target_layer)
        except Exception as e:
            print(f"❌ Grad-CAM 初始化失败: {e}")
            self.grad_cam = None

        self.ig = IntegratedGradients(self.model)
        # NoiseTunnel 用于 SmoothGrad
        self.nt = NoiseTunnel(self.ig)

    def _process_heatmap(self, attr, img_size):
        """
        统一的热力图后处理：聚合通道 -> 插值 -> 归一化
        :param attr: [1, C, H, W] or [1, 1, H, W] 张量
        :param img_size: (H, W) 目标尺寸
        """
        # 1. 聚合通道 (如果有多通道)
        if attr.shape[1] > 1:
            attr = torch.sum(torch.abs(attr), dim=1, keepdim=True)

        # 2. 插值回原图尺寸
        # 使用双线性插值使热力图更平滑
        attr = F.interpolate(attr, size=img_size, mode='bilinear', align_corners=False)

        # 3. 转 Numpy 并移除 Batch 维度
        heatmap = attr.detach().cpu().numpy()[0, 0]

        # 4. 鲁棒归一化 (Min-Max)
        min_v, max_v = heatmap.min(), heatmap.max()
        if max_v - min_v > 1e-8:
            heatmap = (heatmap - min_v) / (max_v - min_v)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def _generate_single(self, input_tensor, method='gradcam'):
        """单次解释生成 (Internal)"""
        input_tensor = input_tensor.to(self.device)
        if not input_tensor.requires_grad:
            input_tensor.requires_grad = True

        img_size = (input_tensor.shape[2], input_tensor.shape[3])

        if method == 'gradcam':
            if self.grad_cam is None: return np.zeros(img_size)
            # relu_attributions=True 意味着只保留正向激活（对预测有正贡献的特征）
            attr = self.grad_cam.attribute(input_tensor, relu_attributions=True)
            return self._process_heatmap(attr, img_size)

        elif method == 'ig':
            # 使用 SmoothGrad (NoiseTunnel) 增强 IG 的平滑度
            # nt_samples=2 保持低开销，如果是单次生成可以用更高
            try:
                attr = self.nt.attribute(input_tensor, nt_type='smoothgrad', nt_samples=2, target=0)
            except:
                attr = self.ig.attribute(input_tensor, n_steps=15, target=0)
            return self._process_heatmap(attr, img_size)

        return np.zeros(img_size)

    def generate(self, input_tensor, method='gradcam'):
        """确定性生成接口"""
        self.model.eval()
        return self._generate_single(input_tensor, method)

    def generate_bayesian(self, input_tensor, method='gradcam', num_samples=10):
        """
        流式贝叶斯生成 (Memory Efficient)
        使用 Welford 算法或简单的累积法在线计算均值和方差，
        避免保存所有采样结果导致显存溢出。
        """
        img_h, img_w = input_tensor.shape[2], input_tensor.shape[3]

        # 强制开启 Dropout
        enable_dropout(self.model)

        # 初始化累积器
        sum_heatmap = np.zeros((img_h, img_w), dtype=np.float64)
        sum_sq_heatmap = np.zeros((img_h, img_w), dtype=np.float64)

        success_count = 0

        for _ in range(num_samples):
            # 每次生成一张热力图
            heatmap = self._generate_single(input_tensor, method)

            # 累积
            sum_heatmap += heatmap
            sum_sq_heatmap += heatmap ** 2
            success_count += 1

        self.model.eval()  # 恢复 Eval

        if success_count == 0:
            return np.zeros((img_h, img_w)), np.zeros((img_h, img_w))

        # 计算统计量
        # Mean = E[X]
        mean_map = sum_heatmap / success_count

        # Var = E[X^2] - (E[X])^2
        # 防止计算误差导致负值
        var_map = (sum_sq_heatmap / success_count) - (mean_map ** 2)
        var_map = np.maximum(var_map, 0)
        std_map = np.sqrt(var_map)

        # 再次归一化 Mean Map 以便于可视化和指标计算
        if mean_map.max() > 0:
            mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min())

        return mean_map, std_map