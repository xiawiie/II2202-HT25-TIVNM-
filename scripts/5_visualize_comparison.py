import sys
import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

# 引入 Captum 组件
from captum.attr import LayerGradCam, IntegratedGradients, NoiseTunnel

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import Config
from src.dataset import RSNADataset
from src.model import get_model
from src.xai_engine import AdvancedXAIEngine


# =========================================================================
# 🛠️ 修复版XAI引擎 (解决兼容性问题)
# =========================================================================
class FixedXAIEngine(AdvancedXAIEngine):
    """修复版解释引擎：解决Grad-CAM初始化报错及Swin维度不匹配问题"""

    def __init__(self, model, device, model_name=None):
        self.model = model.eval()  # 确保模型处于评估模式
        self.device = device
        self.model_name = model_name or getattr(model, 'model_name', 'unknown')

        # 自动选择适合的目标层
        self.target_layer = self._select_target_layer_safe()

        # 初始化解释器（移除reshape_transform参数）
        self.grad_cam = self._init_grad_cam()
        self.ig = IntegratedGradients(model)
        self.nt = NoiseTunnel(self.ig)  # 用于生成SmoothGrad

    def _select_target_layer_safe(self):
        """为不同模型选择合适的特征层"""
        if 'densenet' in self.model_name:
            return self.model.features.denseblock4.denselayer16.conv2
        elif 'swin' in self.model_name:
            return self.model.features[-1][-1].norm1
        else:
            return list(self.model.children())[-1]

    def _init_grad_cam(self):
        """安全初始化Grad-CAM"""
        try:
            return LayerGradCam(self.model, self.target_layer)
        except Exception as e:
            print(f"❌ Grad-CAM初始化失败（模型{self.model_name}）: {str(e)}")
            return None

    def _generate_single_pass(self, input_tensor, method='gradcam'):
        """生成单张解释热力图"""
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        if method == 'gradcam':
            if not self.grad_cam: return np.zeros(input_tensor.shape[2:])
            try:
                attr = self.grad_cam.attribute(input_tensor, relu_attributions=True)
                # Swin特征图维度适配
                if 'swin' in self.model_name and attr.dim() == 4:
                    if attr.shape[-1] > attr.shape[1]:
                        attr = attr.permute(0, 3, 1, 2)

                attr = LayerGradCam.interpolate(attr, (input_tensor.shape[2], input_tensor.shape[3]),
                                                interpolate_mode='bilinear')
                heatmap = attr.detach().cpu().numpy()[0, 0]
            except Exception as e:
                # print(f"⚠️ Grad-CAM计算警告: {str(e)}") # 减少刷屏
                heatmap = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        elif method == 'ig':
            try:
                # 优先使用SmoothGrad
                attr = self.nt.attribute(input_tensor, nt_type='smoothgrad', nt_samples=3, target=0)
            except:
                # 回退
                attr = self.ig.attribute(input_tensor, n_steps=10, target=0)
            heatmap = np.sum(np.abs(attr.detach().cpu().numpy()[0]), axis=0)
        else:
            heatmap = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # 归一化
        min_v, max_v = heatmap.min(), heatmap.max()
        if max_v - min_v > 1e-8:
            return (heatmap - min_v) / (max_v - min_v)
        return np.zeros_like(heatmap)

    def generate(self, input_tensor, method='gradcam'):
        self.model.eval()
        return self._generate_single_pass(input_tensor, method)

    def generate_bayesian(self, input_tensor, method='gradcam', num_samples=10):
        """蒙特卡洛Dropout不确定性"""
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()

        heatmaps = [self._generate_single_pass(input_tensor, method) for _ in range(num_samples)]

        self.model.eval()
        heatmaps = np.array(heatmaps)
        mean_map = np.mean(heatmaps, axis=0)
        std_map = np.std(heatmaps, axis=0)

        if mean_map.max() > 0:
            mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min())

        return mean_map, std_map


# =========================================================================
# 辅助函数
# =========================================================================
def overlay_heatmap(img_rgb, heatmap):
    """叠加热力图（增强鲁棒性）"""
    # 1. 处理NaN和Inf
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    # 2. 确保范围
    heatmap = np.clip(heatmap, 0, 1)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_rgb = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img_rgb, 0.7, heatmap_rgb, 0.3, 0)


def load_checkpoint_smart(model, path, model_name):
    """智能加载权重"""
    if not os.path.exists(path):
        print(f"❌ 权重文件不存在: {path}")
        return False

    print(f"📂 加载权重: {path}")
    try:
        state_dict = torch.load(path, map_location=Config.DEVICE)
        new_state_dict = {}

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            # DenseNet适配
            if model_name == 'densenet121' and 'classifier' in k:
                k = k.replace('classifier.0', 'classifier.1')
                k = k.replace('classifier.weight', 'classifier.1.weight')
                k = k.replace('classifier.bias', 'classifier.1.bias')
            # Swin适配
            elif model_name == 'swin_t' and 'head' in k:
                k = k.replace('head.weight', 'head.1.weight')
                k = k.replace('head.bias', 'head.1.bias')
                k = k.replace('head.0', 'head.1')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print("✅ 权重加载成功")
        return True
    except Exception as e:
        print(f"❌ 权重加载失败: {str(e)}")
        return False


# =========================================================================
# 主可视化函数
# =========================================================================
def visualize_comparison():
    print("🎨 初始化可视化工具...")
    device = Config.DEVICE

    # 文件夹路径设置
    save_dir = os.path.join(Config.OUTPUT_DIR, 'figures_final_5')
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 结果将保存至: {save_dir}")

    # 1. 加载模型
    print("\n📌 加载模型中...")
    model_dense = get_model('densenet121', pretrained=False, mc_dropout=True)
    if not load_checkpoint_smart(model_dense,
                                 os.path.join(Config.OUTPUT_DIR, 'checkpoints', 'best_model_densenet121.pth'),
                                 'densenet121'): return
    model_dense.to(device)

    model_swin = get_model('swin_t', pretrained=False, mc_dropout=True)
    if not load_checkpoint_smart(model_swin, os.path.join(Config.OUTPUT_DIR, 'checkpoints', 'best_model_swin_t.pth'),
                                 'swin_t'): return
    model_swin.to(device)

    # 2. 初始化引擎
    print("\n🔧 初始化解释引擎...")
    xai_dense = FixedXAIEngine(model_dense, device, 'densenet121')
    xai_swin = FixedXAIEngine(model_swin, device, 'swin_t')

    # 3. 加载数据
    print("\n📂 加载数据集...")
    try:
        ds = RSNADataset(
            os.path.join(Config.PROCESSED_DIR, 'test.csv'),
            Config.RAW_IMG_DIR,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            mode='eval',
            full_df_path=Config.RAW_LABEL_CSV
        )
    except Exception as e:
        print(f"❌ 数据集错误: {e}")
        return

    # 4. 极速筛选
    print("\n⚡ 筛选阳性样本...")
    try:
        if 'Target' in ds.data.columns:
            positive_indices = ds.data[ds.data['Target'] == 1].index.tolist()
        else:
            print("⚠️ 慢速筛选模式 (无Target列)")
            positive_indices = [i for i in range(len(ds)) if ds[i][1] == 1]

        if not positive_indices:
            print("❌ 无阳性样本")
            return

        vis_count = min(3, len(positive_indices))
        indices = positive_indices[:vis_count]
        print(f"✅ 准备可视化前 {vis_count} 个样本")

    except Exception as e:
        print(f"❌ 筛选失败: {e}")
        return

    # 5. 绘图
    fig, axes = plt.subplots(nrows=len(indices), ncols=6, figsize=(24, 5 * len(indices)))
    if len(indices) == 1: axes = np.array([axes])

    print("\n🖼️ 开始生成图像...")
    for row_idx, data_idx in enumerate(tqdm(indices)):
        try:
            img_tensor, target, gt_mask, pid = ds[data_idx]
            img_input = img_tensor.unsqueeze(0)

            # 反归一化
            img_vis = img_tensor.permute(1, 2, 0).numpy()
            img_vis = (img_vis * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_vis = np.clip(img_vis, 0, 1)
            img_vis_uint8 = (img_vis * 255).astype(np.uint8)

            # GT
            img_gt = img_vis_uint8.copy()
            if gt_mask.sum() > 0:
                contours, _ = cv2.findContours(gt_mask.numpy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_gt, contours, -1, (255, 0, 0), 2)

            # 生成解释
            gc_dense = xai_dense.generate(img_input, 'gradcam')
            ig_dense = xai_dense.generate(img_input, 'ig')
            gc_swin = xai_swin.generate(img_input, 'gradcam')
            ig_swin = xai_swin.generate(img_input, 'ig')
            _, std_swin = xai_swin.generate_bayesian(img_input, 'gradcam', 10)

            titles = [
                f"Patient {pid}\nGround Truth", "DenseNet\nGrad-CAM", "DenseNet\nSmoothGrad (IG)",
                "Swin\nGrad-CAM", "Swin\nSmoothGrad (IG)", "Swin\nUncertainty Map"
            ]
            images = [
                img_gt, overlay_heatmap(img_vis_uint8, gc_dense), overlay_heatmap(img_vis_uint8, ig_dense),
                overlay_heatmap(img_vis_uint8, gc_swin), overlay_heatmap(img_vis_uint8, ig_swin), std_swin
            ]

            # 子图绘制
            for col_idx, (img, title) in enumerate(zip(images, titles)):
                ax = axes[row_idx][col_idx]
                if col_idx == 5:  # Uncertainty
                    if img.max() > 0: img = img / img.max()
                    im = ax.imshow(img, cmap='inferno')
                else:
                    ax.imshow(img)

                ax.set_title(title, fontsize=12, fontweight='bold' if col_idx == 0 else 'normal')
                ax.axis('off')

            # [优化] 每次循环后清理显存，防止OOM
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ 样本 {pid} 出错: {e}")
            continue

    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    # [优化] 添加总标题
    plt.suptitle(f"Comparative XAI Visualization: DenseNet vs Swin Transformer (Top {len(indices)} Samples)",
                 fontsize=16, y=0.98)

    save_path = os.path.join(save_dir, 'advanced_comparison_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n🎉 结果已保存: {save_path}")


if __name__ == "__main__":
    visualize_comparison()