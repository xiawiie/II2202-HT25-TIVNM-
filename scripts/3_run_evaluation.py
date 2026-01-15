import sys
import os
import argparse
import logging
import torch
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast

# --- 0. 环境适配 Environment Setup ---
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# --- 路径适配 Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import Config
from src.dataset import RSNADataset
from src.model import get_model
# Don't import the buggy engine, we will fix it or wrap it
from src.xai_engine import AdvancedXAIEngine
from src.metrics import calculate_iou, calculate_agreement, FidelityAuditor, calculate_pointing_game
from captum.attr import LayerGradCam, IntegratedGradients, NoiseTunnel  # Import explicitly for fix

# --- 性能核心设置 Performance Settings ---
torch.backends.cudnn.benchmark = True
# Allow TF32 for newer GPUs (RTX 30xx/40xx)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- 日志配置 Logging ---
log_dir = os.path.join(Config.OUTPUT_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)
root_logger = logging.getLogger()
root_logger.handlers = []

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, 'evaluation_process.log'), mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# --- 🛠️ FIX: Define a Fixed XAI Engine locally to override the bug ---
class FixedXAIEngine(AdvancedXAIEngine):
    """
    Repairing the reshape_transform bug in the original engine.
    For Swin, we handle the permutation manually after attribute generation.
    """

    def __init__(self, model, device, model_name=None):
        # Bypass parent __init__ to avoid the crash
        # Re-implementing necessary parts of __init__ from your provided context
        self.model = model
        self.device = device
        self.model_name = model_name or getattr(model, 'model_name', 'unknown')
        self.model.eval()

        # 1. Detect Target Layer
        if 'densenet' in self.model_name:
            # features -> denseblock4 -> denselayer16 -> conv2
            self.target_layer = self.model.features.denseblock4.denselayer16.conv2
        elif 'swin' in self.model_name:
            # features -> 7 (stage 4) -> 1 (block) -> norm1
            # Note: Swin output at norm1 is usually [B, H, W, C]
            self.target_layer = self.model.features[-1][-1].norm1
        else:
            self.target_layer = list(self.model.children())[-1]

        # 2. Init Explainers (THE FIX: No reshape_transform in __init__)
        try:
            self.grad_cam = LayerGradCam(self.model, self.target_layer)
        except Exception as e:
            logger.error(f"Grad-CAM Init Failed: {e}")
            self.grad_cam = None

        self.ig = IntegratedGradients(self.model)
        self.nt = NoiseTunnel(self.ig)

    def _generate_single_pass(self, input_tensor, method='gradcam'):
        """Overriding generation to handle Swin shapes correctly"""
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)
        if not input_tensor.requires_grad:
            input_tensor.requires_grad = True

        if method == 'gradcam':
            if self.grad_cam is None: return np.zeros(input_tensor.shape[2:])

            # Compute attributions
            # relu_attributions=True for GradCAM
            attr = self.grad_cam.attribute(input_tensor, relu_attributions=True)

            # --- SWIN SHAPE FIX ---
            # If Swin, attr might be [1, 7, 7, 768] (B, H, W, C)
            # We need [1, 768, 7, 7] (B, C, H, W) for interpolation
            if 'swin' in self.model_name and attr.dim() == 4:
                if attr.shape[-1] > attr.shape[1]:  # Heuristic: C is likely last
                    attr = attr.permute(0, 3, 1, 2)

            # Interpolate to image size
            attr = LayerGradCam.interpolate(attr, (input_tensor.shape[2], input_tensor.shape[3]),
                                            interpolate_mode='bilinear')

            # Detach
            map_ = attr.detach().cpu().numpy()[0, 0]

        elif method == 'ig':
            # IG logic remains similar
            try:
                attr = self.nt.attribute(input_tensor, nt_type='smoothgrad', nt_samples=2, target=0)
            except:
                attr = self.ig.attribute(input_tensor, n_steps=15, target=0)
            map_ = np.sum(np.abs(attr.detach().cpu().numpy()[0]), axis=0)
        else:
            return np.zeros(input_tensor.shape[2:])

        # Normalize
        min_v, max_v = map_.min(), map_.max()
        if max_v - min_v > 1e-8:
            map_ = (map_ - min_v) / (max_v - min_v)
        else:
            map_ = np.zeros_like(map_)
        return map_


def parse_args():
    parser = argparse.ArgumentParser(description="RSNA XAI 极速评估脚本 (RTX 5060 Optimized)")
    parser.add_argument('--model', type=str, default='all', choices=['densenet121', 'swin_t', 'all'])
    parser.add_argument('--limit', type=int, default=0, help="0=全量")

    # 显存安全批次
    parser.add_argument('--audit_batch_size', type=int, default=32, help="Deletion Game 推理批次")

    # 极速参数
    parser.add_argument('--steps', type=int, default=15, help="审计步数")
    parser.add_argument('--samples', type=int, default=5, help="贝叶斯采样次数")

    parser.add_argument('--num_workers', type=int, default=4, help="数据读取线程")
    parser.add_argument('--save_freq', type=int, default=50, help="保存频率")

    return parser.parse_args()


def load_checkpoint_smart(model, model_name):
    """智能权重加载"""
    ckpt_path = os.path.join(Config.OUTPUT_DIR, 'checkpoints', f'best_model_{model_name}.pth')
    if not os.path.exists(ckpt_path):
        logger.warning(f"⚠️ 缺权重: {ckpt_path}，随机初始化(仅测试)！")
        return

    try:
        state_dict = torch.load(ckpt_path, map_location=Config.DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            # DenseNet Mapping
            if model_name == 'densenet121' and 'classifier' in k:
                k = k.replace('classifier.0', 'classifier.1').replace('classifier.weight',
                                                                      'classifier.1.weight').replace('classifier.bias',
                                                                                                     'classifier.1.bias')
            # Swin Mapping
            elif model_name == 'swin_t' and 'head' in k:
                k = k.replace('head.weight', 'head.1.weight').replace('head.bias', 'head.1.bias').replace('head.0',
                                                                                                          'head.1')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"✅ 权重载入: {model_name}")
    except Exception as e:
        logger.error(f"❌ 权重失败: {e}")


def evaluate_single_model(model_name, args):
    logger.info(f"\n🚀 极速评估: {model_name} | Steps={args.steps} | Samples={args.samples}")

    # 1. 模型
    model = get_model(model_name, pretrained=False, mc_dropout=True)
    load_checkpoint_smart(model, model_name)
    model.to(Config.DEVICE)
    model.eval()

    # 2. 引擎 (Use FIXED Engine)
    # Replaced AdvancedXAIEngine with FixedXAIEngine to fix the bug
    engine = FixedXAIEngine(model, Config.DEVICE, model_name)
    auditor = FidelityAuditor(model, Config.DEVICE)

    # 3. 数据
    test_csv_path = os.path.join(Config.PROCESSED_DIR, 'test.csv')
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_ds = RSNADataset(test_csv_path, Config.RAW_IMG_DIR, transform=transform, mode='eval',
                          full_df_path=Config.RAW_LABEL_CSV)

    if 'Target' in full_ds.data.columns:
        full_ds.data['Target'] = pd.to_numeric(full_ds.data['Target'], errors='coerce').fillna(0).astype(int)
        pos_indices = full_ds.data[full_ds.data['Target'] == 1].index.tolist()
    else:
        pos_indices = list(range(len(full_ds)))

    if args.limit > 0:
        pos_indices = pos_indices[:args.limit]
        logger.info(f"⚡ Limit: {len(pos_indices)}")
    else:
        logger.info(f"🔥 Full: {len(pos_indices)}")

    # 4. DataLoader
    subset = Subset(full_ds, pos_indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    results = []
    output_file = os.path.join(Config.OUTPUT_DIR, 'results', f'audit_{model_name}.csv')

    start_time = time.time()
    pbar = tqdm(loader, desc=f"Run {model_name}", ncols=100)

    METHOD_GC = 'gradcam'
    METHOD_IG = 'ig'

    for i, batch_data in enumerate(pbar):
        try:
            img_tensor, target, gt_mask_tensor, pid_tuple = batch_data
            img_input = img_tensor.to(Config.DEVICE, non_blocking=True)
            pid = pid_tuple[0]
            gt_np = gt_mask_tensor[0].numpy()

            if gt_np.sum() == 0: continue

            # --- 🚀 Accelerated Execution ---
            with autocast():
                # A. Explain
                gc_mean, gc_std = engine.generate_bayesian(img_input, METHOD_GC, args.samples)
                ig_mean, ig_std = engine.generate_bayesian(img_input, METHOD_IG, num_samples=3)

                # B. Audit
                auc_gc, _ = auditor.compute_auc(
                    img_input, gc_mean, 'deletion',
                    steps=args.steps,
                    batch_size=args.audit_batch_size
                )
                auc_ig, _ = auditor.compute_auc(
                    img_input, ig_mean, 'deletion',
                    steps=args.steps,
                    batch_size=args.audit_batch_size
                )

            # C. Metrics
            res = {
                'patientId': pid, 'model': model_name,
                'iou_gc': calculate_iou(gc_mean, gt_np),
                'hit_gc': calculate_pointing_game(gc_mean, gt_np),
                'fidelity_gc': auc_gc,
                'uncertainty_gc': gc_std[gc_mean > 0.2].mean() if (gc_mean > 0.2).any() else 0.0,
                'iou_ig': calculate_iou(ig_mean, gt_np),
                'fidelity_ig': auc_ig,
                'agreement': calculate_agreement(gc_mean, ig_mean)
            }
            results.append(res)

            if (i + 1) % args.save_freq == 0:
                pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8-sig')
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Err {pid}: {e}")
            continue

    if results:
        pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8-sig')
        total_m = (time.time() - start_time) / 60
        logger.info(f"✅ Done. Time: {total_m:.1f}m. Saved to {output_file}")
    else:
        logger.warning("❌ No results.")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.join(Config.OUTPUT_DIR, 'results'), exist_ok=True)
    # models = ['densenet121', 'swin_t'] if args.model == 'all' else [args.model]
    models = ['swin_t'] if args.model == 'all' else [args.model]
    for m in models: evaluate_single_model(m, args)