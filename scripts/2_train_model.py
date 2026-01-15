import sys
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.config import Config
from src.utils import seed_everything
from src.dataset import RSNADataset
from src.model import get_model


# --- 配置日志 (优化版) ---
def setup_logging():
    # 确保日志目录存在
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志，使用UTF-8编码避免中文/特殊字符问题
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log"), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 明确指定stdout避免编码问题
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class TrainingConfig:
    # 模型选择: 'densenet121' 或 'swin_t'
    MODEL_NAME = 'swin_t'

    # 训练参数
    EPOCHS = 10  # 总训练轮数 (有早停，设大点没事)
    BATCH_SIZE = 32  # 显存不够就改小，比如 16
    LR = 5e-5  # 学习率

    # 数据控制
    DATA_PCT = 0.5  # 使用 50% 的数据进行训练
    DATA_LIMIT = 0  # 强制限制使用多少张图片 (例如 100)，用于快速测试。设为 0 则不启用

    # 显存优化
    ACCUM_ITER = 1  # 梯度累积步数
    PATIENCE = 8  # 早停耐心值

    # 日志控制
    LOG_INTERVAL = 10  # 每多少个batch打印一次训练信息


class EarlyStopping:
    """优化版早停机制：增加更多状态信息和灵活性"""

    def __init__(self, patience=5, delta=0, verbose=True, path='checkpoint.pth', metric='loss', mode='min'):
        """
        Args:
            patience: 容忍多少个epoch没有改进
            delta: 最小改进值，小于此值视为没有改进
            verbose: 是否打印信息
            path: 模型保存路径
            metric: 监控的指标 ('loss' 或 'auc'等)
            mode: 优化模式 ('min' 表示指标越小越好，'max'表示越大越好)
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.metric = metric
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        # 根据模式初始化最佳值
        if mode == 'min':
            self.best_value = np.Inf
        else:  # max
            self.best_value = -np.Inf

    def __call__(self, current_value, model, epoch):
        # 根据模式计算分数
        if self.mode == 'min':
            score = -current_value
        else:
            score = current_value

        if self.best_score is None:
            self.best_score = score
            self.best_value = current_value
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_value = current_value
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.verbose:
            logger.info(f"Best {self.metric} improved ({self.best_value:.6f}). Saving model to {self.path}")
        # 确保保存目录存在
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)

    def get_best_info(self):
        """返回最佳状态信息"""
        return {
            'best_epoch': self.best_epoch,
            'best_value': self.best_value,
            'metric': self.metric,
            'mode': self.mode
        }


def parse_args():
    parser = argparse.ArgumentParser(description="RSNA Pneumonia Training Script")
    parser.add_argument('--model', type=str, default=TrainingConfig.MODEL_NAME, choices=['densenet121', 'swin_t'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=TrainingConfig.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=TrainingConfig.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=TrainingConfig.LR, help='Learning rate')
    parser.add_argument('--accum_iter', type=int, default=TrainingConfig.ACCUM_ITER, help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=TrainingConfig.PATIENCE, help='Early stopping patience')
    return parser.parse_args()


def train(args):
    seed_everything(Config.SEED)
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # 配置参数
    data_pct = TrainingConfig.DATA_PCT
    data_limit = TrainingConfig.DATA_LIMIT
    log_interval = TrainingConfig.LOG_INTERVAL

    # 打印训练配置信息
    logger.info("=" * 50)
    logger.info("Training Configuration:")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr:.2e}")
    logger.info(f"Gradient accumulation: {args.accum_iter}")
    logger.info(f"Data percentage: {data_pct * 100}%")
    logger.info(f"Early stopping patience: {args.patience}")
    logger.info("=" * 50)

    # 1. 准备数据
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_csv = os.path.join(Config.PROCESSED_DIR, 'train.csv')
    full_train_ds = RSNADataset(train_csv, Config.RAW_IMG_DIR, transform=train_transform)
    full_val_ds = RSNADataset(os.path.join(Config.PROCESSED_DIR, 'val.csv'), Config.RAW_IMG_DIR,
                              transform=val_transform)

    # 子集逻辑
    train_ds, val_ds = full_train_ds, full_val_ds

    # 应用数据比例限制
    if data_pct < 1.0:
        train_indices = list(range(len(full_train_ds)))
        random.shuffle(train_indices)
        train_size = int(len(full_train_ds) * data_pct)
        train_ds = Subset(full_train_ds, train_indices[:train_size])

        val_indices = list(range(len(full_val_ds)))
        random.shuffle(val_indices)
        val_size = int(len(full_val_ds) * data_pct)
        val_ds = Subset(full_val_ds, val_indices[:val_size])

    # 应用数据数量限制
    if data_limit > 0:
        train_indices = list(range(len(train_ds)))
        random.shuffle(train_indices)
        train_ds = Subset(train_ds, train_indices[:data_limit])

        val_indices = list(range(len(val_ds)))
        random.shuffle(val_indices)
        val_ds = Subset(val_ds, val_indices[:int(data_limit * 0.2)])

    logger.info(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}")

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 2. 计算类别权重
    logger.info("Calculating class weights...")
    try:
        if isinstance(train_ds, Subset):
            targets = full_train_ds.data['Target'].iloc[train_ds.indices].values
        else:
            targets = full_train_ds.data['Target'].values

        num_pos = np.sum(targets)
        num_neg = len(targets) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos]).to(device)
        logger.info(
            f"Class distribution - Positive: {num_pos}, Negative: {num_neg}, Ratio: {num_pos / len(targets):.2%}")
    except Exception as e:
        logger.warning(f"Failed to calculate class weights: {str(e)}. Using default weight of 1.0")
        pos_weight = torch.tensor([1.0]).to(device)

    logger.info(f"Using positive weight: {pos_weight.item():.2f}")

    # 3. 模型与优化器
    model = get_model(model_name=args.model, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 混合精度训练 (修复弃用警告)
    scaler = torch.amp.GradScaler('cuda')  # 改为新的API

    # 早停初始化 (监控AUC，越大越好)
    save_path = os.path.join(Config.OUTPUT_DIR, 'checkpoints', f'best_model_{args.model}.pth')
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=save_path,
        metric='auc',  # 改为监控AUC更合理
        mode='max'  # AUC越大越好
    )

    # 4. 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        batch_count = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (imgs, targets) in enumerate(loop):
            imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
            batch_count += 1

            # 混合精度训练 (修复弃用警告)
            with torch.amp.autocast('cuda'):  # 改为新的API
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss = loss / args.accum_iter  # 梯度累积标准化

            scaler.scale(loss).backward()

            # 梯度累积更新
            if ((i + 1) % args.accum_iter == 0) or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * args.accum_iter
            loop.set_postfix(loss=train_loss / batch_count)

            # 定期打印训练信息
            if (i + 1) % log_interval == 0:
                logger.debug(f"Batch {i + 1}/{len(train_loader)}, Current Loss: {loss.item() * args.accum_iter:.4f}")

        # 验证循环
        model.eval()
        val_loss = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)

                with torch.amp.autocast('cuda'):  # 改为新的API
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())

        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        try:
            val_auc = roc_auc_score(all_targets, all_preds)
            val_acc = accuracy_score(all_targets, (all_preds > 0.5).astype(int))
            val_f1 = f1_score(all_targets, (all_preds > 0.5).astype(int))
        except Exception as e:
            logger.warning(f"Failed to calculate metrics: {str(e)}")
            val_auc = 0.5
            val_acc = 0
            val_f1 = 0

        curr_lr = scheduler.get_last_lr()[0]

        # 打印 epoch 总结
        logger.info("\n" + "=" * 50)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} Results:")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        logger.info(f"Learning Rate: {curr_lr:.2e}")
        logger.info("=" * 50 + "\n")

        scheduler.step()

        # 早停检查 (使用AUC作为判断指标)
        early_stopping(val_auc, model, epoch + 1)
        if early_stopping.early_stop:
            best_info = early_stopping.get_best_info()
            logger.info(
                f"Early stopping triggered! Best {best_info['metric']} at epoch {best_info['best_epoch']}: {best_info['best_value']:.4f}")
            break

    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        args = parse_args()
        train(args)
    except Exception as e:
        logger.error("Training failed with exception:", exc_info=True)
        raise