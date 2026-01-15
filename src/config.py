import os
import torch


class Config:
    # --- 路径设置 ---
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    RAW_IMG_DIR = os.path.join(ROOT_DIR, 'stage_2_train_images')
    RAW_LABEL_CSV = os.path.join(ROOT_DIR, 'stage_2_train_labels.csv')

    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')

    # --- 训练超参数 ---
    SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- XAI 评估超参数 ---
    IOU_THRESHOLD = 0.15  # 热力图二值化阈值
    EVAL_SAMPLES = 200  # 评估样本数量 (设为 -1 则跑全量)
    AUDIT_STEPS = 50  # Deletion Game 的步数


# 自动创建目录
for d in [Config.PROCESSED_DIR,
          os.path.join(Config.OUTPUT_DIR, 'checkpoints'),
          os.path.join(Config.OUTPUT_DIR, 'results')]:
    os.makedirs(d, exist_ok=True)


# class Config:
#     # --- 1. 自动获取项目根目录 (避免硬编码 '.') ---
#     # 获取 src/config.py 的上一级(src)的上一级(项目根目录)
#     ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
#     # --- 2. 路径设置 ---
#     # 数据存放目录
#     DATA_DIR = os.path.join(ROOT_DIR, 'data')
#
#     # 原始数据路径 (请确保这些文件夹真实存在)
#     RAW_IMG_DIR = os.path.join(ROOT_DIR, 'stage_2_train_images')
#     RAW_LABEL_CSV = os.path.join(ROOT_DIR, 'stage_2_train_labels.csv')
#
#     # 处理后数据路径
#     PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
#     OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
#     MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'checkpoints', 'best_model.pth')
#
#     # --- 3. 超参数 ---
#     SEED = 42
#     IMG_SIZE = 224
#     BATCH_SIZE = 32
#     LEARNING_RATE = 1e-4
#     NUM_EPOCHS = 10
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # --- 4. XAI 设置 ---
#     IOU_THRESHOLD = 0.15  # 热力图二值化阈值
#
#
# # 自动创建目录
# os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
# os.makedirs(os.path.join(Config.OUTPUT_DIR, 'checkpoints'), exist_ok=True)
# os.makedirs(os.path.join(Config.OUTPUT_DIR, 'results'), exist_ok=True)