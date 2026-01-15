import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Global Seed set to {seed}")

def enable_dropout(model):
    """
    [核心优化] 强制开启模型中的 Dropout 层，但在推理时保持 BatchNorm 层为 Eval 模式。
    这是实现 Monte Carlo Dropout (Bayesian Inference) 的关键。
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()