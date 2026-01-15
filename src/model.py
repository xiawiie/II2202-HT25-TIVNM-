import torch
import torch.nn as nn
from torchvision import models


def _modify_classifier(base_model, model_type, dropout_rate=0.2):
    """
    内部工具函数：修改模型的分类头以支持 MC Dropout
    """
    if model_type == 'densenet121':
        in_features = base_model.classifier.in_features
        # 替换原有分类器：Dropout -> Linear
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1)
        )

    elif model_type == 'swin_t':
        in_features = base_model.head.in_features
        # Swin 的分类头名为 head
        base_model.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1)
        )

    return base_model


def get_model(model_name='densenet121', pretrained=True, mc_dropout=False):
    """
    模型工厂函数
    :param mc_dropout: 是否开启蒙特卡洛 Dropout (用于贝叶斯推断)
    """
    weights = 'DEFAULT' if pretrained else None
    dropout_p = 0.2 if mc_dropout else 0.0

    # 1. 实例化基座模型
    if model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        model.model_name = 'densenet121'  # 注入属性方便XAI识别

    elif model_name == 'swin_t':
        model = models.swin_t(weights=weights)
        model.model_name = 'swin_t'

    else:
        raise ValueError(f"不支持的模型架构: {model_name}")

    # 2. 修改分类头 (无论是否 MC Dropout，我们都重建头以确保输出维度为1)
    # 如果 mc_dropout=False，dropout_p为0，相当于没有 Dropout，逻辑统一
    model = _modify_classifier(model, model_name, dropout_rate=dropout_p)

    return model