from abc import ABC, abstractmethod
import torch

class BaseExplainer(ABC):
    """XAI 解释器基类"""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def generate(self, input_tensor: torch.Tensor) -> tuple:
        pass

class BaseMetric(ABC):
    """评估指标基类"""
    pass