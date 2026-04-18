from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.utils.eval_stats import EvaluationStats


class BaseTrainer(ABC):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
            device: str="cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

        if isinstance(criterion, nn.Module):
            self.criterion.to(self.device)
        
        self.stats = EvaluationStats()

        self.n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abstractmethod 
    def save_stats(self) -> None:
        raise NotImplementedError

    def save_model(self) -> None:
        torch.save(self.model.state_dict(), self.model.save_name()) 