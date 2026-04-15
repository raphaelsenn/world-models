from abc import ABC, abstractmethod

import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils.eval_stats import EvaluationStats


class BaseTrainer(ABC):
    def __init__(
            self, 
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
            device: str="cpu",
            eval_every: int=1,
            save_every: int=1,
            verbose: bool=True
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # Compute model params
        self.n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.eval_every = eval_every
        self.save_every = save_every
        self.verbose = verbose
        
        self.model_name = "none"
        self.stats = EvaluationStats()

    @abstractmethod
    def train(self, dataloader: DataLoader) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> float:
        raise NotImplementedError

    def handle_periodic_tasks(
        self,
        step: int,
        trainval_loader: DataLoader | None,
        val_loader: DataLoader | None,
    ) -> None:
        did_eval = False

        if step % self.eval_every == 0:
            self.stats.append_step(step)

            if trainval_loader is not None:
                train_eval_loss = self.evaluate(trainval_loader)
                self.stats.append_train(train_eval_loss)

            if val_loader is not None:
                val_eval_loss = self.evaluate(val_loader)
                self.stats.append_val(val_eval_loss)

            self.model.train()
            did_eval = True

        if did_eval and self.verbose:
            msg = f"Step: {step:10d}"

            if trainval_loader is not None:
                msg += f"\tTrain eval loss: {self.stats.last_train_loss:10.4f}"

            if val_loader is not None:
                msg += f"\tVal loss: {self.stats.last_val_loss:10.4f}"

            print(msg)

        if step % self.save_every == 0:
            name = self.model_name
            self.save_model(name)
            self.save_stats(name.replace(".pt", ".csv"))

    def save_model(self, file_name: str) -> None:
        torch.save(self.model.state_dict(), file_name)
    
    def save_stats(self, file_name: str) -> None:
        if len(self.stats) == 0: return 
        data = {
            "step": self.stats.step, 
            "train_loss": self.stats.train_loss,
            "val_loss" : self.stats.val_loss
        }
        pd.DataFrame.from_dict(data).to_csv(file_name, index=False)


