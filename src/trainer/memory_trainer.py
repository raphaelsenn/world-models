import math

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pandas as pd

from src.trainer.base_trainer import BaseTrainer


class MemoryTrainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
            epochs: int,
            device: str="cpu",
            eval_every: int=1,
            save_every: int=1,
            verbose: bool=True,
    ) -> None:
        super().__init__(
            model, 
            criterion, 
            optimizer, 
            device, 
        )

        self.epochs = epochs
        self.eval_every = eval_every
        self.save_every = save_every
        self.verbose = verbose

    def train_one_epoch(self, dataloader: DataLoader) -> None:
        self.model.train() 
        
        for latents, actions in dataloader:
            latents = latents.to(self.device)   # [B, L, z_dim] 
            actions = actions.to(self.device)   # [B, L, action_dim]

            input_latents = latents[:, :-1, :]  # [B, L - 1, z_dim],    z_1, z_2, ... z_L-1
            input_actions = actions[:, :-1, :]  # [B, L - 1, 3],        a_1, a_2, ..., a_L-1
            targets = latents[:, 1:, :]         # [B, L - 1, z_dim],    z_2, z_3, ..., z_L

            pi_logits, mu, std = self.model(input_latents, input_actions)
            loss = self.criterion(pi_logits, mu, std, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(
            self, 
            train_loader: DataLoader, 
            trainval_loader: DataLoader | None = None,
            val_loader: DataLoader | None = None
    ) -> None:
        """
        Args:
        train_loader    : Full training data
        trainval_loader : Subset of training data for evaluation - since train_loader might be large
        val_loader      : Validation data (data the model never has seen before)
        """ 
        for epoch in range(self.epochs):
            self.train_one_epoch(train_loader)
            self.handle_periodic_tasks(epoch + 1, trainval_loader, val_loader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval() 

        total_samples = 0
        total_loss = 0.0

        for latents, actions in dataloader:
            latents = latents.to(self.device)
            actions = actions.to(self.device)

            input_latents = latents[:, :-1, :]  # [B, L - 1, z_dim],    z_1, z_2, ... z_L-1
            input_actions = actions[:, :-1, :]  # [B, L - 1, 3],        a_1, a_2, ..., a_L-1
            targets = latents[:, 1:, :]         # [B, L - 1, z_dim],    z_2, z_3, ..., z_L

            pi_logits, mu, std = self.model(input_latents, input_actions) 

            loss = self.criterion(pi_logits, mu, std, targets)

            batch_size = latents.size(0) 
            pred_len = targets.size(1)
            total_loss += loss.item() * batch_size * pred_len
            total_samples += batch_size * pred_len

        average_loss = total_loss / total_samples

        return average_loss

    def save_stats(self) -> None:
        save_name: str = self.model.save_name()
        save_name = save_name.replace(".pt", "_report.csv")
        if len(self.stats) == 0: return 
        data = {
            "step": self.stats.step, 
            "train_loss": self.stats.train_loss,
            "val_loss" : self.stats.val_loss
        }
        pd.DataFrame.from_dict(data).to_csv(save_name, index=False)
    
    def handle_periodic_tasks(
        self,
        epoch: int,
        trainval_loader: DataLoader | None,
        val_loader: DataLoader | None,
    ) -> None:
        did_eval = False
        msg = f"Epoch: {epoch:10d}\t"

        if epoch % self.eval_every == 0:
            if trainval_loader is not None:
                train_loss = self.evaluate(trainval_loader)
                msg += f"Train loss: {train_loss:10.6f}\t" 
                did_eval = True
            else:
                train_loss = train_loss if trainval_loader is not None else math.nan

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                msg += f"Val loss: {val_loss:10.6f}\t" 
                did_eval = True
            else: 
                val_loss = val_loss if val_loader is not None else math.nan
            
            self.stats.append_step(epoch)
            self.stats.append_val(val_loss)
            self.stats.append_train(train_loss)

        if did_eval and self.verbose:
            print(msg)

        if epoch % self.save_every == 0:
            self.save_model()
            self.save_stats()