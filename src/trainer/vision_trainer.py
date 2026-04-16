import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.trainer.base_trainer import BaseTrainer
from src.utils.eval_stats import EvaluationStats


class VisionTrainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
            epochs: int,
            device: str="cpu",
            eval_every: int=10,
            save_every: int=10,
            verbose: bool=True
    ) -> None:
        super().__init__(
            model, 
            criterion, 
            optimizer, 
            device, 
            eval_every, 
            save_every, 
            verbose
        )

        self.epochs = epochs
        self.model_name = "vae-img64-z32.pt"

    def train(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader | None = None
    ) -> None:
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            for x_img in train_loader:
                batch_size = x_img.size(0) 
                
                x_img = x_img.to(self.device)
                x_recon, kl = self.model(x_img)
                
                loss = self.criterion(x_img, x_recon, kl)
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.handle_periodic_tasks(epoch + 1, total_loss / total_samples, val_loader) 
        
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for x_img in dataloader:
            x_img = x_img.to(self.device)
            batch_size = x_img.size(0)

            x_recon, kl = self.model(x_img)
            loss = self.criterion(x_img, x_recon, kl)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Dataloader is empty.")

        return total_loss / total_samples