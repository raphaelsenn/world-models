import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.trainer.base_trainer import BaseTrainer


class MemoryTrainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
            epochs: int,
            device: str="cpu",
            eval_every: int=1,  # Epochs (not gradient steps)
            save_every: int=1,  # Epochs (not gradient steps)
            verbose: bool=True,
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
        self.model_name = "rdn-rnn-z32-h256.pt"

    def train(
            self, 
            train_loader: DataLoader, 
            trainval_loader: DataLoader | None = None,
            val_loader: DataLoader | None = None
    ) -> None:
        for epoch in range(self.epochs):
            self.model.train() 
            
            for latents, actions in train_loader:
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

            self.handle_periodic_tasks(epoch + 1, trainval_loader, val_loader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval() 

        total_samples = 0
        total_loss = 0.0

        for latents, actions in dataloader:
            batch_size = latents.size(0) 
            seq_len = latents.size(1)

            latents = latents.to(self.device)
            actions = actions.to(self.device)

            pi_logits, mu, std = self.model(latents, actions) 

            loss = self.criterion(pi_logits, mu, std, targets=latents)

            total_loss += loss.item() * batch_size * seq_len
            total_samples += batch_size * seq_len

        average_loss = total_loss / total_samples

        return average_loss