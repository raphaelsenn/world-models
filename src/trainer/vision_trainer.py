import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import gymnasium as gym

from src.world_model.vision import ConvVAE
from src.loss import VisionLoss
from src.data.vision_buffer import VisionBuffer
from src.trainer.base_trainer import BaseTrainer
from src.utils.prepro import preprocess_observation


class VisionTrainer(BaseTrainer):
    def __init__(
            self,
            model: ConvVAE,
            epochs: int,
            in_channels: int=3,
            n_timesteps: int=10_000_000,
            horizon: int=100_000,
            learning_rate: float=0.001,
            kl_weight: float=0.0001,
            batch_size: int=64,
            device: str="cpu",
            n_workers: int=0,
            pin_memory: bool=False, 
            seed: int=0,
            verbose: bool=True
    ) -> None:
        super().__init__(
            model, 
            device, 
        )

        self.criterion = VisionLoss(kl_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buffer = VisionBuffer(in_channels, horizon)

        self.in_channels = in_channels
        self.epochs = epochs
        self.n_timesteps = n_timesteps
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight 
        self.horizon = horizon
        self.batch_size = batch_size

        self.n_workers = n_workers 
        self.pin_memory = pin_memory 
        self.seed = seed
        self.verbose = verbose

        self.obs_ = None

    def collect_data(self, env: gym.Env) -> None:
        for _ in range(0, self.horizon):
            obs = preprocess_observation(self.obs_)
            
            action = env.action_space.sample()
            obs_nxt, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.buffer.push(obs)
            self.obs_ = obs_nxt

            if done:
                self.obs_, _ = env.reset()

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train() 
        
        total_loss = 0.0 
        for x_img in dataloader:
            x_img = x_img[0].to(self.device)
            batch_size = x_img.size(0) 
            
            x_recon, kl = self.model(x_img)

            loss = self.criterion(x_img, x_recon, kl)
            total_loss += float(batch_size * loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        mean_loss = total_loss / self.horizon
        
        return mean_loss

    def train_n_epochs(self, dataloader: DataLoader, step: int) -> None:
        total_loss = 0.0 
        for epoch in range(self.epochs):
            total_loss += self.train_one_epoch(dataloader)

        mean_loss = total_loss / self.epochs
        self.stats.append_step(step)
        self.stats.append_train(mean_loss)

    def train(
            self,
            env: gym.Env,
    ) -> None:
        env.action_space.seed(seed=self.seed)
        self.obs_, _ = env.reset(seed=self.seed)
        
        for step in range(1, self.n_timesteps + 1, self.horizon):
            self.collect_data(env)
            dataloader = self._make_dataloader()
            self.train_n_epochs(dataloader, step)

            self.save_model()
            self.save_stats()

            if self.verbose:
                print(f"T: {step:9d}\tLoss: {self.stats.last_train_loss:10.8f}")

    def save_stats(self) -> None:
        save_name: str = self.model.save_name()
        save_name = save_name.replace(".pt", "_report.csv")
        data = {"step": self.stats.step, "train_loss": self.stats.train_loss}
        pd.DataFrame.from_dict(data).to_csv(save_name, index=False)
    
    def _make_dataloader(self) -> DataLoader:
        dataset = TensorDataset(self.buffer.dataset()) 
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory
        ) 
        return dataloader