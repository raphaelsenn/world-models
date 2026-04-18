from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder parametrized by a convolutional neural net.

    Reference:
    ---------- 
    World Models, Ha and Schmidhuber 2018.
    https://arxiv.org/abs/1803.10122
    """   
    def __init__(self, n_channels: int, z_dim: int) -> None:
        super().__init__()
        assert n_channels > 0, f"in_channels should be > 0, got: {n_channels}"
        assert z_dim > 0, f"z_dim should be > 0, got: {z_dim}"
        self.n_channels = n_channels
        self.z_dim = z_dim

        self.cnn = nn.Sequential(
            # [B, n_channels, 64, 64] -> [B, 32, 31, 31] 
            nn.Conv2d(n_channels, 32, 4, 2),
            nn.ReLU(True),

            # [B, 32, 31, 31] -> [B, 64, 14, 14] 
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),

            # [B, 64, 14, 14] -> [B, 128, 6, 6] 
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(True),
            
            # [B, 128, 6, 6] -> [B, 256, 2, 2]
            nn.Conv2d(128, 256, 4, 2),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(1024, z_dim)
        self.log_std = nn.Linear(1024, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x).view(-1, 1024)      # [B, 1024]
        mu = self.mu(h)                     # [B, z_dim]
        log_std = self.log_std(h)           # [B, z_dim]
        std = F.softplus(log_std) + 1e-6    # [B, z_dim]
        return mu, std


class Decoder(nn.Module):
    """Decoder model parametrized by a deconvolutional neural net.

    Reference:
    ----------
    World Models, Ha and Schmidhuber 2018.
    https://arxiv.org/abs/1803.10122
    """ 
    def __init__(self, n_channels: int, z_dim: int) -> None:
        super().__init__()
        assert n_channels > 0, f"in_channels should be > 0, got: {n_channels}"
        assert z_dim > 0, f"z_dim should be > 0, got: {z_dim}"
        self.n_channels = n_channels
        self.z_dim = z_dim

        # [B, n_channels] -> [B, 1024]
        self.dense = nn.Linear(z_dim, 1024)

        self.dcnn = nn.Sequential(
            # [B, 1024, 1, 1] -> [B, 128, 5, 5]
            nn.ConvTranspose2d(1024, 128, 5, 2),
            nn.ReLU(True),
            
            # [B, 128, 5, 5] -> [B, 64, 13, 13]
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(True),
            
            # [B, 64, 5, 5] -> [B, 32, 30, 30]
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(True),
            
            # [B, 32, 30, 30] -> [B, n_channels, 64, 64]
            nn.ConvTranspose2d(32, n_channels, 6, 2),
            nn.Sigmoid() 
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dense(z)                  # [B, 1024]
        h = h.view(-1, 1024, 1, 1)         # [B, 1024, 1, 1]
        x = self.dcnn(h)                   # [B, n_channels, 64, 64] 
        return x 


class ConvVAE(nn.Module):
    def __init__(self, n_channels: int, z_dim: int) -> None:
        super().__init__()
        assert n_channels > 0, f"in_channels should be > 0, got: {n_channels}"
        assert z_dim > 0, f"z_dim should be > 0, got: {z_dim}"
        self.n_channels = n_channels
        self.z_dim = z_dim

        self.encoder = Encoder(n_channels, z_dim)  
        self.decoder = Decoder(n_channels, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.encoder(x)                                               # [B, z_dim], [B, z_dim]
        z = mu + std * torch.randn_like(std)                                    # [B, z_dim]
        kl = -0.5 * (mu.pow(2) + std.pow(2) - torch.log(std.pow(2) + 1e-8) - 1) # [B, z_dim]
        kl = torch.sum(kl, dim=1)                                               # [B]
        x_recon = self.decoder(z)                                               # [B, n_channels, 64, 64]
        return x_recon, kl

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        mu, std = self.encoder(x)                                               # [B, z_dim], [B, z_dim]
        z = mu + std * torch.randn_like(std)                                    # [B, z_dim]
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)                                                 # [B, z_dim]
        return mu 
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)                                                  # [B, n_channels, 64, 64]

    def save_name(self) -> str:
        save_name = f"vae-cin{self.n_channels}"
        save_name += f"-z{self.z_dim}.pt"
        return save_name