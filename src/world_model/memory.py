from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class MDN(nn.Module):
    """
    Mixture Density Network (gaussian).   

    NOTE: 
    * latent : Compressed current observation (spatial)
    * hidden : Comppressed past context (temporal)
    """
    def __init__(self, z_dim: int, hidden_dim: int, n_mixtures: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures

        self.pi = nn.Linear(hidden_dim, n_mixtures)
        self.mu = nn.Linear(hidden_dim, n_mixtures * z_dim)
        self.logstd = nn.Linear(hidden_dim, n_mixtures * z_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, hidden_dim] or [B, L, hidden_dim]

        Returns: 
        -------- 
        pi   : [B, K] or [B, L, K], where K := n_mixtures
        mu   : [B, K, z_dim] or [B, L, K, z_dim]
        std  : [B, K, z_dim] or [B, L, K, z_dim] 
        """ 
        pi_logits = self.pi(hidden)
        mu = self.mu(hidden)
        log_std = self.logstd(hidden)
        log_std = log_std.clamp(-20, 2)
        std = torch.exp(log_std)

        new_shape = hidden.shape[:-1] + (self.n_mixtures, self.z_dim)
        mu = mu.view(*new_shape)
        std = std.view(*new_shape)

        return pi_logits, mu, std


class Memory(nn.Module):
    """
    Memory component (MDN-RNN) as described in the paper.    

    Reference:
    ----------
    World Models, Ha and Schmidhuber, 2018
    https://arxiv.org/abs/1803.10122

    NOTE: 
    * latent : Compressed current observation
    * hidden : Comppressed past context
    """
    def __init__(
            self, 
            z_dim: int, 
            action_dim: int, 
            hidden_dim: int, 
            n_mixtures: int
   ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures

        # NOTE: Ha and Schmidhuber used a LSTM, i use a GRU.
        self.rnn = nn.GRU(z_dim + action_dim,  hidden_dim, batch_first=True)
        self.mdn = MDN(z_dim, hidden_dim, n_mixtures)

    def forward(
            self, 
            latent: torch.Tensor, 
            action: torch.Tensor,
    ) -> torch.Tensor:
        """
        TRAINING - Expects sequential input
        
        Input:
        ------ 
        latent : [B, L, z_dim]
        action : [B, L, action_dim]
        """ 
        cat = torch.cat([latent, action], dim=-1)
        output, _ = self.rnn(cat)
        pi_logits, mu, std = self.mdn(output)
        return pi_logits, mu, std

    @torch.no_grad() 
    def step(
            self, 
            latent_prev: torch.Tensor, 
            action_prev: torch.Tensor,
            hidden_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        INFERENCE - Expects non-sequential input

        Input:
        ------ 
        latent      : [B, z_dim]
        action      : [B, action_dim]
        hidden_prev : [B, hidden_dim]
        """ 
        cat = torch.cat([latent_prev, action_prev], dim=-1)       # [B, z_dim + action_dim]
        
        cat = cat.unsqueeze(1)                          # [B, 1, z_dim + action_dim]
        hidden_prev = hidden_prev.unsqueeze(1)          # [B, 1, hidden_dim]
        hidden_prev = hidden_prev.permute(1, 0, 2)      # [1, B, hidden_dim]

        output, hidden = self.rnn(cat, hidden_prev)
        out = output[:, -1, :]                          # [B, hidden_dim]

        pi_logits, mu, std = self.mdn(out)              # [B, K], [B, K, z_dim], [B, K, z_dim]
        
        hidden = hidden.squeeze(0)                      # [B, hidden_dim]

        return pi_logits, mu, std, hidden

    @torch.no_grad()
    def encode(
            self, 
            latent_prev: torch.Tensor, 
            action_prev: torch.Tensor, 
            hidden_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        INFERENCE - Expects non-sequential input
        
        Input:
        ------ 
        latent      : [B, z_dim]
        action      : [B, action_dim]
        hidden_prev : [B, hidden_dim]
        """  
        _, _, _, hidden = self.step(latent_prev, action_prev, hidden_prev) 
        return hidden
    
    @torch.no_grad()
    def sample(
            self, 
            latent_prev: torch.Tensor, 
            action_prev: torch.Tensor, 
            hidden_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        INFERENCE - Expects non-sequential input
        
        Input:
        ------ 
        latent      : [B, z_dim]
        action      : [B, action_dim]
        hidden_prev : [B, hidden_dim]
        """  
        pi_logits, mu, std, hidden = self.step(latent_prev, action_prev, hidden_prev) 
        
        mix = Categorical(logits=pi_logits)                          # [B]
        comp_idx = mix.sample()                                      # [B]

        batch_idx = torch.arange(latent_prev.size(0), device=latent_prev.device)
        mu_sel = mu[batch_idx, comp_idx, :]                          # [B, z_dim]
        std_sel = std[batch_idx, comp_idx, :]                        # [B, z_dim]

        latent_next = Normal(mu_sel, std_sel).sample()               # [B, z_dim]
        return latent_next, hidden
    
    def save_name(self) -> str:
        save_name = f"mdn-rnn-z{self.z_dim}"
        save_name += f"-h{self.hidden_dim}.pt"
        return save_name