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
        log_std = log_std.clamp(-7, 2)
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
            latent_dim: int, 
            action_dim: int, 
            hidden_dim: int, 
            n_mixtures: int
   ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures

        # NOTE: Ha and Schmidhuber used LSTM, i use a GRU.
        # self.rnn = nn.LSTM(latent_dim + action_dim,  hidden_dim, batch_first=True)
        self.rnn = nn.GRU(latent_dim + action_dim,  hidden_dim, batch_first=True)
        self.mdn = MDN(latent_dim, hidden_dim, n_mixtures)

    def forward(
            self, 
            latent: torch.Tensor, 
            action: torch.Tensor,
    ) -> torch.Tensor:
        """
        TRAINING - Expects sequential input
        
        Input:
        ------ 
        latent : [B, L, latent_dim]
        action : [B, L, action_dim]
        """ 
        cat = torch.cat([latent, action], dim=-1)       # [B, L, z_dim + action_dim]
        output, _ = self.rnn(cat)                       # [B, L, hidden_dim]
        pi_logits, mu, std = self.mdn(output)           # [B, K], [B, K, action_dim] (mu and std)
        return pi_logits, mu, std

    @torch.no_grad() 
    def predict_next_dist(
            self, 
            latent_prev: torch.Tensor, 
            action_prev: torch.Tensor,
            hidden_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        INFERENCE - Expects non-sequential input

        Input:
        ------ 
        latent      : [B, latent_dim]
        action      : [B, action_dim]
        hidden_prev : [B, hidden_dim]
        """ 
        cat = torch.cat([
            latent_prev, action_prev], dim=-1
        ).unsqueeze(1)                                  # [B, 1, z_dim + action_dim]

        hidden_prev = hidden_prev.unsqueeze(0)          # [1, B, hidden_dim]

        _, hidden = self.rnn(cat, hidden_prev)          # [1, B, hidden_dim]
        hidden = hidden.squeeze(0)                      # [B, hidden_dim]
        pi_logits, mu, std = self.mdn(hidden)           # [B, K], [B, K, z_dim], [B, K, z_dim]

        return pi_logits, mu, std, hidden

    @torch.no_grad()
    def predict_next_hidden(
            self, 
            latent_prev: torch.Tensor, 
            action_prev: torch.Tensor, 
            hidden_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        INFERENCE - Expects non-sequential input
        
        Input:
        ------ 
        latent      : [B, latent_dim]
        action      : [B, action_dim]
        hidden_prev : [B, hidden_dim]
        """  
        _, _, _, hidden = self.predict_next_dist(
            latent_prev, action_prev, hidden_prev
        ) 
        return hidden
    
    @torch.no_grad()
    def sample_next_latent(
            self, 
            latent_prev: torch.Tensor, 
            action_prev: torch.Tensor, 
            hidden_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        INFERENCE - Expects non-sequential input
        
        Input:
        ------ 
        latent      : [B, latent_dim]
        action      : [B, action_dim]
        hidden_prev : [B, hidden_dim]
        """  
        B = latent_prev.size(0) 
        pi_logits, mu, std, hidden = self.predict_next_dist(
            latent_prev, action_prev, hidden_prev
        ) 
        
        mix = Categorical(logits=pi_logits)                          # [B]
        comp_idx = mix.sample()                                      # [B]

        batch_idx = torch.arange(B, device=latent_prev.device)
        mu_sel = mu[batch_idx, comp_idx, :]                          # [B, z_dim]
        std_sel = std[batch_idx, comp_idx, :]                        # [B, z_dim]

        latent_next = Normal(mu_sel, std_sel).sample()               # [B, z_dim]
        return latent_next, hidden

    def save_name(self) -> str:
        save_name = f"mdn-rnn-z{self.latent_dim}"
        save_name += f"-h{self.hidden_dim}.pt"
        return save_name