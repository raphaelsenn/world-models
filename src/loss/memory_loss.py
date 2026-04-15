import torch
import torch.nn as nn
from torch.distributions import Normal


class MemoryLoss(nn.Module):
    def __init__(self, reduction: str="mean") -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("Unkown reduction.")
        self.reduction = reduction

    def forward(
            self, 
            pi_logits: torch.Tensor, 
            mu: torch.Tensor, 
            std: torch.Tensor, 
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        NOTE: Let B := batch_size and L := sequence_length
        pi          : Coefficients of shape     [B, L, n_mixtures]
        mu          : Means of shape            [B, L, n_mixtures, z_dim]
        std         : Std deviations of shape   [B, L, n_mixtures, z_dim] 
        targets     : Latents of shape          [B, L, z_dim]  
        """ 
        B, L, n_mixtures, z_dim = mu.shape

        pi_logits = pi_logits.view(B*L, n_mixtures) # [B * L, n_mixtures]
        log_pi = torch.log_softmax(
            pi_logits, dim=-1
        )                                           # [B * L, n_mixtures] 
        
        mu = mu.reshape(B * L, n_mixtures, z_dim)   # [B * L, n_mixtures, z_dim]
        std = std.reshape(B * L, n_mixtures, z_dim) # [B * L, n_mixtures, z_dim]
        targets = targets.reshape(B * L, 1, z_dim)  # [B * L, 1, z_dim]

        dist = Normal(mu, std)
        log_probs = dist.log_prob(targets)          # [B * L, n_mixtures, z_dim]
        log_probs = log_probs.sum(dim=-1)           # [B * L, n_mixtures] 

        loss = -torch.logsumexp(
            log_pi + log_probs, dim=-1
        )                                           # [B * L]
        
        if self.reduction == "mean": 
            loss = loss.mean()                      # [1]
        elif self.reduction == "sum":
            loss = loss.sum()                       # [1]

        return loss                                 # [1] or [B * L]