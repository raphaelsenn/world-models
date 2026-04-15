import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionLoss(nn.Module):
    """ELBO objective function.""" 
    def __init__(self, kl_weight: float, reduction: str="mean") -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("Unkown reduction.")
        self.reduction = reduction

        assert kl_weight >= 0.0, f"kl_weight should be >= 0.0, got: {kl_weight}"
        self.kl_weight = kl_weight
    
    def forward(
            self, 
            x_recon: torch.Tensor, 
            x_img: torch.Tensor, 
            kl: torch.Tensor
    ) -> torch.Tensor:
        """
        NOTE: Let B := batch_size
        x_recon     : Reconstructed image       [B, 3, 64, 64]
        x_img       : Original image            [B, 3, 64, 64]
        kl          : KL divergence             [B] 
        """    
        if self.reduction == "mean":
            loss_recon = F.mse_loss(x_recon, x_img, reduction="mean") 
            loss_kl = self.kl_weight * kl.mean()
        else: 
            loss_recon = F.mse_loss(x_recon, x_img, reduction="sum") 
            loss_kl = self.kl_weight * kl.sum()

        loss = loss_recon - loss_kl                 # [1] or [B * L]
        return loss