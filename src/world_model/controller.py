import torch
import torch.nn as nn


class Controller(nn.Module):
    """
    Controller component (C) as described in the paper.    

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
            action_dim: int, 
            z_dim: int, 
            hidden_dim: int, 
            action_scale: float=1.0
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.action_scale = action_scale

        self.linear = nn.Linear(z_dim + hidden_dim, action_dim)

    @torch.no_grad() 
    def forward(self, latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([latent, hidden], dim=-1)
        action = self.linear(cat)
        action = action.clamp(-self.action_scale, self.action_scale)
        return action