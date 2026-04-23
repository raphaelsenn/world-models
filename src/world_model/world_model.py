import numpy as np
import torch

from src.utils.prepro import preprocess_observation
from src.world_model.memory import Memory
from src.world_model.vision import ConvVAE


class WorldModel:
    """Simple world model described by Ha and Schmidhuber.
    
    Reference:
    ----------
    World Models, Ha and Schmidhuber, 2018
    https://arxiv.org/abs/1803.10122
    """ 
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 256,
        n_mixtures: int = 5,
        device: str = "cpu",
        weight_vision: str | None = None,
        weight_memory: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim

        self.vision = ConvVAE(
            in_channels, latent_dim
        ).to(self.device)
        self.memory = Memory(
            latent_dim, action_dim, hidden_dim, n_mixtures
        ).to(self.device)

        self._load_weights(weight_vision, weight_memory)

        self.hidden: torch.Tensor | None = None
        self.latent: torch.Tensor | None = None
    
    @torch.no_grad()
    def reset(self, obs: np.ndarray) -> np.ndarray:
        self.vision.eval(); self.memory.eval()
        hidden = self._init_hidden()                                    # [1, hidden_dim]
        latent = self._encode_obs(obs)                                  # [1, latent_dim]

        state = torch.cat([latent, hidden], dim=-1)                     # [1, latent_dim + hidden_dim]
        state = state.squeeze(0).cpu().numpy()                          # [latent_dim + hidden_dim]

        self.hidden = hidden
        self.latent = latent

        return state

    @torch.no_grad()
    def step(self, action: np.ndarray, obs_next: np.ndarray) -> np.ndarray: 
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = action.unsqueeze(0)

        hidden_next = self.memory.encode(self.latent, action, self.hidden)
        latent_next = self._encode_obs(obs_next)

        state_next = torch.cat([latent_next, hidden_next], dim=-1)      # [1, latent_dim + hidden_dim]
        state_next = state_next.squeeze(0).cpu().numpy()                # [latent_dim + hidden_dim]

        self.hidden = hidden_next
        self.latent = latent_next

        return state_next

    @torch.no_grad()
    def _encode_obs(self, obs: np.ndarray) -> torch.Tensor: 
        obs = preprocess_observation(obs)                               # [3, 64, 64]                   
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).div(255.0).unsqueeze(0)                                       # [1, 3, 64, 64]               
        latent = self.vision.encode(obs_t)                              # [1, latent_dim]
        return latent
    
    def _load_weights(self, weight_vision: str | None, weight_memory: str | None) -> None:
        if weight_vision is not None:
            state_dict = torch.load(
                weight_vision, map_location="cpu", weights_only=True
            )
            self.vision.load_state_dict(state_dict)

        if weight_memory is not None:
            state_dict = torch.load(
                weight_memory, map_location="cpu", weights_only=True
            )
            self.memory.load_state_dict(state_dict)

    def _init_hidden(self) -> torch.Tensor:
        return torch.zeros((1, self.hidden_dim), dtype=torch.float32, device=self.device)
    
    def _to(self, device: torch.device) -> None:
        self.device = device
        self.vision.to(self.device)
        self.memory.to(device)