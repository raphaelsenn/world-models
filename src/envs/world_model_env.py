from typing import Dict, Tuple, Any

import numpy as np
import torch
import gymnasium as gym

from src.world_model.memory import Memory
from src.world_model.vision import ConvVAE
from src.envs.utils import (
    obs_to_tensor, 
    action_to_tensor
)


class WorldModelEnv(gym.Wrapper):
    """Simple world model environment (described by Ha and Schmidhuber).
    
    Reference:
    ----------
    World Models, Ha and Schmidhuber, 2018
    https://arxiv.org/abs/1803.10122
    """ 
    def __init__(
        self,
        env: gym.Env,
        in_channels: int = 3,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_mixtures: int = 5,
        weight_vision: str | None = None,
        weight_memory: str | None = None,
        device: str = "cpu",
        train_mode: bool=True
    ) -> None:
        super().__init__(env)

        self.action_dim = env.action_space.shape[0]
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures
        self.device = torch.device(device)
        self.train_mode = train_mode

        self.vision = ConvVAE(
            in_channels, latent_dim
        ).to(self.device)
        self.memory = Memory(
            latent_dim, self.action_dim, hidden_dim, n_mixtures
        ).to(self.device)
        self.load_weights(weight_vision, weight_memory)

        self.hidden: torch.Tensor | None = None
        self.latent: torch.Tensor | None = None

    @torch.no_grad()
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool]:
        obs_next, reward, terminated, truncated, info = self.env.step(action)

        # Penalize for turning to frequently, as done by Ha and Schmidhuber.
        #extra_reward = 0.0
        #if self.train_mode:
        #    extra_reward -= np.abs(action[0]) / 10.0
        #reward += extra_reward

        obs_next = obs_to_tensor(obs_next, self.device)
        action = action_to_tensor(action, self.device)

        hidden_next = self.memory.predict_next_hidden(self.latent, action, self.hidden)
        latent_next = self.vision.encode(obs_next)

        state_next = torch.cat([latent_next, hidden_next], dim=-1)
        state_next = state_next.squeeze(0).cpu().numpy()

        self.hidden = hidden_next
        self.latent = latent_next

        return state_next, reward, terminated, truncated, info


    @torch.no_grad()
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Input:
        ------ 
        obs    : [3, 64, 64]  (np.uint8)
        """  
        obs, info = self.env.reset(seed=seed)
        obs= obs_to_tensor(obs, self.device)

        latent = self.vision.encode(obs)
        hidden = torch.zeros(
            (1, self.hidden_dim), dtype=torch.float32, device=self.device
        )

        state = torch.cat([latent, hidden], dim=-1)
        state = state.squeeze(0).cpu().numpy()

        self.hidden = hidden
        self.latent = latent
        
        return state, info

    def set_eval_mode(self) -> None:
        self.vision.eval()
        self.memory.eval()

        if self.device.type == "cuda":
            self.memory.rnn.flatten_parameters()

    def load_weights(self, weight_vision: str | None, weight_memory: str | None) -> None:
        if weight_vision is not None:
            state_dict = torch.load(
                weight_vision, map_location=self.device, weights_only=True
            )
            self.vision.load_state_dict(state_dict)

        if weight_memory is not None:
            state_dict = torch.load(
                weight_memory, map_location=self.device, weights_only=True
            )
            self.memory.load_state_dict(state_dict)