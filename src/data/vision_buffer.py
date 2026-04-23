import numpy as np
import torch


class VisionBuffer:
    """Rollout buffer for vision component (V) training.""" 
    def __init__(self, n_channels: int, horizon: int) -> None:
        self.n_channels = n_channels 
        self.horizon = horizon

        self.position = 0
        self.observations = np.empty((horizon, n_channels, 64, 64), dtype=np.uint8)

    def push(self, obs: np.ndarray) -> None:
        i = self.position 
        self.observations[i] = obs
        self.position = (i + 1) % self.horizon

    def dataset(self) -> torch.Tensor:
        dataset = torch.as_tensor(self.observations, dtype=torch.float32).div(255.)
        return dataset