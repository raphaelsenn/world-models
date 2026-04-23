import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def action(self, action: np.ndarray) -> np.ndarray:
        steer = action[0]
        gas_or_brake = action[1]
        if gas_or_brake >= 0.0:
            return np.array([steer, gas_or_brake, 0.0], dtype=np.float32)
        return np.array([steer, 0.0, -gas_or_brake], dtype=np.float32)