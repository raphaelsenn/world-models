from typing import Tuple

import numpy as np
import torch


class ControllerBuffer:
    """Replay buffer for controller component (C) training.""" 
    def __init__(
            self, 
            capacity: int,
            batch_size: int,
            state_dim: int, 
            action_dim: int,
            device: torch.device
    ) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.position = 0
        
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)

        self.device = device

    def push(
            self, 
            state: np.ndarray, 
            action: np.float32, 
            reward: float, 
            next_state: np.float32, 
            done: bool
    ) -> None:
        i = self.position
        
        self.states[i] = state.astype(np.float32)
        self.actions[i] = action.astype(np.float32)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state.astype(np.float32)
        self.dones[i] = float(done)

        self.position = (i + 1) % self.capacity

    def sample(self) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.position, size=(self.batch_size,))

        states = torch.as_tensor(
            self.states[indices], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            self.actions[indices], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            self.rewards[indices], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            self.next_states[indices], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            self.dones[indices], dtype=torch.float32, device=self.device
        )

        return (states, actions, rewards, next_states, dones)