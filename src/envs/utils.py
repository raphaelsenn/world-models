import numpy as np
import cv2
import torch


def random_action(t: int) -> np.ndarray:
    """
    Generate pseudo-random actions based on the current time step (t).
    
    Returns:
        numpy.ndarray: Action vector [steering, acceleration, brake]
    """
    if t < 20:
        return np.array([-0.1, 1.0, 0.0])

    actions = [
        np.array([0.0, np.random.random(), 0.0]),       # Random Accelerate
        np.array([-np.random.random(), 0.0, 0.0]),      # Random Turn Left
        np.array([np.random.random(), 0.0, 0.0]),       # Random Turn Right
        np.array([0.0, 0.0, np.random.random()]),       # Random Brake
    ]

    # Select a random action based on probabilities
    probabilities = [.35, .3, .3, .05]                  # Probabilities for each action
    selected_action = np.random.choice(len(actions), p=probabilities)
    return actions[selected_action]


def preprocess_observation(obs: np.ndarray, obs_tgt_size: int=64) -> np.ndarray:
    """Transform observation [96, 96, 3] -> [3, 64, 64]""" 
    obs = obs[:84, :, :]
    obs = cv2.resize(obs, (obs_tgt_size, obs_tgt_size), interpolation=cv2.INTER_AREA)
    obs = obs.transpose(2, 0, 1)
    return obs.astype(np.uint8)


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Input obersvation of shape [96, 96, 3] with dtype=np.uint8."""
    obs = preprocess_observation(obs)                                   # [3, 64, 64]
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)    # [3, 64, 64]
    return obs_t.div(255.0).unsqueeze(0)                                # [1, 3, 64, 64


def action_to_tensor(action: np.ndarray, device: torch.device) -> torch.Tensor:
    """Input of shape [96, 96, 3] with dtype=np.uint8."""
    action = torch.as_tensor(action, dtype=torch.float32, device=device)# [action_dim]
    return action.unsqueeze(0)