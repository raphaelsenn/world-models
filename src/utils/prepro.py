import cv2
import numpy as np


def random_action(t: int) -> np.ndarray:
    """
    Generate pseudo-random actions based on the current time step (t).
    
    Returns:
        numpy.ndarray: Action vector [steering, acceleration/brake]
    """
    if t < 20:
        return np.array([-0.1, 1.0])

    actions = [
        np.array([0, np.random.random()]),    # Random Accelerate
        np.array([-np.random.random(), 0]),   # Random Turn Left
        np.array([np.random.random(), 0]),    # Random Turn Right
        np.array([0, -np.random.random()]),   # Random Brake
    ]
    probabilities = [.35, .3, .3, .05]        # Probabilities for each action

    # Select a random action based on the defined probabilities
    selected_action = np.random.choice(len(actions), p=probabilities)
    return actions[selected_action]


def preprocess_observation(obs: np.ndarray, obs_tgt_size: int=64) -> np.ndarray:
    """Transform observation [96, 96, 3] -> [3, 64, 64]""" 
    obs = obs[:84, :, :]
    obs = cv2.resize(obs, (obs_tgt_size, obs_tgt_size), interpolation=cv2.INTER_AREA)
    obs = obs.transpose(2, 0, 1)
    return obs.astype(np.uint8)