import cv2
import numpy as np


def preprocess_observation(obs: np.ndarray, obs_tgt_size: int=64) -> np.ndarray:
    """Transform observation [96, 96, 3] -> [3, 64, 64]""" 
    obs = cv2.resize(obs, (obs_tgt_size, obs_tgt_size), interpolation=cv2.INTER_AREA)
    obs = obs.transpose(2, 0, 1)
    return obs.astype(np.uint8)