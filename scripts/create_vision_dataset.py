from argparse import Namespace, ArgumentParser
import os

import cv2
import numpy as np
import gymnasium as gym


def preprocess_observation(obs: np.ndarray, obs_tgt_size: int=64) -> np.ndarray:
    """Transform observation [96, 96, 3] -> [3, 64, 64]""" 
    obs = cv2.resize(obs, (obs_tgt_size, obs_tgt_size), interpolation=cv2.INTER_AREA)
    obs = obs.transpose(2, 0, 1)
    return obs.astype(np.uint8)


def create_dataset(folder_name: str, env: gym.Env, n_episodes: int, seed: int, verbose: bool=True) -> None:
    """Creates a dataset for the variational autoencoder (simple script).""" 
    if not isinstance(env, gym.Env):
        raise ValueError("Please use gymnasium environments.") 
    if env.spec is None:
        raise ValueError("Environment needs a valid `spec`.")
    os.makedirs(folder_name, exist_ok=True)

    total_frames = 0
    for episode in range(1, n_episodes + 1):
        done = False
        env.action_space.seed(seed + (episode - 1)) 
        obs, _ = env.reset(seed=seed + (episode - 1))
        ep_frame = 1

        while not done: 
            obs = preprocess_observation(obs) 
            
            action = env.action_space.sample()
            obs_nxt, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            np.save(os.path.join(folder_name, f"episode{episode}_frame{ep_frame:06d}.npy"), obs)
            total_frames += 1 
            ep_frame += 1

            obs = obs_nxt

        if verbose and episode % 10 == 0:
            print(f"Episode: {episode:6d}/{n_episodes}\tTotal frames: {total_frames:9d}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="VAE-Dataset creation")
    parser.add_argument("--folder_name", type=str, default="Vision-CarRacing-v3-Train")

    parser.add_argument("--env_id", type=str, default="CarRacing-v3")
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=100_000)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    env = gym.make(args.env_id, render_mode="rgb_array")
    create_dataset(args.folder_name, env, args.n_episodes, args.seed)


if __name__ == "__main__":
    main()