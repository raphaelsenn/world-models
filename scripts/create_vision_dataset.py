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


def create_folder(env_id: str) -> str:
    """Creates a dataset folder.""" 
    folder_name = f"VAE-Dataset-{env_id}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def create_dataset(env: gym.Env, n_episodes: int, chunk_size: int, seed: int, verbose: bool=True) -> None:
    """Creates a dataset for the variational autoencoder (simple script).""" 
    if not isinstance(env, gym.Env):
        raise ValueError("Please use gymnasium environments.") 
    if env.spec is None:
        raise ValueError("Environment needs a valid `spec`.")
    folder_name = create_folder(env.spec.id)

    total_chunks = 0
    chunks = []
    for episode in range(1, n_episodes + 1):
        done = False
        env.action_space.seed(seed + (episode - 1)) 
        obs, _ = env.reset(seed=seed + (episode - 1))
        
        while not done: 
            chunks.append(preprocess_observation(obs)) 
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if len(chunks) == chunk_size:
                chunks = np.asarray(chunks, dtype=np.uint8)
                np.save(os.path.join(folder_name, f"chunk_{total_chunks:06d}.npy"), chunks)
                total_chunks += 1 
                chunks = []

        if verbose and episode % 10 == 0:
            print(f"Episode: {episode:6d}/{n_episodes}\tTotal chunks: {total_chunks:9d}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="VAE-Dataset creation")

    parser.add_argument("--env_id", type=str, default="CarRacing-v3")
    parser.add_argument("--n_episodes", type=int, default=10_000)
    parser.add_argument("--chunk_size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=100_000)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    env = gym.make(args.env_id, render_mode="rgb_array")
    create_dataset(env, args.n_episodes, args.chunk_size, args.seed)


if __name__ == "__main__":
    main()