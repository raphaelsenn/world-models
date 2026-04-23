from argparse import Namespace, ArgumentParser
import os

import cv2
import numpy as np
import gymnasium as gym

import torch

from src.world_model import ConvVAE
from src.utils.wrappers import ActionWrapper


def preprocess_observation(obs: np.ndarray, obs_tgt_size: int=64) -> np.ndarray:
    """Transform observation [96, 96, 3] -> [3, 64, 64]""" 
    obs = cv2.resize(obs, (obs_tgt_size, obs_tgt_size), interpolation=cv2.INTER_AREA)
    obs = obs.transpose(2, 0, 1)
    return obs.astype(np.uint8)


def create_folder(env_id: str) -> str:
    """Creates a dataset folder.""" 
    folder_name = f"MDN-RNN-Dataset-{env_id}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


@torch.no_grad()
def create_dataset(
        vae: ConvVAE, 
        env: gym.Env, 
        n_episodes: int, 
        seed: int, 
        verbose: bool=True,
        device: str="cpu"
    ) -> None:
    """Creates a dataset for the MDN-RNN (simple script).""" 
    if not isinstance(env, gym.Env):
        raise ValueError("Please use gymnasium environments.") 
    if env.spec is None:
        raise ValueError("Environment needs a valid `spec`.")
    folder_name = create_folder(env.spec.id)

    # Select device and set model to correct device
    device = torch.device(device)
    vae.to(device)
    vae.eval()

    total_steps = 0 
    latents, actions = [], [] 
    for episode in range(1, n_episodes + 1):
        done = False
        env.action_space.seed(seed + (episode - 1)) 
        obs, _ = env.reset(seed=seed + (episode - 1))
        
        while not done: 
            obs_t = torch.as_tensor(
                preprocess_observation(obs), dtype=torch.float32, device=device
            ).div_(255.0)

            z_t = vae.encode(obs_t).view(-1).cpu().numpy()
            a_t = env.action_space.sample()

            latents.append(z_t)
            actions.append(a_t)

            obs, _, terminated, truncated, _ = env.step(a_t)
            done = terminated or truncated
            
            if done:
                path = os.path.join(folder_name, f"episode_{episode:06d}")
                
                latents = np.asarray(latents, dtype=np.float32)
                actions = np.asarray(actions, dtype=np.float32)
                np.savez(path, latents=latents, actions=actions) 

                total_steps += latents.shape[0]
                latents, actions = [], [] 

        if verbose and episode % 10 == 0:
            print(f"Episode: {episode:6d}/{n_episodes}\tTotal steps: {total_steps:9d}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="MDN-RNN-Dataset creation")

    parser.add_argument("--vae_weights", type=str, default="vae-cin3-z32.pt")
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--z_dim", type=int, default=32)

    parser.add_argument("--env_id", type=str, default="CarRacing-v3")
    parser.add_argument("--n_episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    vae = ConvVAE(args.n_channels, args.z_dim)
    vae.load_state_dict(torch.load(args.vae_weights, map_location="cpu", weights_only=True))

    env = gym.make(args.env_id, render_mode="rgb_array")
    env = ActionWrapper(env)

    create_dataset(vae, env, args.n_episodes, args.seed, args.verbose, args.device)


if __name__ == "__main__":
    main()