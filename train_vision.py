from argparse import Namespace, ArgumentParser

import numpy as np
import gymnasium as gym
import torch

from src import ConvVAE, VisionTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Training configuration for the vision component")
    parser.add_argument("--env_id", type=str, default="CarRacing-v3")

    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--z_dim", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n_timesteps", type=int, default=10_000_000)
    parser.add_argument("--horizon", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)

    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    
    vae = ConvVAE(args.n_channels, args.z_dim)
    env = gym.make(args.env_id, render_mode="rgb_array")

    vae_trainer = VisionTrainer(
        model=vae,
        epochs=args.epochs,
        in_channels=args.n_channels,
        n_timesteps=args.n_timesteps,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight, 
        batch_size=args.batch_size,
        device=args.device,
        n_workers=args.n_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
        verbose=args.verbose,
    )
    vae_trainer.train(env)

    env.close()


if __name__ == "__main__":
    main()