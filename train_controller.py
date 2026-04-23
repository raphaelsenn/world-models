from argparse import Namespace, ArgumentParser

import numpy as np
import gymnasium as gym
import torch

from src import WorldModel, Controller, ControllerTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Training configuration for the controller component")

    parser.add_argument("--env_id", type=str, default="CarRacing-v3")

    # World model / controller dims
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--controller_fc_dim", type=int, default=64)
    parser.add_argument("--n_mixtures", type=int, default=5)

    # Pre-trained weights of the world model
    parser.add_argument("--weight_vision", type=str, default="vae-cin3-z32.pt")
    parser.add_argument("--weight_memory", type=str, default="mdn-rnn-z32-h256.pt")

    # Training
    parser.add_argument("--n_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_gradient_steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_scale", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.995)
    parser.add_argument("--buffer_capacity", type=int, default=1_000_000)
    parser.add_argument("--buffer_start_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=0)

    # Optimization
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)

    # Eval
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    env = gym.make(args.env_id)

    controller = Controller(
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        fc_dim=args.controller_fc_dim,
    )

    world_model = WorldModel(
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        n_mixtures=args.n_mixtures,
        device=args.device,
        weight_vision=args.weight_vision,
        weight_memory=args.weight_memory,
    )

    trainer = ControllerTrainer(
        model=controller,
        world_model=world_model,
        state_dim=args.latent_dim + args.hidden_dim,
        action_dim=args.action_dim,
        n_timesteps=args.n_timesteps,
        n_gradient_steps=args.n_gradient_steps,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        tau=args.tau,
        buffer_capacity=args.buffer_capacity,
        buffer_start_size=args.buffer_start_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size,
        device=args.device,
        n_eval_episodes=args.n_eval_episodes,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        verbose=args.verbose,
    )

    trainer.train(env)
    env.close()


if __name__ == "__main__":
    main()