from argparse import Namespace, ArgumentParser

import numpy as np
import gymnasium as gym
import torch

from src import WorldModelEnv, Controller, ControllerTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Training configuration for the controller component")

    parser.add_argument("--env_id", type=str, default="CarRacing-v3")

    # World model / controller dims
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=3)    # [-1, 1] x [-1, 1], (Read more in ./src/utils/wrappers.py)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--controller_fc_dim", type=int, default=512)
    parser.add_argument("--n_mixtures", type=int, default=5)

    # Pre-trained weights of the world model
    parser.add_argument("--weight_vision", type=str, default="vae-cin3-z32.pt")
    parser.add_argument("--weight_memory", type=str, default="mdn-rnn-z32-h256.pt")

    # Training
    parser.add_argument("--n_timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_gradient_steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--buffer_capacity", type=int, default=1_000_000)
    parser.add_argument("--buffer_start_size", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=10_000)

    # Optimization
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--lr_alpha", type=float, default=3e-4)

    # Eval
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--save_every", type=int, default=5_000)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def play(env, controller) -> None:
    for ep in range(10): 
        state, _ = env.reset()
        done = False

        t = 0
        total_reward = 0.0
        while not done: 
            state = torch.as_tensor(state).unsqueeze(0) 
            action = controller.act(state, deterministic=False).cpu().flatten().numpy()

            state_next, reward, terminated, truncated, _ = env.step(action)
            # print(t, total_reward, action) 
            done = terminated or truncated
            state = state_next
            t += 1
            total_reward += reward
        print(total_reward)

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    env = gym.make(args.env_id, render_mode="human")
    env = WorldModelEnv(
        env=env,
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_mixtures=args.n_mixtures,
        device=args.device,
        weight_vision=args.weight_vision,
        weight_memory=args.weight_memory,
    )

    controller = Controller(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        fc_dim=args.controller_fc_dim,
    )

    controller.load_state_dict(torch.load("controller-actor-critic-z32-h256-fc512.ptt1035000-seed2.pt"))
    play(env, controller)



if __name__ == "__main__":
    main()