import copy

import torch
import torch.nn.functional as F

import gymnasium as gym
import numpy as np
import pandas as pd

from src.trainer.base_trainer import BaseTrainer
from src.data.controller_buffer import ControllerBuffer
from src.world_model.controller import Controller
from src.envs.world_model_env import WorldModelEnv
from src.stats.eval_stats import EnvEvaluationStats
from src.envs.utils import random_action


class ControllerTrainer(BaseTrainer):
    """
    Controller training implemented as Soft Actor-Critic (SAC) and using the World Model.

    Reference:
    ----------
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
    with a Stochastic Actor, Haarnoja et al., 2018.
    https://arxiv.org/abs/1801.01290
    
    World Models, Ha and Schmidhuber, 2018
    https://arxiv.org/abs/1803.10122
    """ 
    def __init__(
        self,
        model: Controller,
        state_dim: int = 32 + 256,
        action_dim: int = 3,
        n_timesteps: int = 1_000_000,
        n_gradient_steps: int = 1,
        buffer_capacity: int = 1_000_000,
        buffer_start_size: int = 10_000,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        initial_alpha: float = 0.1,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        device: str = "cpu",
        eval_every: int = 5_000,
        save_every: int = 5_000,
        n_eval_episodes: int = 10,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        super().__init__(model, device)

        self.n_timesteps = n_timesteps
        self.n_gradient_steps = n_gradient_steps
        self.buffer_start_size = buffer_start_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha 
        self.initial_alpha = initial_alpha
        self.gamma = gamma
        self.tau = tau
        
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.nn.Parameter(
            torch.tensor([np.log(initial_alpha)], dtype=torch.float32, device=self.device)
        )

        self.eval_every = eval_every
        self.save_every = save_every
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        self.seed = seed
        
        self.optimizer_actor = torch.optim.Adam(
            self.model.actor.parameters(), lr=lr_actor
        )
        self.optimizer_critic = torch.optim.Adam(
            self.model.critic.parameters(), lr=lr_critic
        )
        self.optimizer_alpha = torch.optim.Adam(
            [self.log_alpha], lr=lr_alpha
        )
        
        self.buffer = ControllerBuffer(
            capacity=buffer_capacity,
            batch_size=batch_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
        )
        self.env_eval_stats = EnvEvaluationStats()

    def train(self, env: WorldModelEnv) -> None:
        self.eval_env = copy.deepcopy(env)
        self.init_buffer(env)

        state, _ = env.reset()
        episode = 1

        for step in range(self.n_timesteps):
            state, done = self.collect_transition(env, state)
            self.train_n_steps()
            self.handle_periodic_tasks(episode, step)
            
            if done: 
                state, _ = env.reset()
                episode += 1

        env.close()
        self.eval_env.close()

    def init_buffer(self, env: gym.Env) -> None:
        env.action_space.seed(self.seed) 
        state, _ = env.reset(seed=self.seed)
        done = False

        for step in range(self.buffer_start_size):
            action = random_action(step) 

            state_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_td = terminated

            self.buffer.push(state, action, reward, state_next, done_td)
            state = state_next

            if done:
                state, _ = env.reset()
                done = False

    def collect_transition(self, env: gym.Env, state: np.ndarray) -> tuple[np.ndarray, bool]:
        self.model.eval()
        action = self.act(state, deterministic=False)

        state_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_td = terminated

        self.buffer.push(state, action, reward, state_next, done_td)

        return state_next, done

    def train_n_steps(self) -> None:
        self.model.train()
        for _ in range(self.n_gradient_steps):
            s, a, r, s_nxt, d = self.buffer.sample()
            alpha = self.log_alpha.exp().detach()

            # =============== Compute targets for Q functions =================
            with torch.no_grad():
                a_tilde, logp_a_tilde = self.model.sample(s_nxt)                    # [B, action_dim], [B]
                q1_nxt, q2_nxt = self.model.critic_target(s_nxt, a_tilde)           # [B], [B]
                q_target = torch.min(q1_nxt, q2_nxt) - alpha * logp_a_tilde         # [B]
                td_target = r + self.gamma * (1.0 - d) * q_target                   # [B]

            # ===================== Update Q functions ========================
            q1, q2 = self.model.critic(s, a)
            loss_q1 = F.mse_loss(q1, td_target)
            loss_q2 = F.mse_loss(q2, td_target)
            loss_q = loss_q1 + loss_q2
            
            self.optimizer_critic.zero_grad()
            loss_q.backward()
            self.optimizer_critic.step()

            for p in self.model.critic.parameters():
                p.requires_grad = False

            # ========================= Update policy =========================
            a_tilde, logp_a_tilde = self.model.sample(s)                           # [B_action_dim], [B]
            q1, q2 = self.model.critic(s, a_tilde)                                 # [B], [B]
            q_target = torch.min(q1, q2) - alpha * logp_a_tilde                    # [B]
            loss_actor = torch.mean(-q_target)
            
            self.optimizer_actor.zero_grad() 
            loss_actor.backward()
            self.optimizer_actor.step()
            
            # =================== Update entropy coeficient ===================
            # Read more here: https://arxiv.org/abs/1812.05905
            loss_alpha = torch.mean(
                -self.log_alpha * logp_a_tilde.detach() - self.log_alpha * self.target_entropy
            )

            self.optimizer_alpha.zero_grad() 
            loss_alpha.backward()
            self.optimizer_alpha.step()

            for p in self.model.critic.parameters():
                p.requires_grad = True

            # ==================== Update target networks =====================
            self.update_target_networks()

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool=False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_t = self.model.act(state_t.unsqueeze(0), deterministic)
        return action_t.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def update_target_networks(self) -> None:
        params = zip(self.model.critic.parameters(), self.model.critic_target.parameters())
        for theta, theta_old in params:
            theta_old.data.copy_(self.tau * theta.data + (1.0 - self.tau) * theta_old.data) 

    def handle_periodic_tasks(self, episode: int, step: int) -> None:
        if step % self.eval_every == 0:
            self.evaluate(step)
            if self.verbose:
                print(
                    f"T: {step:9d}\t"
                    f"Episode: {episode:6d}\t"  
                    f"Average Reward: {self.env_eval_stats.last_return:.4f}\t"
                    f"Alpha: {self.log_alpha.exp().item():.4f}"
                )
        
        if step % self.save_every == 0:
            self.save_stats(step)

    @torch.no_grad()
    def evaluate(self, step: int) -> None:
        self.eval_env.set_eval_mode()
        returns = np.zeros(self.n_eval_episodes, dtype=np.float32)    
        for ep in range(self.n_eval_episodes): 
            state, _ = self.eval_env.reset(seed=self.seed + 1000 + ep)
            done = False

            while not done: 
                action = self.act(state, deterministic=True)
                state_next, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated

                state = state_next
                returns[ep] += reward

        average_return = float(np.mean(returns).item())
        self.env_eval_stats.append_step(step)
        self.env_eval_stats.append_return(average_return)

    def save_stats(self, step: int) -> None:
        file_name = self.model.save_name()
        file_name += f"t{step}-seed{self.seed}.pt"
        state = self.model.state_dict()
        torch.save(state, file_name)

        csv_name = f"controller-train-report-seed{self.seed}.csv"
        data = {
            "Timesteps": self.env_eval_stats.step, 
            "Average Return": self.env_eval_stats.average_return
        }
        pd.DataFrame.from_dict(data).to_csv(csv_name, index=False)