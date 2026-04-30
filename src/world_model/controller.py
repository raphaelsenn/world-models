import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np


class Actor(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            fc_dim: int
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, fc_dim),
            nn.ReLU(True),

            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(fc_dim, action_dim)
        self.log_std = nn.Linear(fc_dim, action_dim)

    def forward(
            self, 
            state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: state_dim = latent_dim + hidden_dim

        Input:
        ------ 
        state : [B, state_dim] (np.float32)
        """  
        h = self.mlp(state)

        mu = self.mu(h)                     # [B, action_dim]
        log_std = self.log_std(h)           # [B, action_dim]
        log_std = log_std.clamp(-20, 2)     # [B, action_dim]
        std = torch.exp(log_std)            # [B, action_dim]

        return mu, std
    
    def act(self, state: torch.Tensor, deterministic: bool=False) -> torch.Tensor:
        """
        NOTE: state_dim = latent_dim + hidden_dim

        Input:
        ------ 
        state : [B, state_dim] (np.float32)
        """   
        mu, std = self(state)

        if deterministic:
            return torch.tanh(mu)

        dist = Normal(mu, std)
        a_pre_tanh = dist.sample()          # [B, action_dim]
        return torch.tanh(a_pre_tanh)       # [B, action_dim]

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: state_dim = latent_dim + hidden_dim

        Input:
        ------ 
        state : [B, state_dim] (np.float32)
        """   
        mu, std = self(state)               # [B, action_dim] (both)

        dist = Normal(mu, std)
        a_pre_tanh = dist.rsample()         # [B, action_dim]
        a_tanh = torch.tanh(a_pre_tanh)     # [B, action_dim]

        # Log-prob correction
        # Deriving this is fun! 
        log_prob = dist.log_prob(a_pre_tanh)# [B, action_dim]
        log_prob = log_prob.sum(dim=-1)     # [B]
        log_prob -= (
            2*(np.log(2) - a_pre_tanh - F.softplus(-2*a_pre_tanh))
        ).sum(dim=-1)                      # [B]

        return a_tanh, log_prob


class Critic(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            fc_dim: int=64
    ) -> None:
        super().__init__()

        self.Q1 = nn.Sequential(
            # [B, state_dim + action_dim] -> [B, fc_dim] 
            nn.Linear(state_dim + action_dim, fc_dim),
            nn.ReLU(True),

            # [B, fc_dim] -> [B, fc_dim] 
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),

            # [B, fc_dim] -> [B, 1] 
            nn.Linear(fc_dim, 1) 
        )
        
        self.Q2 = nn.Sequential(
            # [B, state_dim + action_dim] -> [B, fc_dim] 
            nn.Linear(state_dim + action_dim, fc_dim),
            nn.ReLU(True),

            # [B, fc_dim] -> [B, fc_dim]
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),

            # [B, fc_dim] -> [B, 1]
            nn.Linear(fc_dim, 1) 
        )

    def forward(
            self, 
            state: torch.Tensor, 
            action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat([state, action], dim=-1)    # [1, state_dim + action_dim] 
        q1 = self.Q1(cat)                           # [B, 1]
        q2 = self.Q2(cat)                           # [B, 1]
        return q1.view(-1), q2.view(-1)             # [B], [B]

    def q1(
            self, 
            state: torch.Tensor,
            action: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
        ------ 
        state   : [B, state_dim]    (np.float32)
        action  : [B, action_dim]   (np.float32) 
        """   
        cat = torch.cat([state, action], dim=-1)    # [1, state_dim + action_dim] 
        q1 = self.Q1(cat)                           # [B, 1]
        return q1.view(-1)                          # [B]
    
    def q2(
            self, 
            state: torch.Tensor,
            action: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
        ------ 
        state   : [B, state_dim]    (np.float32)
        action  : [B, action_dim]   (np.float32) 
        """    
        cat = torch.cat([state, action], dim=-1)    # [1, state_dim + action_dim] 
        q2 = self.Q2(cat)                           # [B, 1]
        return q2.view(-1)                          # [B]


class Controller(nn.Module):
    """
    Controller component (C) implemented using SAC.    

    Reference:
    ----------
    World Models, Ha and Schmidhuber, 2018
    https://arxiv.org/abs/1803.10122

    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
    with a Stochastic Actor, Haarnoja et al., 2018.
    https://arxiv.org/abs/1801.01290

    NOTE: 
    * latent : Compressed current observation (spatial)
    * hidden : Comppressed past context (temporal)
    """ 
    def __init__(
            self, 
            action_dim: int, 
            latent_dim: int, 
            hidden_dim: int, 
            fc_dim: int, 
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.state_dim = latent_dim + hidden_dim

        self.actor = Actor(self.state_dim, action_dim, fc_dim)
        self.critic = Critic(self.state_dim, action_dim, fc_dim)
        self.critic_target = copy.deepcopy(self.critic)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
        ------ 
        state   : [B, state_dim]    (np.float32)
        """    
        return self.actor(state) 

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool=True) -> torch.Tensor:
        """Returns a non-differentiable action w.r.t. actor parameters.
        
        Input:
        ------ 
        state   : [B, state_dim]    (np.float32)
        """
        return self.actor.act(state, deterministic)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a differentiable action w.r.t. actor parameters.
        
        Input:
        ------ 
        state   : [B, state_dim]    (np.float32) 
        """
        return self.actor.sample(state)

    def q(
            self, 
            state: torch.Tensor, 
            action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
        ------ 
        state   : [B, state_dim]    (np.float32)
        action  : [B, action_dim]   (np.float32)
        """
        return self.critic(state, action)

    def save_name(self) -> str:
        save_name = f"controller-actor-critic-z{self.latent_dim}"
        save_name += f"-h{self.hidden_dim}-fc{self.fc_dim}.pt"
        return save_name