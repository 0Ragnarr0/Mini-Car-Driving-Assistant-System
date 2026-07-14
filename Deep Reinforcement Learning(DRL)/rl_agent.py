"""Deep RL agent for continuous driving control.

Implements TD3 (Twin Delayed DDPG), an upgrade over plain DDPG that fixes the
two things that make DDPG unstable:
  1. Twin critics + take the minimum target Q  -> reduces Q-value overestimation.
  2. Target-policy smoothing (clipped noise on the target action) -> regularizes.
  3. Delayed actor/target updates (every `policy_delay` critic steps) -> stability.

The public interface (select_action / store_transition / update / save / load and
the `warmup_steps`, `replay`, `replay_size` attributes) is kept identical to the
previous DDPG agent, so the Webots controller works unchanged. A `DDPGAgent`
alias is provided at the bottom for backward compatibility.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy RL."""

    def __init__(self, state_dim, action_dim, capacity=100000):
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_state[idx] = next_state
        self.done[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_state[idxs],
            self.done[idxs],
        )

    def __len__(self):
        return self.size


class MLPActor(nn.Module):
    """Actor outputs bounded continuous actions in [-1, 1].

    LayerNorm stabilizes training and normalizes raw observations of varied scale;
    the small init on the pre-Tanh layer keeps initial actions near 0 so the policy
    does NOT saturate to a fixed action early (the collapse seen in the first run).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)

    def forward(self, state):
        return self.net(state)


class TwinCritic(nn.Module):
    """Two independent Q networks (TD3 uses the min of the two as the target)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, state, action):
        """Single-critic evaluation used for the actor (policy) objective."""
        return self.q1(torch.cat([state, action], dim=-1))


class TD3Agent:
    """TD3 agent for continuous steering and speed actions."""

    def __init__(self, state_dim, action_dim=2, config=None, device=None):
        cfg = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = 1.0  # actions live in [-1, 1] (Tanh output)

        # Core RL hyperparameters
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.batch_size = int(cfg.get("batch_size", 128))
        self.replay_size = int(cfg.get("replay_size", 100000))
        self.warmup_steps = int(cfg.get("warmup_steps", 2000))
        self.action_noise = float(cfg.get("action_noise", 0.15))
        self.hidden_dim = int(cfg.get("hidden_dim", 256))

        # TD3-specific hyperparameters
        self.policy_delay = max(1, int(cfg.get("policy_delay", 2)))
        self.target_noise = float(cfg.get("target_noise", 0.2))
        self.noise_clip = float(cfg.get("noise_clip", 0.5))

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor = MLPActor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.actor_target = MLPActor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim, self.hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.get("actor_lr", 1e-4)))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.get("critic_lr", 1e-3)))

        self.replay = ReplayBuffer(state_dim, action_dim, self.replay_size)
        self.total_updates = 0
        self._last_actor_loss = 0.0  # cached for steps where the actor is not updated

    def select_action(self, state, explore=True, noise_scale=None):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if explore:
            if noise_scale is None:
                sigma = np.full((self.action_dim,), float(self.action_noise), dtype=np.float32)
            else:
                sigma = np.asarray(noise_scale, dtype=np.float32)
                if sigma.shape == ():
                    sigma = np.full((self.action_dim,), float(sigma), dtype=np.float32)
                elif sigma.shape != (self.action_dim,):
                    sigma = np.full((self.action_dim,), float(self.action_noise), dtype=np.float32)
            noise = np.random.normal(0.0, sigma, size=self.action_dim).astype(np.float32)
            action = action + noise

        return np.clip(action, -self.max_action, self.max_action).astype(np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # --- Critic update (twin critics + target-policy smoothing) ---
        with torch.no_grad():
            noise = (torch.randn_like(actions_t) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states_t) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1, target_q2 = self.critic_target(next_states_t, next_actions)
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # --- Delayed actor + target network update ---
        actor_loss_value = self._last_actor_loss
        if (self.total_updates % self.policy_delay) == 0:
            actor_loss = -self.critic.Q1(states_t, self.actor(states_t)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optim.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            actor_loss_value = float(actor_loss.item())
            self._last_actor_loss = actor_loss_value

        self.total_updates += 1
        return {
            "actor_loss": actor_loss_value,
            "critic_loss": float(critic_loss.item()),
            "updates": self.total_updates,
        }

    def _soft_update(self, source_net, target_net):
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), source_net.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def save(self, path_prefix):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")

    def load(self, path_prefix):
        actor_path = f"{path_prefix}_actor.pth"
        critic_path = f"{path_prefix}_critic.pth"

        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


# Backward-compatible alias: existing code that imports DDPGAgent now gets TD3.
DDPGAgent = TD3Agent
