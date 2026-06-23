"""Deep reinforcement learning agent (DDPG) for continuous driving control."""

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
    """Actor network that outputs bounded continuous actions in [-1, 1]."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class MLPCritic(nn.Module):
    """Critic network that estimates Q(state, action)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DDPGAgent:
    """DDPG agent for continuous steering and speed actions."""

    def __init__(self, state_dim, action_dim=2, config=None, device=None):
        cfg = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.batch_size = int(cfg.get("batch_size", 128))
        self.replay_size = int(cfg.get("replay_size", 100000))
        self.warmup_steps = int(cfg.get("warmup_steps", 2000))
        self.action_noise = float(cfg.get("action_noise", 0.15))
        self.hidden_dim = int(cfg.get("hidden_dim", 256))

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor = MLPActor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.actor_target = MLPActor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = MLPCritic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = MLPCritic(state_dim, action_dim, self.hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.get("actor_lr", 1e-4)))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.get("critic_lr", 1e-3)))

        self.replay = ReplayBuffer(state_dim, action_dim, self.replay_size)
        self.total_updates = 0

    def select_action(self, state, explore=True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if explore:
            noise = np.random.normal(0.0, self.action_noise, size=self.action_dim).astype(np.float32)
            action = action + noise

        return np.clip(action, -1.0, 1.0).astype(np.float32)

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

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            td_target = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        actor_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, actor_actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.total_updates += 1
        return {
            "actor_loss": float(actor_loss.item()),
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
