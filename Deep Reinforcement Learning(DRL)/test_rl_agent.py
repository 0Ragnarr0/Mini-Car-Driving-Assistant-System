"""Standalone test harness for the TD3 agent (rl_agent.py).

This verifies the *algorithm* works without needing Webots or CARLA:

  1. Smoke tests  -> shapes, action bounds, replay buffer, save/load round-trip,
                     and that update() returns finite losses.
  2. Learning test -> trains TD3 on a small 2D point-mass "reach the target"
                      environment (pure numpy) and checks that the average return
                      improves significantly from start to finish.

Run:  python test_rl_agent.py
Needs only: numpy, torch
"""

import os
import tempfile
import numpy as np
import torch

from rl_agent import TD3Agent, ReplayBuffer


# ---------------------------------------------------------------------------
# Toy environment: 2D point mass must reach a random target.
#   state  = [pos_x, pos_y, vel_x, vel_y, target_x, target_y]   (dim 6)
#   action = [accel_x, accel_y] in [-1, 1]                       (dim 2)
#   reward = -distance each step, +10 bonus for reaching target.
# A working continuous-control algorithm should learn to drive distance down.
# ---------------------------------------------------------------------------
class PointMassEnv:
    def __init__(self, dt=0.1, damping=0.9, max_steps=80, seed=0):
        self.dt = dt
        self.damping = damping
        self.max_steps = max_steps
        self.state_dim = 6
        self.action_dim = 2
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.pos = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.target = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.concatenate([self.pos, self.vel, self.target]).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.vel = self.vel * self.damping + action * self.dt
        self.pos = self.pos + self.vel
        self.t += 1

        dist = float(np.linalg.norm(self.pos - self.target))
        reward = -dist
        reached = dist < 0.15
        if reached:
            reward += 10.0
        done = reached or (self.t >= self.max_steps)
        return self._obs(), reward, done, {"dist": dist}


def make_agent(state_dim, action_dim, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = {
        "gamma": 0.99,
        "tau": 0.01,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "batch_size": 128,
        "replay_size": 50000,
        "warmup_steps": 1000,
        "hidden_dim": 128,
        "policy_delay": 2,
        "target_noise": 0.2,
        "noise_clip": 0.5,
        "action_noise": 0.1,
    }
    return TD3Agent(state_dim=state_dim, action_dim=action_dim, config=cfg, device="cpu")


def smoke_tests():
    print("=== Smoke tests ===")
    env = PointMassEnv(seed=1)
    agent = make_agent(env.state_dim, env.action_dim, seed=1)

    # Replay buffer
    buf = ReplayBuffer(env.state_dim, env.action_dim, capacity=10)
    s = env.reset()
    a = agent.select_action(s, explore=True)
    assert a.shape == (env.action_dim,), f"action shape {a.shape}"
    assert np.all(a >= -1.0) and np.all(a <= 1.0), "action out of [-1,1]"
    ns, r, d, _ = env.step(a)
    buf.add(s, a, r, ns, d)
    assert len(buf) == 1
    print("  [ok] action shape/bounds + replay buffer add")

    # update() returns None until warmup is satisfied
    assert agent.update() is None, "update should be None before warmup"
    print("  [ok] update() gated by warmup")

    # Fill buffer past warmup and confirm finite losses + actor delay
    s = env.reset()
    for _ in range(1500):
        a = agent.select_action(s, explore=True)
        ns, r, d, _ = env.step(a)
        agent.store_transition(s, a, r, ns, d)
        s = ns if not d else env.reset()
    losses = agent.update()
    assert losses is not None, "update should run after warmup"
    assert np.isfinite(losses["actor_loss"]), "actor_loss not finite"
    assert np.isfinite(losses["critic_loss"]), "critic_loss not finite"
    print(f"  [ok] update() losses finite: {losses}")

    # Deterministic action (explore=False) is reproducible
    s = env.reset()
    a1 = agent.select_action(s, explore=False)
    a2 = agent.select_action(s, explore=False)
    assert np.allclose(a1, a2), "deterministic action not reproducible"
    print("  [ok] deterministic action reproducible")

    # Save / load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "td3_test")
        agent.save(prefix)
        assert os.path.exists(prefix + "_actor.pth")
        assert os.path.exists(prefix + "_critic.pth")
        agent2 = make_agent(env.state_dim, env.action_dim, seed=99)
        agent2.load(prefix)
        b = agent2.select_action(s, explore=False)
        assert np.allclose(a1, b, atol=1e-5), "loaded policy differs from saved"
    print("  [ok] save/load round-trip preserves policy")
    print("Smoke tests PASSED\n")


def learning_test(num_episodes=160):
    print("=== Learning test (TD3 on PointMass) ===")
    env = PointMassEnv(seed=2)
    agent = make_agent(env.state_dim, env.action_dim, seed=2)

    returns = []
    total_steps = 0
    for ep in range(num_episodes):
        s = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            # Random actions during warmup encourage broad exploration early.
            if total_steps < agent.warmup_steps:
                a = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            else:
                a = agent.select_action(s, explore=True, noise_scale=0.1)
            ns, r, done, _ = env.step(a)
            agent.store_transition(s, a, r, ns, done)
            agent.update()
            s = ns
            ep_return += r
            total_steps += 1
        returns.append(ep_return)
        if (ep + 1) % 20 == 0:
            print(f"  episode {ep+1:3d} | avg return (last 20): {np.mean(returns[-20:]):.2f}")

    first = float(np.mean(returns[:20]))
    last = float(np.mean(returns[-20:]))
    print(f"\n  first-20-ep avg return: {first:.2f}")
    print(f"  last-20-ep  avg return: {last:.2f}")
    improvement = last - first
    print(f"  improvement: {improvement:+.2f}")

    # The agent should clearly improve. Threshold is loose to avoid flakiness.
    assert improvement > 2.0, (
        f"TD3 did not learn enough (improvement {improvement:.2f} <= 2.0). "
        "Something is wrong with the algorithm."
    )
    print("Learning test PASSED\n")
    return improvement


if __name__ == "__main__":
    smoke_tests()
    learning_test()
    print("ALL TESTS PASSED")
