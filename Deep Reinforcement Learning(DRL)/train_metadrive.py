"""Train the TD3 agent in MetaDrive (local, no dedicated GPU needed).

Prerequisite:
    pip install metadrive-simulator

Run:
    python train_metadrive.py --episodes 500

Checkpoints: <models_dir>/td3_metadrive_ep_<N>_actor.pth / _critic.pth
"""

import os
import sys
import argparse
import time
import datetime
import numpy as np

from config import RL_CONFIG, DATA_CONFIG
from rl_agent import TD3Agent
from metadrive_env import MetaDriveCarEnv


class _Tee:
    """Write all print() output to both the console and a log file."""

    def __init__(self, file_handle):
        self._console = sys.stdout
        self._file = file_handle

    def write(self, data):
        self._console.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._console.flush()
        self._file.flush()


try:
    import matplotlib
    matplotlib.use("Agg")  # headless: save PNG progress plots without a GUI window
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _rolling(x, window):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - window + 1): i + 1].mean()
    return out


def save_progress_plot(path, returns, avg10s, successes, collisions, window=20):
    """Save a 2-panel learning-progress chart (like watching football DRL learn)."""
    if plt is None or not returns:
        return
    episodes = list(range(1, len(returns) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(episodes, returns, color="#9bbcff", alpha=0.5, label="episode return")
    ax1.plot(episodes, avg10s, color="#1f6feb", linewidth=2, label="avg(10)")
    ax1.set_ylabel("return"); ax1.grid(alpha=0.3); ax1.legend(loc="upper left")
    ax1.set_title("MetaDrive TD3 — training progress")
    ax2.plot(episodes, _rolling(successes, window), color="#2ea043", linewidth=2,
             label=f"arrive rate ({window})")
    ax2.plot(episodes, _rolling(collisions, window), color="#d9534f", linewidth=2,
             label=f"crash rate ({window})")
    ax2.set_ylabel("rate"); ax2.set_xlabel("episode"); ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3); ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Train TD3 in MetaDrive")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--horizon", type=int, default=1000, help="max steps per episode")
    p.add_argument("--num-scenarios", type=int, default=1000,
                   help="number of distinct maps to train on (>1 prevents the policy "
                        "from memorizing a single route)")
    p.add_argument("--traffic-density", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=3000,
                   help="random-action steps before learning starts")
    p.add_argument("--updates-per-step", type=int, default=2,
                   help="gradient updates per env step (higher = fewer env steps "
                        "needed to learn, but slower per step)")
    p.add_argument("--steer-noise", type=float, default=0.2,
                   help="exploration noise std on steering")
    p.add_argument("--throttle-noise", type=float, default=0.3,
                   help="exploration noise std on throttle")
    p.add_argument("--save-every", type=int, default=20, help="episodes per checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="show the simulator window")
    p.add_argument("--resume", type=str, default="",
                   help="checkpoint prefix to resume (without _actor.pth)")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-log every run so there is always a file to inspect afterwards.
    logs_dir = DATA_CONFIG.get("logs_dir", os.path.join(os.getcwd(), "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"train_metadrive_{stamp}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(log_file)
    print(f"[Train] Logging to: {log_path}")

    env = MetaDriveCarEnv(
        use_render=args.render,
        traffic_density=args.traffic_density,
        horizon=args.horizon,
        seed=args.seed,
        num_scenarios=args.num_scenarios,
    )

    cfg = dict(RL_CONFIG)
    cfg["warmup_steps"] = args.warmup_steps

    agent = TD3Agent(state_dim=env.state_dim, action_dim=env.action_dim, config=cfg)
    print(f"[Train] TD3 agent | state_dim={env.state_dim}, action_dim={env.action_dim}, "
          f"device={agent.device}")

    if args.resume:
        try:
            agent.load(args.resume)
            print(f"[Train] Resumed from {args.resume}")
        except Exception as exc:
            print(f"[Train] Could not resume from {args.resume}: {exc}")

    models_dir = DATA_CONFIG["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    ckpt_prefix = os.path.join(models_dir, "td3_metadrive")

    # Live progress tracking: CSV + auto-updating plot (watch it learn).
    metrics_path = os.path.join(logs_dir, f"train_metadrive_{stamp}.csv")
    plot_path = os.path.join(logs_dir, f"train_metadrive_{stamp}_progress.png")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        mf.write("episode,total_steps,return,avg10,arrive,crash\n")
    print(f"[Train] Metrics CSV : {metrics_path}")
    print(f"[Train] Progress plot: {plot_path}  (updates every {args.save_every} eps)")
    print(f"[Train] Live view    : python plot_progress.py   (in another terminal)")

    episode_returns = []
    avg10s, successes, collisions = [], [], []
    total_steps = 0
    start = time.time()

    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            ep_return = 0.0
            ep_steps = 0
            done = False

            expl_noise = np.array([args.steer_noise, args.throttle_noise], dtype=np.float32)
            while not done:
                if total_steps < agent.warmup_steps:
                    action = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
                else:
                    action = agent.select_action(state, explore=True, noise_scale=expl_noise)

                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, info["applied_action"], reward, next_state, done)
                # Update-to-data ratio: several gradient steps per env step improves
                # sample efficiency (learn from fewer environment steps).
                if total_steps >= agent.warmup_steps:
                    for _ in range(args.updates_per_step):
                        agent.update()

                state = next_state
                ep_return += reward
                ep_steps += 1
                total_steps += 1

            episode_returns.append(ep_return)
            avg10 = float(np.mean(episode_returns[-10:]))
            arrive = bool(info.get("arrive_dest", False))
            crash = bool(info.get("crash", info.get("crash_vehicle", False)))
            avg10s.append(avg10)
            successes.append(1 if arrive else 0)
            collisions.append(1 if crash else 0)
            with open(metrics_path, "a", encoding="utf-8") as mf:
                mf.write(f"{ep},{total_steps},{ep_return:.4f},{avg10:.4f},"
                         f"{int(arrive)},{int(crash)}\n")
            print(
                f"[Train] ep {ep:4d} | steps {ep_steps:4d} | return {ep_return:8.2f} | "
                f"avg10 {avg10:8.2f} | arrive={arrive} crash={crash} | "
                f"total_steps {total_steps} | buffer {len(agent.replay)}"
            )

            if ep % args.save_every == 0:
                prefix = f"{ckpt_prefix}_ep_{ep}"
                agent.save(prefix)
                save_progress_plot(plot_path, episode_returns, avg10s, successes, collisions)
                print(f"[Train] saved checkpoint + updated progress plot: {plot_path}")

    except KeyboardInterrupt:
        print("\n[Train] Interrupted - saving final checkpoint...")
        agent.save(f"{ckpt_prefix}_interrupted")
    finally:
        save_progress_plot(plot_path, episode_returns, avg10s, successes, collisions)
        env.close()
        elapsed = time.time() - start
        print(f"[Train] Done. {total_steps} steps in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
