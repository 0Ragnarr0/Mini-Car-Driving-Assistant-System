"""Train the TD3 agent in CARLA.

Prerequisites:
  1. A running CARLA server, e.g.:   ./CarlaUE4.sh -quality-level=Low
  2. The CARLA Python API installed:  pip install carla   (match server version)

Run:
  python train_carla.py --episodes 500 --town Town03

Checkpoints are written to <models_dir>/td3_carla_ep_<N>_actor.pth / _critic.pth
(same naming/loading convention as the rest of the project).
"""

import os
import argparse
import time
import numpy as np

from config import RL_CONFIG, DATA_CONFIG
from rl_agent import TD3Agent
from carla_env import CarlaEnv


def parse_args():
    p = argparse.ArgumentParser(description="Train TD3 in CARLA")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default="Town03")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--target-speed", type=float, default=30.0, help="km/h")
    p.add_argument("--lidar-bins", type=int, default=12)
    p.add_argument("--warmup-steps", type=int, default=2000,
                   help="random-action steps before learning starts")
    p.add_argument("--save-every", type=int, default=20, help="episodes per checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default="",
                   help="checkpoint prefix to resume (without _actor.pth)")
    return p.parse_args()


def main():
    args = parse_args()

    env = CarlaEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        dt=args.dt,
        lidar_bins=args.lidar_bins,
        max_episode_steps=args.max_episode_steps,
        target_speed_kmh=args.target_speed,
        seed=args.seed,
    )

    # Use the project RL config but let CLI override warmup for CARLA.
    cfg = dict(RL_CONFIG)
    cfg["warmup_steps"] = args.warmup_steps

    agent = TD3Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=cfg,
    )
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
    ckpt_prefix = os.path.join(models_dir, "td3_carla")

    episode_returns = []
    total_steps = 0
    start = time.time()

    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            ep_return = 0.0
            ep_steps = 0
            done = False

            while not done:
                if total_steps < agent.warmup_steps:
                    # Broad random exploration before the policy is trained.
                    action = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
                else:
                    # Steering noise small, throttle noise larger (like the old setup).
                    action = agent.select_action(
                        state, explore=True, noise_scale=np.array([0.1, 0.3], dtype=np.float32)
                    )

                next_state, reward, done, info = env.step(action)
                # Store the action actually applied (env clipped it once).
                agent.store_transition(state, info["applied_action"], reward, next_state, done)
                agent.update()

                state = next_state
                ep_return += reward
                ep_steps += 1
                total_steps += 1

            episode_returns.append(ep_return)
            avg10 = np.mean(episode_returns[-10:])
            end_reason = ("collision" if info.get("collision") else
                          "off_lane" if info.get("off_lane") else
                          "timeout" if info.get("timeout") else "done")
            print(
                f"[Train] ep {ep:4d} | steps {ep_steps:4d} | return {ep_return:8.2f} | "
                f"avg10 {avg10:8.2f} | end={end_reason} | total_steps {total_steps} | "
                f"buffer {len(agent.replay)}"
            )

            if ep % args.save_every == 0:
                prefix = f"{ckpt_prefix}_ep_{ep}"
                agent.save(prefix)
                print(f"[Train] saved checkpoint: {prefix}_actor.pth / _critic.pth")

    except KeyboardInterrupt:
        print("\n[Train] Interrupted - saving final checkpoint...")
        agent.save(f"{ckpt_prefix}_interrupted")
    finally:
        env.close()
        elapsed = time.time() - start
        print(f"[Train] Done. {total_steps} steps in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
