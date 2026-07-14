"""Evaluate a trained TD3 checkpoint in MetaDrive (deterministic, no exploration).

Use this to SEE what the policy actually does (e.g. confirm whether it drives
straight off the road, or follows the lane).

Examples:
    # measure 10 episodes, no window:
    python eval_metadrive.py --ckpt trained_models/td3_metadrive_ep_80 --episodes 10

    # watch a few episodes in a window:
    python eval_metadrive.py --ckpt trained_models/td3_metadrive_ep_80 --episodes 3 --render

The --ckpt prefix is the path WITHOUT the _actor.pth / _critic.pth suffix.
"""

import os
import argparse
import numpy as np

from config import RL_CONFIG, DATA_CONFIG
from rl_agent import TD3Agent
from metadrive_env import MetaDriveCarEnv


def resolve_ckpt(ckpt):
    """Find the checkpoint, trying the given path then the project models_dir.

    Lets you pass just `td3_metadrive_ep_900` regardless of which folder you run
    from (checkpoints are saved to config.DATA_CONFIG['models_dir']).
    """
    if os.path.exists(ckpt + "_actor.pth"):
        return ckpt
    alt = os.path.join(DATA_CONFIG["models_dir"], os.path.basename(ckpt))
    if os.path.exists(alt + "_actor.pth"):
        return alt
    raise FileNotFoundError(
        f"Could not find '{ckpt}_actor.pth' (also tried '{alt}_actor.pth'). "
        f"Checkpoints live in: {DATA_CONFIG['models_dir']}"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a TD3 checkpoint in MetaDrive")
    p.add_argument("--ckpt", type=str, required=True,
                   help="checkpoint prefix (without _actor.pth)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--traffic-density", type=float, default=0.0,
                   help="match training conditions (train used 0)")
    p.add_argument("--seed", type=int, default=10000,
                   help="start map seed. Default is HELD-OUT (outside the default "
                        "training range 0-999) to test generalization. Use --seed 0 "
                        "--num-scenarios 1 to replay the exact training map.")
    p.add_argument("--num-scenarios", type=int, default=20,
                   help="number of distinct maps to evaluate on")
    p.add_argument("--render", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    env = MetaDriveCarEnv(
        use_render=args.render,
        traffic_density=args.traffic_density,
        horizon=args.horizon,
        seed=args.seed,
        num_scenarios=args.num_scenarios,
    )

    agent = TD3Agent(state_dim=env.state_dim, action_dim=env.action_dim, config=dict(RL_CONFIG))
    ckpt = resolve_ckpt(args.ckpt)
    agent.load(ckpt)
    print(f"[Eval] Loaded {ckpt} | state_dim={env.state_dim}")

    returns, lengths, arrivals, crashes = [], [], [], []
    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            ep_return, ep_steps, done = 0.0, 0, False
            while not done:
                # Deterministic policy: no exploration noise.
                action = agent.select_action(state, explore=False)
                state, reward, done, info = env.step(action)
                ep_return += reward
                ep_steps += 1
            arrive = bool(info.get("arrive_dest", False))
            crash = bool(info.get("crash", info.get("crash_vehicle", False)))
            returns.append(ep_return)
            lengths.append(ep_steps)
            arrivals.append(arrive)
            crashes.append(crash)
            print(f"[Eval] ep {ep:3d} | steps {ep_steps:4d} | return {ep_return:8.2f} | "
                  f"arrive={arrive} crash={crash}")
    finally:
        env.close()

    if returns:
        print("\n[Eval] Summary over", len(returns), "episodes:")
        print(f"  mean return : {np.mean(returns):8.2f}  (min {np.min(returns):.2f}, "
              f"max {np.max(returns):.2f})")
        print(f"  mean length : {np.mean(lengths):8.1f} steps")
        print(f"  arrive rate : {np.mean(arrivals)*100:5.1f}%")
        print(f"  crash rate  : {np.mean(crashes)*100:5.1f}%")


if __name__ == "__main__":
    main()
