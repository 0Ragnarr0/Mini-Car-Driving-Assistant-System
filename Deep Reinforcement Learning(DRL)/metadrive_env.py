"""MetaDrive environment wrapper (lightweight, runs on integrated GPU / CPU).

Same gym-style interface as carla_env.py so the TD3 agent plugs in unchanged:
    env = MetaDriveCarEnv()
    state = env.reset()
    next_state, reward, done, info = env.step(action)   # action = [steer, throttle] in [-1, 1]
    env.close()

MetaDrive is already a complete RL env: its action space is [steer, throttle/brake]
in [-1, 1] (matches the TD3 actor exactly) and it provides a built-in driving
reward (forward speed + lane keeping, with penalties for crashes / leaving the
road). The default observation is a flat vector (ego state + lidar-like rays),
which the MLP policy consumes directly.

Install:  pip install metadrive-simulator
Runs locally without a dedicated GPU.
"""

import numpy as np

try:
    from metadrive.envs.metadrive_env import MetaDriveEnv as _MetaDriveEnv
except Exception:  # pragma: no cover
    try:
        from metadrive import MetaDriveEnv as _MetaDriveEnv
    except Exception as exc:
        _MetaDriveEnv = None
        _METADRIVE_IMPORT_ERROR = exc


class MetaDriveCarEnv:
    def __init__(self, config=None, use_render=False, traffic_density=0.1,
                 horizon=1000, seed=0, num_scenarios=1):
        if _MetaDriveEnv is None:
            raise ImportError(
                "MetaDrive is not installed. Install it with "
                "`pip install metadrive-simulator`.\n"
                f"Original error: {_METADRIVE_IMPORT_ERROR}"
            )

        # num_scenarios = how many DIFFERENT maps the env cycles through. Training
        # on 1 map causes the policy to memorize one route (no generalization);
        # train on many (e.g. 1000) so it learns general driving. Eval on a
        # held-out seed range to measure true generalization.
        env_config = {
            "use_render": use_render,
            "traffic_density": traffic_density,
            "horizon": horizon,
            "start_seed": seed,
            "num_scenarios": int(num_scenarios),
            # Keep it a driving-policy task for now (no camera obs); the camera /
            # YOLOP perception upgrade comes later, matching the CARLA track.
            "image_observation": False,
        }
        if config:
            env_config.update(config)

        self.env = _MetaDriveEnv(env_config)
        self.action_dim = 2

        # Probe a reset to discover the observation dimension robustly across
        # MetaDrive versions (gym vs gymnasium return signatures).
        probe = self.reset()
        self.state_dim = int(probe.shape[0])

    def reset(self):
        out = self.env.reset()
        state = out[0] if isinstance(out, tuple) else out
        return np.asarray(state, dtype=np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        out = self.env.step(action)
        if len(out) == 5:  # gymnasium: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
        else:              # gym: obs, reward, done, info
            obs, reward, done, info = out
            done = bool(done)
        info = dict(info) if info else {}
        info["applied_action"] = action
        return np.asarray(obs, dtype=np.float32), float(reward), done, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
