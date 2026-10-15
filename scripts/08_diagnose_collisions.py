#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 08_diagnose_collisions.py
#
# Runs a trained policy for a batch of episodes and reports exactly what
# each collision hit, so we can tell ground/target/building collisions
# apart - now generalized to any algorithm (was SAC-only before), since we
# need this again to check whether the obstacle sensor actually changed
# WHAT gets collided with, not just the raw collision rate.
# ---------------------------------------------------------------------------
import sys
import argparse
import numpy as np
from collections import Counter

sys.path.append("/mnt/disks/data/aerotail")
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from envs.airsim_env import AirSimTargetFollowEnv

ON_POLICY_ALGOS = {"ppo", "recurrent_ppo"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "ppo", "recurrent_ppo"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vecnormalize", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    cls = {"sac": SAC, "ppo": PPO, "recurrent_ppo": RecurrentPPO}[args.algo]
    base_env = DummyVecEnv([lambda: AirSimTargetFollowEnv()])
    if args.algo in ON_POLICY_ALGOS:
        env = VecNormalize.load(args.vecnormalize, base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env
    model = cls.load(args.checkpoint, env=env)

    collision_objects = Counter()
    collision_altitudes = []

    for i in range(args.episodes):
        obs = env.env_method("reset", seed=10_000 + i)[0][0]
        obs = np.expand_dims(obs, axis=0)
        if isinstance(env, VecNormalize):
            obs = env.normalize_obs(obs)

        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        while not done:
            if args.algo == "recurrent_ppo":
                action, lstm_state = model.predict(
                    obs, state=lstm_state, episode_start=episode_start, deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, infos = env.step(action)
            done = bool(done_arr[0])
            episode_start = done_arr
            info = infos[0]
            if info.get("collision", False):
                obj = info.get("collision_object", "unknown")
                alt = info.get("altitude", None)
                obs_dist = info.get("obstacle_distance", None)
                collision_objects[obj] += 1
                collision_altitudes.append(alt)
                print(f"episode {i}: collided with '{obj}' at altitude {alt:.2f}m "
                      f"(obstacle sensor read {obs_dist:.2f}m just before)")

    print(f"\n== Collision breakdown over {args.episodes} episodes ==")
    for obj, count in collision_objects.most_common():
        print(f"  {obj}: {count}")
    if collision_altitudes:
        print(f"\nAltitude at collision: mean={np.mean(collision_altitudes):.2f}m "
              f"min={np.min(collision_altitudes):.2f}m max={np.max(collision_altitudes):.2f}m")

    env.close()


if __name__ == "__main__":
    main()
