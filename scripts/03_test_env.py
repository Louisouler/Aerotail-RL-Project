#!/usr/bin/env python3
# Quick sanity check: random actions for a few hundred steps, confirm no
# crashes and that reward/observation shapes look right.
import sys
sys.path.append("/mnt/disks/data/aerotail")
from envs.airsim_env import AirSimTargetFollowEnv

env = AirSimTargetFollowEnv()
obs, info = env.reset()
print("obs shape:", obs.shape, "sample obs:", obs)

for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0:
        print(f"step {i}: reward={reward:.3f} distance={info['distance']:.2f} visible={info['visible']}")
    if terminated or truncated:
        print("episode ended early")
        break

env.close()
print("smoke test done")
