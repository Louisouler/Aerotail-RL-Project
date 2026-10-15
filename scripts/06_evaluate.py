#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 06_evaluate.py
#
# Quantitative evaluation matching our Testing Plan: target-centring error,
# following-distance error, target-visibility ratio, control smoothness,
# and collision rate, averaged over a batch of held-out episodes with fixed
# seeds (not used during training) so runs are reproducible and comparable
# across algorithms/stages.
#
# Metrics are read from the environment's own info dict (ground truth),
# not from `obs`, since obs may be transformed by VecNormalize for
# ppo/recurrent_ppo but not for sac - reading metrics from info keeps
# comparisons fair and consistent regardless of which wrapper an algorithm
# happens to use.
#
# Usage:
#   python3 scripts/06_evaluate.py --algo sac --stage 0 \
#       --checkpoint checkpoints/rl/sac_stage0/final_model.zip
#   python3 scripts/06_evaluate.py --algo ppo --stage 1 \
#       --checkpoint checkpoints/rl/ppo_stage1/final_model.zip \
#       --vecnormalize checkpoints/rl/ppo_stage1/vecnormalize.pkl
# ---------------------------------------------------------------------------
import argparse
import sys
import numpy as np

sys.path.append("/mnt/disks/data/aerotail")

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from envs.airsim_env import AirSimTargetFollowEnv


CURRICULUM_STAGES = {
    0: dict(obs_noise_std=0.0, detection_dropout_prob=0.0, latency_steps=0,
            occlusion_prob=0.0, occlusion_max_steps=0),
    1: dict(obs_noise_std=0.03, detection_dropout_prob=0.01, latency_steps=0,
            occlusion_prob=0.0, occlusion_max_steps=0),
    2: dict(obs_noise_std=0.06, detection_dropout_prob=0.05, latency_steps=1,
            occlusion_prob=0.01, occlusion_max_steps=10),
    3: dict(obs_noise_std=0.10, detection_dropout_prob=0.10, latency_steps=2,
            occlusion_prob=0.02, occlusion_max_steps=20),
}

ON_POLICY_ALGOS = {"ppo", "recurrent_ppo"}
EVAL_SEEDS_BASE = 10_000


def make_env(stage):
    def _init():
        return AirSimTargetFollowEnv(**CURRICULUM_STAGES[stage])
    return _init


def load_policy(algo, checkpoint, stage, vecnormalize_path=None):
    cls = {"sac": SAC, "ppo": PPO, "recurrent_ppo": RecurrentPPO}[algo]
    base_env = DummyVecEnv([make_env(stage)])

    if algo in ON_POLICY_ALGOS:
        if vecnormalize_path is None:
            raise ValueError(f"{algo} needs --vecnormalize, it was trained with VecNormalize")
        env = VecNormalize.load(vecnormalize_path, base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env

    model = cls.load(checkpoint, env=env)
    return model, env


def run_episode(model, env, algo, seed):
    # env_method("reset", seed=...) reaches through the VecEnv wrapper(s)
    # straight to our own env's reset(), which is what actually accepts a
    # seed - this is the one call that both seeds AND returns the first obs.
    obs = env.env_method("reset", seed=seed)[0][0]
    obs = np.expand_dims(obs, axis=0)
    if isinstance(env, VecNormalize):
        obs = env.normalize_obs(obs)

    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    centring_errors, distance_errors, visibilities, smoothness = [], [], [], []
    prev_action = np.zeros(4, dtype=np.float32)
    total_reward = 0.0
    steps = 0
    collided = False
    done = False

    while not done:
        if algo == "recurrent_ppo":
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=episode_start, deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done_arr, infos = env.step(action)
        done = bool(done_arr[0])
        episode_start = done_arr
        info = infos[0]

        total_reward += float(np.asarray(reward).flatten()[0])
        steps += 1

        raw_action = np.asarray(action).reshape(-1)[:4].astype(np.float32)
        centring_errors.append(float(np.hypot(info["bearing_h"], info["bearing_v"])))
        distance_errors.append(abs(info["distance"] - 8.0))  # desired_distance default
        visibilities.append(info["visible"])
        smoothness.append(float(np.sum((raw_action - prev_action) ** 2)))
        prev_action = raw_action
        if info.get("collision", False):
            collided = True

    return dict(
        steps=steps,
        total_reward=total_reward,
        mean_centring_error=float(np.mean(centring_errors)),
        mean_distance_error=float(np.mean(distance_errors)),
        visibility_ratio=float(np.mean(visibilities)),
        mean_smoothness_penalty=float(np.mean(smoothness)),
        collided=collided,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "ppo", "recurrent_ppo"], required=True)
    parser.add_argument("--stage", type=int, choices=list(CURRICULUM_STAGES.keys()), default=0)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    model, env = load_policy(args.algo, args.checkpoint, args.stage, args.vecnormalize)

    results = []
    for i in range(args.episodes):
        seed = EVAL_SEEDS_BASE + i
        r = run_episode(model, env, args.algo, seed)
        results.append(r)
        print(f"episode {i:2d} (seed {seed}): steps={r['steps']:3d} "
              f"reward={r['total_reward']:8.1f} centring_err={r['mean_centring_error']:.3f} "
              f"dist_err={r['mean_distance_error']:.3f} visibility={r['visibility_ratio']:.2f} "
              f"collided={r['collided']}")

    print(f"\n== Aggregate over {args.episodes} episodes ({args.algo} stage {args.stage}) ==")
    for key in ["steps", "total_reward", "mean_centring_error", "mean_distance_error",
                "visibility_ratio", "mean_smoothness_penalty"]:
        vals = [r[key] for r in results]
        print(f"  {key:>24}: mean={np.mean(vals):8.3f}  std={np.std(vals):8.3f}")
    collision_rate = np.mean([r["collided"] for r in results])
    print(f"  {'collision_rate':>24}: {collision_rate:.2%}")

    env.close()


if __name__ == "__main__":
    main()
