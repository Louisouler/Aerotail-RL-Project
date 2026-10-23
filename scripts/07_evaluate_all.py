#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 07_evaluate_all.py
#
# Runs 06_evaluate.py's evaluation logic across every checkpoint we have,
# automatically, and writes one CSV row per (algorithm, stage) combination -
# this is the table we pull directly into the report's results section for
# the SAC vs PPO (vs RecurrentPPO later) comparison our testing plan calls
# for. We reuse the exact same env/policy-loading logic as 06_evaluate.py
# (duplicated here rather than imported, since 06_evaluate.py's filename
# starts with a digit and isn't import-friendly).
#
# Add a new entry to CONFIGS below every time a new checkpoint finishes
# training (e.g. stage 2/3, recurrent_ppo) and rerun this script - it will
# append to the existing CSV rather than overwrite it, so the comparison
# table grows as training progresses.
# ---------------------------------------------------------------------------
import csv
import sys
import os
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
EPISODES_PER_CONFIG = 20

CKPT_ROOT = "/mnt/disks/data/aerotail/checkpoints/rl/active"
OUTPUT_CSV = "/mnt/disks/data/aerotail/checkpoints/eval_results.csv"


def latest_run_files(run_dir):
    # A finished run has final_model.zip + vecnormalize.pkl. If a run was
    # stopped before finishing, fall back to its latest periodic checkpoint
    # and the VecNormalize stats saved alongside it (ckpt_N_steps.zip ->
    # ckpt_vecnormalize_N_steps.pkl).
    final_model = os.path.join(run_dir, "final_model.zip")
    if os.path.exists(final_model):
        return final_model, os.path.join(run_dir, "vecnormalize.pkl")
    steps = []
    for name in os.listdir(run_dir) if os.path.isdir(run_dir) else []:
        if name.startswith("ckpt_") and name.endswith("_steps.zip"):
            num = name[len("ckpt_"):-len("_steps.zip")]
            if num.isdigit():
                steps.append(int(num))
    if not steps:
        return final_model, os.path.join(run_dir, "vecnormalize.pkl")  # reported as SKIPPED
    n = max(steps)
    return (os.path.join(run_dir, f"ckpt_{n}_steps.zip"),
            os.path.join(run_dir, f"ckpt_vecnormalize_{n}_steps.pkl"))


RPPO_STAGE0_CKPT, RPPO_STAGE0_VECNORM = latest_run_files(f"{CKPT_ROOT}/recurrent_ppo_stage0")
RPPO_STAGE1_CKPT, RPPO_STAGE1_VECNORM = latest_run_files(f"{CKPT_ROOT}/recurrent_ppo_stage1")
RPPO_STAGE2_CKPT, RPPO_STAGE2_VECNORM = latest_run_files(f"{CKPT_ROOT}/recurrent_ppo_stage2")

# Every checkpoint to evaluate. Add rows here as new stages/algorithms
# finish training - existing entries are safe to re-run (they'll just
# produce a fresh row with the same config, useful if you retrain).
CONFIGS = [
    dict(algo="sac", stage=0, checkpoint=f"{CKPT_ROOT}/sac_stage0/final_model.zip", vecnormalize=None),
    dict(algo="sac", stage=1, checkpoint=f"{CKPT_ROOT}/sac_stage1/final_model.zip", vecnormalize=None),
    dict(algo="sac", stage=2, checkpoint=f"{CKPT_ROOT}/sac_stage2/final_model.zip", vecnormalize=None),
    dict(algo="sac", stage=3, checkpoint=f"{CKPT_ROOT}/sac_stage3/final_model.zip", vecnormalize=None),
    dict(algo="ppo", stage=0, checkpoint=f"{CKPT_ROOT}/ppo_stage0/final_model.zip",
         vecnormalize=f"{CKPT_ROOT}/ppo_stage0/vecnormalize.pkl"),
    dict(algo="ppo", stage=1, checkpoint=f"{CKPT_ROOT}/ppo_stage1/final_model.zip",
         vecnormalize=f"{CKPT_ROOT}/ppo_stage1/vecnormalize.pkl"),
    dict(algo="ppo", stage=2, checkpoint=f"{CKPT_ROOT}/ppo_stage2/final_model.zip",
         vecnormalize=f"{CKPT_ROOT}/ppo_stage2/vecnormalize.pkl"),
    dict(algo="ppo", stage=3, checkpoint=f"{CKPT_ROOT}/ppo_stage3/final_model.zip",
         vecnormalize=f"{CKPT_ROOT}/ppo_stage3/vecnormalize.pkl"),
    # RecurrentPPO is trained on stages 0-2 only.
    dict(algo="recurrent_ppo", stage=0, checkpoint=RPPO_STAGE0_CKPT, vecnormalize=RPPO_STAGE0_VECNORM),
    dict(algo="recurrent_ppo", stage=1, checkpoint=RPPO_STAGE1_CKPT, vecnormalize=RPPO_STAGE1_VECNORM),
    dict(algo="recurrent_ppo", stage=2, checkpoint=RPPO_STAGE2_CKPT, vecnormalize=RPPO_STAGE2_VECNORM),
]


def make_env(stage):
    def _init():
        # Drone2/Target2, not Drone1/Target1 - avoids conflicting with any
        # training currently running on Drone1 against the same simulator.
        return AirSimTargetFollowEnv(
            drone_name="Drone2", target_name="Target2",
            target_spawn=(10.0, 100.0, -1.0),
            **CURRICULUM_STAGES[stage]
        )
    return _init


def load_policy(algo, checkpoint, stage, vecnormalize_path=None):
    cls = {"sac": SAC, "ppo": PPO, "recurrent_ppo": RecurrentPPO}[algo]
    base_env = DummyVecEnv([make_env(stage)])
    if algo in ON_POLICY_ALGOS:
        env = VecNormalize.load(vecnormalize_path, base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env
    model = cls.load(checkpoint, env=env)
    return model, env


def run_episode(model, env, algo, seed):
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
        distance_errors.append(abs(info["distance"] - 8.0))
        visibilities.append(info["visible"])
        smoothness.append(float(np.sum((raw_action - prev_action) ** 2)))
        prev_action = raw_action
        if info.get("collision", False):
            collided = True

    return dict(
        steps=steps, total_reward=total_reward,
        mean_centring_error=float(np.mean(centring_errors)),
        mean_distance_error=float(np.mean(distance_errors)),
        visibility_ratio=float(np.mean(visibilities)),
        mean_smoothness_penalty=float(np.mean(smoothness)),
        collided=collided,
    )


def evaluate_config(cfg):
    print(f"\n--- Evaluating {cfg['algo']} stage {cfg['stage']} ---")
    print(f"  checkpoint: {cfg['checkpoint']}")
    if not os.path.exists(cfg["checkpoint"]):
        print(f"  SKIPPED: checkpoint not found at {cfg['checkpoint']}")
        return None

    model, env = load_policy(cfg["algo"], cfg["checkpoint"], cfg["stage"], cfg["vecnormalize"])

    results = []
    for i in range(EPISODES_PER_CONFIG):
        seed = EVAL_SEEDS_BASE + i
        r = run_episode(model, env, cfg["algo"], seed)
        results.append(r)
        print(f"  episode {i:2d}: steps={r['steps']:3d} reward={r['total_reward']:8.1f} "
              f"centring={r['mean_centring_error']:.3f} dist_err={r['mean_distance_error']:.3f} "
              f"visibility={r['visibility_ratio']:.2f} collided={r['collided']}")

    env.close()

    row = {"algo": cfg["algo"], "stage": cfg["stage"], "episodes": EPISODES_PER_CONFIG}
    for key in ["steps", "total_reward", "mean_centring_error", "mean_distance_error",
                "visibility_ratio", "mean_smoothness_penalty"]:
        vals = [r[key] for r in results]
        row[f"{key}_mean"] = float(np.mean(vals))
        row[f"{key}_std"] = float(np.std(vals))
    row["collision_rate"] = float(np.mean([r["collided"] for r in results]))
    return row


def main():
    rows = []
    for cfg in CONFIGS:
        row = evaluate_config(cfg)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No configs evaluated - nothing to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {OUTPUT_CSV}")

    print("\n== Summary ==")
    print(f"{'algo':<8}{'stage':<7}{'ep_len':<10}{'reward':<10}{'centring':<10}"
          f"{'dist_err':<10}{'visib':<8}{'collision%':<10}")
    for r in rows:
        print(f"{r['algo']:<8}{r['stage']:<7}{r['steps_mean']:<10.1f}{r['total_reward_mean']:<10.1f}"
              f"{r['mean_centring_error_mean']:<10.3f}{r['mean_distance_error_mean']:<10.3f}"
              f"{r['visibility_ratio_mean']:<8.2f}{r['collision_rate']*100:<10.1f}")


if __name__ == "__main__":
    main()
