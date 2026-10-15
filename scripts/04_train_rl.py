#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 04_train_rl.py
#
# Trains the target-following policy in AirSim, across the curriculum
# stages from our proposal (Stage 3 of the implementation plan).
#
# We only run ONE environment instance (single AirSim simulator), wrapped
# in DummyVecEnv since SB3 requires a VecEnv even for one env.
#
# VecNormalize note: PPO/RecurrentPPO are on-policy and learn their value
# function directly from raw returns, which makes them much more sensitive
# to reward scale/variance than SAC (off-policy, uses a replay buffer +
# automatic entropy tuning). Our reward mixes small continuous penalties
# with a lumpy -10 collision penalty - exactly the kind of unnormalized,
# spiky signal that causes PPO's value function to fail to converge (we
# saw this directly: explained_variance stuck near 0 after 200k steps on
# our first PPO run). VecNormalize (running mean/std normalization of obs
# and reward) is standard practice for PPO in continuous control and fixes
# this. We don't apply it to SAC, which trained fine without it.
#
# Usage examples:
#   python3 scripts/04_train_rl.py --algo sac   --stage 0 --timesteps 200000
#   python3 scripts/04_train_rl.py --algo ppo   --stage 0 --timesteps 200000
#   python3 scripts/04_train_rl.py --algo recurrent_ppo --stage 3 \
#       --timesteps 300000 --load-from checkpoints/rl/ppo_stage2/final_model.zip
# ---------------------------------------------------------------------------
import argparse
import sys
from pathlib import Path

sys.path.append("/mnt/disks/data/aerotail")

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
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


def make_env(stage, drone_name="Drone1", target_name="Target1"):
    def _init():
        # Monitor tracks per-episode reward/length so SB3 can log
        # rollout/ep_rew_mean and ep_len_mean - the metrics we need for a
        # fair SAC vs PPO vs RecurrentPPO comparison across curriculum
        # stages (per our testing plan). drone_name/target_name let two
        # algorithms train genuinely concurrently against the same running
        # simulator, each owning its own vehicle pair.
        target_spawn = (10.0, 0.0, -1.0) if drone_name == "Drone1" else (10.0, 100.0, -1.0)
        return Monitor(AirSimTargetFollowEnv(
            drone_name=drone_name, target_name=target_name,
            target_spawn=target_spawn, **CURRICULUM_STAGES[stage]
        ))
    return _init


def build_env(algo, stage, drone_name="Drone1", target_name="Target1"):
    base_env = DummyVecEnv([make_env(stage, drone_name, target_name)])
    if algo in ON_POLICY_ALGOS:
        # norm_obs + norm_reward: running normalization, updated online as
        # training progresses. clip_reward bounds the normalized reward to
        # avoid rare large spikes (e.g. right after a collision) from still
        # dominating the value target.
        return VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    return base_env


def build_model(algo, env, tensorboard_log):
    if algo == "sac":
        return SAC(
            "MlpPolicy", env, verbose=1,
            learning_rate=3e-4, buffer_size=200_000, batch_size=256,
            train_freq=1, gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=tensorboard_log,
        )
    elif algo == "ppo":
        return PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=tensorboard_log,
        )
    elif algo == "recurrent_ppo":
        return RecurrentPPO(
            "MlpLstmPolicy", env, verbose=1,
            learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10,
            policy_kwargs=dict(net_arch=[256, 256], lstm_hidden_size=128),
            tensorboard_log=tensorboard_log,
        )
    else:
        raise ValueError(f"unknown algo {algo}")


def load_model(algo, path, env, learning_rate=None):
    cls = {"sac": SAC, "ppo": PPO, "recurrent_ppo": RecurrentPPO}[algo]
    model = cls.load(path, env=env)
    if learning_rate is not None:
        # SB3 does NOT read the optimizer's own param_groups during
        # training - every train() call recomputes the LR from
        # model.lr_schedule (a callable) and overwrites the optimizer with
        # that value. Patching the optimizer directly (what we tried first)
        # gets silently clobbered back to the original schedule on the very
        # next gradient step. We have to replace lr_schedule itself.
        model.lr_schedule = get_schedule_fn(learning_rate)
        model.learning_rate = learning_rate
        print(f"Overrode lr_schedule to a constant {learning_rate}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "ppo", "recurrent_ppo"], required=True)
    parser.add_argument("--stage", type=int, choices=list(CURRICULUM_STAGES.keys()), default=0)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--load-from", type=str, default=None,
                         help="path to a .zip checkpoint to continue training from")
    parser.add_argument("--vehicle", type=str, default="Drone1",
                         help="which drone to control (Drone1 or Drone2) - lets two "
                              "algorithms train concurrently against the same simulator")
    parser.add_argument("--target", type=str, default="Target1",
                         help="matching target vehicle name (Target1 or Target2)")
    parser.add_argument("--learning-rate", type=float, default=None,
                         help="override the optimizer learning rate when continuing "
                              "training (e.g. a much smaller LR when fine-tuning into "
                              "a new curriculum stage, to avoid destabilizing on "
                              "out-of-distribution transitions)")
    parser.add_argument("--vecnormalize-from", type=str, default=None,
                         help="path to a saved VecNormalize .pkl to continue stats from "
                              "(only relevant for ppo/recurrent_ppo)")
    parser.add_argument("--checkpoint-dir", type=str,
                         default="/mnt/disks/data/aerotail/checkpoints/rl/active")
    parser.add_argument("--log-dir", type=str,
                         default="/mnt/disks/data/aerotail/logs")
    args = parser.parse_args()

    run_name = f"{args.algo}_stage{args.stage}"
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args.algo, args.stage, args.vehicle, args.target)

    # When resuming from a periodic checkpoint, prefer the VecNormalize stats
    # saved alongside that exact checkpoint (ckpt_N_steps.zip ->
    # ckpt_vecnormalize_N_steps.pkl) over --vecnormalize-from, so a crash
    # resume continues with the stats the policy was actually trained against.
    vecnormalize_path = args.vecnormalize_from
    if args.load_from:
        load_stem = Path(args.load_from).stem
        if load_stem.startswith("ckpt_") and load_stem.endswith("_steps"):
            step_num = load_stem[len("ckpt_"):-len("_steps")]
            matching = Path(args.load_from).parent / f"ckpt_vecnormalize_{step_num}_steps.pkl"
            if matching.exists():
                vecnormalize_path = str(matching)
    if vecnormalize_path and isinstance(env, VecNormalize):
        env = VecNormalize.load(vecnormalize_path, env.venv)
        print(f"Loaded VecNormalize stats from {vecnormalize_path}")

    if args.load_from:
        print(f"Continuing training from {args.load_from}")
        model = load_model(args.algo, args.load_from, env, learning_rate=args.learning_rate)
        # Restore the replay buffer too, if one was saved alongside this
        # checkpoint - avoids the cold-buffer performance dip on resume.
        # CheckpointCallback names its buffer file after the checkpoint
        # itself (e.g. ckpt_95000_steps.zip -> ckpt_replay_buffer_95000_steps.pkl),
        # while our own final-save uses a plain replay_buffer.pkl - check both.
        load_stem = Path(args.load_from).stem  # e.g. "ckpt_95000_steps"
        candidates = [Path(args.load_from).parent / "replay_buffer.pkl"]
        if load_stem.startswith("ckpt_") and load_stem.endswith("_steps"):
            step_num = load_stem[len("ckpt_"):-len("_steps")]
            candidates.append(Path(args.load_from).parent / f"ckpt_replay_buffer_{step_num}_steps.pkl")
        buffer_path = next((c for c in candidates if c.exists()), None)
        if hasattr(model, "load_replay_buffer") and buffer_path is not None:
            model.load_replay_buffer(str(buffer_path))
            print(f"Restored replay buffer from {buffer_path}")
        elif hasattr(model, "load_replay_buffer"):
            print("No matching replay buffer found - resuming with a cold buffer")
    else:
        model = build_model(args.algo, env, tensorboard_log=args.log_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000, save_path=str(ckpt_dir), name_prefix="ckpt",
        save_replay_buffer=True, save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        reset_num_timesteps=(args.load_from is None),
    )

    final_path = ckpt_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}")

    # For off-policy algorithms (SAC), also save the replay buffer - without
    # it, resuming from a checkpoint after a crash starts with an EMPTY
    # buffer, which we found causes a real (if usually temporary)
    # performance dip while it refills, since a near-converged policy
    # trains on a tiny, unrepresentative buffer via train_freq=1 updates.
    if hasattr(model, "save_replay_buffer"):
        buffer_path = ckpt_dir / "replay_buffer.pkl"
        model.save_replay_buffer(str(buffer_path))
        print(f"Saved replay buffer to {buffer_path}")

    # Save VecNormalize running stats alongside the model - needed to
    # correctly evaluate or continue training this policy later, since the
    # normalization statistics are part of what the policy learned against.
    if isinstance(env, VecNormalize):
        vecnorm_path = ckpt_dir / "vecnormalize.pkl"
        env.save(str(vecnorm_path))
        print(f"Saved VecNormalize stats to {vecnorm_path}")

    env.close()


if __name__ == "__main__":
    main()
