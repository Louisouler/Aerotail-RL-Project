#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 10_dashboard_demo.py
#
# Dashboard-compatible single-model inference launcher. Runs deterministic
# inference with ONE existing checkpoint and publishes per-step telemetry
# to dashboard/runtime/telemetry.jsonl for the Dash monitor (dashboard/app.py).
#
# The control loop is a line-for-line mirror of run_episode() in
# 07_evaluate_all.py (same env construction, same VecNormalize settings
# with training=False / norm_reward=False, same seeded reset through
# env_method, same RecurrentPPO lstm_state / episode_start handling, same
# metric definitions). Curriculum settings are imported from 07 rather than
# copied, so they cannot drift. Nothing here trains, saves, or mutates a
# model or VecNormalize statistics.
#
# Telemetry taps (read-only, pass-through, monitoring only):
#   * the env's AirSim client getMultirotorState is wrapped so the drone
#     state the env ALREADY fetched inside _compute_observation() is kept
#     for telemetry - no extra RPC call, identical return value;
#   * the raw env's step() is wrapped to snapshot that state and the
#     target position at the end of each step, BEFORE DummyVecEnv's
#     auto-reset on a terminal step overwrites them.
# Pass --no-telemetry to run with neither tap installed (baseline for
# overhead measurement). SIGINT (Ctrl+C / the dashboard's Stop button) ends
# the run cleanly: the unfinished episode is dropped and run_end is written.
#
# Usage:
#   python3 scripts/10_dashboard_demo.py --algo recurrent_ppo --stage 0 --episodes 1
#   python3 scripts/10_dashboard_demo.py --algo ppo --stage 1 --episodes 3
#   python3 scripts/10_dashboard_demo.py --algo sac --stage 0 --vehicle Drone1 --target Target1
# ---------------------------------------------------------------------------
import argparse
import importlib.util
import json
import math
import os
import signal
import sys
import time

import numpy as np

PROJECT_DIR = "/mnt/disks/data/aerotail"
sys.path.append(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "dashboard"))

import airsim  # noqa: E402
from stable_baselines3 import SAC, PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from sb3_contrib import RecurrentPPO  # noqa: E402

from envs.airsim_env import AirSimTargetFollowEnv  # noqa: E402
from telemetry import TelemetryWriter, DEFAULT_TELEMETRY_PATH  # noqa: E402

# 07_evaluate_all.py is the authoritative evaluation code; its filename
# starts with a digit, so load it by path. Importing it only defines
# constants/functions (no evaluation runs at import time).
_spec = importlib.util.spec_from_file_location(
    "evaluate_all", os.path.join(PROJECT_DIR, "scripts", "07_evaluate_all.py"))
evaluate_all = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_all)

CURRICULUM_STAGES = evaluate_all.CURRICULUM_STAGES
ON_POLICY_ALGOS = evaluate_all.ON_POLICY_ALGOS
EVAL_SEEDS_BASE = evaluate_all.EVAL_SEEDS_BASE
CKPT_ROOT = evaluate_all.CKPT_ROOT

ALGO_CLASSES = {"sac": SAC, "ppo": PPO, "recurrent_ppo": RecurrentPPO}
ALGO_DISPLAY = {"sac": "SAC", "ppo": "PPO", "recurrent_ppo": "RecurrentPPO"}


def make_env(stage, vehicle, target, ip):
    def _init():
        # Same target_spawn rule as 04_train_rl.py (Drone2/Target2 ->
        # (10, 100, -1), as used by 07_evaluate_all.py).
        target_spawn = (10.0, 0.0, -1.0) if vehicle == "Drone1" else (10.0, 100.0, -1.0)
        return AirSimTargetFollowEnv(
            ip_address=ip, drone_name=vehicle, target_name=target,
            target_spawn=target_spawn, **CURRICULUM_STAGES[stage]
        )
    return _init


def load_policy(algo, checkpoint, stage, vecnormalize_path, vehicle, target, ip):
    # Same as 07_evaluate_all.load_policy, with a configurable vehicle pair.
    cls = ALGO_CLASSES[algo]
    base_env = DummyVecEnv([make_env(stage, vehicle, target, ip)])
    if algo in ON_POLICY_ALGOS:
        if not vecnormalize_path:
            raise ValueError(f"{algo} needs --vecnormalize-path, it was trained with VecNormalize")
        env = VecNormalize.load(vecnormalize_path, base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env
    model = cls.load(checkpoint, env=env)
    return model, env


class StateTap:
    """Pass-through wrapper that remembers the last getMultirotorState result."""

    def __init__(self, client):
        self._orig = client.getMultirotorState
        self.last = None
        client.getMultirotorState = self  # instance attribute shadows the method

    def __call__(self, *args, **kwargs):
        state = self._orig(*args, **kwargs)
        self.last = state
        return state


class StepTap:
    """Pass-through wrapper on the raw env's step(): snapshots the drone state
    used for this step's observation and the target position, before any
    DummyVecEnv auto-reset."""

    def __init__(self, raw_env, state_tap):
        self._env = raw_env
        self._orig = raw_env.step
        self._state_tap = state_tap
        self.drone_state = None
        self.target_pos = None
        raw_env.step = self

    def __call__(self, action):
        out = self._orig(action)
        self.drone_state = self._state_tap.last
        self.target_pos = np.array(self._env._target_pos, dtype=np.float64).copy()
        return out


def run_episode(model, env, algo, seed, episode_idx, ctx):
    """Mirror of 07_evaluate_all.run_episode with telemetry emission."""
    tel, raw_env, step_tap = ctx["telemetry"], ctx["raw_env"], ctx["step_tap"]
    sample_every = ctx["sample_every"]

    if tel:
        tel.emit(dict(type="episode_reset", timestamp=time.time(), episode=episode_idx, seed=seed))

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

    prev_target_pos = np.array(raw_env._target_pos, dtype=np.float64).copy() if tel else None
    lstm_resets = []  # episode_step indices at which predict() was told episode_start=True
    t_loop0 = time.perf_counter()

    while not done:
        lstm_reset_flag = bool(episode_start[0])
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

        r = float(np.asarray(reward).flatten()[0])
        total_reward += r
        steps += 1

        raw_action = np.asarray(action).reshape(-1)[:4].astype(np.float32)
        centring = float(np.hypot(info["bearing_h"], info["bearing_v"]))
        dist_err = abs(info["distance"] - 8.0)
        smooth = float(np.sum((raw_action - prev_action) ** 2))
        centring_errors.append(centring)
        distance_errors.append(dist_err)
        visibilities.append(info["visible"])
        smoothness.append(smooth)
        prev_action = raw_action
        if info.get("collision", False):
            collided = True

        if lstm_reset_flag:
            lstm_resets.append(steps)

        if tel and (steps % sample_every == 0 or done or steps == 1):
            rec = dict(
                type="step", timestamp=time.time(),
                run_mode="inference", algorithm=ctx["algo_display"], stage=ctx["stage"],
                episode=episode_idx, seed=seed, episode_step=steps,
                global_step=ctx["global_step"] + 1,
                reward=r, episode_reward=total_reward,
                collision=bool(info.get("collision", False)),
                collision_object=info.get("collision_object"),
                terminal=done,
                visibility=float(info["visible"]),
                centring_error=centring, distance_error=dist_err,
                smoothness_penalty=smooth,
                distance=float(info["distance"]),
                bearing_h=float(info["bearing_h"]), bearing_v=float(info["bearing_v"]),
                altitude=float(info["altitude"]),
            )
            ds = step_tap.drone_state
            if ds is not None:
                k = ds.kinematics_estimated
                _, _, yaw = airsim.to_eularian_angles(k.orientation)
                vx, vy, vz = k.linear_velocity.x_val, k.linear_velocity.y_val, k.linear_velocity.z_val
                rec.update(
                    drone_x=k.position.x_val, drone_y=k.position.y_val, drone_z=k.position.z_val,
                    drone_vx=vx, drone_vy=vy, drone_vz=vz,
                    drone_speed=math.sqrt(vx * vx + vy * vy + vz * vz),
                    drone_yaw_deg=math.degrees(yaw),
                )
            else:
                rec.update(drone_x=None, drone_y=None, drone_z=None, drone_vx=None,
                           drone_vy=None, drone_vz=None, drone_speed=None, drone_yaw_deg=None)
            tp = step_tap.target_pos
            if tp is not None:
                tv = (tp - prev_target_pos) / raw_env.dt
                rec.update(target_x=tp[0], target_y=tp[1], target_z=tp[2],
                           target_vx=tv[0], target_vy=tv[1], target_vz=tv[2])
            for i in range(4):
                rec[f"action_{i}"] = float(raw_action[i])
            rec["lstm_episode_start"] = lstm_reset_flag
            if algo == "recurrent_ppo" and lstm_state is not None:
                # Read-only diagnostic: L2 norm of the actor LSTM hidden state h.
                rec["lstm_hidden_norm"] = float(np.linalg.norm(np.asarray(lstm_state[0])))
            else:
                rec["lstm_hidden_norm"] = None
            tel.emit(rec)
        if tel and step_tap.target_pos is not None:
            prev_target_pos = step_tap.target_pos
        ctx["global_step"] += 1

    loop_time = time.perf_counter() - t_loop0
    result = dict(
        steps=steps, total_reward=total_reward,
        mean_centring_error=float(np.mean(centring_errors)),
        mean_distance_error=float(np.mean(distance_errors)),
        visibility_ratio=float(np.mean(visibilities)),
        mean_smoothness_penalty=float(np.mean(smoothness)),
        collided=collided,
        loop_time_s=loop_time,
        lstm_reset_steps=lstm_resets,
    )
    if tel:
        tel.emit(dict(type="episode_end", timestamp=time.time(), episode=episode_idx, seed=seed,
                      **result))
    return result


def main():
    p = argparse.ArgumentParser(description="Single-model inference run with dashboard telemetry.")
    p.add_argument("--algo", choices=sorted(ALGO_CLASSES), default="recurrent_ppo")
    p.add_argument("--stage", type=int, choices=sorted(CURRICULUM_STAGES), default=0)
    p.add_argument("--model-path", default=None,
                   help="default: final model of checkpoints/rl/active/<algo>_stage<stage>")
    p.add_argument("--vecnormalize-path", default=None,
                   help="default: the VecNormalize stats saved with --model-path's run (ppo/recurrent_ppo)")
    p.add_argument("--vehicle", default="Drone2")
    p.add_argument("--target", default="Target2")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=EVAL_SEEDS_BASE,
                   help="episode i uses seed seed_base + i (07 uses 10000)")
    p.add_argument("--ip", default="127.0.0.1")
    p.add_argument("--telemetry-path", default=DEFAULT_TELEMETRY_PATH)
    p.add_argument("--no-telemetry", action="store_true",
                   help="run without any telemetry taps (baseline for overhead measurement)")
    p.add_argument("--sample-every", type=int, default=1,
                   help="publish one step record every N env steps (first/terminal always published)")
    p.add_argument("--flush-every", type=int, default=10)
    p.add_argument("--summary-json", default=None, help="optional path to write the run summary")
    args = p.parse_args()

    # A process started in the background by a non-interactive shell inherits
    # SIGINT=ignored; restore the default so Ctrl+C / the dashboard's Stop
    # button raise KeyboardInterrupt and the run shuts down cleanly.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    model_path, vecnorm_path = args.model_path, args.vecnormalize_path
    if model_path is None:
        model_path, default_vecnorm = evaluate_all.latest_run_files(
            f"{CKPT_ROOT}/{args.algo}_stage{args.stage}")
        if vecnorm_path is None and args.algo in ON_POLICY_ALGOS:
            vecnorm_path = default_vecnorm
    if not os.path.exists(model_path):
        sys.exit(f"model not found: {model_path}")

    model, env = load_policy(args.algo, model_path, args.stage, vecnorm_path,
                             args.vehicle, args.target, args.ip)
    base = env.venv if isinstance(env, VecNormalize) else env
    raw_env = base.envs[0]

    tel = None
    step_tap = None
    if not args.no_telemetry:
        step_tap = StepTap(raw_env, StateTap(raw_env.client))
        tel = TelemetryWriter(args.telemetry_path, flush_every=args.flush_every)
        tel.emit(dict(
            type="run_start", timestamp=time.time(), run_mode="inference",
            algorithm=ALGO_DISPLAY[args.algo], algo_key=args.algo, stage=args.stage,
            stage_settings=CURRICULUM_STAGES[args.stage],
            model_path=model_path, vecnormalize_path=vecnorm_path,
            vehicle=args.vehicle, target=args.target, episodes=args.episodes,
            seed_base=args.seed_base, deterministic=True,
            episode_len_steps=raw_env.episode_len_steps, dt=raw_env.dt,
            desired_distance=raw_env.desired_distance,
            max_speed=raw_env.max_speed, max_yaw_rate=raw_env.max_yaw_rate,
            action_labels=[
                f"vx body forward (x{raw_env.max_speed:g} m/s)",
                f"vy body right (x{raw_env.max_speed:g} m/s)",
                f"vz body down, NED (x{raw_env.max_speed:g} m/s)",
                f"yaw rate (x{raw_env.max_yaw_rate:g} deg/s)",
            ],
            frame="AirSim NED (x north, y east, z down), env coordinate convention",
            pid=os.getpid(),
        ))

    ctx = dict(telemetry=tel, raw_env=raw_env, step_tap=step_tap, sample_every=max(1, args.sample_every),
               algo_display=ALGO_DISPLAY[args.algo], stage=args.stage, global_step=0)

    print(f"model: {model_path}\nvecnormalize: {vecnorm_path}\n"
          f"vehicle: {args.vehicle}/{args.target}  telemetry: "
          f"{'OFF' if tel is None else args.telemetry_path}")
    results = []
    interrupted = False
    t0 = time.perf_counter()
    try:
        for i in range(args.episodes):
            seed = args.seed_base + i
            try:
                r = run_episode(model, env, args.algo, seed, i, ctx)
            except KeyboardInterrupt:
                # SIGINT (Ctrl+C or the dashboard's Stop button): stop after
                # recording what was completed; the partial episode is dropped.
                interrupted = True
                print("interrupted - stopping", flush=True)
                break
            results.append(r)
            print(f"episode {i:2d} (seed {seed}): steps={r['steps']:3d} reward={r['total_reward']:8.1f} "
                  f"centring={r['mean_centring_error']:.3f} dist_err={r['mean_distance_error']:.3f} "
                  f"visibility={r['visibility_ratio']:.2f} collided={r['collided']} "
                  f"loop={r['steps'] / r['loop_time_s']:.2f} steps/s "
                  f"lstm_resets_at={r['lstm_reset_steps']}", flush=True)
    finally:
        wall = time.perf_counter() - t0
        total_steps = sum(r["steps"] for r in results)
        loop_time = sum(r["loop_time_s"] for r in results)
        summary = dict(
            algo=args.algo, stage=args.stage, telemetry=tel is not None,
            episodes_completed=len(results), interrupted=interrupted, total_steps=total_steps,
            loop_steps_per_s=(total_steps / loop_time) if loop_time else None,
            wall_steps_per_s_incl_resets=(total_steps / wall) if wall else None,
            telemetry_dropped=tel.dropped if tel else None,
        )
        if tel:
            tel.emit(dict(type="run_end", timestamp=time.time(), **summary))
            tel.close()
            summary["telemetry_written"] = tel.written
        print("SUMMARY " + json.dumps(summary))
        if args.summary_json:
            with open(args.summary_json, "w") as f:
                json.dump(dict(summary, episodes=results), f, indent=2)
        try:
            env.close()
        except RuntimeError:
            # An interrupt that lands inside an RPC call leaves this client's
            # event loop busy; release the drone exactly as env.close() does,
            # through a fresh client.
            client = airsim.MultirotorClient(ip=args.ip)
            client.armDisarm(False, args.vehicle)
            client.enableApiControl(False, args.vehicle)
            print("env.close() via fresh RPC client after interrupt", flush=True)


if __name__ == "__main__":
    main()
