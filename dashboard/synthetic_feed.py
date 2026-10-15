#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# synthetic_feed.py - TEST ONLY.
#
# Streams a short, clearly labelled SYNTHETIC telemetry run (no AirSim, no
# model) into the dashboard's telemetry file so the UI can be checked
# offline. Values are made up and labelled algorithm="SYNTHETIC-TEST"; never
# use them as results. Also injects a malformed line and a partially
# written line to exercise the reader's robustness.
#
#   python3 dashboard/synthetic_feed.py --steps 120 --episodes 2 --rate 20
# ---------------------------------------------------------------------------
import argparse
import math
import os
import random
import time

from telemetry import TelemetryWriter, DEFAULT_TELEMETRY_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry-path", default=DEFAULT_TELEMETRY_PATH)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--rate", type=float, default=20.0, help="records per second")
    p.add_argument("--no-run-end", action="store_true", help="stop without run_end (to test STALE)")
    args = p.parse_args()

    rng = random.Random(0)
    w = TelemetryWriter(args.telemetry_path, flush_every=5)
    w.emit(dict(type="run_start", timestamp=time.time(), run_mode="synthetic-test",
                algorithm="SYNTHETIC-TEST", stage=0,
                stage_settings=dict(obs_noise_std=0.0, detection_dropout_prob=0.0,
                                    latency_steps=0, occlusion_prob=0.0),
                model_path="synthetic/none", vehicle="Drone2", target="Target2",
                episodes=args.episodes, episode_len_steps=args.steps, dt=0.1, desired_distance=8.0,
                action_labels=["vx body forward (x5 m/s)", "vy body right (x5 m/s)",
                               "vz body down, NED (x5 m/s)", "yaw rate (x60 deg/s)"]))
    gs = 0
    for ep in range(args.episodes):
        w.emit(dict(type="episode_reset", timestamp=time.time(), episode=ep, seed=10000 + ep))
        time.sleep(1.0)
        total = 0.0
        tx, ty, heading = 10.0, 100.0, rng.uniform(0, 2 * math.pi)
        dx, dy = 0.0, 100.0
        prev_a = [0.0] * 4
        vis_n = 0
        for k in range(1, args.steps + 1):
            heading += rng.uniform(-0.05, 0.05)
            tx += 2.0 * math.cos(heading) * 0.1
            ty += 2.0 * math.sin(heading) * 0.1
            dx += 0.25 * (tx - 8 * math.cos(heading) - dx)
            dy += 0.25 * (ty - 8 * math.sin(heading) - dy)
            dist = math.hypot(tx - dx, ty - dy)
            visible = 0.0 if 40 <= k <= 48 else 1.0
            vis_n += visible
            a = [max(-1, min(1, math.sin(k / 10 + i) * 0.6 + rng.gauss(0, 0.05))) for i in range(4)]
            smooth = sum((x - y) ** 2 for x, y in zip(a, prev_a))
            prev_a = a
            centring = abs(rng.gauss(0.05, 0.03)) if visible else math.sqrt(2)
            collision = ep == args.episodes - 1 and k == args.steps
            r = -centring - abs(dist - 8) / 8 + (2 if visible else -2) - 0.1 * smooth - (10 if collision else 0)
            total += r
            gs += 1
            w.emit(dict(type="step", timestamp=time.time(), run_mode="synthetic-test",
                        algorithm="SYNTHETIC-TEST", stage=0, episode=ep, seed=10000 + ep,
                        episode_step=k, global_step=gs, reward=r, episode_reward=total,
                        collision=collision, collision_object="SyntheticWall" if collision else None,
                        terminal=k == args.steps, visibility=visible, centring_error=centring,
                        distance_error=abs(dist - 8), smoothness_penalty=smooth, distance=dist,
                        altitude=3 + 0.2 * math.sin(k / 15),
                        drone_x=dx, drone_y=dy, drone_z=-(3 + 0.2 * math.sin(k / 15)),
                        target_x=tx, target_y=ty, target_z=-1.0,
                        drone_vx=2 * math.cos(heading), drone_vy=2 * math.sin(heading), drone_vz=0.0,
                        drone_speed=2.0, target_vx=2 * math.cos(heading),
                        target_vy=2 * math.sin(heading), target_vz=0.0,
                        action_0=a[0], action_1=a[1], action_2=a[2], action_3=a[3],
                        lstm_episode_start=k == 1, lstm_hidden_norm=3 + math.sin(k / 20)))
            time.sleep(1.0 / args.rate)
        w.emit(dict(type="episode_end", timestamp=time.time(), episode=ep, seed=10000 + ep,
                    steps=args.steps, total_reward=total, mean_centring_error=0.1,
                    mean_distance_error=0.5, visibility_ratio=vis_n / args.steps,
                    mean_smoothness_penalty=0.01, collided=ep == args.episodes - 1))
    w.close()
    # Robustness: a malformed line and a partially written (unterminated) line.
    with open(args.telemetry_path, "a") as f:
        f.write("this is not json\n")
        f.write('{"type":"step","partial')
        f.flush()
    if not args.no_run_end:
        time.sleep(2)
        with open(args.telemetry_path, "a") as f:
            # Completes the partial line as valid JSON that is NOT a complete
            # step record -> must be rejected by the reader's schema check.
            f.write('":1}\n')
            f.write('{"type":"run_end","timestamp":%f,"episodes_completed":%d,"loop_steps_per_s":%f}\n'
                    % (time.time(), args.episodes, args.rate))
    print("synthetic feed done:", os.path.abspath(args.telemetry_path))


if __name__ == "__main__":
    main()
