# AeroTail live monitoring dashboard

A Plotly Dash dashboard for watching a single-model inference run of the AirSim
target-following policy (RecurrentPPO, PPO or SAC): live telemetry charts, a live view of
the simulator window, and optional start/stop of a demo run.

```
AirSim Blocks or AirSimNH (windowed on Xvfb :99)
   |  RPC (only from the launcher)              |  X display pixels
scripts/10_dashboard_demo.py                     |
   |  TelemetryWriter.emit(): non-blocking (~17 us/step)
   v                                             |
dashboard/runtime/telemetry.jsonl (append-only)  |
   |  file tail, every 1 s                       |  MJPEG /sim.mjpg
   v                                             v
dashboard/app.py (separate process and venv)  ->  http://127.0.0.1:8050
```

The dashboard never calls the AirSim API, never loads a model, never calls
`predict()`/`learn()` and never touches VecNormalize or LSTM state. Closing it does not
affect the simulator or a running demo (tested).

## Files

| File | Purpose |
|---|---|
| `scripts/10_dashboard_demo.py` | Inference launcher: one model, deterministic, publishes telemetry. |
| `dashboard/demo.sh` | One-command start/stop/status of Xvfb, the simulator and the dashboard. |
| `dashboard/app.py` | The Dash app. |
| `dashboard/telemetry.py` | `TelemetryWriter` (non-blocking, drops instead of blocking) and `TelemetryReader` (incremental tail; tolerates partial/malformed lines; detects new runs). Standard library only. |
| `dashboard/simview.py` | Simulator view: MJPEG stream of the X display at `/sim.mjpg` (pixels only). |
| `dashboard/control.py` | Run control: starts/stops the launcher as a separate process. |
| `dashboard/record_demo.py` | Records a backup video of one demo run. |
| `dashboard/synthetic_feed.py` | TEST ONLY: streams a labelled synthetic run for offline UI checks. |
| `dashboard/assets/` | `style.css` (dark theme) and `simview.js` (enlarge/fullscreen); served by Dash automatically. |
| `dashboard/runtime/` | `telemetry.jsonl` (current run; previous runs move to `archive/`) and `demo.log`. |
| `dashboard/recordings/` | Output of `record_demo.py`. |
| `dashboard/backup/` | Earlier versions of the files above, kept for reference. |

No pre-existing project file was modified.

## Setup

The dashboard has its **own** venv, so the experiment venv (`/mnt/disks/data/aerotail-venv`) is unchanged:

```bash
python3 -m venv /mnt/disks/data/aerotail-dashboard-venv
/mnt/disks/data/aerotail-dashboard-venv/bin/pip install dash pillow          # dashboard + simulator view
/mnt/disks/data/aerotail-dashboard-venv/bin/pip install imageio-ffmpeg playwright   # recording only
/mnt/disks/data/aerotail-dashboard-venv/bin/python -m playwright install chromium   # recording only
```

The launcher needs nothing new (`telemetry.py` uses only the standard library).

## Quick start

```bash
/mnt/disks/data/aerotail/dashboard/demo.sh start     # Xvfb :99 -> Blocks (windowed) -> dashboard; skips what is running
/mnt/disks/data/aerotail/dashboard/demo.sh start --env nh   # same with AirSimNH (houses, trees, parked cars)
/mnt/disks/data/aerotail/dashboard/demo.sh status
/mnt/disks/data/aerotail/dashboard/demo.sh restart   # restart the dashboard only
/mnt/disks/data/aerotail/dashboard/demo.sh stop      # stop the dashboard (add --all for simulator + Xvfb;
                                                     # refused while a training job runs)
```

`start` prints the SSH tunnel command; then open http://localhost:8050 on your own computer.
Only one environment runs at a time; to switch, `stop --all` and `start --env <blocks|nh>`.
AirSimNH is `AirSimNH.zip` from the AirSim v1.7.0-linux GitHub release, unpacked into
`sim/AirSimNH/`; it uses the same `settings.json`. The models were trained in Blocks only;
in AirSimNH they track the target as well but collide more often with trees and power lines.
Logs: `runtime/{xvfb,blocks,airsimnh,dashboard}.log`.

## Run manually

```bash
cd /mnt/disks/data/aerotail
Xvfb :99 -screen 0 1280x720x24 &
(cd sim/Blocks/Blocks/LinuxNoEditor && DISPLAY=:99 ./Blocks.sh -windowed -ResX=1280 -ResY=720 > /tmp/blocks.log 2>&1 &)
/mnt/disks/data/aerotail-dashboard-venv/bin/python dashboard/app.py
ssh -L 8050:127.0.0.1:8050 <user>@<vm>        # on your own computer, then open http://localhost:8050
```

Start a demo with the control bar, or from the command line:

```bash
/mnt/disks/data/aerotail-venv/bin/python scripts/10_dashboard_demo.py --algo recurrent_ppo --stage 0 --episodes 1
```

Launcher options: `--algo {recurrent_ppo,ppo,sac}`, `--stage N`, `--model-path`, `--vecnormalize-path`
(default: final model and VecNormalize stats of `checkpoints/rl/active/<algo>_stage<N>`, resolved as in 07),
`--vehicle Drone2 --target Target2`, `--episodes`, `--seed-base 10000`, `--sample-every N`,
`--no-telemetry` (no taps; overhead baseline), `--summary-json PATH`.

Dashboard options: `--port 8050 --refresh-ms 1000 --history 600 --action-window 100 --stale-after 15`,
`--read-only`, `--sim-display :99 --sim-fps 30 --sim-width 854 --sim-hd-width 1280 --sim-hd-fps 30`,
`--no-sim-view`.

## Dashboard

**Layout:** header (run identity and status) → run control bar → 8 KPI cards → Simulator view |
Trajectory → Tracking error | Reward | Flight state → Policy actions | Recurrent state → Completed episodes.
Time-series panels show the **current episode** on a fixed 0..300 step axis; finished episodes go to
the table. Each panel header shows the latest value; red vertical lines mark collisions. Figures are
rebuilt only when new telemetry arrives and are shared by all open tabs.

**States:** WAITING FOR TELEMETRY, STARTING, LIVE, RESETTING (reset/takeoff in progress),
STALE DATA (no new records for `--stale-after` s, or 3× that during a reset), RUN FINISHED,
RUN STOPPED (stopped by the user).

**Simulator view:** pixels of the X display (Pillow `ImageGrab`), pushed as MJPEG on the dashboard's
own port: 854x480 at 30 fps (~470 KB/s). A pushed stream is used instead of VNC, which is
request/response and caps at ~5 fps over a ~190 ms round trip. **Enlarge** widens the panel inside
the dashboard; **Fullscreen** (or double-click; Esc exits) fills the screen. Both switch to a
1280x720 stream (`/sim.mjpg?q=hd`, ~1 MB/s) and back afterwards; each stream is only captured while
watched. The panel reports "unavailable" if the display does not exist (e.g. `-RenderOffscreen`).

**Run control** (Model · Vehicle · Episodes · Start · Stop) starts the launcher as a separate process
(own session; log in `runtime/demo.log`). Rules in `control.py`:
- only finished models in `checkpoints/rl/active` are listed (on-policy ones need `vecnormalize.pkl`);
- at most one demo at a time; Start is refused while `04_train_rl.py` / `09_train_with_retry.sh` runs or
  when AirSim's RPC port 41451 is closed;
- Stop sends SIGINT: the unfinished episode is dropped, `run_end` is written (`interrupted: true`) and
  the drone is released; SIGTERM follows only after 20 s;
- only enabled on a loopback host; `--host 0.0.0.0` or `--read-only` gives a read-only dashboard.

Vehicle: Drone1/Target1 is followed by the simulator's default camera (best for a visual demo; do not
use it while training runs on Drone1). Drone2/Target2 is the evaluation pair used by 07.

## Backup recording

```bash
/mnt/disks/data/aerotail-dashboard-venv/bin/python dashboard/record_demo.py --model recurrent_ppo:0 --episodes 1
```

Starts one demo (same rules as the Start button), records until it ends and writes to `recordings/`:
`<stamp>_<model>_sim.mp4` (simulator display, 1280x720, 30 fps, H.264) and
`<stamp>_<model>_dashboard.mp4` (the dashboard page, 1080p). Requires the dashboard to be running.

`--seed-base N` replaces the 07 evaluation seeds (default 10000) and adds `_seedN` to the file
names. The seed only selects the target's random route. In AirSimNH the evaluation seeds often send
the target through houses and gardens (it ignores collisions), which the chase camera cannot show;
`--seed-base 31539` sends it straight north along the street from Drone1's spawn point, which makes
a readable video. Such a hand-picked route is not representative of the evaluation results.

## Faithfulness to the evaluation

`10_dashboard_demo.py` follows `run_episode()` in `07_evaluate_all.py` line for line:
`VecNormalize.load(...)` with `training=False, norm_reward=False`; seeded reset through
`env.env_method("reset", seed=...)` plus `normalize_obs`;
`model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)` with
`lstm_state=None, episode_start=ones` at episode start and `episode_start = done` afterwards.
`CURRICULUM_STAGES`, `EVAL_SEEDS_BASE` and the checkpoint lookup are imported from 07, not copied.

Two pass-through taps, installed on the instances at runtime (no source change), expose fields the
env computes but does not put in `info`:
1. `env.client.getMultirotorState` is wrapped to keep the drone state `_compute_observation()`
   **already fetched** (no extra RPC call, return value unchanged);
2. `raw_env.step` is wrapped to snapshot that state and `env._target_pos` before DummyVecEnv's
   auto-reset overwrites them on a terminal step.

## Telemetry schema (`type: "step"`)

| Field | Panel | Source |
|---|---|---|
| `timestamp` | status | wall clock at emit |
| `run_mode`, `algorithm`, `stage` | cards | launcher (`run_mode` is always `"inference"`) |
| `episode`, `episode_step`, `global_step`, `seed` | cards, x-axes | launcher loop counters (seed = 10000 + episode, as in 07) |
| `reward`, `episode_reward` | Reward, cards | `env.step` reward (raw, `norm_reward=False`) and its running sum |
| `collision`, `collision_object` | card, red lines | `info["collision"]`, `info["collision_object"]` |
| `visibility` | card, Flight state | `info["visible"]` (env flag incl. FOV, dropout, occlusion) |
| `centring_error` | Tracking error | `hypot(info["bearing_h"], info["bearing_v"])`, as in 07 |
| `distance_error` | Tracking error | `abs(info["distance"] - 8.0)` m, as in 07 |
| `smoothness_penalty` | table | `sum((a_t - a_{t-1})^2)`, as in 07 (reward term is `-0.1 ×` this) |
| `distance`, `bearing_h`, `bearing_v`, `altitude` | Flight state (altitude) | `info` |
| `drone_x/y/z`, `drone_vx/vy/vz`, `drone_speed`, `drone_yaw_deg` | Trajectory, Flight state | `kinematics_estimated` of the env's own state call (tap 1) |
| `target_x/y/z`, `target_vx/vy/vz` | Trajectory | `env._target_pos` (tap 2); velocity = displacement / dt, `vz` always 0 |
| `action_0..3` | Policy actions | `model.predict` output: vx forward, vy right, vz down (×5 m/s), yaw rate (×60 deg/s) |
| `lstm_episode_start`, `lstm_hidden_norm` | Recurrent state | `episode_start` passed to `predict`; L2 norm of the LSTM `h` (RecurrentPPO only, else null) |

Coordinates are the env's own (AirSim NED: x north, y east, z down). Other record types:
`run_start` (metadata), `episode_reset`, `episode_end` (07-style summary), `run_end`.

**Not shown:** obstacle distance (sensors are off in the validated 10-D configuration); separate reward
components (with latency > 0 the env's centring term uses the delayed observation). AirSim runs in real
time, so the same seed does not reproduce an episode exactly, with or without telemetry.

## Validation (2026-10-14, RecurrentPPO stage 0, seeds 10000-10001)

- No telemetry file → WAITING FOR TELEMETRY; synthetic run → all panels update, malformed/partial lines
  skipped, RUN FINISHED.
- Real run vs env: `|target - drone|` = `info.distance` (max diff 1.8e-15); env reward formula recomputed
  from published fields = `reward` (max diff 1.5e-7); per-episode sums = `episode_end`;
  `lstm_episode_start` true only at step 1 of each episode.
- Killing the dashboard mid-run: the demo finishes. SIGKILL of the launcher: STALE DATA; AirSim keeps running.
- Stop button: clean exit, RUN STOPPED; switching between RecurrentPPO, PPO and SAC works.
- Overhead: ~17 µs of in-loop telemetry work per step (~0.02% of a 77 ms step). End-to-end loop
  steps/s: telemetry off 13.44, 14.40; on (dashboard open) 12.97, 13.47, 12.85. The episode-to-episode
  spread (11.9-15.4) is larger than the gap.
