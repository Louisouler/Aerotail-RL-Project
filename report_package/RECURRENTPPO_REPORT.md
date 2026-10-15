# RecurrentPPO (curriculum stages 0-1): training, evaluation and deviations

This document covers the RecurrentPPO part of the project: how it was trained,
how it was evaluated against the existing SAC/PPO baselines, and every place
where the procedure deviates from the original setup. All times are UTC.

## 1. Summary

- RecurrentPPO was trained on curriculum **stage 0 for 1,001,000 timesteps** and
  then **stage 1 for 1,000,448 timesteps**, continuing from the stage-0 model
  (cumulative 2,001,448). Stages 2-3 were out of scope by design.
- All algorithms (SAC, PPO, RecurrentPPO) were re-evaluated with the existing
  methodology (`scripts/07_evaluate_all.py`) on the same reconstructed runtime,
  producing the updated `checkpoints/eval_results.csv`. The table produced on the
  original VM is preserved as `checkpoints/eval_results_original_vm.csv`.
- Result: RecurrentPPO scores between SAC and PPO on total reward, has the
  lowest distance error of all three algorithms, degrades only slightly from
  stage 0 to stage 1, but has 10-15 % collisions and much less smooth control
  than PPO - despite ~6.6x more training than PPO received for stages 0-1.
- The original training VM was not accessible, so the runtime was rebuilt on a
  new VM. A reset/takeoff bug surfaced on the rebuilt runtime and was fixed; the
  rebuilt runtime was validated against the original evaluation table before
  training (Section 4).

## 2. Training budget: 1M + 1M instead of 5M + 5M

The task asked for 5M timesteps per stage, estimated at ~1 day each. That
estimate does not hold on this setup: training is bound by the simulator
running in real time, not by the GPU.

| Measurement | Value |
|---|---|
| Environment step (`dt` = 0.1 s sim time at `ClockSpeed` 2) incl. RPC + inference | ~59.7 ms |
| RecurrentPPO update (10 epochs per 1024-step rollout), amortised | ~6.1 ms/step |
| Episode reset incl. takeoff | ~1.7-5.7 s per episode |
| Sustained throughput (original VM logs and this run) | **~11-17 steps/s** |
| 5M timesteps at ~11.5 steps/s | ~5 days per stage |

The GPU accounts for ~13 % of wall time, so a faster or additional GPU does
not help. The ~1 day/5M estimate likely came from an early PPO log
(`ppo_stage0_3`, ~46 steps/s) recorded during the concurrent-access bug
described in `checkpoints/rl/README.md`, when episodes were artificially short.

Within the time available for this task (~2 days), 1M + 1M was the largest
budget that fits on the canonical runtime (`ClockSpeed` 2, single
environment). Faster options were benchmarked but rejected because they change
experimental conditions (Section 6).

For context, the baselines' actual budgets (read from each model's
`num_timesteps`):

| Algorithm | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| PPO | 200,704 | +100,352 | +200,704 | +301,056 |
| SAC | 500,000 (stage 1 was branched from the 200k stage-0 checkpoint) | +100,000 | +150,000 | diverged at ~630k cumulative |
| RecurrentPPO | 1,001,000 | +1,000,448 | - | - |

RecurrentPPO therefore received ~6.6x the stage 0-1 training of PPO. A
budget-matched comparison is possible from the saved checkpoints (every 5,000
steps, each with its VecNormalize stats) but has not been run.

## 3. Training configuration and run

Hyperparameters are unchanged from `scripts/04_train_rl.py`:
`MlpLstmPolicy`, `net_arch=[256, 256]`, `lstm_hidden_size=128`, `n_steps=1024`,
`batch_size=128`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0`,
constant learning rate, `VecNormalize(norm_obs=True, norm_reward=True,
clip_reward=10)`, vehicle `Drone1`, unseeded (same as the baselines).

| | Stage 0 | Stage 1 |
|---|---|---|
| Learning rate | 3e-4 (as PPO stage 0) | 3e-5 (as PPO and SAC stage 1) |
| Initialisation | from scratch | stage-0 `final_model.zip` + `vecnormalize.pkl` |
| Wall time | 2026-10-03 00:23 - 2026-10-04 01:36 (~25 h) | 2026-10-04 01:38 - 2026-10-05 01:52 (~24 h) |
| Simulator crashes | 2 (recovered) | 0 |
| Final `ep_len_mean` / `ep_rew_mean` (training, stochastic) | 226.6 / 355.6 | 238.2 / 363.6 |

Training curves (training-time statistics, not evaluation):

| Stage 0 timestep | ep_len_mean | ep_rew_mean | | Stage 1 timestep (cumulative) | ep_len_mean | ep_rew_mean |
|---|---|---|---|---|---|---|
| 1,024 | 33.8 | -9.1 | | 1,002,024 | 226.9 | 355.5 |
| 97,944 | 196.9 | 274.2 | | 1,101,352 | 189.5 | 273.7 |
| 248,360 | 198.9 | 304.2 | | 1,251,880 | 208.6 | 311.6 |
| 499,240 | 196.6 | 311.4 | | 1,501,736 | 227.8 | 342.9 |
| 750,120 | 207.3 | 328.0 | | 1,751,592 | 218.8 | 331.7 |
| 1,001,000 | 226.6 | 355.6 | | 2,001,448 | 238.2 | 363.6 |

Stage 0 reached most of its performance within ~100-250k steps and improved
slowly afterwards. Stage 1 dipped when observation noise was introduced and
recovered to the stage-0 level.

### Crashes and recovery

The simulator (Unreal Engine 4.25) crashed twice during stage 0 with a
render-thread segfault in the particle system
(`FParticleSystemSceneProxy::UpdateData`, signal 11) - an engine bug unrelated
to project code, consistent with the unexplained crashes that motivated
`scripts/09_train_with_retry.sh`.

- **2026-10-03 00:57** - the trainer did not exit but blocked on the AirSim RPC
  client's default 3600 s timeout. It was terminated manually at 01:17 and
  `09_train_with_retry.sh` resumed from `ckpt_15000_steps.zip`. This checkpoint
  predates the VecNormalize fix (Section 5.2), so **VecNormalize statistics
  restarted from zero at 15,000 steps**; training reward dipped briefly and fully
  recovered (98 before -> 27 after -> 157 by 35k steps).
- **2026-10-03 ~03:40** - detected by `ops/train_watchdog.sh` within ~60 s;
  resumed from `ckpt_105000_steps.zip` **with its matching VecNormalize stats**.

Roughly 20 minutes and ~3-8k timesteps were repeated in total. Stage 1 ran
without crashes.

## 4. Runtime reconstruction and fidelity validation

The original VM (`instance-20260908-063817`) and its disks could not be
accessed. The runtime was rebuilt from `gs://brats-validation-aerotail/aerotail-backup/`
(the newest backup) plus public components:

| Component | Source / version |
|---|---|
| Project state | `aerotail-backup/` (newer than `aerotail/`: includes `settings.json`, the 7-row eval table, PPO stage 3, the Drone2-based `07_evaluate_all.py`) |
| AirSim settings | `settings.json` from the backup (md5 `9782df76edfe7514e5152d9825b8faf7`), installed at `~/Documents/AirSim/settings.json`; `ClockSpeed` 2, Drone1/Target1/Drone2/Target2 |
| Simulator | Microsoft AirSim `v1.7.0-linux` `Blocks.zip` (UE 4.25.1), SHA256 `b1374222195a40bfb1d16d9d6c725950ce81ab9b482878c5f90e5db88fad5396`. Chosen because only the v1.4-v1.8.0 Linux zips unpack to the path hard-coded in `09_train_with_retry.sh` (`sim/Blocks/Blocks/LinuxNoEditor/`); the v1.8.0-linux zip ships the identical binary. No direct record of the original build exists. |
| Python | 3.10.12; torch 2.14.0+cu130, stable-baselines3 2.9.0, sb3-contrib 2.9.0, numpy 2.2.6, gymnasium 1.3.0, cloudpickle 3.1.2 (all match the metadata stored in the original checkpoints); airsim 1.8.1, msgpack-rpc-python 0.4.1, tornado 4.5.3, msgpack-python 0.5.6. Full list: `reconstruction_pip_freeze.txt` |
| System | `libnvidia-gl-580-server` matching the NVIDIA 580.178.04 driver (Vulkan on the L4 under Xvfb), `xvfb`, `libvulkan1` |

**Fidelity check.** Before training, the existing PPO and SAC stage 0-1 models
were re-evaluated on the rebuilt runtime with the original methodology and
compared with the original table (`checkpoints/fidelity/`). With the reset fix
in Section 5.1, PPO reproduced the original almost exactly (stage 0:
300.0 +/- 0 steps, reward 521.9 vs 520.7, 0 % collisions; stage 1: 491.8 vs 492.0),
and SAC matched within run-to-run variation. The final evaluation (Section 7)
confirms this for stages 2-3 as well.

## 5. Code changes

Four files differ from the team backup. Originals are kept under
`checkpoints/rl/archive/`. Observation semantics, reward, curriculum, action
semantics and collision logic are unchanged.

### 5.1 `envs/airsim_env.py` - reset/takeoff fix (affects behaviour)

`simSetVehiclePose()` keeps the drone's previous velocity. If the drone is
still moving faster than 0.05 m/s, AirSim's `takeoff()` raises "Cannot perform
takeoff because vehicle is already moving"; the RPC server suppresses the
exception and `.join()` does not surface it, so `reset()` returned with the
drone on the ground. This happened after any episode that ended while moving
and caused early ground collisions in ~10 % of evaluation episodes (8/80 in the
pre-fix fidelity run vs 0/80 in the original table).

Fix: after `armDisarm(True)` and before `takeoffAsync()`, poll the drone's
linear speed every 0.05 s until it is <= 0.05 m/s (at most 5 s). Diagnosis and
evidence: `ISSUE.md`, `checkpoints/fidelity/reset_probe*.log`. The takeoff
logic is identical in AirSim v1.7.0 and v1.8.1, so this is not caused by the
choice of simulator build. Why the original runtime did not show the problem
is unknown (most likely timing).

### 5.2 `scripts/04_train_rl.py` - VecNormalize saved with every checkpoint

`CheckpointCallback(..., save_vecnormalize=True)`, and when resuming from
`ckpt_N_steps.zip` the matching `ckpt_vecnormalize_N_steps.pkl` is loaded in
preference to `--vecnormalize-from`. Previously only the end-of-run save
contained the normalisation stats, so every crash-resume restarted them from
zero, and no intermediate checkpoint could be evaluated correctly. No effect on
learning in a run without crashes.

### 5.3 `scripts/09_train_with_retry.sh` - stage-relative progress

Checkpoints are named by cumulative timesteps. For a stage bootstrapped from a
previous stage, the first crash made the script read e.g. 1,005,000 >=
1,000,000 and stop with "Target already reached". The script now records the
stage's starting cumulative timestep in `stage_start_timesteps.txt` and
measures progress relative to it.

### 5.4 `scripts/07_evaluate_all.py` - RecurrentPPO rows

Adds RecurrentPPO stages 0 and 1 to `CONFIGS` (using `final_model.zip` +
`vecnormalize.pkl`, or the latest checkpoint + its VecNormalize stats if a run
had not finished) and prints the evaluated checkpoint path. Evaluation
methodology and CSV format are unchanged.

### Operational scripts (not part of the experiment)

`ops/` holds the helper scripts used during the run: `train_watchdog.sh`
(terminates a trainer whose simulator has crashed or whose log stalls for
15 min, so `09_train_with_retry.sh` restarts within ~1 min instead of after the
3600 s RPC timeout), `chain_stage1.sh` (starts stage 1 automatically when
stage 0 finishes, budget read from `stage1_timesteps.txt`) and `final_eval.sh`
(runs `07_evaluate_all.py` once stage 1 finishes).

## 6. Rejected speed-ups

| Option | Measured effect | Reason not used |
|---|---|---|
| CPU vs GPU for the policy | < 1 % | negligible |
| Zero velocity on teleport (`simSetKinematics`) | resets 5.7 s -> 1.7 s, ~16 % faster overall | further environment change; not adopted |
| `ClockSpeed` 4 | 1.93x raw throughput | changes simulated time per step (+10 %) and the simulation conditions the baselines were trained under; one unexplained takeoff failure observed |
| Parallel environments | ~Nx | changes data collection relative to the single-environment baselines |

## 7. Evaluation results

`scripts/07_evaluate_all.py`, unchanged methodology: Drone2/Target2, 20
episodes, seeds 10000-10019, deterministic actions, each model evaluated at its
own curriculum stage. Run 2026-10-05 01:52-02:47 on the rebuilt runtime.
Values are means (+/- std across episodes); original-VM values in brackets.

| Config | Episode length | Total reward | Collision | Centring err | Distance err | Visibility | Smoothness pen. |
|---|---|---|---|---|---|---|---|
| SAC 0 | 286.7 +/- 40.1 (299.3) | 403.7 +/- 390.2 (416.5) | 10 % (5 %) | 0.275 | 2.67 | 0.964 | 0.193 |
| SAC 1 | 265.8 +/- 90.0 (284.0) | 389.3 +/- 144.0 (413.7) | 15 % (10 %) | 0.293 | 2.07 | 0.987 | 0.392 |
| SAC 2 | 11.0 +/- 1.5 (21.1) | 4.8 +/- 5.3 (12.0) | 100 % (100 %) | 0.244 | 1.74 | 0.949 | 0.718 |
| PPO 0 | 300.0 +/- 0.0 (300.0) | 522.7 +/- 6.3 (520.7) | 0 % (0 %) | 0.252 | 1.37 | 1.000 | 0.189 |
| PPO 1 | 300.0 +/- 0.0 (300.0) | 493.0 +/- 12.9 (492.0) | 0 % (0 %) | 0.250 | 1.26 | 0.990 | 0.780 |
| PPO 2 | 300.0 +/- 0.0 (285.4) | 291.0 +/- 94.2 (289.6) | 0 % (5 %) | 0.490 | 1.33 | 0.902 | 1.158 |
| PPO 3 | 23.2 +/- 32.1 (31.9) | -1.7 +/- 22.4 (-11.7) | 100 % (100 %) | 0.387 | 1.68 | 0.871 | 1.880 |
| **RecurrentPPO 0** | 279.1 +/- 62.9 | 446.1 +/- 108.9 | 10 % | 0.298 | **1.23** | 1.000 | 1.392 |
| **RecurrentPPO 1** | 279.1 +/- 66.1 | 428.2 +/- 106.3 | 15 % | 0.264 | **1.24** | 0.990 | 1.632 |

SAC stage 3 has no final model (diverged) and is skipped, as in the original
table.

### Observations

- **Total reward:** RecurrentPPO sits between SAC and PPO at both stages
  (stage 0: 446 vs SAC 404 / PPO 523; stage 1: 428 vs 389 / 493).
- **Distance keeping:** RecurrentPPO has the lowest distance error of all
  algorithms (1.23 / 1.24). Centring error and visibility are close to PPO.
- **Control smoothness:** RecurrentPPO's smoothness penalty is ~7x PPO's at
  stage 0 (1.39 vs 0.19) and ~2x at stage 1, i.e. much jerkier actions; this
  term is part of the reward and explains part of the reward gap to PPO.
- **Collisions:** 2/20 (stage 0, at steps 85 and 96) and 3/20 (stage 1, at
  steps 2, 235 and 245), comparable to SAC; PPO has none at these stages.
- **Robustness to observation noise:** from stage 0 to stage 1 RecurrentPPO's
  reward drops by 4 % (446 -> 428) with unchanged episode length; PPO drops by
  6 % and SAC by 4 % with more collisions.
- **Training budget:** RecurrentPPO did not outperform PPO despite ~6.6x more
  stage 0-1 training.
- **Reproduction of the baselines:** PPO stages 0-3 match the original table
  closely and SAC stages 0-1 within run-to-run variation. SAC stage 2 episodes
  are shorter (11 vs 21 steps), but both runs collide in 100 % of episodes, so
  the stage-2 conclusion is unchanged.

## 8. Caveats

1. **Reconstructed runtime.** The exact original simulator build and the
   original Python environment could not be recovered; Section 4 lists what was
   used and how it was validated.
2. **Environment fix.** The reset fix (5.1) changes behaviour at episode start
   relative to the code that trained the SAC/PPO models. SAC/PPO were trained on
   the original runtime and evaluated here with the fix; RecurrentPPO was
   trained and evaluated with the fix.
3. **Seed 10012 at stage 1.** SAC and RecurrentPPO both collided at step 2 of
   episode 12 (seed 10012) at stage 1, while PPO completed it. The cause was not
   investigated (possibly a residual reset effect or that seed's start state).
   It affects both algorithms equally and accounts for 5 of RecurrentPPO's
   15 % stage-1 collision rate.
4. **VecNormalize restart at 15k steps** in stage 0 (Section 3).
5. **Unequal training budgets** across algorithms (Section 2).
6. **Unseeded training**, as for the baselines: runs are not bit-reproducible.

## 9. Files

| Path | Contents |
|---|---|
| `checkpoints/rl/active/recurrent_ppo_stage{0,1}/` | `final_model.zip`, `vecnormalize.pkl`, checkpoints every 5,000 steps with `ckpt_vecnormalize_*` (stage 0 checkpoints before 20k lack them) |
| `checkpoints/eval_results.csv` | updated comparison table (this runtime, all algorithms) |
| `checkpoints/eval_results_original_vm.csv` | original 7-row table from the original VM |
| `checkpoints/fidelity/` | fidelity runs, reset-bug probes, throughput benchmarks |
| `checkpoints/rl/archive/` | pre-change copies of every modified file; the 5k RecurrentPPO checkpoint from the original VM |
| `logs/recurrent_ppo_stage0_1/`, `logs/recurrent_ppo_stage1_0/` | TensorBoard logs (SB3 numbers the continued stage-1 run `_0`) |
| `logs/train_recurrent_ppo_stage{0,1}.out`, `logs/eval_all_final.out` | training and evaluation console output |
| `logs/train_watchdog.log`, `logs/train_chain.log`, `logs/final_eval_runner.log` | operational logs |
| `ops/` | watchdog, stage chaining and final-evaluation helper scripts |
| `ISSUE.md` | reset/takeoff issue write-up |
| `reconstruction_pip_freeze.txt` | Python environment of the rebuilt runtime |

## 10. Reproducing

```bash
source /mnt/disks/data/aerotail-venv/bin/activate
cd /mnt/disks/data/aerotail
# stage 0
bash scripts/09_train_with_retry.sh recurrent_ppo 0 1000000 Drone1 Target1
# stage 1, continuing from stage 0
bash scripts/09_train_with_retry.sh recurrent_ppo 1 1000000 Drone1 Target1 3e-5 \
    checkpoints/rl/active/recurrent_ppo_stage0/final_model.zip \
    checkpoints/rl/active/recurrent_ppo_stage0/vecnormalize.pkl
# evaluation (simulator must be running; 09 leaves it running)
python3 scripts/07_evaluate_all.py
```

Run `ops/train_watchdog.sh` alongside training to recover from simulator
crashes quickly. Do not run commands whose command line contains "Blocks" or
"Xvfb" during training: `09_train_with_retry.sh` restarts the simulator with
`pkill -9 -f Blocks` / `pkill -9 -f Xvfb`, which also kills such processes.
