# RecurrentPPO report package

Material for writing up the RecurrentPPO part of the project (curriculum
stages 0-1) and its comparison with SAC and PPO. The full write-up, including
every deviation from the original procedure, is in `RECURRENTPPO_REPORT.md`;
start there. The final RecurrentPPO models are included; the intermediate
checkpoints (every 5,000 steps, about 1.6 GB) are not - they are in
`checkpoints/rl/active/recurrent_ppo_stage{0,1}/` on the VM.

## Key numbers

| | RecurrentPPO | PPO | SAC |
|---|---|---|---|
| Training, stage 0 / stage 1 (timesteps) | 1,001,000 / 1,000,448 | 200,704 / 100,352 | 500,000 / 100,000 |
| Total reward, stage 0 / 1 | 446.1 / 428.2 | 522.7 / 493.0 | 403.7 / 389.3 |
| Collision rate, stage 0 / 1 | 10 % / 15 % | 0 % / 0 % | 10 % / 15 % |
| Distance error (m), stage 0 / 1 | 1.23 / 1.24 | 1.37 / 1.26 | 2.67 / 2.07 |
| Smoothness penalty, stage 0 / 1 | 1.39 / 1.63 | 0.19 / 0.78 | 0.19 / 0.39 |

Evaluation: `scripts/07_evaluate_all.py`, Drone2/Target2, 20 episodes,
seeds 10000-10019, deterministic actions, all algorithms on the same rebuilt
runtime.

### Stage 2 (added later)

RecurrentPPO stage 2 was trained for 500,736 timesteps (2,502,184 cumulative) from the stage-1 final
model and VecNormalize stats (learning rate 5e-5, as for PPO stage 2; model in
`checkpoints/rl/active/recurrent_ppo_stage2/`) and evaluated with the same
protocol:

| Stage 2 | RecurrentPPO | PPO | SAC |
|---|---|---|---|
| Training (timesteps) | 500,736 | 200,704 | 150,000 |
| Total reward | 34.4 | 291.0 | 4.8 |
| Collision rate | 50 % | 0 % | 100 % |
| Distance error (m) | 3.09 | 1.33 | 1.74 |
| Visibility ratio | 0.86 | 0.90 | 0.95 |

### Zero-shot generalisation to AirSimNH

All models (trained in Blocks only) were evaluated without retraining in the
AirSimNH neighbourhood environment (houses, trees, hedges, parked cars, power
lines; AirSim v1.7.0 Linux release, same `settings.json`), with the same
protocol, via `ops/eval_all_to.py`. Blocks -> AirSimNH:

| Model | Episode length | Reward | Distance error (m) | Visibility | Collision rate |
|---|---|---|---|---|---|
| SAC 0 | 287 -> 145 | 404 -> 221 | 2.67 -> 1.81 | 0.96 -> 0.98 | 10 % -> 70 % |
| SAC 1 | 266 -> 172 | 389 -> 233 | 2.07 -> 2.06 | 0.99 -> 0.98 | 15 % -> 70 % |
| SAC 2 | 11 -> 9 | 5 -> 1 | 1.74 -> 1.75 | 0.95 -> 0.94 | 100 % -> 100 % |
| PPO 0 | 300 -> 179 | 523 -> 306 | 1.37 -> 1.43 | 1.00 -> 1.00 | 0 % -> 55 % |
| PPO 1 | 300 -> 195 | 493 -> 316 | 1.26 -> 1.31 | 0.99 -> 0.99 | 0 % -> 50 % |
| PPO 2 | 300 -> 226 | 291 -> 226 | 1.33 -> 1.30 | 0.90 -> 0.92 | 0 % -> 40 % |
| PPO 3 | 23 -> 9 | -2 -> -2 | 1.68 -> 1.55 | 0.87 -> 0.88 | 100 % -> 100 % |
| RecurrentPPO 0 | 279 -> 105 | 446 -> 158 | 1.23 -> 1.49 | 1.00 -> 1.00 | 10 % -> 95 % |
| RecurrentPPO 1 | 279 -> 141 | 428 -> 210 | 1.24 -> 1.31 | 0.99 -> 0.99 | 15 % -> 85 % |
| RecurrentPPO 2 | 227 -> 100 | 34 -> 70 | 3.09 -> 1.86 | 0.86 -> 0.90 | 50 % -> 90 % |

Tracking transfers (distance error and visibility are almost unchanged, since
the observation does not depend on the scene's appearance), but obstacle
avoidance does not: collision rates rise to 40-95 % for every model that
completed episodes in Blocks, mostly against trees, bushes and power lines.
PPO generalises best and RecurrentPPO worst. The reward of RecurrentPPO stage 2
rises only because episodes end earlier, before the target is lost. Caveat: the
target is moved kinematically and passes through obstacles, so some collisions
are hard for any follower to avoid.

## Contents

| Path | What it is |
|---|---|
| `RECURRENTPPO_REPORT.md` | Full report: training budget, run, runtime reconstruction, code changes, results, caveats, reproduction |
| `ISSUE.md` | Write-up of the reset/takeoff bug found on the rebuilt runtime |
| `results/eval_results.csv` | Final comparison table (all algorithms, rebuilt runtime) - same format as the original table. The `recurrent_ppo` stage 2 row was added later (see below) without re-running the other rows |
| `results/eval_results_original_vm.csv` | The original 7-row table produced on the original VM, for reference |
| `results/eval_all_final.out` | Per-episode output of the final evaluation run (reward, steps, errors, collision per episode) |
| `results/eval_results_airsimnh.csv`, `results/eval_all_airsimnh.out` | Zero-shot evaluation of all models in AirSimNH (table and per-episode output) |
| `results/eval_recurrent_ppo_stage2.out` | Per-episode output of the RecurrentPPO stage 2 evaluation (`ops/eval_single.py`, same protocol as `07_evaluate_all.py`) |
| `models/recurrent_ppo_stage{0,1}/` | Final RecurrentPPO models: `final_model.zip` (stage 0: 1,001,000 timesteps; stage 1: 2,001,448 cumulative) and the matching `vecnormalize.pkl`. Load with `RecurrentPPO.load(...)` and `VecNormalize.load(...)` (set `training=False`, `norm_reward=False` for evaluation) |
| `tensorboard/` | Raw TensorBoard event files for both stages (`tensorboard --logdir tensorboard`). Stage 0 has three files because training resumed twice after simulator crashes |
| `training_curves/recurrent_ppo_stage{0,1}_training_curves.csv` | All TensorBoard scalars per rollout (episode reward/length, losses, KL, entropy, explained variance, learning rate, fps) |
| `figures/fig1_eval_comparison_stage0_1.png` | Evaluation metrics for PPO, SAC and RecurrentPPO at stages 0-1 |
| `figures/fig2_recurrent_ppo_training_curves.png` | RecurrentPPO training reward and episode length over both stages, with the stage switch and the two crash-resume points marked |
| `figures/fig3_fidelity_original_vs_rebuilt.png` | SAC/PPO results on the original VM vs the rebuilt runtime (validation of the rebuilt runtime) |
| `fidelity/step9_*` | Fidelity check before the reset fix (shows the early ground collisions) |
| `fidelity/step9b_*` | Fidelity check after the reset fix (matches the original table) |
| `fidelity/reset_probe*.log` | Diagnosis of the reset/takeoff bug (takeoff fails whenever the drone is still moving) |
| `fidelity/benchB_reset.log`, `fidelity/benchC_clock*.log` | Throughput benchmarks behind the training-budget decision |
| `code_changes/*.patch` | Unified diffs of the four changed files against the team backup |
| `code_changes/ops/` | Helper scripts used to run training unattended (crash watchdog, stage chaining, final evaluation) |
| `environment/` | Python package list and the AirSim `settings.json` used |
| `logs/` | Training console output for both stages and the operational logs |

## Figure captions (suggested)

- **Figure 1.** Evaluation of PPO, SAC and RecurrentPPO on curriculum stages 0
  and 1 (20 deterministic episodes per model, seeds 10000-10019). RecurrentPPO
  keeps the target distance best but has 10-15 % collisions and much less smooth
  control than PPO.
- **Figure 2.** RecurrentPPO training curves (mean over training rollouts,
  stochastic policy). Stage 1 continues from the final stage-0 model with a lower
  learning rate and noisy observations; markers show the two simulator crashes
  that were resumed from checkpoints.
- **Figure 3.** The existing SAC and PPO models evaluated on the original VM and
  on the rebuilt runtime used for RecurrentPPO, showing the rebuilt runtime
  reproduces the original results.

## Points the write-up should mention

1. Training budget is 1M + 1M timesteps rather than the requested 5M + 5M:
   training is bound by real-time simulation (~11-17 steps/s), not the GPU.
2. RecurrentPPO received ~6.6x more stage 0-1 training than PPO and still
   did not outperform it.
3. The runtime was rebuilt on a new VM and validated against the original
   evaluation table.
4. `envs/airsim_env.py` was changed to fix a reset/takeoff bug (waits for the
   drone to stop before takeoff); SAC/PPO were trained before this fix and
   evaluated with it.
5. Caveats listed in Section 8 of `RECURRENTPPO_REPORT.md`.
