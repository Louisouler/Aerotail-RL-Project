# AeroTail - UAV target following with reinforcement learning

A quadrotor learns to follow a moving target in Microsoft AirSim (Blocks environment), trained with
Stable-Baselines3 / sb3-contrib (SAC, PPO, RecurrentPPO) over a four-stage observation-noise curriculum.

## Repository layout

| Path | Contents |
|---|---|
| `envs/airsim_env.py` | Gymnasium environment: observations, actions, reward, curriculum noise, collision handling |
| `scripts/` | Environment test, training (`04_train_rl.py`, `09_train_with_retry.sh`), evaluation (`06`, `07_evaluate_all.py`), collision diagnostics, dashboard demo launcher (`10_dashboard_demo.py`) |
| `checkpoints/rl/active/<algo>_stage<N>/` | Final models (`final_model.zip`) and, for PPO/RecurrentPPO, `vecnormalize.pkl` |
| `dashboard/` | Live monitoring dashboard (Plotly Dash): telemetry, simulator view, run control - see `dashboard/README.md` |
| `report_package/` | RecurrentPPO report material: results, figures, training curves, TensorBoard logs - see `report_package/README.md` |
| `RECURRENTPPO_REPORT.md`, `ISSUE.md` | RecurrentPPO write-up and the reset/takeoff bug report |
| `ops/` | Helper scripts used to run the long training jobs |
| `settings.json` | AirSim settings (two drone/target pairs, ClockSpeed 2) |
| `reconstruction_pip_freeze.txt` | Python package versions of the training environment |

Not included: the AirSim Blocks binaries (`sim/`), intermediate checkpoints and raw training logs.

## Quick start

```bash
# Evaluate all final models (needs AirSim Blocks running with settings.json)
python scripts/07_evaluate_all.py

# Live demo: simulator + dashboard, then open http://localhost:8050 through an SSH tunnel
dashboard/demo.sh start
```
