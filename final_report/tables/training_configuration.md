# Training configuration

## Training budget per curriculum stage

Read from each final model (`num_timesteps`, `learning_rate`). Each stage continues from the previous stage's final model (and VecNormalize statistics for PPO/RecurrentPPO).

| Algorithm | Stage | Timesteps in stage | Cumulative timesteps | Learning rate (stored in model) |
|---|---|---|---|---|
| SAC | 0 | 500,000 | 500,000 | 5e-05 |
| SAC | 1 | 100,000 | 300,000 | 3e-05 |
| SAC | 2 | 150,000 | 450,000 | 5e-05 |
| PPO | 0 | 200,704 | 200,704 | 0.0003 |
| PPO | 1 | 100,352 | 301,056 | 3e-05 |
| PPO | 2 | 200,704 | 501,760 | 5e-05 |
| PPO | 3 | 301,056 | 802,816 | 5e-05 |
| RecurrentPPO | 0 | 1,001,000 | 1,001,000 | 0.0003 |
| RecurrentPPO | 1 | 1,000,448 | 2,001,448 | 3e-05 |
| RecurrentPPO | 2 | 500,736 | 2,502,184 | 5e-05 |
| SAC | 3 | diverged at ~630k cumulative | - | - |

SAC stage 1 was branched from the 200k stage-0 checkpoint, not from the 500k final model. The SAC stage-0 model stores learning rate 5e-5: its TensorBoard log (`logs/sac_stage0_13`) shows 3e-4 up to about 115k steps, after which the run was resumed from the 65.9k checkpoint with 5e-5 and continued to 500k.

## Algorithm hyperparameters (Stable-Baselines3 / sb3-contrib)

| | SAC | PPO | RecurrentPPO |
|---|---|---|---|
| Policy | MlpPolicy, [256, 256] | MlpPolicy, [256, 256] | MlpLstmPolicy, [256, 256], LSTM hidden 128 |
| Initial learning rate | 3e-4 | 3e-4 | 3e-4 |
| Fine-tuning learning rate (stages 1 / 2 / 3) | 3e-5 / 5e-5 / - | 3e-5 / 5e-5 / 5e-5 | 3e-5 / 5e-5 / - |
| Discount gamma | 0.99 | 0.99 | 0.99 |
| Batch size | 256 | 256 | 128 |
| Rollout length n_steps | - | 2048 | 1024 |
| Epochs per update | - | 10 | 10 |
| GAE lambda | - | 0.95 | 0.95 |
| Clip range | - | 0.2 | 0.2 |
| Entropy coefficient | auto (target entropy -4) | 0.0 | 0.0 |
| Value coefficient | - | 0.5 | 0.5 |
| Replay buffer | 200,000 | - | - |
| Soft update tau | 0.005 | - | - |
| Train frequency / gradient steps | 1 / 1 | - | - |
| Learning starts | 100 | - | - |
| Observation / reward normalisation | none | VecNormalize (obs + reward, clip 10) | VecNormalize (obs + reward, clip 10) |

Raw values: `model_hyperparameters_raw.csv`.

## Curriculum stages (`CURRICULUM_STAGES` in scripts/04_train_rl.py and 07_evaluate_all.py)

| Stage | Observation noise std | Detection dropout prob | Latency (steps) | Occlusion prob | Max occlusion (steps) |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0 | 0.00 | 0 |
| 1 | 0.03 | 0.01 | 0 | 0.00 | 0 |
| 2 | 0.06 | 0.05 | 1 | 0.01 | 10 |
| 3 | 0.10 | 0.10 | 2 | 0.02 | 20 |

## Environment (envs/airsim_env.py, AirSimTargetFollowEnv)

| Item | Value |
|---|---|
| Simulator | Microsoft AirSim v1.7.0 (Linux binaries), Unreal Engine 4; Blocks for training, AirSimNH for zero-shot evaluation; ClockSpeed 2.0 |
| Vehicles | Drone1/Target1 (training, demos), Drone2/Target2 at Y=100 (evaluation) |
| Control step dt | 0.1 s (simulated); episode 30 s = 300 steps |
| Action (4-D, [-1, 1]) | body-frame velocity vx, vy, vz (x 5 m/s) and yaw rate (x 60 deg/s), via moveByVelocityBodyFrameAsync |
| Observation (10-D, [-1, 1]) | horizontal and vertical bearing to target (normalised by 45 deg FOV half-angle), size proxy (8 m / distance - 1), visibility flag, relative velocity x/y (implemented as minus the drone velocity in world frame: the target velocity is not included), previous action (4) |
| Visibility | 0 if the target is outside 1.5 x the FOV half-angle, during a simulated occlusion, or on a detection dropout; bearing/size set to -1 when not visible. No ray-cast occlusion by scene geometry |
| Target | moved kinematically with simSetVehiclePose(ignore_collision=True) from (10, 0, -1) (pair 1) at 1-3 m/s, random heading changes (p=0.01/step, +-90 deg) and stops (p=0.005/step, 1-3 s); route set by the reset seed |
| Reward | -(bearing_h^2 + bearing_v^2) - abs(distance - 8)/8 + 2 x (+1 visible / -1 not visible) - 0.1 x sum(action change^2); -10 on collision |
| Termination | collision (terminated) or 300 steps (truncated) |
| Desired distance | 8 m |

## Evaluation protocol (scripts/07_evaluate_all.py)

Drone2/Target2, 20 episodes per model, seeds 10000-10019, deterministic actions, curriculum stage settings of the model's own stage, VecNormalize with training=False and norm_reward=False. Metrics per episode: steps survived, total reward, mean centring error (hypot of bearings), mean |distance - 8 m|, visibility ratio, mean squared action change, collision flag. AirSim is not deterministic between runs, so repeated evaluations of the same model differ slightly.
