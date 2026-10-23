# Results tables

## Evaluation in Blocks (training environment)

20 episodes per model, seeds 10000-10019, deterministic actions, Drone2/Target2.

| Model | Episode length (steps) | Total reward | Centring error | Distance error (m) | Visibility ratio | Smoothness penalty | Collision rate |
|---|---|---|---|---|---|---|---|
| SAC stage 0 | 286.7 ± 40.1 | 403.7 ± 390.2 | 0.275 ± 0.186 | 2.67 ± 5.26 | 0.96 ± 0.11 | 0.19 ± 0.28 | 10 % |
| SAC stage 1 | 265.8 ± 90.0 | 389.3 ± 144.0 | 0.293 ± 0.129 | 2.07 ± 0.81 | 0.99 ± 0.02 | 0.39 ± 0.18 | 15 % |
| SAC stage 2 | 11.0 ± 1.5 | 4.8 ± 5.3 | 0.244 ± 0.105 | 1.74 ± 0.78 | 0.95 ± 0.07 | 0.72 ± 0.44 | 100 % |
| PPO stage 0 | 300.0 ± 0.0 | 522.7 ± 6.3 | 0.252 ± 0.010 | 1.37 ± 0.18 | 1.00 ± 0.00 | 0.19 ± 0.05 | 0 % |
| PPO stage 1 | 300.0 ± 0.0 | 493.0 ± 12.9 | 0.250 ± 0.015 | 1.26 ± 0.18 | 0.99 ± 0.00 | 0.78 ± 0.11 | 0 % |
| PPO stage 2 | 300.0 ± 0.0 | 291.0 ± 94.2 | 0.490 ± 0.056 | 1.33 ± 0.71 | 0.90 ± 0.04 | 1.16 ± 0.14 | 0 % |
| PPO stage 3 | 23.2 ± 32.1 | -1.7 ± 22.4 | 0.387 ± 0.174 | 1.68 ± 0.65 | 0.87 ± 0.12 | 1.88 ± 0.91 | 100 % |
| RecurrentPPO stage 0 | 279.1 ± 62.9 | 446.1 ± 108.9 | 0.298 ± 0.056 | 1.23 ± 0.35 | 1.00 ± 0.00 | 1.39 ± 0.20 | 10 % |
| RecurrentPPO stage 1 | 279.1 ± 66.1 | 428.2 ± 106.3 | 0.264 ± 0.050 | 1.24 ± 0.30 | 0.99 ± 0.00 | 1.63 ± 0.12 | 15 % |
| RecurrentPPO stage 2 | 226.7 ± 90.4 | 34.4 ± 240.0 | 0.592 ± 0.188 | 3.09 ± 2.55 | 0.86 ± 0.11 | 1.23 ± 0.26 | 50 % |

## Zero-shot evaluation in AirSimNH

20 episodes per model, seeds 10000-10019, deterministic actions, Drone2/Target2.

| Model | Episode length (steps) | Total reward | Centring error | Distance error (m) | Visibility ratio | Smoothness penalty | Collision rate |
|---|---|---|---|---|---|---|---|
| SAC stage 0 | 145.3 ± 110.4 | 220.7 ± 223.3 | 0.253 ± 0.146 | 1.81 ± 1.68 | 0.98 ± 0.07 | 0.30 ± 0.41 | 70 % |
| SAC stage 1 | 172.3 ± 104.2 | 233.4 ± 155.8 | 0.393 ± 0.147 | 2.06 ± 1.52 | 0.98 ± 0.02 | 0.46 ± 0.28 | 70 % |
| SAC stage 2 | 8.5 ± 1.1 | 0.9 ± 5.5 | 0.220 ± 0.132 | 1.75 ± 0.65 | 0.94 ± 0.09 | 0.74 ± 0.52 | 100 % |
| PPO stage 0 | 179.0 ± 121.3 | 305.8 ± 216.7 | 0.258 ± 0.036 | 1.43 ± 0.17 | 1.00 ± 0.00 | 0.15 ± 0.05 | 55 % |
| PPO stage 1 | 195.3 ± 110.6 | 316.2 ± 186.4 | 0.242 ± 0.033 | 1.31 ± 0.23 | 0.99 ± 0.01 | 0.88 ± 0.48 | 50 % |
| PPO stage 2 | 226.2 ± 98.5 | 226.4 ± 110.1 | 0.468 ± 0.050 | 1.30 ± 0.36 | 0.92 ± 0.04 | 1.16 ± 0.16 | 40 % |
| PPO stage 3 | 9.2 ± 1.9 | -1.8 ± 8.4 | 0.329 ± 0.194 | 1.55 ± 0.62 | 0.88 ± 0.15 | 1.81 ± 1.00 | 100 % |
| RecurrentPPO stage 0 | 105.2 ± 77.1 | 158.1 ± 127.2 | 0.257 ± 0.067 | 1.49 ± 0.58 | 1.00 ± 0.00 | 1.49 ± 0.27 | 95 % |
| RecurrentPPO stage 1 | 140.8 ± 96.2 | 210.1 ± 155.3 | 0.257 ± 0.036 | 1.31 ± 0.37 | 0.99 ± 0.01 | 1.74 ± 0.19 | 85 % |
| RecurrentPPO stage 2 | 100.0 ± 74.7 | 70.0 ± 98.7 | 0.488 ± 0.171 | 1.86 ± 0.76 | 0.90 ± 0.08 | 1.45 ± 0.38 | 90 % |

## Blocks vs AirSimNH

| Model | Collision rate | Total reward | Distance error (m) | Visibility ratio |
|---|---|---|---|---|
| SAC stage 0 | 10 % -> 70 % | 403.7 -> 220.7 | 2.67 -> 1.81 | 0.96 -> 0.98 |
| SAC stage 1 | 15 % -> 70 % | 389.3 -> 233.4 | 2.07 -> 2.06 | 0.99 -> 0.98 |
| SAC stage 2 | 100 % -> 100 % | 4.8 -> 0.9 | 1.74 -> 1.75 | 0.95 -> 0.94 |
| PPO stage 0 | 0 % -> 55 % | 522.7 -> 305.8 | 1.37 -> 1.43 | 1.00 -> 1.00 |
| PPO stage 1 | 0 % -> 50 % | 493.0 -> 316.2 | 1.26 -> 1.31 | 0.99 -> 0.99 |
| PPO stage 2 | 0 % -> 40 % | 291.0 -> 226.4 | 1.33 -> 1.30 | 0.90 -> 0.92 |
| PPO stage 3 | 100 % -> 100 % | -1.7 -> -1.8 | 1.68 -> 1.55 | 0.87 -> 0.88 |
| RecurrentPPO stage 0 | 10 % -> 95 % | 446.1 -> 158.1 | 1.23 -> 1.49 | 1.00 -> 1.00 |
| RecurrentPPO stage 1 | 15 % -> 85 % | 428.2 -> 210.1 | 1.24 -> 1.31 | 0.99 -> 0.99 |
| RecurrentPPO stage 2 | 50 % -> 90 % | 34.4 -> 70.0 | 3.09 -> 1.86 | 0.86 -> 0.90 |

SAC stage 3 is not listed: its training diverged at about 630k cumulative timesteps and no final model was saved (the diverged checkpoint is kept in the GCS backup under `checkpoints/rl/known_issues/sac_stage3_diverged/`). RecurrentPPO was trained on stages 0-2 only.
