# Material for the Part-E final report

Everything the Part-E template (`.orca/drops/Project-FinalReport-Template-Part-E.docx`)
asks for, collected in one place and mapped to its sections. All numbers come from the
files in this folder; nothing here needs a new experiment.

```
final_report/
  README.md                 this index
  references.md             reference list [1]-[16]
  figures/                  report figures (PNG, 200 dpi)
  screenshots/              environment, dashboard and limitation images
  tables/                   result tables (CSV + Markdown) and training configuration
  training_curves/          per-rollout training reward/length for every model (CSV)
  yolo/                     perception side study: metrics table and Ultralytics plots
  scripts/                  scripts that generate the figures
```

Demo videos (not in git, 58 MB): `dashboard/recordings/final_demos/` - one run per
algorithm and environment, simulator view and dashboard view.

## Section-by-section

| Template section | Material |
|---|---|
| Title page, Appendix A/B (team, student IDs, contribution split, signatures) | **Not available here - the team has to provide it.** |
| Abstract | Key findings below. |
| 1.1-1.3 Problem, motivation, application | Part-D report (`.orca/drops/Project-Part-D.docx`) and `RECURRENTPPO_REPORT.md` section 1. Problem: a UAV follows a moving ground/low-altitude target at 8 m, keeping it centred and in view, under noisy, delayed and intermittent target detections. |
| 1.4 Environment, sample images | `screenshots/env_blocks_chase_view.png`, `env_airsimnh_chase_view.png`, `env_airsimnh_chase_view_2.png`, `airsimnh_object_map.png`; environment table in `tables/training_configuration.md` |
| 2.1 Workflow | `figures/fig_workflow.png` |
| 2.2 System architecture, experiments | `figures/fig_system_architecture.png`; algorithms and hyperparameters in `tables/training_configuration.md` |
| 2.3 GUI / environment design | `screenshots/dashboard_blocks_ppo_run.png`, `dashboard_airsimnh_ppo_run.png`; dashboard description in `dashboard/README.md` |
| 3.1 Experimental settings | `tables/training_configuration.md`: training budget per stage, hyperparameters, curriculum stages, environment, evaluation protocol |
| 3.2 Pre-processing | Observation construction and normalisation (`VecNormalize` for PPO/RecurrentPPO, none for SAC) in `tables/training_configuration.md`; YOLO/VisDrone training in `yolo/yolo_results.md` |
| 3.3 Experimental results | `tables/results_tables.md` (Blocks, AirSimNH, Blocks vs AirSimNH); `figures/fig_eval_blocks_all_stages.png`, `fig_generalisation_blocks_vs_airsimnh.png`, `fig_training_curves_all.png`; earlier figures in `report_package/figures/` (stage 0-1 comparison, RecurrentPPO curves, original-VM vs rebuilt-runtime check) |
| 3.4 Limitations | List below; images `screenshots/limitation_follower_beside_house_target_out_of_view.png`, `limitation_follower_under_eaves.png` |
| 4 Discussion, future work | Key findings and limitations below |
| 5 References | `references.md` |

## Key findings (all from `tables/`)

1. **Clean observations (stages 0-1): all three algorithms learn to follow.** PPO is best
   in Blocks: 0 % collisions, reward 523 / 493, distance error 1.37 / 1.26 m. SAC and
   RecurrentPPO collide in 10-15 % of episodes.
2. **Noisier stages break the policies.** SAC stage 2 and PPO stage 3 collide in 100 % of
   episodes (on average 11 and 23 steps), RecurrentPPO stage 2 in 50 %. PPO stage 2 is the only
   later-stage model that keeps 0 % collisions (reward 291). The training curves show the
   same drop at each stage switch; SAC stage 3 diverged and has no model.
3. **The recurrent policy did not pay off.** RecurrentPPO got about 6.6x more stage 0-1
   training than PPO (2.0M vs 0.3M timesteps) and keeps the distance best (1.23 m), but
   collides more and its control is much less smooth (smoothness penalty 1.39 vs 0.19).
4. **Tracking transfers to an unseen environment, obstacle avoidance does not.** In
   AirSimNH (zero-shot), distance error and visibility hardly change, but collision rates
   rise to 40-95 % for every model that completed episodes in Blocks (in an 8-episode
   diagnostic run of PPO stage 0 the collisions were with trees/bushes and a power line). PPO transfers best (stage 2: 0 % -> 40 %), RecurrentPPO worst
   (stage 0: 10 % -> 95 %).

## Limitations (for 3.4 / 4)

- The observation has no obstacle information. Four distance sensors are configured in
  `settings.json` but disabled in the environment (`use_obstacle_sensors=False`).
- The target is moved kinematically with collisions ignored: in AirSimNH it can pass
  through houses and hedges, which no follower can do.
- "Visibility" is geometric (angle and curriculum dropout/occlusion), not ray-cast, so a
  target hidden behind a house still counts as visible.
- The "relative velocity" observation is minus the drone's own world-frame velocity; the
  target's velocity is not included (comment in the code says body-frame target velocity).
- The YOLO detector is trained but not connected to the RL loop.
- Training is bound by real-time simulation (about 11-17 steps/s at ClockSpeed 2), which
  limited budgets (e.g. RecurrentPPO stage 3 was not trained) and prevented seed repeats:
  every result is one training run per configuration.
- AirSim is not deterministic, so re-evaluating the same model gives slightly different
  numbers; evaluation uses 20 episodes per model.
- PPO stage 0 has no training-reward curve: the run that produced the final model logged
  only losses (`logs/ppo_stage0_2`).
- The demo videos use hand-picked target routes (seeds) so the chase camera can show the
  tracking; they are illustrations, not evaluation results.

## Provenance

- Evaluation tables: `checkpoints/eval_results.csv` (Blocks) and
  `checkpoints/eval_results_airsimnh.csv`, produced by `scripts/07_evaluate_all.py` and
  `ops/eval_all_to.py`.
- Training budgets and hyperparameters: read from each `final_model.zip`.
- Training curves: TensorBoard logs in `logs/`, one run per stage (the run that produced
  the final model; where a run was resumed, the resumed part replaces the earlier part).
- Figures: `scripts/report_figs.py` (result charts, from `tables/` and `training_curves/`) and
  `scripts/report_diagrams.py` (workflow and architecture). They need matplotlib, which is not
  in the project venvs: `python3 -m venv /tmp/mpl && /tmp/mpl/bin/pip install matplotlib`.
