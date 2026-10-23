"""Evaluate one algo/stage with exactly the scripts/07_evaluate_all.py protocol
(Drone2/Target2, 20 deterministic episodes, seeds 10000-10019) and add its row
to checkpoints/eval_results.csv without re-running the other models.

Usage: python3 ops/eval_single.py <algo> <stage>

The existing CSV is backed up first; a row for the same algo/stage is replaced.
The row is also written on its own to checkpoints/eval_results_<algo>_stage<N>.csv.
"""
import csv
import importlib.util
import shutil
import sys
import time

PROJECT_DIR = "/mnt/disks/data/aerotail"
spec = importlib.util.spec_from_file_location("eval_all", f"{PROJECT_DIR}/scripts/07_evaluate_all.py")
eval_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_all)

algo, stage = sys.argv[1], int(sys.argv[2])
checkpoint, vecnormalize = eval_all.latest_run_files(f"{eval_all.CKPT_ROOT}/{algo}_stage{stage}")
row = eval_all.evaluate_config(dict(algo=algo, stage=stage, checkpoint=checkpoint,
                                    vecnormalize=vecnormalize if algo in eval_all.ON_POLICY_ALGOS else None))
if row is None:
    sys.exit(1)

with open(f"{PROJECT_DIR}/checkpoints/eval_results_{algo}_stage{stage}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)

csv_path = eval_all.OUTPUT_CSV
shutil.copy2(csv_path, csv_path.replace(".csv", f".backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames, rows = reader.fieldnames, list(reader)
rows = [r for r in rows if not (r["algo"] == algo and int(r["stage"]) == stage)]
rows.append({k: row[k] for k in fieldnames})
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{algo} stage {stage}: ep_len {row['steps_mean']:.1f}, reward {row['total_reward_mean']:.1f}, "
      f"dist_err {row['mean_distance_error_mean']:.2f} m, visibility {row['visibility_ratio_mean']:.2f}, "
      f"collisions {row['collision_rate'] * 100:.0f} %")
print(f"Added to {csv_path} (backup saved alongside)")
