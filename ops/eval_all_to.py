"""Run the full scripts/07_evaluate_all.py evaluation (same models, protocol and
seeds) but write the table to a separate CSV, e.g. for another AirSim
environment, leaving checkpoints/eval_results.csv untouched.

Usage: python3 ops/eval_all_to.py <output_csv>
"""
import csv
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("eval_all", "/mnt/disks/data/aerotail/scripts/07_evaluate_all.py")
eval_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_all)

output_csv = sys.argv[1]
rows = []
for cfg in eval_all.CONFIGS:
    row = eval_all.evaluate_config(cfg)
    if row is not None:
        rows.append(row)
        with open(output_csv, "w", newline="") as f:  # rewritten after each model, so partial runs keep results
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to {output_csv}")
for r in rows:
    print(f"{r['algo']:<14}{r['stage']:<3}ep_len {r['steps_mean']:6.1f}  reward {r['total_reward_mean']:7.1f}  "
          f"dist_err {r['mean_distance_error_mean']:5.2f}  visib {r['visibility_ratio_mean']:.2f}  "
          f"collisions {r['collision_rate'] * 100:3.0f} %")
