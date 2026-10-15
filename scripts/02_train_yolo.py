#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 02_train_yolo.py
#
# We train the perception module here: a YOLO detector on VisDrone that
# finds candidate targets (pedestrians, vehicles, etc.) in aerial imagery.
# This model is trained ONCE, offline, and then frozen - it's not part of
# the RL loop itself. Its job in the pipeline is to hand the RL agent a
# clean observation (target bounding box -> position, size, visibility)
# instead of raw pixels.
#
# We use Ultralytics' built-in VisDrone.yaml, which auto-downloads the
# dataset the first time we run this (to wherever we pointed datasets_dir
# via `yolo settings`, i.e. our 400GB data disk).
# ---------------------------------------------------------------------------
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt",
                         help="starting weights - 'n' (nano) is fastest, larger variants "
                              "(s/m/l/x, or newer families like yolo26) trade speed for accuracy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--project", default="/mnt/disks/data/aerotail/checkpoints/yolo")
    parser.add_argument("--name", default="visdrone_yolov8n")
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "false"],
                         help="dataset caching mode - 'ram' is fastest but memory-hungry at "
                              "high imgsz/large models; use 'disk' to avoid exhausting system RAM")
    args = parser.parse_args()

    # We start from pretrained weights rather than random init - the object
    # categories overlap enough (people, vehicles) that transfer learning
    # gives us a big head start over training from scratch.
    model = YOLO(args.model)

    model.train(
        data="VisDrone.yaml",       # ships with ultralytics, auto-downloads on first use
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,                    # our single L4
        project=args.project,
        name=args.name,
        patience=20,                 # early-stop if val mAP stalls for 20 epochs
        workers=8,
        amp=True,                    # mixed precision - L4 handles this well, speeds up training
        cache=(False if args.cache == "false" else args.cache),
    )

    # Quick sanity readout so we can paste mAP straight into the report.
    metrics = model.val()
    print("== Validation metrics ==")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
