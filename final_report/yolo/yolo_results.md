# YOLO detector on VisDrone (perception side study)

Trained offline with Ultralytics (`scripts/02_train_yolo.py`, dataset `VisDrone.yaml`). The detector is NOT connected to the RL environment: the RL observation is computed from simulator ground truth (bearing, distance, visibility) with curriculum noise, dropout, latency and occlusion standing in for detector errors.

| Run | Model | Image size | Epochs planned / completed | Best mAP50 | Best mAP50-95 | Precision / recall at best mAP50-95 |
|---|---|---|---|---|---|---|
| visdrone_yolov8n | YOLOv8n | 640 | 3 / 3 | 0.192 | 0.102 | 0.276 / 0.234 |
| visdrone_yolov8n-2 | YOLOv8n | 640 | 100 / 100 | 0.337 | 0.186 | 0.468 / 0.342 |
| visdrone_yolo26l_960 | YOLO26l | 960 | 100 / 63 | 0.558 | 0.345 | 0.644 / 0.544 |
| visdrone_yolo26x_1280 | YOLO26x | 1280 | 150 / 0 (stopped before the first validation) | - | - | - |

Plots (from Ultralytics) are copied for the two completed-or-longest runs: `yolo26l_960_*` and `yolov8n_100ep_*`.
