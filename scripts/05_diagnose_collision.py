#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 05_diagnose_collision.py
#
# We isolate whether our short-episode problem is a REAL collision (the
# drone genuinely crashing) or a stale/always-updating collision flag from
# AirSim. We bypass our env entirely and just hover in place with the raw
# client, printing collision info every step, so there's no exploratory
# policy action that could cause a genuine crash - if collisions still fire
# here, it's definitely a detection bug, not real behavior.
# ---------------------------------------------------------------------------
import time
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True, "Drone1")
client.armDisarm(True, "Drone1")
client.takeoffAsync(vehicle_name="Drone1").join()

baseline = client.simGetCollisionInfo(vehicle_name="Drone1")
print(f"BASELINE right after takeoff: has_collided={baseline.has_collided} "
      f"time_stamp={baseline.time_stamp} object_name='{baseline.object_name}'")

client.hoverAsync(vehicle_name="Drone1").join()

for i in range(30):
    time.sleep(0.1)
    info = client.simGetCollisionInfo(vehicle_name="Drone1")
    pos = client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position
    is_new = info.has_collided and info.time_stamp > baseline.time_stamp
    print(f"step {i:2d}: has_collided={info.has_collided} time_stamp={info.time_stamp} "
          f"NEW_vs_baseline={is_new} object='{info.object_name}' "
          f"pos=({pos.x_val:.2f},{pos.y_val:.2f},{pos.z_val:.2f})")

client.armDisarm(False, "Drone1")
client.enableApiControl(False, "Drone1")
print("done")
