#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# airsim_env.py
#
# Our Gymnasium wrapper around AirSim. Drone1/Target1 (or Drone2/Target2,
# via the drone_name/target_name constructor args below) is the
# agent/target pair. We added a second vehicle pair to the simulator so
# two algorithms can genuinely train concurrently against the SAME running
# simulator instance, each owning its own drone - earlier we found that
# running two algorithms against a single shared Drone1 corrupted both
# runs' data (RPC contention, timeouts, garbage transitions); giving each
# algorithm its own named vehicle avoids that entirely, since AirSim scopes
# every API call by vehicle_name.
#
# OBSTACLE SENSING v2 (3 angled sensors): our first attempt (a single
# forward-facing DistanceSensor) didn't reduce collisions - diagnosis
# showed 6 of 7 building collisions happened while that single narrow ray
# read "all clear", because the drone is constantly turning to track a
# moving target and the ray doesn't cover what's beside it. We now use
# three sensors (center, +/-35 degrees) per vehicle, and drive the
# obstacle penalty from whichever reads closest.
# ---------------------------------------------------------------------------
import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim


class AirSimTargetFollowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",
        drone_name="Drone1",
        target_name="Target1",
        target_spawn=(10.0, 0.0, -1.0),
        episode_len_s=30.0,
        dt=0.1,
        desired_distance=8.0,
        max_speed=5.0,
        max_yaw_rate=60.0,
        obstacle_sensor_max_range=20.0,
        obstacle_safety_margin=8.0,
        use_obstacle_sensors=False,
        obs_noise_std=0.0,
        detection_dropout_prob=0.0,
        latency_steps=0,
        occlusion_prob=0.0,
        occlusion_max_steps=0,
    ):
        super().__init__()

        self.drone_name = drone_name
        self.target_name = target_name
        self.target_spawn = np.array(target_spawn, dtype=np.float64)

        self.dt = dt
        self.episode_len_steps = int(episode_len_s / dt)
        self.desired_distance = desired_distance
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.obstacle_sensor_max_range = obstacle_sensor_max_range
        self.obstacle_safety_margin = obstacle_safety_margin
        # use_obstacle_sensors=False reverts to the validated, matched 10-D
        # no-sensor environment (both stage 0 and stage 1 were trained
        # together at this size, confirmed compatible) - the sensor
        # investigation (11/13/14-D variants) is documented separately as
        # its own side study and doesn't need to be the active config for
        # the main curriculum.
        self.use_obstacle_sensors = use_obstacle_sensors
        self.obstacle_sensor_names = (
            ["DistanceCenter", "DistanceLeft", "DistanceRight", "DistanceDown"]
            if use_obstacle_sensors else []
        )

        self.obs_noise_std = obs_noise_std
        self.detection_dropout_prob = detection_dropout_prob
        self.latency_steps = latency_steps
        self.occlusion_prob = occlusion_prob
        self.occlusion_max_steps = occlusion_max_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation (all normalised, roughly [-1, 1]):
        #   [0:2]   bearing to target (horizontal, vertical)
        #   [2]     apparent size proxy (inverse distance)
        #   [3]     visibility flag
        #   [4:6]   target relative velocity (x, y) in drone body frame
        #   [6:10]  our own last action
        #   [10:14] obstacle proximities (centre/left/right/down), ONLY if
        #           use_obstacle_sensors=True - omitted entirely otherwise,
        #           giving the validated 10-D observation space.
        n_obstacle_dims = 4 if use_obstacle_sensors else 0
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10 + n_obstacle_dims,), dtype=np.float32
        )

        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()

        self._obs_history = []
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._occlusion_countdown = 0
        self._step_count = 0
        self._collision_baseline_time = 0
        self._rng = np.random.default_rng()

    def _init_target_trajectory(self):
        self._target_pos = self.target_spawn.copy()
        self._target_heading = self._rng.uniform(0, 2 * math.pi)
        self._target_speed = self._rng.uniform(1.0, 3.0)
        self._target_stop_timer = 0

    def _step_target(self):
        if self._target_stop_timer > 0:
            self._target_stop_timer -= 1
            speed = 0.0
        else:
            speed = self._target_speed
            if self._rng.random() < 0.01:
                self._target_heading += self._rng.uniform(-math.pi / 2, math.pi / 2)
            if self._rng.random() < 0.005:
                self._target_stop_timer = int(self._rng.uniform(1, 3) / self.dt)

        self._target_pos[0] += speed * math.cos(self._target_heading) * self.dt
        self._target_pos[1] += speed * math.sin(self._target_heading) * self.dt

        pose = airsim.Pose(
            airsim.Vector3r(*self._target_pos),
            airsim.to_quaternion(0, 0, self._target_heading),
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.target_name)

    def _get_obstacle_distances(self):
        dists = []
        for name in self.obstacle_sensor_names:
            try:
                data = self.client.getDistanceSensorData(name, vehicle_name=self.drone_name)
                d = float(data.distance)
                if d <= 0 or d > self.obstacle_sensor_max_range:
                    d = self.obstacle_sensor_max_range
            except Exception:
                d = self.obstacle_sensor_max_range
            dists.append(d)
        return dists

    def _compute_observation(self):
        drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        drone_pos = drone_state.kinematics_estimated.position
        drone_vel = drone_state.kinematics_estimated.linear_velocity
        _, _, drone_yaw = airsim.to_eularian_angles(drone_state.kinematics_estimated.orientation)

        rel = self._target_pos - np.array([drone_pos.x_val, drone_pos.y_val, drone_pos.z_val])
        distance = float(np.linalg.norm(rel))

        world_bearing = math.atan2(rel[1], rel[0])
        rel_bearing = (world_bearing - drone_yaw + math.pi) % (2 * math.pi) - math.pi
        fov_half = math.radians(45)
        bearing_h = np.clip(rel_bearing / fov_half, -1, 1)
        bearing_v = np.clip((self._target_pos[2] - drone_pos.z_val) / max(distance, 1e-3), -1, 1)

        size_proxy = np.clip(self.desired_distance / max(distance, 1e-3) - 1.0, -1, 1)

        visible = 1.0
        if self._occlusion_countdown > 0:
            self._occlusion_countdown -= 1
            visible = 0.0
        elif self._rng.random() < self.occlusion_prob:
            self._occlusion_countdown = self._rng.integers(1, max(self.occlusion_max_steps, 2))
            visible = 0.0
        if abs(rel_bearing) > fov_half * 1.5:
            visible = 0.0
        if self._rng.random() < self.detection_dropout_prob:
            visible = 0.0

        rel_vel_x = np.clip((0.0 - drone_vel.x_val) / self.max_speed, -1, 1)
        rel_vel_y = np.clip((0.0 - drone_vel.y_val) / self.max_speed, -1, 1)

        if self.use_obstacle_sensors:
            obstacle_dists = self._get_obstacle_distances()
            obstacle_proximities = [
                np.clip(2.0 * (d / self.obstacle_sensor_max_range) - 1.0, -1, 1) for d in obstacle_dists
            ]
        else:
            obstacle_dists = []
            obstacle_proximities = []

        obs = np.array([
            bearing_h, bearing_v, size_proxy, visible,
            rel_vel_x, rel_vel_y,
            *self._prev_action,
            *obstacle_proximities,
        ], dtype=np.float32)

        if self.obs_noise_std > 0:
            noise = self._rng.normal(0, self.obs_noise_std, size=6).astype(np.float32)
            obs[[0, 1, 2, 4, 5]] += noise[[0, 1, 2, 4, 5]]
            obs = np.clip(obs, -1, 1)

        if visible == 0.0:
            obs[0:3] = -1.0

        self._obs_history.append(obs)
        altitude = -drone_pos.z_val
        return obs, distance, visible, altitude, obstacle_dists

    def _get_delayed_observation(self, live_obs):
        if self.latency_steps <= 0 or len(self._obs_history) <= self.latency_steps:
            return live_obs
        return self._obs_history[-1 - self.latency_steps]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # IMPORTANT: client.reset() resets EVERY vehicle in the simulation,
        # not just ours - with two environments running concurrently (one
        # per drone, so both algorithms can genuinely train at once), this
        # was causing each algorithm's frequent episode resets to yank the
        # OTHER algorithm's drone back to its spawn mid-flight, corrupting
        # both runs (we saw this directly: both SAC and PPO collapsed to
        # ~9-step episodes when run concurrently). Instead we manually
        # reset only our own vehicle: teleport it back to a spawn pose,
        # re-arm, and take off, never touching the global sim state.
        spawn_pose = airsim.Pose(
            airsim.Vector3r(self.target_spawn[0] - 10.0, self.target_spawn[1], 0.0),
            airsim.to_quaternion(0, 0, 0),
        )
        self.client.simSetVehiclePose(spawn_pose, ignore_collision=True, vehicle_name=self.drone_name)
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        # Teleporting keeps the drone's velocity, and AirSim's takeoff()
        # refuses (silently, through .join()) if |v| > 0.05 m/s - which left
        # episodes starting on the ground after any episode that ended while
        # moving. Wait for the drone to settle so takeoff actually happens.
        wait_start = time.time()
        while time.time() - wait_start < 5.0:
            v = self.client.getMultirotorState(
                vehicle_name=self.drone_name
            ).kinematics_estimated.linear_velocity
            if math.sqrt(v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2) <= 0.05:
                break
            time.sleep(0.05)
        self.client.takeoffAsync(vehicle_name=self.drone_name).join()

        self._collision_baseline_time = self.client.simGetCollisionInfo(
            vehicle_name=self.drone_name
        ).time_stamp

        self._init_target_trajectory()
        self._obs_history = []
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._occlusion_countdown = 0
        self._step_count = 0

        obs, _, _, _, _ = self._compute_observation()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        vx = float(action[0]) * self.max_speed
        vy = float(action[1]) * self.max_speed
        vz = float(action[2]) * self.max_speed
        yaw_rate = float(action[3]) * self.max_yaw_rate

        self.client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, self.dt,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=self.drone_name,
        ).join()

        self._step_target()

        live_obs, distance, visible, altitude, obstacle_dists = self._compute_observation()
        obs = self._get_delayed_observation(live_obs)

        centring_penalty = -(obs[0] ** 2 + obs[1] ** 2)
        distance_penalty = -abs(distance - self.desired_distance) / self.desired_distance
        visibility_reward = 1.0 if visible else -1.0
        smoothness_penalty = -float(np.sum((action - self._prev_action) ** 2))

        if obstacle_dists:
            closest_obstacle = min(obstacle_dists)
            proximity_ratio = max(0.0, (self.obstacle_safety_margin - closest_obstacle) / self.obstacle_safety_margin)
            obstacle_penalty = -(proximity_ratio ** 2) * 2.0
        else:
            closest_obstacle = self.obstacle_sensor_max_range
            obstacle_penalty = 0.0

        reward = (
            1.0 * centring_penalty
            + 1.0 * distance_penalty
            + 2.0 * visibility_reward
            + 0.1 * smoothness_penalty
            + 1.0 * obstacle_penalty
        )

        self._prev_action = action
        self._step_count += 1

        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        collision = (
            collision_info.has_collided
            and collision_info.time_stamp > self._collision_baseline_time
        )
        terminated = bool(collision)
        truncated = self._step_count >= self.episode_len_steps
        if collision:
            reward -= 10.0

        info = {
            "distance": distance,
            "visible": visible,
            "collision": collision,
            "bearing_h": float(live_obs[0]),
            "bearing_v": float(live_obs[1]),
            "collision_object": collision_info.object_name if collision else None,
            "altitude": altitude,
            "obstacle_distance": closest_obstacle,
            "obstacle_distances": obstacle_dists,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.client.armDisarm(False, self.drone_name)
        self.client.enableApiControl(False, self.drone_name)
