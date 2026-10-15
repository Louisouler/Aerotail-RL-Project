#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# control.py - optional run control for the dashboard (start/stop a demo).
#
# The dashboard never talks to AirSim or loads a model itself. This module
# only starts/stops scripts/10_dashboard_demo.py as a SEPARATE
# process (own session, so closing the dashboard never kills it) and reads
# /proc to see what is running. Safety rules:
#   * at most one demo process at a time;
#   * refuse to start while any training process (04_train_rl.py or
#     09_train_with_retry.sh) is running, so a training run is never
#     disturbed;
#   * refuse to start if the AirSim RPC port is not listening;
#   * Stop sends SIGINT (the launcher finishes cleanly, writes run_end and
#     disarms its own drone) and escalates to SIGTERM only after a timeout.
# ---------------------------------------------------------------------------
import os
import re
import signal
import socket
import subprocess
import threading
import time

PROJECT_DIR = "/mnt/disks/data/aerotail"
CKPT_ROOT = os.path.join(PROJECT_DIR, "checkpoints", "rl", "active")
DEMO_SCRIPT = os.path.join(PROJECT_DIR, "scripts", "10_dashboard_demo.py")
EXPERIMENT_PYTHON = "/mnt/disks/data/aerotail-venv/bin/python"
DEMO_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime", "demo.log")
AIRSIM_PORT = 41451

ALGO_DISPLAY = {"recurrent_ppo": "RecurrentPPO", "ppo": "PPO", "sac": "SAC"}
ON_POLICY = {"ppo", "recurrent_ppo"}
VEHICLES = {"Drone1": "Target1", "Drone2": "Target2"}
TRAINING_MARKERS = ("04_train_rl.py", "09_train_with_retry.sh")
DEMO_MARKER = "10_dashboard_demo.py"


def available_models():
    """[(value, label)] for every finished model in checkpoints/rl/active."""
    out = []
    order = {"recurrent_ppo": 0, "ppo": 1, "sac": 2}
    for name in os.listdir(CKPT_ROOT):
        m = re.fullmatch(r"(recurrent_ppo|ppo|sac)_stage(\d+)", name)
        if not m:
            continue
        algo, stage = m.group(1), int(m.group(2))
        d = os.path.join(CKPT_ROOT, name)
        if not os.path.exists(os.path.join(d, "final_model.zip")):
            continue
        if algo in ON_POLICY and not os.path.exists(os.path.join(d, "vecnormalize.pkl")):
            continue
        out.append((order[algo], stage, f"{algo}:{stage}", f"{ALGO_DISPLAY[algo]} · Stage {stage}"))
    return [(v, label) for _, _, v, label in sorted(out)]


def _processes():
    """[(pid, cmdline)] of all processes except this one."""
    procs = []
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if cmd:
            procs.append((int(entry), cmd))
    return procs


def _runs_script(cmd, marker):
    """True if cmd is `python|bash [flags] .../marker ...` - i.e. the script itself
    is executing, not merely mentioned in some other command line."""
    tokens = cmd.split()
    if not tokens:
        return False
    first = os.path.basename(tokens[0])
    if not (first.startswith("python") or first in ("bash", "sh")):
        return False
    return any(t.endswith(marker) for t in tokens[1:4])


def running_demos():
    return [(p, c) for p, c in _processes() if _runs_script(c, DEMO_MARKER)]


def running_training():
    return [(p, c) for p, c in _processes() if any(_runs_script(c, m) for m in TRAINING_MARKERS)]


def airsim_listening(host="127.0.0.1", port=AIRSIM_PORT, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class DemoController:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None  # Popen of a demo we started (for reaping)

    def _reap(self):
        if self._proc is not None and self._proc.poll() is not None:
            self._proc = None

    def status(self):
        with self._lock:
            self._reap()
            demos = running_demos()
            if demos:
                return dict(running=True, pid=demos[0][0], cmd=demos[0][1])
            return dict(running=False)

    def start(self, model_value, vehicle, episodes):
        with self._lock:
            self._reap()
            if running_demos():
                return False, "A demo is already running - stop it first."
            training = running_training()
            if training:
                return False, f"Training is running (pid {training[0][0]}) - refusing to start a demo."
            valid = dict(available_models())
            if model_value not in valid:
                return False, "Unknown model."
            if vehicle not in VEHICLES:
                return False, "Unknown vehicle."
            try:
                episodes = int(episodes)
            except (TypeError, ValueError):
                return False, "Episodes must be a number."
            if not 1 <= episodes <= 20:
                return False, "Episodes must be between 1 and 20."
            if not airsim_listening():
                return False, f"AirSim is not running (port {AIRSIM_PORT} closed)."
            algo, stage = model_value.split(":")
            cmd = [EXPERIMENT_PYTHON, "-u", DEMO_SCRIPT, "--algo", algo, "--stage", stage,
                   "--vehicle", vehicle, "--target", VEHICLES[vehicle], "--episodes", str(episodes)]
            os.makedirs(os.path.dirname(DEMO_LOG), exist_ok=True)
            log = open(DEMO_LOG, "w")
            self._proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
            log.close()
            return True, f"Started {valid[model_value]} on {vehicle} ({episodes} episode(s)), pid {self._proc.pid}."

    def stop(self, timeout_s=20.0):
        with self._lock:
            demos = running_demos()
            if not demos:
                return False, "No demo is running."
            pid = demos[0][0]
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                return False, "Demo already exited."

        def _escalate():
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if not any(p == pid for p, _ in running_demos()):
                    return
                time.sleep(0.5)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        threading.Thread(target=_escalate, daemon=True).start()
        return True, f"Stopping demo (pid {pid})..."
