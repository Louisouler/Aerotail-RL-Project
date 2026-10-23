"""Workflow and system-architecture diagrams for the Part-E report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/mnt/disks/data/aerotail/final_report/figures"
INK, INK2, LINE = "#0b0b0b", "#52514e", "#8a8984"
FILL = {"sim": "#e8f1fc", "env": "#e6f6ef", "agent": "#fdeee6", "tool": "#f3f2ef", "side": "#ffffff"}
EDGE = {"sim": "#2a78d6", "env": "#1baf7a", "agent": "#eb6834", "tool": "#8a8984", "side": "#8a8984"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def box(ax, x, y, w, h, title, body="", kind="tool", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", facecolor=FILL[kind],
                                edgecolor=EDGE[kind], linewidth=1.6, linestyle=(0, (4, 3)) if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.4, color=INK2, linespacing=1.4)


def arrow(ax, p, q, text="", dashed=False, rad=0.0, tx=0.0, ty=0.1, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=LINE, linewidth=1.4,
                                 linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, ha=ha, va="bottom", fontsize=8.2,
                color=INK2, linespacing=1.3)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


# ---- Workflow ----------------------------------------------------------------------------------
fig, ax = canvas(17, 5.4)
ax.text(0.2, 5.3, "Project workflow", fontsize=13, fontweight="bold", va="top")
steps = [
    ("1. Simulation setup", "AirSim v1.7.0 (UE4), Blocks map\n2 drone/target pairs, ClockSpeed 2\nGCP VM, NVIDIA L4, Xvfb display", "sim"),
    ("2. Environment design", "Gymnasium env: 10-D observation\n4-D velocity + yaw-rate action\nshaped reward, 300-step episodes", "env"),
    ("3. Curriculum training", "SAC, PPO, RecurrentPPO (SB3)\nstages 0 -> 1 -> 2 -> 3, each\nfine-tuned from the previous one", "agent"),
    ("4. Evaluation", "Blocks, Drone2/Target2\n20 episodes, seeds 10000-10019\ndeterministic policy", "tool"),
    ("5. Generalisation", "zero-shot in AirSimNH (houses,\ntrees, cars, power lines)\nsame models, same protocol", "sim"),
]
x0, w, gap, y, h = 0.2, 3.05, 0.35, 2.55, 1.75
for i, (t, b, k) in enumerate(steps):
    box(ax, x0 + i * (w + gap), y, w, h, t, b, k)
    if i:
        arrow(ax, (x0 + i * (w + gap) - gap, y + h / 2), (x0 + i * (w + gap), y + h / 2))
bx = lambda i: x0 + i * (w + gap)
box(ax, 0.2, 0.3, 6.4, 1.55, "Perception side study (offline, not in the RL loop)",
    "YOLOv8n / YOLO26l trained on VisDrone (Ultralytics); the RL observation\nuses simulator ground truth with noise, dropout, latency and occlusion", "side", dashed=True)
box(ax, 7.0, 0.3, 4.6, 1.55, "Monitoring and demos",
    "Plotly Dash dashboard (read-only)\nlive telemetry, simulator view, recordings", "tool")
box(ax, 12.0, 0.3, 4.75, 1.55, "Results and report",
    "evaluation tables, figures, training curves\nlimitations and future work", "tool")
arrow(ax, (bx(2) + w / 2, y), (9.3, 1.85), "trained models", tx=0.15, ty=-0.05, ha="left")
arrow(ax, (bx(3) + w / 2, y), (14.4, 1.85), "metrics", tx=0.15, ty=-0.05, ha="left")
arrow(ax, (bx(3) + 0.5, y + h), (bx(2) + w - 0.5, y + h), dashed=True, rad=0.5)
ax.text((bx(2) + w + bx(3)) / 2, y + h + 0.62, "next stage / retrain", ha="center", fontsize=8.2, color=INK2)
fig.savefig(f"{ROOT}/fig_workflow.png", dpi=200, bbox_inches="tight", facecolor="white")

# ---- System architecture -----------------------------------------------------------------------
fig, ax = canvas(14, 7.4)
ax.text(0.2, 7.3, "System architecture (training and evaluation loop)", fontsize=13, fontweight="bold", va="top")
box(ax, 0.2, 4.3, 3.4, 2.5, "AirSim simulator", "Unreal Engine 4, Blocks / AirSimNH\nSimpleFlight multirotors\nDrone1 + Target1 (training)\nDrone2 + Target2 (evaluation)\nmsgpack-RPC on port 41451", "sim")
box(ax, 5.0, 2.9, 4.4, 3.9, "AirSimTargetFollowEnv (Gymnasium)",
    "target motion: kinematic random walk,\n1-3 m/s, set with simSetVehiclePose\n\nobservation (10-D): bearing h/v, size\nproxy, visibility, relative velocity x/y,\nprevious action (4)\n\ncurriculum: noise, detection dropout,\nlatency, occlusion (stages 0-3)\n\nreward: centring + distance-to-8 m\n+ visibility + smoothness, -10 collision",
    "env")
box(ax, 10.8, 5.4, 3.0, 1.4, "Wrappers", "Monitor, DummyVecEnv\nVecNormalize (PPO, RecurrentPPO)", "tool")
box(ax, 10.8, 2.9, 3.0, 1.9, "RL agent (SB3)", "SAC: MLP 256x256, replay 200k\nPPO: MLP 256x256\nRecurrentPPO: MLP 256x256\n+ LSTM 128 (sb3-contrib)", "agent")
arrow(ax, (3.6, 6.0), (5.0, 6.0), "state,\ncollisions")
arrow(ax, (5.0, 4.7), (3.6, 4.7), "velocity and\nyaw-rate command,\ntarget pose")
arrow(ax, (9.4, 6.1), (10.8, 6.1), "obs, reward")
arrow(ax, (12.3, 5.4), (12.3, 4.8))
arrow(ax, (10.8, 3.6), (9.4, 3.6), "action\nin [-1, 1]^4")
box(ax, 0.2, 0.3, 4.2, 1.9, "Training orchestration", "04_train_rl.py (one stage)\n09_train_with_retry.sh: relaunches the\nsimulator and resumes from the last\ncheckpoint (every 5k steps) after a crash", "tool")
box(ax, 4.8, 0.3, 4.6, 1.9, "Evaluation", "07_evaluate_all.py: 20 seeded episodes\nper model -> eval_results.csv\neval_all_to.py: same protocol in AirSimNH", "tool")
box(ax, 9.8, 0.3, 4.0, 1.9, "Monitoring dashboard", "10_dashboard_demo.py -> telemetry JSONL\nPlotly Dash app (read-only) and the\nsimulator view from the Xvfb display", "tool")
arrow(ax, (1.9, 2.2), (1.9, 4.3), "launches", tx=0.12, ha="left")
arrow(ax, (7.1, 2.2), (7.1, 2.9), "runs episodes", tx=0.12, ty=-0.05, ha="left")
arrow(ax, (11.0, 2.2), (9.4, 3.05), "demo episodes", tx=0.1, ty=-0.1, ha="left")
fig.savefig(f"{ROOT}/fig_system_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
