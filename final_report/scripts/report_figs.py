"""Result figures for the Part-E report (static PNG, light mode)."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/disks/data/aerotail/final_report"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COL = {"sac": "#2a78d6", "ppo": "#eb6834", "recurrent_ppo": "#1baf7a"}
NAME = {"sac": "SAC", "ppo": "PPO", "recurrent_ppo": "RecurrentPPO"}
ALGOS = ["sac", "ppo", "recurrent_ppo"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False})


def load(f):
    return {(r["algo"], int(r["stage"])): r for r in csv.DictReader(open(f"{ROOT}/tables/{f}"))}


B, N = load("eval_blocks.csv"), load("eval_airsimnh.csv")


# ---- Figure: evaluation across curriculum stages (Blocks) -----------------------------------
def eval_stages():
    metrics = [("total_reward_mean", "Total reward (higher is better)", 1),
               ("collision_rate", "Collision rate (lower is better)", 100),
               ("mean_distance_error_mean", "Distance error, m (lower is better)", 1),
               ("visibility_ratio_mean", "Visibility ratio (higher is better)", 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    w = 0.26
    for ax, (key, title, scale) in zip(axes.flat, metrics):
        for i, a in enumerate(ALGOS):
            xs, ys = [], []
            for s in range(4):
                if (a, s) in B:
                    xs.append(s + (i - 1) * w); ys.append(float(B[(a, s)][key]) * scale)
                else:
                    ax.text(s + (i - 1) * w, 0, "n/a", ha="center", va="bottom", fontsize=7, color=INK2, rotation=90)
            ax.bar(xs, ys, width=w - 0.02, color=COL[a], label=NAME[a], edgecolor=SURF, linewidth=1)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"Stage {s}" for s in range(4)])
        ax.grid(axis="x", visible=False)
        if key == "collision_rate": ax.set_ylim(0, 105); ax.set_ylabel("%")
    axes[0, 0].legend(ncol=3, loc="upper right", fontsize=9)
    fig.suptitle("Evaluation in Blocks per curriculum stage (20 episodes, seeds 10000-10019)", x=0.01, ha="left",
                 fontweight="bold")
    fig.text(0.01, 0.005, "n/a: SAC stage 3 diverged during training; RecurrentPPO was trained on stages 0-2 only.",
             color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(f"{ROOT}/figures/fig_eval_blocks_all_stages.png", dpi=200)


# ---- Figure: Blocks -> AirSimNH (dumbbell) ---------------------------------------------------
def generalisation():
    order = [(a, s) for a in ALGOS for s in range(4) if (a, s) in B]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, (key, title, scale) in zip(axes, [("collision_rate", "Collision rate (%)", 100),
                                             ("mean_distance_error_mean", "Distance error (m)", 1)]):
        for y, (a, s) in enumerate(order):
            b, n = float(B[(a, s)][key]) * scale, float(N[(a, s)][key]) * scale
            ax.plot([b, n], [y, y], color=GRID, linewidth=3, zorder=1)
            ax.scatter([b], [y], s=60, facecolor=SURF, edgecolor=COL[a], linewidth=2, zorder=2)
            ax.scatter([n], [y], s=60, color=COL[a], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
        if key == "collision_rate": ax.set_xlim(-5, 105)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{NAME[a]} stage {s}" for a, s in order]); axes[0].invert_yaxis()
    h1 = axes[1].scatter([], [], s=60, facecolor=SURF, edgecolor=INK2, linewidth=2, label="Blocks (training environment)")
    h2 = axes[1].scatter([], [], s=60, color=INK2, label="AirSimNH (zero-shot)")
    fig.legend(handles=[h1, h2], loc="upper left", ncol=2, fontsize=9, bbox_to_anchor=(0.005, 0.95))
    fig.suptitle("Zero-shot transfer from Blocks to AirSimNH (same models, same protocol)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{ROOT}/figures/fig_generalisation_blocks_vs_airsimnh.png", dpi=200)


# ---- Figure: training curves -------------------------------------------------------------------
def training_curves():
    rows = list(csv.DictReader(open(f"{ROOT}/training_curves/training_curves_all.csv")))
    start = {"sac": {0: 0, 1: 200000, 2: 300000, 3: 450000},
             "ppo": {0: 0, 1: 200704, 2: 301056, 3: 501760},
             "recurrent_ppo": {0: 0, 1: 1001000, 2: 2001448}}
    length = {"sac": {0: 500000, 1: 100000, 2: 150000, 3: 182544},
              "ppo": {0: 200704, 1: 100352, 2: 200704, 3: 301056},
              "recurrent_ppo": {0: 1001000, 1: 1000448, 2: 500736}}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharey=True)
    for ax, a in zip(axes, ALGOS):
        off = 0
        for s in sorted(length[a]):
            pts = [(off + int(r["timestep"]) - start[a][s], float(r["ep_rew_mean"])) for r in rows
                   if r["algo"] == a and int(r["stage"]) == s]
            if pts:
                ax.plot(*zip(*pts), color=COL[a], linewidth=1.6)
            else:
                ax.text(off + length[a][s] / 2, 0, "no episode-reward\nlog for this run", ha="center", va="center",
                        fontsize=8, color=INK2)
            ax.axvline(off, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
            label = f"Stage {s}" + (" (diverged)" if (a, s) == ("sac", 3) else "")
            ax.text(off + length[a][s] / 2, 1.02, label, transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=8.5, color=INK2)
            off += length[a][s]
        ax.set_xlim(0, off)
        ax.set_title(f"{NAME[a]}", loc="left", pad=16)
        ax.set_ylabel("Mean episode reward")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"))
    axes[-1].set_xlabel("Training timesteps (curriculum stages laid end to end)")
    fig.suptitle("Training reward (SB3 rollout/ep_rew_mean: mean of the last 100 training episodes, stochastic policy)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{ROOT}/figures/fig_training_curves_all.png", dpi=200)


eval_stages(); generalisation(); training_curves()
print("ok")
