import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


def plot_forgetting_trajectory(
    trajectories: Dict[str, List[Tuple[int, float]]],
    forgetting_rates: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Module Forgetting Trajectory",
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax_traj = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    for idx, (module_name, trajectory) in enumerate(trajectories.items()):
        if len(trajectory) == 0:
            continue
        steps = [t[0] for t in trajectory]
        mmd_values = [t[1] for t in trajectory]
        ax_traj.plot(steps, mmd_values, color=colors[idx], label=module_name, linewidth=2)

        if len(mmd_values) > 5:
            window = min(10, len(mmd_values))
            ema = []
            ema_val = mmd_values[0]
            decay = 0.9
            for v in mmd_values:
                ema_val = decay * ema_val + (1 - decay) * v
                ema.append(ema_val)
            ax_traj.plot(steps, ema, color=colors[idx], linestyle="--", alpha=0.6, linewidth=1.5)

    ax_traj.set_xlabel("Training Step", fontsize=12)
    ax_traj.set_ylabel("MMD Score", fontsize=12)
    ax_traj.set_title(title, fontsize=14)
    ax_traj.legend(fontsize=10)
    ax_traj.grid(True, alpha=0.3)

    ax_rates = axes[1]
    module_names = list(forgetting_rates.keys())
    rates = [forgetting_rates[name] for name in module_names]

    bar_colors = []
    max_rate = max(rates) if rates else 1.0
    for r in rates:
        intensity = r / max_rate if max_rate > 0 else 0.0
        bar_colors.append(plt.cm.RdYlGn_r(intensity))

    bars = ax_rates.barh(module_names, rates, color=bar_colors)
    ax_rates.set_xlabel("Forgetting Rate", fontsize=12)
    ax_rates.set_title("Module Forgetting Rates", fontsize=12)
    ax_rates.grid(True, alpha=0.3, axis="x")

    for bar, rate in zip(bars, rates):
        ax_rates.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return fig


def plot_interference_weights(
    interference_weights: Dict[str, float],
    forgetting_rates: Dict[str, float],
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    modules = list(interference_weights.keys())
    weights = [interference_weights[m] for m in modules]
    rates = [forgetting_rates.get(m, 0.0) for m in modules]

    x = np.arange(len(modules))
    width = 0.35

    ax.bar(x - width / 2, rates, width, label="Forgetting Rate", color="coral", alpha=0.8)
    ax.bar(x + width / 2, weights, width, label="Interference Weight", color="steelblue", alpha=0.8)

    ax.set_xlabel("Module", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Forgetting Rate vs Interference Weight per Module", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return fig


def plot_loss_history(
    loss_history: List[Dict[str, float]],
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = [h["step"] for h in loss_history]
    new_losses = [h["new_task_loss"] for h in loss_history]
    replay_losses = [h.get("replay_loss", 0.0) for h in loss_history]
    interference_losses = [h.get("interference_loss", 0.0) for h in loss_history]

    ax.plot(steps, new_losses, label="New Task Loss", linewidth=2)
    ax.plot(steps, replay_losses, label="Replay Loss", linewidth=2, linestyle="--")
    ax.plot(steps, interference_losses, label="Interference Loss", linewidth=2, linestyle=":")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss History", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return fig


def plot_multi_task_trajectory(
    task_trajectories: Dict[int, Dict[str, List[Tuple[int, float]]]],
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, len(task_trajectories), figsize=(6 * len(task_trajectories), 5))
    if len(task_trajectories) == 1:
        axes = [axes]

    for ax, (task_id, trajectories) in zip(axes, task_trajectories.items()):
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        for idx, (module_name, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) == 0:
                continue
            steps = [t[0] for t in trajectory]
            mmd_values = [t[1] for t in trajectory]
            ax.plot(steps, mmd_values, color=colors[idx], label=module_name, linewidth=2)

        ax.set_xlabel("Step")
        ax.set_ylabel("MMD Score")
        ax.set_title(f"Task {task_id} Forgetting Trajectory")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)

    return fig
