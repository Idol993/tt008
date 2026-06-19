import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import argparse
import os
import json

from model import ModularContinualNet
from replay_buffer import ReplayBuffer
from forgetting_detector import ForgettingTraceDetector
from adversarial_generator import AdversarialInterferenceGenerator
from interference_trainer import TraceReinforcedInterferenceTrainer
from visualization import (
    plot_forgetting_trajectory,
    plot_interference_weights,
    plot_loss_history,
    plot_multi_task_trajectory,
)


def generate_synthetic_task(
    task_id: int,
    num_samples: int = 1000,
    num_classes: int = 5,
    input_channels: int = 3,
    input_size: int = 32,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed + task_id * 1000)
    inputs = torch.from_numpy(
        rng.randn(num_samples, input_channels, input_size, input_size).astype(np.float32)
    )
    labels = torch.from_numpy(
        rng.randint(0, num_classes, size=num_samples).astype(np.int64)
    )
    offset = task_id * 0.5
    inputs = inputs + offset
    return inputs, labels


def evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, inputs.size(0), batch_size):
            batch_x = inputs[i : i + batch_size].to(device)
            batch_y = labels[i : i + batch_size].to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def run_continual_learning(
    num_tasks: int = 5,
    epochs_per_task: int = 10,
    batch_size: int = 64,
    lr: float = 0.01,
    replay_buffer_size: int = 500,
    replay_weight: float = 1.0,
    interference_base_weight: float = 0.5,
    forgetting_rate_scale: float = 10.0,
    max_interference_weight: float = 2.0,
    top_k_forgetting: int = 1,
    use_orthogonal_projection: bool = True,
    adv_epsilon: float = 0.05,
    adv_steps: int = 5,
    num_classes_per_task: int = 5,
    input_size: int = 32,
    output_dir: str = "outputs",
    device_str: str = "auto",
    seed: int = 42,
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    total_classes = num_tasks * num_classes_per_task
    model = ModularContinualNet(
        in_channels=3,
        num_classes=total_classes,
        input_size=input_size,
    ).to(device)

    module_names = model.get_module_names()
    print(f"Model modules: {module_names}")

    replay_buffer = ReplayBuffer(
        max_size_per_task=replay_buffer_size,
        module_names=module_names,
    )

    forgetting_detector = ForgettingTraceDetector(
        module_names=module_names,
        ema_decay=0.9,
        forgetting_rate_window=5,
    )

    adversarial_generator = AdversarialInterferenceGenerator(
        epsilon=adv_epsilon,
        num_steps=adv_steps,
        module_names=module_names,
    )

    trainer = TraceReinforcedInterferenceTrainer(
        model=model,
        replay_buffer=replay_buffer,
        forgetting_detector=forgetting_detector,
        adversarial_generator=adversarial_generator,
        lr=lr,
        replay_weight=replay_weight,
        interference_base_weight=interference_base_weight,
        forgetting_rate_scale=forgetting_rate_scale,
        max_interference_weight=max_interference_weight,
        top_k_forgetting=top_k_forgetting,
        use_orthogonal_projection=use_orthogonal_projection,
        device=device,
    )

    task_datasets: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    task_accuracies: Dict[int, Dict[int, float]] = {}

    global_step = 0

    for task_id in range(num_tasks):
        print(f"\n{'='*60}")
        print(f"Training on Task {task_id}")
        print(f"{'='*60}")

        inputs, labels = generate_synthetic_task(
            task_id=task_id,
            num_samples=1000,
            num_classes=num_classes_per_task,
            input_size=input_size,
            seed=seed,
        )

        task_datasets[task_id] = (inputs, labels)

        trainer.set_current_task(task_id)

        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs_per_task):
            epoch_losses = []
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                loss_info = trainer.train_step(
                    new_inputs=batch_x,
                    new_labels=batch_y,
                    step=global_step,
                )
                epoch_losses.append(loss_info)
                global_step += 1

            avg_loss = np.mean([l["new_task_loss"] for l in epoch_losses])
            avg_replay = np.mean([l["replay_loss"] for l in epoch_losses])
            avg_interference = np.mean([l["interference_loss"] for l in epoch_losses])
            print(
                f"  Epoch {epoch+1}/{epochs_per_task} | "
                f"Loss: {avg_loss:.4f} | Replay: {avg_replay:.4f} | "
                f"Interference: {avg_interference:.4f}"
            )

        trainer.update_replay_buffer(inputs, labels, task_id)

        print(f"\n  Evaluating on all seen tasks after Task {task_id}...")
        task_accs = {}
        for prev_task in range(task_id + 1):
            test_inputs, test_labels = task_datasets[prev_task]
            acc = evaluate(model, test_inputs, test_labels, device)
            task_accs[prev_task] = acc
            print(f"    Task {prev_task} Accuracy: {acc:.4f}")
        task_accuracies[task_id] = task_accs

        trajectories = forgetting_detector.get_all_trajectories()
        forgetting_rates = forgetting_detector.get_forgetting_rates()

        plot_forgetting_trajectory(
            trajectories,
            forgetting_rates,
            save_path=os.path.join(output_dir, f"forgetting_trajectory_task{task_id}.png"),
            title=f"Forgetting Trajectory after Task {task_id}",
        )

        if any(v > 0 for v in forgetting_rates.values()):
            interference_weights = {
                name: trainer._compute_dynamic_interference_weight(rate)
                for name, rate in forgetting_rates.items()
            }
            plot_interference_weights(
                interference_weights,
                forgetting_rates,
                save_path=os.path.join(output_dir, f"interference_weights_task{task_id}.png"),
            )

    loss_history = trainer.get_loss_history()
    plot_loss_history(
        loss_history,
        save_path=os.path.join(output_dir, "loss_history.png"),
    )

    plot_multi_task_trajectory(
        {task_id: forgetting_detector.get_all_trajectories()},
        save_path=os.path.join(output_dir, "multi_task_trajectory.png"),
    )

    results = {
        "task_accuracies": {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in task_accuracies.items()},
        "final_forgetting_rates": forgetting_rates,
        "config": {
            "num_tasks": num_tasks,
            "epochs_per_task": epochs_per_task,
            "lr": lr,
            "replay_buffer_size": replay_buffer_size,
            "interference_base_weight": interference_base_weight,
            "forgetting_rate_scale": forgetting_rate_scale,
            "max_interference_weight": max_interference_weight,
            "top_k_forgetting": top_k_forgetting,
            "use_orthogonal_projection": use_orthogonal_projection,
        },
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for task_id, accs in task_accuracies.items():
        print(f"After Task {task_id}: {accs}")
    print(f"Final forgetting rates: {forgetting_rates}")
    print(f"Results saved to {output_dir}/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module-Specific Forgetting Trace Continual Learning")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--epochs_per_task", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--replay_buffer_size", type=int, default=500)
    parser.add_argument("--replay_weight", type=float, default=1.0)
    parser.add_argument("--interference_base_weight", type=float, default=0.5)
    parser.add_argument("--forgetting_rate_scale", type=float, default=10.0)
    parser.add_argument("--max_interference_weight", type=float, default=2.0)
    parser.add_argument("--top_k_forgetting", type=int, default=1)
    parser.add_argument("--use_orthogonal_projection", action="store_true", default=True)
    parser.add_argument("--no_orthogonal_projection", action="store_false", dest="use_orthogonal_projection")
    parser.add_argument("--adv_epsilon", type=float, default=0.05)
    parser.add_argument("--adv_steps", type=int, default=5)
    parser.add_argument("--num_classes_per_task", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_continual_learning(**vars(args))
