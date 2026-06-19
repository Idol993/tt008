import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math


def _gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_list: Optional[List[float]] = None,
) -> torch.Tensor:
    if sigma_list is None:
        sigma_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)

    xx = torch.mm(x_flat, x_flat.t())
    yy = torch.mm(y_flat, y_flat.t())
    xy = torch.mm(x_flat, y_flat.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dist_xx = rx.t() + rx - 2.0 * xx
    dist_yy = ry.t() + ry - 2.0 * yy
    dist_xy = rx.t() + ry - 2.0 * xy

    result = torch.zeros(1, device=x.device)
    for sigma in sigma_list:
        gamma = 1.0 / (2.0 * sigma ** 2)
        k_xx = torch.exp(-gamma * dist_xx)
        k_yy = torch.exp(-gamma * dist_yy)
        k_xy = torch.exp(-gamma * dist_xy)

        mmd_xx = k_xx.sum() / (x.size(0) * x.size(0))
        mmd_yy = k_yy.sum() / (y.size(0) * y.size(0))
        mmd_xy = k_xy.sum() / (x.size(0) * y.size(0))

        result = result + mmd_xx + mmd_yy - 2.0 * mmd_xy

    result = result / len(sigma_list)
    return torch.clamp(result, min=0.0)


def compute_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    sigma_list: Optional[List[float]] = None,
) -> torch.Tensor:
    return _gaussian_kernel(source, target, sigma_list)


class ForgettingTraceDetector:
    def __init__(
        self,
        module_names: List[str],
        ema_decay: float = 0.9,
        forgetting_rate_window: int = 5,
        sigma_list: Optional[List[float]] = None,
    ):
        self.module_names = module_names
        self.ema_decay = ema_decay
        self.forgetting_rate_window = forgetting_rate_window
        self.sigma_list = sigma_list

        self._trajectory: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self._mmd_ema: Dict[str, float] = {}
        self._forgetting_rates: Dict[str, float] = {}
        self._step_counter: int = 0

        for name in module_names:
            self._mmd_ema[name] = 0.0
            self._forgetting_rates[name] = 0.0

    @torch.no_grad()
    def detect(
        self,
        current_outputs: Dict[str, torch.Tensor],
        reference_outputs: Dict[str, torch.Tensor],
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        if step is not None:
            self._step_counter = step
        else:
            self._step_counter += 1

        mmd_scores = {}
        for name in self.module_names:
            if name not in current_outputs or name not in reference_outputs:
                mmd_scores[name] = 0.0
                continue

            cur = current_outputs[name]
            ref = reference_outputs[name]

            min_samples = min(cur.size(0), ref.size(0))
            if min_samples < 2:
                mmd_scores[name] = 0.0
                continue

            cur = cur[:min_samples]
            ref = ref[:min_samples]

            mmd_val = compute_mmd(cur, ref, self.sigma_list)
            mmd_float = mmd_val.item()

            mmd_scores[name] = mmd_float

            if self._mmd_ema[name] == 0.0:
                self._mmd_ema[name] = mmd_float
            else:
                self._mmd_ema[name] = (
                    self.ema_decay * self._mmd_ema[name]
                    + (1.0 - self.ema_decay) * mmd_float
                )

            self._trajectory[name].append((self._step_counter, mmd_float))

        self._update_forgetting_rates()

        return mmd_scores

    def _update_forgetting_rates(self):
        for name in self.module_names:
            trajectory = self._trajectory[name]
            if len(trajectory) < 2:
                self._forgetting_rates[name] = 0.0
                continue

            window = trajectory[-self.forgetting_rate_window :]
            if len(window) < 2:
                self._forgetting_rates[name] = 0.0
                continue

            values = [v for _, v in window]
            if len(values) < 2:
                self._forgetting_rates[name] = 0.0
                continue

            diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            avg_rate = sum(diffs) / len(diffs)

            self._forgetting_rates[name] = max(avg_rate, 0.0)

    def get_forgetting_rates(self) -> Dict[str, float]:
        return dict(self._forgetting_rates)

    def get_forgetting_trajectory(self, module_name: str) -> List[Tuple[int, float]]:
        return list(self._trajectory.get(module_name, []))

    def get_all_trajectories(self) -> Dict[str, List[Tuple[int, float]]]:
        return dict(self._trajectory)

    def get_fastest_forgetting_modules(self, top_k: int = 1) -> List[str]:
        rates = self.get_forgetting_rates()
        sorted_modules = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_modules[:top_k]]

    def get_mmd_ema(self) -> Dict[str, float]:
        return dict(self._mmd_ema)

    def inject_mmd_score(
        self,
        module_name: str,
        mmd_value: float,
        step: int,
    ):
        """Manually inject an MMD score (for testing/debugging the rate logic)."""
        if self._mmd_ema.get(module_name, 0.0) == 0.0:
            self._mmd_ema[module_name] = mmd_value
        else:
            self._mmd_ema[module_name] = (
                self.ema_decay * self._mmd_ema[module_name]
                + (1.0 - self.ema_decay) * mmd_value
            )
        self._trajectory[module_name].append((step, mmd_value))
        self._step_counter = max(self._step_counter, step)
        self._update_forgetting_rates()

    def reset(self):
        for name in self.module_names:
            self._mmd_ema[name] = 0.0
            self._forgetting_rates[name] = 0.0
        self._trajectory = defaultdict(list)
        self._step_counter = 0
