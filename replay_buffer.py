import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random


class ReplayBuffer:
    def __init__(
        self,
        max_size_per_task: int = 500,
        module_names: Optional[List[str]] = None,
    ):
        self.max_size_per_task = max_size_per_task
        self.module_names = module_names or []
        self._data: Dict[int, List[Dict[str, torch.Tensor]]] = defaultdict(list)
        self._task_labels: List[int] = []

    def store(
        self,
        task_id: int,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        module_outputs: Dict[str, torch.Tensor],
    ):
        batch_size = inputs.size(0)
        for i in range(batch_size):
            entry = {
                "input": inputs[i].cpu(),
                "label": labels[i].cpu(),
            }
            for name in self.module_names:
                if name in module_outputs:
                    entry[f"module_{name}"] = module_outputs[name][i].cpu()
            self._data[task_id].append(entry)

        if len(self._data[task_id]) > self.max_size_per_task:
            self._data[task_id] = self._data[task_id][-self.max_size_per_task:]

        if task_id not in self._task_labels:
            self._task_labels.append(task_id)

    def sample(
        self,
        task_id: Optional[int] = None,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Dict[str, torch.Tensor]]:
        if task_id is not None:
            pool = self._data.get(task_id, [])
        else:
            pool = []
            for tid in self._task_labels:
                pool.extend(self._data[tid])

        if len(pool) == 0:
            return None

        batch_size = min(batch_size, len(pool))
        samples = random.sample(pool, batch_size)

        inputs = torch.stack([s["input"] for s in samples]).to(device)
        labels = torch.stack([s["label"] for s in samples]).to(device)

        result = {"input": inputs, "label": labels}
        for name in self.module_names:
            key = f"module_{name}"
            if key in samples[0]:
                result[f"module_{name}"] = torch.stack([s[key] for s in samples]).to(device)

        return result

    def sample_module_outputs(
        self,
        module_name: str,
        task_id: Optional[int] = None,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[torch.Tensor]:
        data = self.sample(task_id, batch_size, device)
        if data is None:
            return None
        key = f"module_{module_name}"
        return data.get(key, None)

    def get_stored_tasks(self) -> List[int]:
        return list(self._task_labels)

    def get_size(self, task_id: Optional[int] = None) -> int:
        if task_id is not None:
            return len(self._data.get(task_id, []))
        return sum(len(v) for v in self._data.values())

    def merge(self, other: "ReplayBuffer"):
        for task_id, entries in other._data.items():
            for entry in entries:
                self._data[task_id].append(entry)
            if len(self._data[task_id]) > self.max_size_per_task:
                self._data[task_id] = self._data[task_id][-self.max_size_per_task:]
            if task_id not in self._task_labels:
                self._task_labels.append(task_id)

    def clear_task(self, task_id: int):
        if task_id in self._data:
            del self._data[task_id]
        if task_id in self._task_labels:
            self._task_labels.remove(task_id)

    def get_reference_outputs(
        self,
        module_name: str,
        task_id: int,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[torch.Tensor]:
        pool = self._data.get(task_id, [])
        if len(pool) == 0:
            return None
        key = f"module_{module_name}"
        if key not in pool[0]:
            return None
        outputs = torch.stack([s[key] for s in pool]).to(device)
        return outputs
