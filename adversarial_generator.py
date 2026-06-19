import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class AdversarialInterferenceGenerator:
    def __init__(
        self,
        epsilon: float = 0.05,
        num_steps: int = 5,
        step_size: Optional[float] = None,
        module_names: Optional[List[str]] = None,
        interference_weight: float = 1.0,
        clamp_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size or (epsilon / num_steps * 2.0)
        self.module_names = module_names or []
        self.interference_weight = interference_weight
        self.clamp_range = clamp_range

    def generate_interference_samples(
        self,
        model: nn.Module,
        replay_inputs: torch.Tensor,
        reference_module_outputs: Dict[str, torch.Tensor],
        target_modules: List[str],
        interference_strength: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = replay_inputs.device
        delta = torch.zeros_like(replay_inputs, device=device)
        delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)

        for step in range(self.num_steps):
            adv_inputs = replay_inputs + delta
            adv_inputs = adv_inputs.clamp(*self.clamp_range)

            model.zero_grad()

            _ = model(adv_inputs)
            current_outputs = model.get_all_module_outputs()

            interference_loss = torch.tensor(0.0, device=device)
            for mod_name in target_modules:
                if mod_name not in current_outputs or mod_name not in reference_module_outputs:
                    continue

                cur_out = current_outputs[mod_name]
                ref_out = reference_module_outputs[mod_name]

                min_batch = min(cur_out.size(0), ref_out.size(0))
                cur_out = cur_out[:min_batch]
                ref_out = ref_out[:min_batch]

                cur_flat = cur_out.reshape(min_batch, -1)
                ref_flat = ref_out.reshape(min_batch, -1)

                cos_sim = F.cosine_similarity(cur_flat, ref_flat, dim=1)
                mod_interference = -cos_sim.mean()

                strength = 1.0
                if interference_strength is not None and mod_name in interference_strength:
                    strength = interference_strength[mod_name]

                interference_loss = interference_loss + strength * mod_interference

            if delta.grad is not None:
                delta.grad = None

            delta_grad = torch.autograd.grad(
                outputs=interference_loss,
                inputs=delta,
                retain_graph=False,
                create_graph=False,
            )[0]

            if delta_grad is not None:
                grad_sign = delta_grad.sign()
                delta_data = delta.detach() + self.step_size * grad_sign
                delta_data = torch.clamp(delta_data, -self.epsilon, self.epsilon)
                delta_data = torch.clamp(
                    replay_inputs + delta_data, *self.clamp_range
                ) - replay_inputs
                delta = delta_data.clone().detach().requires_grad_(True)

        final_adv = (replay_inputs + delta).clamp(*self.clamp_range)

        model.zero_grad()
        with torch.no_grad():
            _ = model(final_adv)
            adv_module_outputs = model.get_all_module_outputs()
            detached_outputs = {k: v.clone() for k, v in adv_module_outputs.items() if v is not None}

        return final_adv.detach(), detached_outputs

    def generate_layer_interference(
        self,
        model: nn.Module,
        replay_inputs: torch.Tensor,
        reference_module_output: torch.Tensor,
        module_name: str,
        interference_strength: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = replay_inputs.device
        delta = torch.zeros_like(replay_inputs, device=device)
        delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)

        ref_flat = reference_module_output.view(reference_module_output.size(0), -1)

        for step in range(self.num_steps):
            adv_inputs = replay_inputs + delta
            adv_inputs = adv_inputs.clamp(*self.clamp_range)
            adv_inputs.requires_grad_(True)

            model.zero_grad()

            module_out = model.forward_to_module(adv_inputs, module_name)

            cur_flat = module_out.view(module_out.size(0), -1)
            min_batch = min(cur_flat.size(0), ref_flat.size(0))
            cur_flat = cur_flat[:min_batch]
            ref_batch = ref_flat[:min_batch]

            cos_sim = F.cosine_similarity(cur_flat, ref_batch, dim=1)
            interference_loss = -interference_strength * cos_sim.mean()

            if delta.grad is not None:
                delta.grad.zero_()
            interference_loss.backward()

            if delta.grad is not None:
                grad_sign = delta.grad.sign()
                delta_data = delta.detach() + self.step_size * grad_sign
                delta_data = torch.clamp(delta_data, -self.epsilon, self.epsilon)
                delta_data = torch.clamp(
                    replay_inputs + delta_data, *self.clamp_range
                ) - replay_inputs
                delta = delta_data.detach().requires_grad_(True)

        final_adv = (replay_inputs + delta).clamp(*self.clamp_range)

        model.zero_grad()
        with torch.no_grad():
            module_out = model.forward_to_module(final_adv, module_name)
            adv_output = module_out.clone()

        return final_adv.detach(), adv_output.detach()
