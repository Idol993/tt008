import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from forgetting_detector import ForgettingTraceDetector
from adversarial_generator import AdversarialInterferenceGenerator
from replay_buffer import ReplayBuffer
from model import ModularContinualNet


class TraceReinforcedInterferenceTrainer:
    def __init__(
        self,
        model: ModularContinualNet,
        replay_buffer: ReplayBuffer,
        forgetting_detector: ForgettingTraceDetector,
        adversarial_generator: AdversarialInterferenceGenerator,
        lr: float = 0.01,
        replay_weight: float = 1.0,
        interference_base_weight: float = 0.5,
        forgetting_rate_scale: float = 10.0,
        max_interference_weight: float = 2.0,
        top_k_forgetting: int = 1,
        use_orthogonal_projection: bool = True,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.forgetting_detector = forgetting_detector
        self.adversarial_generator = adversarial_generator
        self.replay_weight = replay_weight
        self.interference_base_weight = interference_base_weight
        self.forgetting_rate_scale = forgetting_rate_scale
        self.max_interference_weight = max_interference_weight
        self.top_k_forgetting = top_k_forgetting
        self.use_orthogonal_projection = use_orthogonal_projection
        self.device = device
        self.verbose = verbose

        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        self._current_task_id: int = 0
        self._step_count: int = 0
        self._loss_history: List[Dict[str, float]] = []

        self._last_grad_info: Dict[str, float] = {}

    def set_current_task(self, task_id: int):
        self._current_task_id = task_id

    def _compute_dynamic_interference_weight(
        self, forgetting_rate: float
    ) -> float:
        scaled = self.interference_base_weight * (
            1.0 + self.forgetting_rate_scale * forgetting_rate
        )
        return min(scaled, self.max_interference_weight)

    def _sample_and_get_reference_outputs(
        self, batch_size: int
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        replay_data = self.replay_buffer.sample(
            batch_size=min(batch_size, self.replay_buffer.get_size()),
            device=self.device,
        )
        if replay_data is None:
            return None, {}

        reference_outputs: Dict[str, torch.Tensor] = {}
        for mod_name in self.model.get_module_names():
            key = f"module_{mod_name}"
            if key in replay_data:
                reference_outputs[mod_name] = replay_data[key]

        return replay_data, reference_outputs

    @torch.no_grad()
    def _detect_forgetting_this_step(
        self,
        replay_data: Dict[str, torch.Tensor],
        reference_outputs: Dict[str, torch.Tensor],
        step: int,
    ) -> Tuple[Dict[str, float], Dict[str, float], List[str], Dict[str, float]]:
        _ = self.model(replay_data["input"])
        current_module_outputs = {
            k: v.clone() for k, v in self.model.get_all_module_outputs().items()
            if v is not None
        }

        available_ref = {k: v for k, v in reference_outputs.items() if k in current_module_outputs}
        if len(available_ref) == 0:
            return {}, {}, [], {}

        mmd_scores = self.forgetting_detector.detect(
            current_outputs=current_module_outputs,
            reference_outputs=available_ref,
            step=step,
        )

        forgetting_rates = self.forgetting_detector.get_forgetting_rates()
        fastest_modules = self.forgetting_detector.get_fastest_forgetting_modules(
            top_k=self.top_k_forgetting
        )

        interference_weights: Dict[str, float] = {}
        for mod_name in fastest_modules:
            rate = forgetting_rates.get(mod_name, 0.0)
            interference_weights[mod_name] = self._compute_dynamic_interference_weight(rate)

        return mmd_scores, forgetting_rates, fastest_modules, interference_weights

    def _compute_replay_loss(
        self,
        replay_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        inputs = replay_data["input"]
        labels = replay_data["label"]
        outputs = self.model(inputs)
        replay_loss = F.cross_entropy(outputs, labels)
        return replay_loss

    def _compute_interference_loss_wrt_stored_traces(
        self,
        adv_inputs: torch.Tensor,
        replay_data: Dict[str, torch.Tensor],
        target_modules: List[str],
        interference_weights: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            _ = self.model(replay_data["input"])
            normal_outputs = self.model.get_all_module_outputs()

        _ = self.model(adv_inputs)
        adv_outputs = self.model.get_all_module_outputs()

        interference_loss = torch.tensor(0.0, device=self.device)
        per_module_loss: Dict[str, float] = {}

        for mod_name in target_modules:
            if mod_name not in adv_outputs:
                continue
            stored_key = f"module_{mod_name}"
            if stored_key not in replay_data:
                continue

            cur_out = adv_outputs[mod_name]
            ref_out = replay_data[stored_key]

            min_batch = min(cur_out.size(0), ref_out.size(0))
            if min_batch < 1:
                continue

            cur_flat = cur_out[:min_batch].reshape(min_batch, -1)
            ref_flat = ref_out[:min_batch].reshape(min_batch, -1).detach()

            adv_mse = F.mse_loss(cur_flat, ref_flat).item()

            normal_mse = 0.0
            if mod_name in normal_outputs:
                nor = normal_outputs[mod_name][:min_batch].reshape(min_batch, -1)
                normal_mse = F.mse_loss(nor, ref_flat).item()

            weight = interference_weights.get(mod_name, 1.0)
            weighted = weight * F.mse_loss(cur_flat, ref_flat)
            interference_loss = interference_loss + weighted
            per_module_loss[mod_name] = adv_mse
            per_module_loss[f"{mod_name}__normal_mse"] = normal_mse
            per_module_loss[f"{mod_name}__adv_mse"] = adv_mse
            per_module_loss[f"{mod_name}__shift"] = adv_mse - normal_mse
            per_module_loss[f"{mod_name}__weight"] = weight

        return interference_loss, per_module_loss

    def _extract_grads(self) -> Dict[str, torch.Tensor]:
        grads: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
        return grads

    def _apply_grads(self, grads: Dict[str, torch.Tensor]):
        for name, param in self.model.named_parameters():
            if name in grads:
                param.grad = grads[name].clone()

    def _flatten_grads(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for name in sorted(grads.keys()):
            parts.append(grads[name].flatten())
        return torch.cat(parts)

    def _compute_grad_diagnostics(
        self,
        new_grads: Dict[str, torch.Tensor],
        old_grads: Dict[str, torch.Tensor],
        proj_old_grads: Dict[str, torch.Tensor],
        merged_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        g_new = self._flatten_grads(new_grads)
        g_old = self._flatten_grads(old_grads)
        g_proj = self._flatten_grads(proj_old_grads)
        g_merged = self._flatten_grads(merged_grads)

        new_norm = g_new.norm().item()
        old_norm = g_old.norm().item()
        proj_norm = g_proj.norm().item()
        merged_norm = g_merged.norm().item()

        eps = 1e-12
        cos_new_old = (torch.dot(g_new, g_old) / (new_norm * old_norm + eps)).item()
        cos_new_proj = (torch.dot(g_new, g_proj) / (new_norm * proj_norm + eps)).item()
        cos_new_merged = (torch.dot(g_new, g_merged) / (new_norm * merged_norm + eps)).item()

        angle_new_old = float(np.arccos(np.clip(cos_new_old, -1.0, 1.0))) * 180.0 / np.pi
        angle_new_proj = float(np.arccos(np.clip(cos_new_proj, -1.0, 1.0))) * 180.0 / np.pi
        angle_new_merged = float(np.arccos(np.clip(cos_new_merged, -1.0, 1.0))) * 180.0 / np.pi

        new_norm_sq = torch.dot(g_new, g_new).item()
        old_norm_sq = torch.dot(g_old, g_old).item()
        new_component_in_merged = (torch.dot(g_merged, g_new) / (new_norm_sq + eps)).item()

        dot_new_old = torch.dot(g_new, g_old).item()
        dot_new_proj = torch.dot(g_new, g_proj).item()
        dot_new_merged = torch.dot(g_merged, g_new).item()

        return {
            "g_new_norm": new_norm,
            "g_old_norm": old_norm,
            "g_proj_norm": proj_norm,
            "g_merged_norm": merged_norm,
            "cos_new_old": cos_new_old,
            "cos_new_proj": cos_new_proj,
            "cos_new_merged": cos_new_merged,
            "angle_new_old_deg": angle_new_old,
            "angle_new_proj_deg": angle_new_proj,
            "angle_new_merged_deg": angle_new_merged,
            "new_component_ratio": new_component_in_merged,
            "dot_new_old": dot_new_old,
            "dot_new_proj": dot_new_proj,
            "dot_new_merged": dot_new_merged,
        }

    def _orthogonal_project_old_onto_new(
        self,
        new_grads: Dict[str, torch.Tensor],
        old_grads: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        projected_old: Dict[str, torch.Tensor] = {}
        info: Dict[str, float] = {}

        total_old_norm_sq = 0.0
        total_proj_norm_sq = 0.0
        total_dot_positive = 0.0
        total_params = 0

        for name in new_grads:
            if name not in old_grads:
                continue

            g_new = new_grads[name].clone().flatten()
            g_old = old_grads[name].clone().flatten()

            dot = torch.dot(g_old, g_new)
            new_norm_sq = torch.dot(g_new, g_new)
            old_norm_sq = torch.dot(g_old, g_old)

            if new_norm_sq > 1e-12:
                proj_coef = dot / new_norm_sq
                g_old_proj = g_old - proj_coef * g_new
            else:
                g_old_proj = g_old.clone()

            old_norm = torch.sqrt(old_norm_sq + 1e-16)
            proj_norm = g_old_proj.norm() + 1e-16
            if old_norm > 1e-10 and proj_norm > 1e-10:
                g_old_proj = g_old_proj * (old_norm / proj_norm)

            projected_old[name] = g_old_proj.view_as(new_grads[name])

            total_old_norm_sq += old_norm_sq.item()
            total_proj_norm_sq += g_old_proj.dot(g_old_proj).item()
            total_dot_positive += max(dot.item(), 0.0)
            total_params += 1

        info["avg_old_norm"] = (total_old_norm_sq / max(total_params, 1)) ** 0.5
        info["avg_proj_norm"] = (total_proj_norm_sq / max(total_params, 1)) ** 0.5
        info["total_dot_positive"] = total_dot_positive
        info["num_params"] = total_params

        return projected_old, info

    def train_step(
        self,
        new_inputs: torch.Tensor,
        new_labels: torch.Tensor,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        self._step_count += 1
        current_step = step if step is not None else self._step_count

        new_inputs = new_inputs.to(self.device)
        new_labels = new_labels.to(self.device)

        # =========================================================
        # PHASE 1: Compute new-task gradient (g_new) — ALONE
        # =========================================================
        self.model.zero_grad()
        new_outputs = self.model(new_inputs)
        new_task_loss = F.cross_entropy(new_outputs, new_labels)
        new_task_loss.backward()

        new_task_grads: Dict[str, torch.Tensor] = self._extract_grads()
        new_grad_norm = 0.0
        for g in new_task_grads.values():
            new_grad_norm += g.flatten().dot(g.flatten()).item()
        new_grad_norm = new_grad_norm ** 0.5

        # =========================================================
        # PHASE 2: MMD forgetting detection — THIS STEP's replay samples
        # =========================================================
        has_old_tasks = len(self.replay_buffer.get_stored_tasks()) > 0

        replay_loss_val = 0.0
        interference_loss_val = 0.0
        mmd_scores: Dict[str, float] = {}
        forgetting_rates: Dict[str, float] = {}
        fastest_modules: List[str] = []
        interference_weights: Dict[str, float] = {}
        per_module_interference: Dict[str, float] = {}
        grad_proj_info: Dict[str, float] = {}
        grad_diag: Dict[str, float] = {}

        old_task_grads: Dict[str, torch.Tensor] = {}
        merged_grads: Dict[str, torch.Tensor] = {}

        if has_old_tasks:
            # (2a) Sample replay + collect STORED module traces as reference
            replay_data, stored_reference_outputs = self._sample_and_get_reference_outputs(batch_size=32)

            if replay_data is not None and len(stored_reference_outputs) > 0:
                # (2b) MMD detection on *this* step's replay data — before any training
                mmd_scores, forgetting_rates, fastest_modules, interference_weights = (
                    self._detect_forgetting_this_step(
                        replay_data=replay_data,
                        reference_outputs=stored_reference_outputs,
                        step=current_step,
                    )
                )

                # Log: which modules were selected this step and their rates/weights
                if self.verbose:
                    log_str = f"[Step {current_step}] MMD → "
                    for mod in fastest_modules:
                        log_str += (
                            f"{mod}[rate={forgetting_rates.get(mod,0):.5f}, "
                            f"weight={interference_weights.get(mod,0):.3f}, "
                            f"MMD={mmd_scores.get(mod,0):.4f}]  "
                        )
                    print(log_str)

                # =========================================================
                # PHASE 3: Generate adversarial samples targeting fastest_modules
                #          Reference = STORED old traces, NOT current model outputs
                # =========================================================
                adv_inputs = None
                if len(fastest_modules) > 0:
                    ref_for_adv: Dict[str, torch.Tensor] = {}
                    for mod_name in fastest_modules:
                        stored_key = f"module_{mod_name}"
                        if stored_key in replay_data:
                            ref_for_adv[mod_name] = replay_data[stored_key]

                    if len(ref_for_adv) > 0:
                        adv_inputs, _ = (
                            self.adversarial_generator.generate_interference_samples(
                                model=self.model,
                                replay_inputs=replay_data["input"],
                                reference_module_outputs=ref_for_adv,
                                target_modules=fastest_modules,
                                interference_strength=interference_weights,
                            )
                        )

                # =========================================================
                # PHASE 4: Compute old-task gradient (g_old) — ALONE
                #          (replay classification + interference w.r.t. STORED traces)
                # =========================================================
                self.model.zero_grad()

                replay_loss = self._compute_replay_loss(replay_data)
                total_old_loss = self.replay_weight * replay_loss
                replay_loss_val = replay_loss.item()

                if adv_inputs is not None and len(fastest_modules) > 0:
                    interference_loss, per_module_interference = (
                        self._compute_interference_loss_wrt_stored_traces(
                            adv_inputs=adv_inputs,
                            replay_data=replay_data,
                            target_modules=fastest_modules,
                            interference_weights=interference_weights,
                        )
                    )
                    interference_loss_val = interference_loss.item()
                    total_old_loss = total_old_loss + interference_loss

                total_old_loss.backward()
                old_task_grads = self._extract_grads()

                # =========================================================
                # PHASE 5: Gradient orthogonal projection + merge
                # =========================================================
                if self.use_orthogonal_projection and len(new_task_grads) > 0 and len(old_task_grads) > 0:
                    projected_old_grads, grad_proj_info = self._orthogonal_project_old_onto_new(
                        new_task_grads, old_task_grads
                    )

                    self.model.zero_grad()
                    merged_grads: Dict[str, torch.Tensor] = {}
                    for name, param in self.model.named_parameters():
                        combined = torch.zeros_like(param)
                        if name in new_task_grads:
                            combined = combined + new_task_grads[name]
                        if name in projected_old_grads:
                            combined = combined + projected_old_grads[name]
                        param.grad = combined
                        merged_grads[name] = combined.clone()

                    grad_diag = self._compute_grad_diagnostics(
                        new_task_grads, old_task_grads, projected_old_grads, merged_grads
                    )
                else:
                    self.model.zero_grad()
                    merged_grads = {}
                    proj_for_diag: Dict[str, torch.Tensor] = {}
                    for name, param in self.model.named_parameters():
                        combined = torch.zeros_like(param)
                        if name in new_task_grads:
                            combined = combined + new_task_grads[name]
                        if name in old_task_grads:
                            combined = combined + old_task_grads[name]
                        param.grad = combined
                        merged_grads[name] = combined.clone()
                        if name in old_task_grads:
                            proj_for_diag[name] = old_task_grads[name].clone()

                    grad_diag = self._compute_grad_diagnostics(
                        new_task_grads, old_task_grads, proj_for_diag, merged_grads
                    )

                if self.verbose and len(old_task_grads) > 0:
                    a_old = grad_diag.get("angle_new_old_deg", 0)
                    a_proj = grad_diag.get("angle_new_proj_deg", 0)
                    a_mrg = grad_diag.get("angle_new_merged_deg", 0)
                    ncr = grad_diag.get("new_component_ratio", 0)
                    d_old = grad_diag.get("dot_new_old", 0)
                    d_proj = grad_diag.get("dot_new_proj", 0)
                    conflict = "⚡CONFLICT" if d_old < 0 else ""
                    print(
                        f"  Grad: ‖new‖={grad_diag['g_new_norm']:.3f} "
                        f"‖old‖={grad_diag['g_old_norm']:.3f} "
                        f"‖proj‖={grad_diag['g_proj_norm']:.3f} "
                        f"‖merged‖={grad_diag['g_merged_norm']:.3f} | "
                        f"∠(new,old)={a_old:.1f}° "
                        f"∠(new,proj)={a_proj:.1f}° "
                        f"∠(new,merged)={a_mrg:.1f}° | "
                        f"dot(n,o)={d_old:+.2f} dot(n,p)={d_proj:+.2f} "
                        f"new_comp={ncr:.3f} {conflict}"
                    )

        # =========================================================
        # PHASE 6: Optimizer step
        # =========================================================
        self.optimizer.step()
        self.model.zero_grad()

        # =========================================================
        # Build detailed loss info dict
        # =========================================================
        loss_info: Dict[str, float] = {
            "new_task_loss": new_task_loss.item(),
            "replay_loss": replay_loss_val,
            "interference_loss": interference_loss_val,
            "step": current_step,
            "num_fastest_modules": len(fastest_modules),
        }

        for mod in fastest_modules:
            loss_info[f"mmd__{mod}"] = mmd_scores.get(mod, 0.0)
            loss_info[f"rate__{mod}"] = forgetting_rates.get(mod, 0.0)
            loss_info[f"weight__{mod}"] = interference_weights.get(mod, 0.0)
            if mod in per_module_interference:
                loss_info[f"interf_loss__{mod}"] = per_module_interference[mod]

        for k, v in grad_proj_info.items():
            loss_info[f"proj_{k}"] = v

        for k, v in grad_diag.items():
            loss_info[f"grad_{k}"] = v

        self._loss_history.append(loss_info)
        self._last_grad_info = grad_proj_info

        return loss_info

    def update_replay_buffer(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task_id: Optional[int] = None,
    ):
        tid = task_id if task_id is not None else self._current_task_id
        with torch.no_grad():
            _ = self.model(inputs)
            module_outputs = self.model.get_all_module_outputs()
            detached = {k: v.clone() for k, v in module_outputs.items() if v is not None}
        self.replay_buffer.store(tid, inputs, labels, detached)

    def get_loss_history(self) -> List[Dict[str, float]]:
        return self._loss_history

    def get_last_grad_info(self) -> Dict[str, float]:
        return self._last_grad_info

    def state_dict(self) -> Dict:
        return {
            "current_task_id": self._current_task_id,
            "step_count": self._step_count,
            "loss_history": self._loss_history,
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self._current_task_id = state["current_task_id"]
        self._step_count = state["step_count"]
        self._loss_history = state["loss_history"]
        self.optimizer.load_state_dict(state["optimizer_state"])
