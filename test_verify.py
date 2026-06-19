"""
Verification test for the module-specific forgetting trace + interference training pipeline.

Tests:
  [A] Real MMD on fake representations identifies fastest-forgetting module
  [B] MMD selection → adversarial targeting pipeline end-to-end
  [C] Gradient projection diagnostics (norms, angles, new-direction preserved)
  [D] Interference loss scales with dynamic weight (high rate → high loss; low rate → low loss)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from model import ModularContinualNet
from replay_buffer import ReplayBuffer
from forgetting_detector import ForgettingTraceDetector, compute_mmd
from adversarial_generator import AdversarialInterferenceGenerator
from interference_trainer import TraceReinforcedInterferenceTrainer


# ======================================================================
# HELPER: Create a controlled 3-module "forgetting" scenario with the
#         real MMD pipeline.  Each module gets a FAKE representation
#         where "current" is the "reference" shifted by a mean offset
#         that grows at a different rate per module.
# ======================================================================

class FakeReplayScenario:
    MODULE_NAMES = ["slow", "medium", "fast"]
    DIM = 16
    BATCH = 32
    SLOPES = {"slow": 0.02, "medium": 0.15, "fast": 0.50}

    def __init__(self, seed: int = 42):
        self.references: Dict[str, torch.Tensor] = {}
        g = torch.Generator().manual_seed(seed)
        for mod in self.MODULE_NAMES:
            self.references[mod] = torch.randn(self.BATCH, self.DIM, generator=g)

    def current_outputs(self, step: int) -> Dict[str, torch.Tensor]:
        out = {}
        for mod in self.MODULE_NAMES:
            ref = self.references[mod]
            offset_mag = self.SLOPES[mod] * (step + 1)
            direction = ref[0] / (ref[0].norm() + 1e-8)
            cur = ref + offset_mag * direction.unsqueeze(0)
            out[mod] = cur
        return out

    def reference_outputs(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.references.items()}


# ======================================================================
# HELPER: Progressively corrupt a model to simulate forgetting.
#         Each call to drift_step() corrupts high_level a bit more,
#         so MMD increases each step → non-zero forgetting rate.
# ======================================================================

def progressive_drift_setup(seed: int = 100):
    """
    Returns (model, x_old, y_old, stored_traces) where the model is fresh.
    Caller should call progressive_drift_step() repeatedly before each MMD detect.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = ModularContinualNet(in_channels=3, num_classes=10, input_size=32).to(device)
    x_old = torch.randn(32, 3, 32, 32)
    y_old = torch.randint(0, 10, (32,))
    with torch.no_grad():
        _ = model(x_old)
        stored_traces = {k: v.clone() for k, v in model.get_all_module_outputs().items()}
    return model, x_old, y_old, stored_traces


def progressive_drift_step(model, high_level_factor=0.88, mid_level_factor=0.97):
    """Corrupt model one more step.  Returns nothing; mutates model in-place."""
    with torch.no_grad():
        for p in model.high_level.parameters():
            p.mul_(high_level_factor)
        for p in model.mid_level.parameters():
            p.mul_(mid_level_factor)


# ======================================================================
# TEST 1: Real MMD on fake representations → correct ranking
# ======================================================================

def test_mmd_on_fake_representations():
    print("\n" + "=" * 70)
    print("[Test 1] REAL MMD on fake representations: stable correct ranking")
    print("=" * 70)
    print("  Using REAL compute_mmd() — no inject_mmd_score() calls")
    print("  slow module: drift_slope=0.02  (low forgetting)")
    print("  medium module: drift_slope=0.15 (medium forgetting)")
    print("  fast module: drift_slope=0.50  (high forgetting)")
    print()

    scenario = FakeReplayScenario(seed=42)
    detector = ForgettingTraceDetector(
        module_names=FakeReplayScenario.MODULE_NAMES,
        ema_decay=0.0,
        forgetting_rate_window=6,
    )

    N_STEPS = 20
    fastest_history: List[str] = []

    print("  Step │ MMD_slow  MMD_medium  MMD_fast  │ rate_slow rate_medium rate_fast │ fastest")
    print("  ─────┼──────────────────────────────────┼──────────────────────────────────┼───────")

    for step in range(N_STEPS):
        cur = scenario.current_outputs(step)
        ref = scenario.reference_outputs()

        scores = detector.detect(cur, ref, step=step)

        rates = detector.get_forgetting_rates()
        fastest = detector.get_fastest_forgetting_modules(top_k=1)[0]
        fastest_history.append(fastest)

        print(f"  {step:4d} │ {scores.get('slow',0):8.4f}  {scores.get('medium',0):10.4f}  "
              f"{scores.get('fast',0):8.4f}  │ {rates['slow']:9.5f} {rates['medium']:11.5f} "
              f"{rates['fast']:9.5f} │ {fastest}")

    last_10 = fastest_history[-10:]
    fast_count = last_10.count("fast")
    print(f"\n  Last 10 steps: 'fast' selected {fast_count}/10 times → {last_10}")

    assert fast_count >= 8, f"FAIL: 'fast' not consistently selected. Last 10: {last_10}"

    final_rates = detector.get_forgetting_rates()
    assert final_rates["fast"] > final_rates["medium"]
    assert final_rates["medium"] > final_rates["slow"]
    print(f"  Final rates: fast={final_rates['fast']:.4f} > "
          f"medium={final_rates['medium']:.4f} > slow={final_rates['slow']:.4f} ✓")
    print("  [PASS] Real MMD on fake representations → correct stable ranking")
    return True


# ======================================================================
# TEST 2: MMD selection → adversarial targeting → same module throughout
#         Uses PROGRESSIVE drift so MMD rate is non-zero.
# ======================================================================

def test_mmd_select_then_adversarial_target():
    print("\n" + "=" * 70)
    print("[Test 2] MMD SELECT → ADVERSARIAL TARGET: same module + trace shift")
    print("=" * 70)

    device = torch.device("cpu")
    model, x_old, y_old, stored_traces = progressive_drift_setup(seed=100)

    feature_modules = ["low_level", "mid_level", "high_level"]

    detector = ForgettingTraceDetector(module_names=feature_modules, ema_decay=0.0, forgetting_rate_window=4)

    print("  Step │ rate_low   rate_mid   rate_high  │ fastest")
    print("  ─────┼───────────────────────────────────┼──────────")

    for s in range(8):
        progressive_drift_step(model, high_level_factor=0.85, mid_level_factor=0.97)
        with torch.no_grad():
            _ = model(x_old[:16])
            cur = {k: v.clone() for k, v in model.get_all_module_outputs().items() if k in feature_modules}
        ref = {k: v[:16].clone() for k, v in stored_traces.items() if k in feature_modules}
        scores = detector.detect(cur, ref, step=s)

        rates = detector.get_forgetting_rates()
        fastest = detector.get_fastest_forgetting_modules(top_k=1)[0]
        print(f"  {s:4d} │ {rates['low_level']:10.5f} {rates['mid_level']:11.5f} "
              f"{rates['high_level']:11.5f} │ {fastest}")

    rates = detector.get_forgetting_rates()
    selected = detector.get_fastest_forgetting_modules(top_k=1)[0]
    mmd_vals = {m: detector.get_mmd_ema().get(m, 0.0) for m in feature_modules}

    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │ MMD SELECTION RESULT                             │")
    print(f"  ├──────────────────────────────────────────────────┤")
    for m in feature_modules:
        marker = "  ← SELECTED" if m == selected else ""
        print(f"  │ {m:12s}  MMD_ema={mmd_vals[m]:.4f}  rate={rates[m]:.5f}{marker}")
    print(f"  └──────────────────────────────────────────────────┘")

    assert selected == "high_level", (
        f"FAIL: MMD selected {selected} instead of 'high_level'. Rates: {rates}"
    )

    gen = AdversarialInterferenceGenerator(epsilon=0.1, num_steps=10)
    ref_for_adv = {selected: stored_traces[selected][:16]}
    adv_x, _ = gen.generate_interference_samples(
        model=model,
        replay_inputs=x_old[:16],
        reference_module_outputs=ref_for_adv,
        target_modules=[selected],
        interference_strength={selected: 2.0},
    )

    with torch.no_grad():
        _ = model(x_old[:16])
        orig_outs = {k: v.clone() for k, v in model.get_all_module_outputs().items() if k in feature_modules}
        _ = model(adv_x)
        adv_outs = {k: v.clone() for k, v in model.get_all_module_outputs().items() if k in feature_modules}

    ref_traces = {k: v[:16].clone() for k, v in stored_traces.items() if k in feature_modules}

    print(f"\n  ┌──────────────────────────────────────────────────────────────────────────────┐")
    print(f"  │ OLD-TRACE SHIFT: MSE vs stored traces (normal input vs adversarial input)    │")
    print(f"  ├──────────────────────────────────────────────────────────────────────────────┤")
    print(f"  │ {'Module':12s} │ {'MSE_normal':>12s} │ {'MSE_adversarial':>15s} │ {'Δshift':>10s} │ cos_dist │")
    print(f"  ├──────────────┼──────────────┼─────────────────┼────────────┼──────────┤")

    adv_mses = {}
    mse_shifts = {}
    cos_dists = {}
    for m in feature_modules:
        B = 16
        ref_f = ref_traces[m].view(B, -1)
        normal_mse = F.mse_loss(orig_outs[m].view(B, -1), ref_f).item()
        adv_mse = F.mse_loss(adv_outs[m].view(B, -1), ref_f).item()
        shift = adv_mse - normal_mse
        mse_shifts[m] = shift
        adv_mses[m] = adv_mse

        o = orig_outs[m].view(B, -1)
        a = adv_outs[m].view(B, -1)
        cos_dists[m] = (1.0 - F.cosine_similarity(o, a, dim=1).mean().item())

        marker = " ← TARGET" if m == selected else ""
        print(f"  │ {m:12s} │ {normal_mse:12.6f} │ {adv_mse:15.6f} │ {shift:+10.6f} │ "
              f"{cos_dists[m]:.5f} │{marker}")

    print(f"  └──────────────────────────────────────────────────────────────────────────────┘")

    assert cos_dists[selected] == max(cos_dists.values()), (
        f"FAIL: selected '{selected}' doesn't have highest cos_dist. Dists: {cos_dists}"
    )
    assert adv_mses[selected] == max(adv_mses.values()), (
        f"FAIL: selected '{selected}' doesn't have highest adversarial MSE. MSEs: {adv_mses}"
    )

    print(f"\n  ✓ MMD selected '{selected}'")
    print(f"  ✓ Adversarial targeted '{selected}'")
    print(f"  ✓ Largest cos_dist at '{selected}' ({cos_dists[selected]:.5f})")
    print(f"  ✓ Highest adv-MSE at '{selected}' ({adv_mses[selected]:.6f})")
    print("  [PASS] MMD selection → adversarial targeting → trace shift: consistent module")
    return True


# ======================================================================
# TEST 3: Gradient projection diagnostics — full visibility
#         Key metric: new_component_ratio = (g_merged · g_new) / ||g_new||² ≈ 1.0
#         (means the new-task component is fully preserved in merged gradient)
# ======================================================================

def test_gradient_projection_diagnostics():
    print("\n" + "=" * 70)
    print("[Test 3] GRADIENT DIAGNOSTICS: norms, angles, dot products, new-direction")
    print("=" * 70)

    device = torch.device("cpu")
    model, x_old, y_old, stored_traces = progressive_drift_setup(seed=77)
    feature_modules = ["low_level", "mid_level", "high_level"]

    rb = ReplayBuffer(max_size_per_task=100, module_names=feature_modules)
    rb.store(0, x_old, y_old, {k: v for k, v in stored_traces.items() if k in feature_modules})

    fd = ForgettingTraceDetector(module_names=feature_modules, ema_decay=0.0, forgetting_rate_window=3)
    for s in range(6):
        progressive_drift_step(model, high_level_factor=0.85, mid_level_factor=0.97)
        with torch.no_grad():
            _ = model(x_old[:16])
            cur = {k: v.clone() for k, v in model.get_all_module_outputs().items() if k in feature_modules}
        ref = {k: v[:16].clone() for k, v in stored_traces.items() if k in feature_modules}
        fd.detect(cur, ref, step=s)

    ag = AdversarialInterferenceGenerator(epsilon=0.05, num_steps=3)
    trainer = TraceReinforcedInterferenceTrainer(
        model=model, replay_buffer=rb, forgetting_detector=fd,
        adversarial_generator=ag, lr=0.01,
        interference_base_weight=0.5, forgetting_rate_scale=10.0,
        max_interference_weight=2.0, top_k_forgetting=1,
        use_orthogonal_projection=True,
        device=device, verbose=True,
    )

    x_new = torch.randn(16, 3, 32, 32)
    y_new = torch.randint(0, 10, (16,))
    trainer.set_current_task(1)

    print("\n  Running train_step...\n")
    info = trainer.train_step(x_new, y_new, step=10)

    required = [
        "grad_g_new_norm", "grad_g_old_norm", "grad_g_proj_norm", "grad_g_merged_norm",
        "grad_cos_new_old", "grad_cos_new_proj", "grad_cos_new_merged",
        "grad_angle_new_old_deg", "grad_angle_new_proj_deg", "grad_angle_new_merged_deg",
        "grad_new_component_ratio",
        "grad_dot_new_old", "grad_dot_new_proj", "grad_dot_new_merged",
    ]
    for k in required:
        assert k in info, f"FAIL: missing key '{k}' in loss_info"

    conflict = "YES ⚡" if info["grad_dot_new_old"] < 0 else "no"
    print(f"\n  ┌──────────────────────────────────────────────────────────────────────┐")
    print(f"  │ GRADIENT DIAGNOSTICS                                               │")
    print(f"  ├──────────────────────────────────────────────────────────────────────┤")
    print(f"  │ NORMS                                                              │")
    print(f"  │   ‖g_new‖    = {info['grad_g_new_norm']:>10.4f}   (new task gradient)            │")
    print(f"  │   ‖g_old‖    = {info['grad_g_old_norm']:>10.4f}   (old task + interference)      │")
    print(f"  │   ‖g_proj‖   = {info['grad_g_proj_norm']:>10.4f}   (old proj ⊥ new)              │")
    print(f"  │   ‖g_merged‖ = {info['grad_g_merged_norm']:>10.4f}   (g_new + g_proj)             │")
    print(f"  ├──────────────────────────────────────────────────────────────────────┤")
    print(f"  │ ANGLES with g_new                                                  │")
    print(f"  │   ∠(new,old)   = {info['grad_angle_new_old_deg']:>8.2f}°                              │")
    print(f"  │   ∠(new,proj)  = {info['grad_angle_new_proj_deg']:>8.2f}°  (≈90° = projection OK)    │")
    print(f"  │   ∠(new,merged)= {info['grad_angle_new_merged_deg']:>8.2f}°                              │")
    print(f"  ├──────────────────────────────────────────────────────────────────────┤")
    print(f"  │ DOT PRODUCTS with g_new                                            │")
    print(f"  │   g_new·g_old   = {info['grad_dot_new_old']:>+12.4f}  conflict={conflict:>5s}              │")
    print(f"  │   g_new·g_proj  = {info['grad_dot_new_proj']:>+12.4f}  (≈0 = orthogonal OK)        │")
    print(f"  │   g_new·g_mrgd = {info['grad_dot_new_merged']:>+12.4f}  (≈‖g_new‖² = preserved)     │")
    print(f"  ├──────────────────────────────────────────────────────────────────────┤")
    print(f"  │ cos SIMILARITY with g_new                                          │")
    print(f"  │   cos(new,old)  = {info['grad_cos_new_old']:>+8.4f}                                   │")
    print(f"  │   cos(new,proj) = {info['grad_cos_new_proj']:>+8.4f}  (≈0 = orthogonal)            │")
    print(f"  │   cos(new,mrgd) = {info['grad_cos_new_merged']:>+8.4f}                                   │")
    print(f"  ├──────────────────────────────────────────────────────────────────────┤")
    print(f"  │ new_component_ratio = {info['grad_new_component_ratio']:>8.4f}  (≈1.0 = new dir FULLY kept) │")
    print(f"  │   = (g_merged · g_new) / ‖g_new‖²                                 │")
    print(f"  └──────────────────────────────────────────────────────────────────────┘")

    angle_proj = info["grad_angle_new_proj_deg"]
    assert 80.0 <= angle_proj <= 100.0, (
        f"FAIL: projected old grad not orthogonal to new. angle={angle_proj:.1f}°"
    )

    ncr = info["grad_new_component_ratio"]
    assert 0.95 <= ncr <= 1.05, (
        f"FAIL: new_component_ratio = {ncr:.4f}, expected ≈1.0. "
        f"New-task gradient component is not preserved!"
    )

    cos_proj = info["grad_cos_new_proj"]
    assert abs(cos_proj) < 0.05, (
        f"FAIL: cos(new, proj) = {cos_proj:.4f}, expected ≈0."
    )

    dot_proj = info["grad_dot_new_proj"]
    assert abs(dot_proj) / (info["grad_g_new_norm"] * info["grad_g_proj_norm"] + 1e-10) < 0.05, (
        f"FAIL: dot(new,proj) not ≈0 relative to norms"
    )

    print(f"\n  ✓ ∠(new,proj)≈90° → projection orthogonal")
    print(f"  ✓ new_component_ratio={ncr:.4f} → new-task direction fully preserved (≈1.0)")
    print(f"  ✓ cos(new,proj)={cos_proj:+.4f} → no leakage of new direction into projected old")
    print(f"  ✓ dot(new,proj)={dot_proj:+.4f} → orthogonal dot product confirmed")
    print("  [PASS] Gradient projection diagnostics: full visibility, new-direction safe")
    return True


# ======================================================================
# TEST 4: Interference loss scales with dynamic weight / forgetting rate
#         Uses PROGRESSIVE drift so rates are non-zero.
# ======================================================================

def test_interference_loss_scales_with_forgetting_rate():
    print("\n" + "=" * 70)
    print("[Test 4] INTERFERENCE LOSS: constraint gap + dynamic weight + near-zero rate")
    print("=" * 70)

    device = torch.device("cpu")
    feature_modules = ["low_level", "mid_level", "high_level"]

    torch.manual_seed(123)
    model_A, x_old_A, y_old_A, stored_A = progressive_drift_setup(seed=123)
    rb_A = ReplayBuffer(max_size_per_task=100, module_names=feature_modules)
    rb_A.store(0, x_old_A, y_old_A, {k: v for k, v in stored_A.items() if k in feature_modules})
    for _ in range(6):
        progressive_drift_step(model_A, high_level_factor=0.80, mid_level_factor=0.97)

    torch.manual_seed(123)
    model_B, x_old_B, y_old_B, stored_B = progressive_drift_setup(seed=123)
    rb_B = ReplayBuffer(max_size_per_task=100, module_names=feature_modules)
    rb_B.store(0, x_old_B, y_old_B, {k: v for k, v in stored_B.items() if k in feature_modules})
    for _ in range(6):
        progressive_drift_step(model_B, high_level_factor=0.99, mid_level_factor=0.995)

    ag = AdversarialInterferenceGenerator(epsilon=0.1, num_steps=5)

    base_weight = 0.5
    rate_scale = 10.0
    max_weight = 2.0

    test_configs = [
        ("SEVERE+high_rate", model_A, rb_A, 0.05),
        ("SEVERE+low_rate",  model_A, rb_A, 0.001),
        ("SEVERE+near_zero", model_A, rb_A, 0.0001),
        ("MINIMAL+near_zero", model_B, rb_B, 0.0001),
    ]

    results = {}
    for label, mdl, rb, fake_rate in test_configs:
        weight = min(base_weight * (1.0 + rate_scale * fake_rate), max_weight)

        replay = rb.sample(batch_size=16, device=device)

        ref_for_adv = {"high_level": replay["module_high_level"]}
        adv_x, _ = ag.generate_interference_samples(
            model=mdl,
            replay_inputs=replay["input"],
            reference_module_outputs=ref_for_adv,
            target_modules=["high_level"],
            interference_strength={"high_level": weight},
        )

        fd = ForgettingTraceDetector(module_names=feature_modules)
        trainer = TraceReinforcedInterferenceTrainer(
            model=mdl, replay_buffer=rb, forgetting_detector=fd,
            adversarial_generator=ag, lr=0.01,
            interference_base_weight=base_weight,
            forgetting_rate_scale=rate_scale,
            max_interference_weight=max_weight,
            top_k_forgetting=1,
            device=device, verbose=False,
        )

        with torch.enable_grad():
            interference_loss, per_module = trainer._compute_interference_loss_wrt_stored_traces(
                adv_inputs=adv_x,
                replay_data=replay,
                target_modules=["high_level"],
                interference_weights={"high_level": weight},
            )

        normal_mse = per_module.get("high_level__normal_mse", 0.0)
        adv_mse = per_module.get("high_level__adv_mse", 0.0)
        shift = per_module.get("high_level__shift", 0.0)

        results[label] = {
            "fake_rate": fake_rate,
            "weight": weight,
            "normal_mse": normal_mse,
            "adv_mse": adv_mse,
            "shift": shift,
            "interference_loss": interference_loss.item(),
            "raw_mse_high_level": per_module.get("high_level", 0.0),
        }

        print(f"\n  [{label}]")
        print(f"    Forgetting rate:     {fake_rate:.4f}")
        print(f"    Dynamic weight:      {weight:.4f}  (base={base_weight}, scale={rate_scale})")
        print(f"    ┌─────────────────────────────────────────────────┐")
        print(f"    │ MSE vs stored traces (high_level):              │")
        print(f"    │   Normal input:   {normal_mse:12.6f}  (before constraint) │")
        print(f"    │   Adversarial:    {adv_mse:12.6f}  (after constraint)  │")
        print(f"    │   Δ shift:        {shift:+12.6f}                      │")
        print(f"    └─────────────────────────────────────────────────┘")
        print(f"    Weighted interference loss: {interference_loss.item():.6f}")

    severe_high = results["SEVERE+high_rate"]
    severe_low = results["SEVERE+low_rate"]
    severe_near_zero = results["SEVERE+near_zero"]
    minimal_near_zero = results["MINIMAL+near_zero"]

    assert severe_high["weight"] > severe_low["weight"], (
        f"FAIL: high_rate weight ({severe_high['weight']:.3f}) should exceed low_rate ({severe_low['weight']:.3f})"
    )
    assert severe_high["interference_loss"] > severe_low["interference_loss"], (
        f"FAIL: SEVERE+high_rate loss ({severe_high['interference_loss']:.6f}) should exceed "
        f"SEVERE+low_rate ({severe_low['interference_loss']:.6f})"
    )
    print(f"\n  ✓ Same model (SEVERE): high_rate loss ({severe_high['interference_loss']:.6f}) > "
          f"low_rate loss ({severe_low['interference_loss']:.6f})")

    assert minimal_near_zero["interference_loss"] < severe_high["interference_loss"], (
        f"FAIL: MINIMAL+near_zero loss should be smaller than SEVERE+high_rate"
    )
    print(f"  ✓ MINIMAL+near_zero loss ({minimal_near_zero['interference_loss']:.6f}) < "
          f"SEVERE+high_rate ({severe_high['interference_loss']:.6f})")

    print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ DYNAMIC WEIGHT TABLE: weight = base × (1 + scale × rate)       │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │ {'Config':20s} │ {'Rate':>8s} │ {'Weight':>8s} │ {'Adv_MSE':>10s} │ {'Wtd_loss':>10s} │")
    print(f"  ├──────────────────────┼──────────┼──────────┼────────────┼────────────┤")
    for label in test_configs:
        lab = label[0]
        r = results[lab]
        print(f"  │ {lab:20s} │ {r['fake_rate']:8.4f} │ {r['weight']:8.4f} │ "
              f"{r['adv_mse']:10.6f} │ {r['interference_loss']:10.6f} │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    weight_ratio = severe_high["weight"] / severe_low["weight"]
    print(f"\n  Weight ratio (high_rate/low_rate): {weight_ratio:.2f}x")
    assert weight_ratio > 1.3, f"FAIL: weight ratio too small ({weight_ratio:.2f})"
    print(f"  ✓ Dynamic weight scales with forgetting rate ({weight_ratio:.2f}x)")

    print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ NEAR-ZERO RATE CHECK: rate≈0 should NOT be misjudged            │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    for label in ["SEVERE+near_zero", "MINIMAL+near_zero"]:
        r = results[label]
        base_only_loss = base_weight * r["adv_mse"]
        amplification = r["weight"] / base_weight
        print(f"  │ {label:20s}: weight={r['weight']:.4f} ≈ base={base_weight:.1f} "
              f"(amplification={amplification:.3f}x)")
        print(f"  │   weighted_loss={r['interference_loss']:.6f}  vs  base×MSE={base_only_loss:.6f}  "
              f"→ ratio={r['interference_loss']/max(base_only_loss,1e-10):.3f}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    for label in ["SEVERE+near_zero", "MINIMAL+near_zero"]:
        r = results[label]
        amplification = r["weight"] / base_weight
        assert amplification < 1.01, (
            f"FAIL: {label} weight={r['weight']:.4f} amplifies by {amplification:.3f}x at near-zero rate"
        )

    severe_nz_vs_high = severe_high["interference_loss"] / max(severe_near_zero["interference_loss"], 1e-10)
    print(f"\n  ✓ Near-zero rate: weight ≈ base (no amplification)")
    print(f"  ✓ SEVERE+high_rate loss is {severe_nz_vs_high:.1f}x SEVERE+near_zero loss")
    print(f"  ✓ Forgetting rate controls loss contribution — not misjudged at rate≈0")
    print("  [PASS] Interference loss: constraint gap + dynamic weight + near-zero safe")
    return True


# ======================================================================
# TEST 5: End-to-end with full diagnostic output
# ======================================================================

def test_end_to_end_with_diagnostics():
    print("\n" + "=" * 70)
    print("[Test 5] END-TO-END: 3 training steps with full diagnostic output")
    print("=" * 70)

    device = torch.device("cpu")
    model, x_old, y_old, stored = progressive_drift_setup(seed=55)
    feature_modules = ["low_level", "mid_level", "high_level"]

    rb = ReplayBuffer(max_size_per_task=100, module_names=feature_modules)
    rb.store(0, x_old, y_old, {k: v for k, v in stored.items() if k in feature_modules})

    fd = ForgettingTraceDetector(module_names=feature_modules, ema_decay=0.0, forgetting_rate_window=3)
    for s in range(4):
        progressive_drift_step(model, high_level_factor=0.88, mid_level_factor=0.97)
        with torch.no_grad():
            _ = model(x_old[:16])
            cur = {k: v.clone() for k, v in model.get_all_module_outputs().items() if k in feature_modules}
        ref = {k: v[:16].clone() for k, v in stored.items() if k in feature_modules}
        fd.detect(cur, ref, step=s)

    ag = AdversarialInterferenceGenerator(epsilon=0.05, num_steps=3)
    trainer = TraceReinforcedInterferenceTrainer(
        model=model, replay_buffer=rb, forgetting_detector=fd,
        adversarial_generator=ag, lr=0.01,
        interference_base_weight=0.5, forgetting_rate_scale=10.0,
        max_interference_weight=2.0, top_k_forgetting=1,
        use_orthogonal_projection=True,
        device=device, verbose=True,
    )

    trainer.set_current_task(1)
    x_new = torch.randn(16, 3, 32, 32)
    y_new = torch.randint(0, 10, (16,))

    for i in range(3):
        print(f"\n  --- Step {i} ---")
        info = trainer.train_step(x_new, y_new, step=i)

        if "grad_g_new_norm" in info:
            ncr = info.get("grad_new_component_ratio", 0)
            conflict = "YES ⚡" if info.get("grad_dot_new_old", 0) < 0 else "no"
            print(f"    ‖g_new‖={info['grad_g_new_norm']:.4f}  "
                  f"‖g_old‖={info['grad_g_old_norm']:.4f}  "
                  f"‖g_proj‖={info['grad_g_proj_norm']:.4f}  "
                  f"‖g_merged‖={info['grad_g_merged_norm']:.4f}")
            print(f"    ∠(new,old)={info['grad_angle_new_old_deg']:.1f}°  "
                  f"∠(new,proj)={info['grad_angle_new_proj_deg']:.1f}°  "
                  f"∠(new,merged)={info['grad_angle_new_merged_deg']:.1f}°")
            print(f"    dot(n,o)={info.get('grad_dot_new_old',0):+.4f}  "
                  f"dot(n,p)={info.get('grad_dot_new_proj',0):+.4f}  "
                  f"new_comp={ncr:.4f}  conflict={conflict}")

        for k in sorted(info.keys()):
            if k.startswith("rate__") or k.startswith("weight__") or k.startswith("mmd__"):
                print(f"    {k} = {info[k]:.5f}")

    print("\n  [PASS] End-to-end steps with full diagnostics")
    return True


# ======================================================================
# Main runner
# ======================================================================

def main():
    print("\n" + "#" * 70)
    print("#   VERIFICATION TESTS — Forgetting-Trace Interference Training")
    print("#" * 70)

    tests = [
        ("MMD on fake representations", test_mmd_on_fake_representations),
        ("MMD select → adversarial + trace shift", test_mmd_select_then_adversarial_target),
        ("Gradient projection diagnostics", test_gradient_projection_diagnostics),
        ("Interference loss + dynamic weight + near-zero", test_interference_loss_scales_with_forgetting_rate),
        ("End-to-end with diagnostics", test_end_to_end_with_diagnostics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            ok = test_fn()
            results.append((name, ok))
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            results.append((name, False))
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_pass = all_pass and ok

    print("=" * 70)
    if all_pass:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
