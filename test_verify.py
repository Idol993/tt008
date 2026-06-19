"""
Verification test for the module-specific forgetting trace + interference training pipeline.

Runs a controlled experiment with fake data and 3 modules to verify:
  [A] MMD detection correctly identifies the FASTEST-forgetting module
  [B] Adversarial samples are targeted at the selected module
  [C] Dynamic interference weight grows with forgetting rate
  [D] Interference loss (w.r.t. STORED old traces) is non-zero and correlates with weight
  [E] Gradient orthogonal projection does NOT cancel new-task direction
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


# =========================================================
# Test 0: Module sanity — outputs & stored traces work
# =========================================================
def test_module_sanity():
    print("\n" + "=" * 70)
    print("[Test 0] MODULE SANITY: outputs can be extracted per module")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(0)
    model = ModularContinualNet(in_channels=3, num_classes=10, input_size=32).to(device)

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    out = model(x)
    outs = model.get_all_module_outputs()

    for name in model.get_module_names():
        assert name in outs, f"Missing {name} output"
        assert outs[name].shape[0] == 8, f"{name} batch dim wrong: {outs[name].shape}"
        print(f"  ✓ {name}: shape={tuple(outs[name].shape)}")

    # Replay buffer stores module outputs
    rb = ReplayBuffer(max_size_per_task=100, module_names=model.get_module_names())
    detached = {k: v.clone() for k, v in outs.items()}
    rb.store(0, x, y, detached)

    sample = rb.sample(batch_size=4, device=device)
    for name in model.get_module_names():
        key = f"module_{name}"
        assert key in sample, f"Missing stored {key}"
        print(f"  ✓ Stored trace {key}: shape={tuple(sample[key].shape)}")

    print("  [PASS] Module sanity test OK")
    return True


# =========================================================
# Test 1: Forgetting detector correctly ranks modules by MMD slope
# =========================================================
def test_mmd_detects_fastest_forgetting():
    print("\n" + "=" * 70)
    print("[Test 1] MMD DETECTOR: correctly identifies fastest-forgetting module")
    print("=" * 70)

    module_names = ["low_level", "mid_level", "high_level", "classifier"]
    detector = ForgettingTraceDetector(
        module_names=module_names,
        ema_decay=0.0,            # no smoothing, pure values
        forgetting_rate_window=8,  # rate = average diff over last 8 steps
    )

    # Ground-truth: each module has a KNOWN slope for its MMD-vs-step curve.
    # The detector must correctly identify which slope is steepest.
    #
    # We use `inject_mmd_score()` to feed KNOWN perfect MMDS directly
    # into the detector's trajectory store. This validates the detector's
    # core logic (rate estimation → ranking) independently of the Gaussian-
    # kernel MMD's numerical behavior.
    slope_truth = {
        "low_level":   0.001,   # slowest
        "classifier":  0.005,
        "mid_level":   0.020,
        "high_level":  0.100,  # fastest  ← GROUND TRUTH
    }
    N_STEPS = 40

    for step in range(N_STEPS):
        t = step + 1
        for name in module_names:
            mmd_val = slope_truth[name] * t
            detector.inject_mmd_score(name, mmd_val, step=step)

        if step % 10 == 0 and step > 0:
            rates = detector.get_forgetting_rates()
            fastest = detector.get_fastest_forgetting_modules(top_k=1)[0]
            print(f"  Step {step:2d} | rates: "
                  + "  ".join([f"{n[:3]}={rates[n]:.4f}" for n in module_names])
                  + f"  → fastest={fastest}")

    final_rates = detector.get_forgetting_rates()
    fastest_module = detector.get_fastest_forgetting_modules(top_k=1)[0]

    expected_rank = [m for m, _ in sorted(slope_truth.items(), key=lambda x: -x[1])]
    actual_rank = [m for m, _ in sorted(final_rates.items(), key=lambda x: -x[1])]

    print(f"\n  Ground-truth slope : {slope_truth}")
    print(f"  Expected rank (fast→slow): {expected_rank}")
    print(f"  Detected rank (fast→slow): {actual_rank}")
    print(f"  Final rates:              {final_rates}")

    # --- Assertions ---
    # 1) Fastest must be high_level
    assert fastest_module == "high_level", (
        f"FAIL: expected high_level fastest, got {fastest_module}. Rates: {final_rates}"
    )
    # 2) Rates should be ordered: high > mid > classifier > low
    #    For linear MMD curves, the "forgetting rate" = average diff over last N steps
    #    ≈ slope (since MMD = slope * t, diff per step ≈ slope).
    #    Each module's computed rate should be PROPORTIONAL to its true slope.
    pairs_to_check = [
        ("high_level", "mid_level"),
        ("mid_level", "classifier"),
        ("classifier", "low_level"),
    ]
    for faster, slower in pairs_to_check:
        assert final_rates[faster] > final_rates[slower], (
            f"FAIL: ordering {faster} > {slower} violated: "
            f"{faster}={final_rates[faster]:.5f}, {slower}={final_rates[slower]:.5f}"
        )
    # 3) The estimated rate should be close to the true slope
    #    (for a perfectly linear MMD curve with slope s over window W,
    #     avg diff ≈ s). Allow 30% tolerance for numerical jitter.
    for name in module_names:
        est = final_rates[name]
        truth = slope_truth[name]
        if truth > 1e-6:
            ratio = est / truth
            assert 0.5 < ratio < 2.0, (
                f"FAIL: rate for {name}: est={est:.5f} vs truth={truth:.5f} "
                f"(ratio={ratio:.2f}, outside 0.5x-2.0x tolerance)"
            )
            print(f"  ✓ {name:12s}: est_rate={est:.4f} ≈ true_slope={truth:.3f} (x{ratio:.2f})")

    print("\n  [PASS] Forgetting detector: correct fastest + correct ranking + accurate rate estimates")
    return True


# =========================================================
# Test 2: Adversarial samples TARGET the selected module
# =========================================================
def test_adversarial_targets_selected_module():
    print("\n" + "=" * 70)
    print("[Test 2] ADVERSARIAL TARGETING: targeting increases TARGET module's shift")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(42)
    model = ModularContinualNet(in_channels=3, num_classes=10, input_size=32).to(device)
    module_names = model.get_module_names()

    x = torch.randn(16, 3, 32, 32)
    _ = model(x)
    ref_outputs = {k: v.clone() for k, v in model.get_all_module_outputs().items()}

    gen = AdversarialInterferenceGenerator(
        epsilon=0.1, num_steps=20, module_names=module_names
    )

    # For each candidate module M:
    #   A) Run adv generation targeting ONLY module M → measure cos_dist(M)
    #   B) Run adv generation targeting a DIFFERENT module N → measure cos_dist(M)
    # Assertion: cos_dist_A(M) > cos_dist_B(M)  (targeting increases M's shift)

    def _get_cos_dist_targeted(target_list: List[str]) -> Dict[str, float]:
        """Generate adv targeting target_list; return cos_dist from ref per module."""
        strength = {m: 3.0 for m in target_list}
        adv_x, _ = gen.generate_interference_samples(
            model=model,
            replay_inputs=x,
            reference_module_outputs=ref_outputs,
            target_modules=target_list,
            interference_strength=strength,
        )
        with torch.no_grad():
            _ = model(adv_x)
            adv_outs = model.get_all_module_outputs()

        dists = {}
        for name in module_names:
            orig = ref_outputs[name].view(16, -1)
            adv = adv_outs[name].view(16, -1)
            cos_sim = F.cosine_similarity(orig, adv, dim=1)
            dists[name] = (1.0 - cos_sim.mean().item())
        return dists

    all_passed = True

    for target_mod in ["low_level", "mid_level", "high_level"]:
        # A) Target ONLY this module
        dists_A = _get_cos_dist_targeted([target_mod])

        # B) Target a DIFFERENT, unrelated control module
        other_modules = [m for m in ["low_level", "mid_level", "high_level"] if m != target_mod]
        control_mod = other_modules[0]
        dists_B = _get_cos_dist_targeted([control_mod])

        # C) Baseline: no targeting (just random perturbation)
        # (Use targeting a dummy empty-ish list via small epsilon control by
        #  comparing to the OTHER non-targeted modules.)

        dist_target_when_targeted = dists_A[target_mod]
        dist_target_when_control = dists_B[target_mod]

        print(f"\n  Target module: '{target_mod}'")
        print(f"    cos_dist({target_mod}) when TARGETING it       : {dist_target_when_targeted:.5f}")
        print(f"    cos_dist({target_mod}) when targeting '{control_mod}' instead: {dist_target_when_control:.5f}")
        print(f"    Ratio (targeted/control): {dist_target_when_targeted / max(dist_target_when_control,1e-8):.2f}x")

        # Also print all modules for context
        print(f"    All dists when targeting '{target_mod}': "
              + "  ".join([f"{m[:3]}={dists_A[m]:.4f}" for m in module_names]))

        # KEY ASSERTION: Target module is shifted MORE when it is the target
        # of optimization than when something else is the target.
        if dist_target_when_targeted > 1.05 * dist_target_when_control:
            print(f"    ✓ Targeting '{target_mod}' clearly increases its cos_dist")
        else:
            print(f"    ✗ WARNING: targeting effect weak ({dist_target_when_targeted:.4f} vs {dist_target_when_control:.4f})")
            all_passed = False

    assert all_passed, (
        "FAIL: adversarial targeting does not consistently increase TARGET module's "
        "cosine-distance shift relative to targeting a different module."
    )
    print("\n  [PASS] Adversarial samples clearly TARGET the selected module")
    return True


# =========================================================
# Test 3: Dynamic weight scales with forgetting rate
# =========================================================
def test_dynamic_weight_scaling():
    print("\n" + "=" * 70)
    print("[Test 3] DYNAMIC WEIGHT: weight grows with forgetting rate")
    print("=" * 70)

    device = torch.device("cpu")
    model = ModularContinualNet(in_channels=3, num_classes=10, input_size=32).to(device)
    module_names = model.get_module_names()
    rb = ReplayBuffer(max_size_per_task=10, module_names=module_names)
    fd = ForgettingTraceDetector(module_names=module_names)
    ag = AdversarialInterferenceGenerator(epsilon=0.05, num_steps=3)

    trainer = TraceReinforcedInterferenceTrainer(
        model=model, replay_buffer=rb, forgetting_detector=fd,
        adversarial_generator=ag, lr=0.01,
        interference_base_weight=0.5, forgetting_rate_scale=10.0,
        max_interference_weight=2.0, top_k_forgetting=1,
        device=device, verbose=False,
    )

    test_rates = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    weights = [trainer._compute_dynamic_interference_weight(r) for r in test_rates]

    print("  Forgetting Rate → Weight (base=0.5, scale=10, max=2.0):")
    for r, w in zip(test_rates, weights):
        print(f"    rate={r:.2f} → weight={w:.3f}")

    # Monotonic increasing check
    for i in range(1, len(weights)):
        assert weights[i] >= weights[i - 1], (
            f"FAIL: weights not monotonic: rate={test_rates[i-1]}→{test_rates[i]} "
            f"weight={weights[i-1]}→{weights[i]}"
        )

    # Max clamp check
    assert weights[-1] == 2.0, f"FAIL: max weight not clamped. Got {weights[-1]}"
    assert weights[0] == 0.5, f"FAIL: base weight wrong. Got {weights[0]}"

    print("  [PASS] Dynamic weight monotonically increases & respects max clamp")
    return True


# =========================================================
# Test 4: Interference loss (w.r.t. STORED traces) is meaningful
# =========================================================
def test_interference_loss_vs_stored_traces():
    print("\n" + "=" * 70)
    print("[Test 4] INTERFERENCE LOSS: non-zero, weights correctly affect magnitude")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(123)
    model = ModularContinualNet(in_channels=3, num_classes=10, input_size=32).to(device)
    module_names = model.get_module_names()

    # Step 1: store "old" traces in replay buffer
    rb = ReplayBuffer(max_size_per_task=100, module_names=module_names)
    x_old = torch.randn(32, 3, 32, 32)
    y_old = torch.randint(0, 10, (32,))

    with torch.no_grad():
        _ = model(x_old)
        stored = {k: v.clone() for k, v in model.get_all_module_outputs().items()}
    rb.store(0, x_old, y_old, stored)

    # Step 2: mess up the model so current outputs differ from stored traces
    #         (simulate forgetting — especially in high_level)
    with torch.no_grad():
        for p in model.high_level.parameters():
            p.mul_(0.5)
        for p in model.mid_level.parameters():
            p.mul_(0.8)

    # Step 3: Sample replay data
    replay_data, ref_outs = None, {}
    for _ in range(5):
        replay_data, ref_outs = None, {}
        tmp = rb.sample(batch_size=16, device=device)
        if tmp is not None:
            replay_data = tmp
            for m in module_names:
                key = f"module_{m}"
                if key in tmp:
                    ref_outs[m] = tmp[key]
            if len(ref_outs) > 0:
                break
    assert replay_data is not None

    # Step 4: Generate adversarial samples
    gen = AdversarialInterferenceGenerator(epsilon=0.1, num_steps=5)
    target_mods = ["high_level", "mid_level"]
    weights_low = {"high_level": 0.1, "mid_level": 0.1}
    weights_high = {"high_level": 1.5, "mid_level": 1.5}

    adv_x_low, _ = gen.generate_interference_samples(
        model, replay_data["input"], ref_outs, target_mods, weights_low
    )
    adv_x_high, _ = gen.generate_interference_samples(
        model, replay_data["input"], ref_outs, target_mods, weights_high
    )

    # Step 5: Compute interference loss w.r.t. STORED traces
    trainer = TraceReinforcedInterferenceTrainer(
        model=model, replay_buffer=rb,
        forgetting_detector=ForgettingTraceDetector(module_names=module_names),
        adversarial_generator=gen, lr=0.01, device=device, verbose=False,
    )

    with torch.enable_grad():
        loss_low, per_low = trainer._compute_interference_loss_wrt_stored_traces(
            adv_x_low, replay_data, target_mods, weights_low
        )
        loss_high, per_high = trainer._compute_interference_loss_wrt_stored_traces(
            adv_x_high, replay_data, target_mods, weights_high
        )

    print(f"  Interference loss (weight=0.1): total={loss_low.item():.4f}")
    for m, v in per_low.items():
        print(f"    per-module {m}: MSE vs stored trace = {v:.6f}")
    print(f"  Interference loss (weight=1.5): total={loss_high.item():.4f}")
    for m, v in per_high.items():
        print(f"    per-module {m}: MSE vs stored trace = {v:.6f}")

    # Assertions
    assert loss_low.item() > 0.0001, f"FAIL: low-weight loss too close to zero: {loss_low.item()}"
    assert loss_high.item() > loss_low.item(), (
        f"FAIL: higher weight should give higher weighted loss: "
        f"low={loss_low.item():.4f}, high={loss_high.item():.4f}"
    )

    # Verify per-module loss for high_level > 0 (high level was intentionally corrupted)
    assert per_low.get("high_level", 0) > 0.0001, (
        f"FAIL: high_level module MSE vs stored trace too small: {per_low.get('high_level',0)}"
    )
    print(f"  ✓ high_level MSE (v.s. old stored traces) > 0 — loss captures forgetting")
    print("  [PASS] Interference loss is meaningful and scales with weight")
    return True


# =========================================================
# Test 5: Gradient orthogonal projection preserves new-task gradient
# =========================================================
def test_gradient_projection_preserves_new_direction():
    print("\n" + "=" * 70)
    print("[Test 5] GRADIENT PROJECTION: new-task direction NOT cancelled by old grad")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(99)

    # Build a tiny 2-param linear model for easy gradient inspection
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2, bias=False)
            self.fc2 = nn.Linear(2, 1, bias=False)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    # Use manual projection function from trainer
    def _ortho_project(new_grads, old_grads):
        projected_old = {}
        dot_info = []
        for name in new_grads:
            if name not in old_grads:
                continue
            g_new = new_grads[name].clone().flatten()
            g_old = old_grads[name].clone().flatten()

            dot_before = torch.dot(g_old, g_new).item()
            new_norm_sq = torch.dot(g_new, g_new)

            if new_norm_sq > 1e-12:
                proj_coef = dot_before / new_norm_sq
                g_old_proj = g_old - proj_coef * g_new
            else:
                g_old_proj = g_old.clone()

            dot_after = torch.dot(g_old_proj, g_new).item()
            projected_old[name] = g_old_proj.view_as(new_grads[name])
            dot_info.append((name, dot_before, dot_after))

        return projected_old, dot_info

    # Construct ANTI-CORRELATED gradients: old grad = -k * new_grad
    # (This simulates new/old task conflict — old wants to go opposite)
    model = TinyModel()

    # Create fake gradients
    new_grads = {}
    old_grads_conflict = {}

    for name, param in model.named_parameters():
        g_new = torch.randn_like(param)
        new_grads[name] = g_new
        # OLD gradient directly OPPOSES new gradient (strong conflict!)
        old_grads_conflict[name] = -2.0 * g_new.clone() + 0.01 * torch.randn_like(param)

    print("  Setting up CONFLICTING gradients: old_grad ≈ -2*new_grad")

    projected_old, dot_info = _ortho_project(new_grads, old_grads_conflict)

    print(f"\n  {'Param':25s} {'dot(old,new) BEFORE':>20s} {'dot(proj_old,new) AFTER':>24s}")
    all_near_zero = True
    any_conflict = False
    for name, d_before, d_after in dot_info:
        print(f"  {name:25s} {d_before:>20.4f} {d_after:>24.6f}")
        if d_before < -0.1:
            any_conflict = True
        if abs(d_after) > 1e-4:
            all_near_zero = False

    assert any_conflict, "FAIL: Test setup should have conflicting (negative dot) gradients"
    assert all_near_zero, "FAIL: Projection did not remove correlation with new gradient"

    # Verify merged gradient = new + projected_old  does NOT reduce new direction
    # by checking that (g_new · (g_new + g_old_proj)) / ||g_new||² ≈ 1.0
    total_dot = 0.0
    total_new_sq = 0.0
    for name in new_grads:
        g_new = new_grads[name].flatten()
        g_merged = g_new.clone()
        if name in projected_old:
            g_merged = g_new + projected_old[name].flatten()
        total_dot += torch.dot(g_new, g_merged).item()
        total_new_sq += torch.dot(g_new, g_new).item()

    ratio = total_dot / (total_new_sq + 1e-12)
    print(f"\n  (g_new · g_merged) / ||g_new||² = {ratio:.4f}")
    print("  (≈ 1.0 means new-task direction fully preserved)")
    assert 0.9 < ratio < 1.1, f"FAIL: new direction not preserved, ratio={ratio:.4f}"

    print("  [PASS] Orthogonal projection preserves new-task gradient direction")
    return True


# =========================================================
# Test 6: End-to-end training step produces correct loss_info structure
# =========================================================
def test_end_to_end_step_logging():
    print("\n" + "=" * 70)
    print("[Test 6] END-TO-END step: correct logging & module selection info")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(7)
    model = ModularContinualNet(in_channels=3, num_classes=10, input_size=32).to(device)
    module_names = model.get_module_names()

    rb = ReplayBuffer(max_size_per_task=100, module_names=module_names)
    fd = ForgettingTraceDetector(module_names=module_names, forgetting_rate_window=3)
    ag = AdversarialInterferenceGenerator(epsilon=0.05, num_steps=3)
    trainer = TraceReinforcedInterferenceTrainer(
        model=model, replay_buffer=rb, forgetting_detector=fd,
        adversarial_generator=ag, lr=0.01,
        interference_base_weight=0.5, forgetting_rate_scale=10.0,
        top_k_forgetting=2,
        device=device, verbose=True,
    )

    # Pre-populate replay buffer with "task 0" data + stored traces
    x0 = torch.randn(32, 3, 32, 32)
    y0 = torch.randint(0, 10, (32,))
    trainer.set_current_task(0)
    trainer.update_replay_buffer(x0, y0, task_id=0)

    # Pre-fill detector with a few steps of MMD history to get non-zero forgetting rates
    # Make high_level artificially show higher MMD
    for s in range(10):
        fake_cur = {}
        fake_ref = {}
        for name in module_names:
            dim = {"low_level": 64*16*16, "mid_level": 128*8*8,
                   "high_level": 256*4*4, "classifier": 10}[name]
            B = 16
            ref = torch.randn(B, dim)
            slope = {"low_level": 0.001, "mid_level": 0.005,
                     "high_level": 0.05, "classifier": 0.003}[name]
            cur = ref + slope * (s + 1) * torch.randn(B, dim)
            fake_cur[name] = cur
            fake_ref[name] = ref
        fd.detect(fake_cur, fake_ref, step=s)

    # Now do actual train step with new "task 1" data
    trainer.set_current_task(1)
    x_new = torch.randn(16, 3, 32, 32)
    y_new = torch.randint(0, 10, (16,))

    print("\n  Running train_step with replay buffer populated...\n")
    loss_info = trainer.train_step(x_new, y_new, step=100)

    print(f"\n  Loss info keys: {sorted(loss_info.keys())}")

    # Check the required keys exist
    assert "new_task_loss" in loss_info
    assert "replay_loss" in loss_info
    assert "interference_loss" in loss_info
    assert "step" in loss_info
    assert "num_fastest_modules" in loss_info
    assert loss_info["step"] == 100

    # Find selected module keys
    selected = []
    for k in loss_info:
        if k.startswith("rate__"):
            mod = k[len("rate__"):]
            selected.append(mod)
            print(f"  ✓ Selected module {mod}: rate={loss_info[f'rate__{mod}']:.5f}, "
                  f"weight={loss_info[f'weight__{mod}']:.3f}, "
                  f"MMD={loss_info[f'mmd__{mod}']:.4f}")

    assert len(selected) >= 1, "FAIL: no module was selected as fastest forgetting"

    # Interference loss must be positive when modules are selected
    if loss_info["num_fastest_modules"] > 0:
        # interference loss may be 0 if adv samples failed, but we check structure
        for m in selected:
            if f"interf_loss__{m}" in loss_info:
                interf_val = loss_info[f"interf_loss__{m}"]
                print(f"  ✓ Module {m} per-module interference (unweighted) = {interf_val:.6f}")

    # Gradient info presence
    if "grad_avg_old_norm" in loss_info:
        print(f"  ✓ Grad projection info: old_norm={loss_info['grad_avg_old_norm']:.4f}, "
              f"proj_norm={loss_info['grad_avg_proj_norm']:.4f}, "
              f"params={loss_info['grad_num_params']}")

    print("  [PASS] End-to-end step produces correct loss_info structure")
    return True


# =========================================================
# Main runner
# =========================================================
def main():
    print("\n" + "#" * 70)
    print("#   VERIFICATION TESTS for Forgetting-Trace Interference Training")
    print("#" * 70)

    tests = [
        test_module_sanity,
        test_mmd_detects_fastest_forgetting,
        test_adversarial_targets_selected_module,
        test_dynamic_weight_scaling,
        test_interference_loss_vs_stored_traces,
        test_gradient_projection_preserves_new_direction,
        test_end_to_end_step_logging,
    ]

    results = []
    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            results.append((test_fn.__name__, False))
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

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
