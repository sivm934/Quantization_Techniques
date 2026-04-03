"""
experiments.py
Replicates all experiments from Section 5 of the paper.

Run:
    python experiments.py

Results are saved as numpy .npz files in results/ for plotting.
"""

import os, time
import numpy as np
from itertools import product

from data import (make_gaussian_vectors, make_hypercube_vectors,
                  make_unit_vectors, make_synthetic_regression,
                  load_ujindoorloc)
from compressors import (NoisySign, HadamardMultiDim, SparseReg, OneBit,
                         RandK, SRQ, Drive, PermK, CorrelatedSRQ,
                         RandKSpatial, RandKSpatialProj, Kashin)
from server import (run_dme, run_power_iteration,
                    run_linear_regression, run_logistic_regression)
from metrics import cosine_distance

os.makedirs("results", exist_ok=True)

N_SEEDS = 5
RNG_SEEDS = list(range(N_SEEDS))

# Paper §D: d = 512 throughout
D = 512
# Paper §D: m = 100 for DME, m = 50 for KMeans/power iteration
M_DME = 100
M_DOWNSTREAM = 50


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def _k_for_budget(budget_bits: float, d: int) -> int:
    """Return K such that 32K + K*log2(d) ≈ budget_bits."""
    k = max(1, int(budget_bits / (32 + np.log2(d))))
    return k


COMM_BUDGET = 2375   # bits/client (paper §D: 2375 ± 25)


def _make_compressors(d: int) -> dict:
    """Build compressor instances tuned to COMM_BUDGET."""
    K = _k_for_budget(COMM_BUDGET, d)
    K = max(K, 1)
    # t for OneBit: t ≈ budget / 1 bit
    t_ob = COMM_BUDGET
    return {
        # Proposed
        "NoisySign":        NoisySign(sigma=3.0),
        "HadamardMultiDim": HadamardMultiDim(B=100.0),
        "SparseReg":        SparseReg(L=8, B=100.0),
        "OneBit":           OneBit(t=min(t_ob, d), technique='II'),
        # Baselines
        "RandK":            RandK(K=K),
        "SRQ":              SRQ(K=max(2, K // d) if K < d else 4),
        "Drive":            Drive(),
        "PermK":            PermK(K=K),
        "CorrelatedSRQ":    CorrelatedSRQ(K=4),
        "RandKSpatial":     RandKSpatial(K=K),
        "RandKSpatialProj": RandKSpatialProj(K=K),
        "Kashin":           Kashin(lam=2.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Exp 1: DME – ℓ₂ error vs Δ₂  (Gaussian vectors, Fig 2a)
# ══════════════════════════════════════════════════════════════════════════════

def exp_dme_l2():
    print("\n── Exp 1: DME ℓ₂ error vs Δ₂ ──")
    delta2_vals = np.logspace(-3, 2, 20)
    compressors_keys = ["RandK", "SRQ", "Drive", "PermK",
                        "CorrelatedSRQ", "SparseReg"]
    results = {k: np.zeros((N_SEEDS, len(delta2_vals))) for k in compressors_keys}

    for s, seed in enumerate(RNG_SEEDS):
        rng = np.random.default_rng(seed)
        comps = _make_compressors(D)
        for j, d2 in enumerate(delta2_vals):
            G, g = make_gaussian_vectors(M_DME, D, delta2=d2, g_norm=100.0, rng=rng)
            for key in compressors_keys:
                err = run_dme(comps[key], G, g, error_metric='l2', rng=rng)
                results[key][s, j] = err
        print(f"  seed {seed} done")

    np.savez("results/dme_l2.npz",
             delta2_vals=delta2_vals, **results)
    print("  Saved results/dme_l2.npz")
    return results, delta2_vals


# ══════════════════════════════════════════════════════════════════════════════
# Exp 2: DME – ℓ∞ error vs Δ∞  (Hypercube vectors, Fig 2b)
# ══════════════════════════════════════════════════════════════════════════════

def exp_dme_linf():
    print("\n── Exp 2: DME ℓ∞ error vs Δ∞ ──")
    delta_inf_vals = np.logspace(-3, 2, 20)
    compressors_keys = ["NoisySign", "HadamardMultiDim",
                        "CorrelatedSRQ", "SRQ", "Drive"]
    results = {k: np.zeros((N_SEEDS, len(delta_inf_vals))) for k in compressors_keys}

    for s, seed in enumerate(RNG_SEEDS):
        rng = np.random.default_rng(seed)
        comps = _make_compressors(D)
        for j, dinf in enumerate(delta_inf_vals):
            G, g = make_hypercube_vectors(M_DME, D, delta_inf=dinf, B=100.0, rng=rng)
            for key in compressors_keys:
                err = run_dme(comps[key], G, g, error_metric='linf', rng=rng)
                results[key][s, j] = err
        print(f"  seed {seed} done")

    np.savez("results/dme_linf.npz",
             delta_inf_vals=delta_inf_vals, **results)
    print("  Saved results/dme_linf.npz")
    return results, delta_inf_vals


# ══════════════════════════════════════════════════════════════════════════════
# Exp 3: DME – Cosine distance vs Δ_corr  (Unit vectors, Fig 2c)
# ══════════════════════════════════════════════════════════════════════════════

def exp_dme_cosine():
    print("\n── Exp 3: DME cosine distance vs Δ_corr ──")
    delta_corr_vals = np.linspace(0.01, 0.4, 20)
    compressors_keys = ["OneBit", "SparseReg", "SRQ", "RandKSpatialProj"]
    results = {k: np.zeros((N_SEEDS, len(delta_corr_vals))) for k in compressors_keys}

    for s, seed in enumerate(RNG_SEEDS):
        rng = np.random.default_rng(seed)
        comps = _make_compressors(D)
        for j, dc in enumerate(delta_corr_vals):
            G, g = make_unit_vectors(M_DME, D, delta_corr=dc, rng=rng)
            for key in compressors_keys:
                err = run_dme(comps[key], G, g, error_metric='cosine', rng=rng)
                results[key][s, j] = err
        print(f"  seed {seed} done")

    np.savez("results/dme_cosine.npz",
             delta_corr_vals=delta_corr_vals, **results)
    print("  Saved results/dme_cosine.npz")
    return results, delta_corr_vals


# ══════════════════════════════════════════════════════════════════════════════
# Exp 4: Linear Regression on Synthetic dataset  (Fig 2i)
# ══════════════════════════════════════════════════════════════════════════════

def exp_linear_regression_synthetic():
    print("\n── Exp 4: Linear Regression (Synthetic) ──")
    compressors_keys = ["HadamardMultiDim", "SparseReg", "SRQ",
                        "RandKSpatialProj", "OneBit"]
    T = 50
    results = {k: np.zeros((N_SEEDS, T)) for k in compressors_keys}

    for s, seed in enumerate(RNG_SEEDS):
        rng = np.random.default_rng(seed)
        client_X, client_y, test_X, test_y = make_synthetic_regression(
            m=M_DOWNSTREAM, d=D, n_per_client=1000, delta2=4.0, rng=rng)
        comps = _make_compressors(D)
        for key in compressors_keys:
            mse_curve = run_linear_regression(
                comps[key], client_X, client_y, test_X, test_y,
                T=T, lr=0.01, rng=rng)
            results[key][s] = mse_curve
        print(f"  seed {seed} done")

    np.savez("results/linreg_synthetic.npz", **results)
    print("  Saved results/linreg_synthetic.npz")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Exp 5: Linear Regression on UJIndoorLoc  (Fig 2f)
# ══════════════════════════════════════════════════════════════════════════════

def exp_linear_regression_ujindoorloc():
    print("\n── Exp 5: Linear Regression (UJIndoorLoc) ──")
    compressors_keys = ["NoisySign", "HadamardMultiDim", "SparseReg",
                        "SRQ", "RandKSpatialProj", "OneBit", "CorrelatedSRQ"]
    T = 50
    results = {k: np.zeros((N_SEEDS, T)) for k in compressors_keys}

    for s, seed in enumerate(RNG_SEEDS):
        rng = np.random.default_rng(seed)
        client_X, client_y, test_X, test_y = load_ujindoorloc(
            m=M_DOWNSTREAM, d=D, rng=rng)
        comps = _make_compressors(D)
        for key in compressors_keys:
            mse_curve = run_linear_regression(
                comps[key], client_X, client_y, test_X, test_y,
                T=T, lr=1e-4, rng=rng)
            results[key][s] = mse_curve
        print(f"  seed {seed} done")

    np.savez("results/linreg_ujindoorloc.npz", **results)
    print("  Saved results/linreg_ujindoorloc.npz")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Exp 6: Logistic Regression on HAR  (Fig 3)
# ══════════════════════════════════════════════════════════════════════════════

def exp_logistic_regression_har():
    print("\n── Exp 6: Logistic Regression (HAR) ──")
    from data import load_har
    compressors_keys = ["HadamardMultiDim", "OneBit", "SparseReg", "RandKSpatialProj"]
    T = 200
    train_results = {k: np.zeros((N_SEEDS, T)) for k in compressors_keys}
    test_results = {k: np.zeros((N_SEEDS, T)) for k in compressors_keys}

    for s, seed in enumerate(RNG_SEEDS):
        rng = np.random.default_rng(seed)
        client_X, client_y, test_X, test_y = load_har(m=20, d=D, rng=rng)
        comps = _make_compressors(D)
        for key in compressors_keys:
            losses, accs = run_logistic_regression(
                comps[key], client_X, client_y, test_X, test_y,
                T=T, lr=0.001, rng=rng)
            train_results[key][s] = losses
            test_results[key][s] = accs
        print(f"  seed {seed} done")

    np.savez("results/logreg_har_train.npz", **train_results)
    np.savez("results/logreg_har_test.npz", **test_results)
    print("  Saved results/logreg_har_*.npz")
    return train_results, test_results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    print("=" * 60)
    print("Collaborative Compressors – Paper Experiments")
    print("=" * 60)

    exp_dme_l2()
    exp_dme_linf()
    exp_dme_cosine()
    exp_linear_regression_synthetic()
    exp_linear_regression_ujindoorloc()
    exp_logistic_regression_har()

    elapsed = time.time() - t0
    print(f"\nAll experiments done in {elapsed/60:.1f} min.")
    print("Run  python plots.py  to generate figures.")
