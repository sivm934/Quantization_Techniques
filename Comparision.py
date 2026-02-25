"""
USQ K=2  vs  USQ K=16  vs  HadamardMultiDim on Gaussian Vectors
================================================================
USQ K=2  : 1 bit/coord  (d bits/client)   — independent
USQ K=16 : 4 bits/coord (4d bits/client)  — independent
Hadamard : 1 bit/coord  (d bits/client)   — collaborative
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE GAUSSIAN VECTORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_gaussian_vectors(m, d, mean_norm=10.0, delta=1.0, seed=0):
    rng    = np.random.default_rng(seed)
    g_true = rng.normal(0, mean_norm / np.sqrt(d), d)
    noise  = rng.normal(0, delta / np.sqrt(d), (m, d))
    return g_true + noise, g_true


# ─────────────────────────────────────────────────────────────────────────────
# 2. USQ  (K levels)
# ─────────────────────────────────────────────────────────────────────────────

def usq_encode(g_i, B, K):
    """Quantize to K uniform levels in [-B, B]."""
    g_c  = np.clip(g_i, -B, B)
    step = 2 * B / (K - 1)
    idx  = np.round((g_c + B) / step).astype(int)
    return -B + np.clip(idx, 0, K - 1) * step


def usq_dme(vectors, B, K):
    """Independent: each client quantizes, server averages."""
    m, d   = vectors.shape
    g_true = vectors.mean(axis=0)
    g_hat  = np.mean([usq_encode(v, B, K) for v in vectors], axis=0)
    mse    = np.mean((g_hat - g_true) ** 2)
    bits   = int(np.ceil(np.log2(K)) * d)
    return g_hat, mse, bits


# ─────────────────────────────────────────────────────────────────────────────
# 3. HADAMARD MULTI-DIM  (Algorithm 3 — collaborative)
# ─────────────────────────────────────────────────────────────────────────────

def _binary_search_bit(scalar, B, level):
    lo, hi = -B, B
    bit = 1.0
    s   = float(np.clip(scalar, -B, B))
    for _ in range(level):
        mid = (lo + hi) / 2.0
        if s < mid:
            bit = -1.0; hi = mid
        else:
            bit =  1.0; lo = mid
    return bit


def hadamard_dme(vectors, B, seed=None):
    """
    Client i encodes at binary-search level rho[i] (1 bit/coord).
    Decode: g_hat^(j) = sum_i  bit_i^(j) * B / 2^rho[i]
    """
    m, d   = vectors.shape
    g_true = vectors.mean(axis=0)
    rng    = np.random.default_rng(seed)
    rho    = rng.permutation(m) + 1

    bits_matrix = np.array([
        [_binary_search_bit(vectors[i, j], B, rho[i]) for j in range(d)]
        for i in range(m)
    ])

    weights = B / (2.0 ** rho.astype(float))
    g_hat   = (bits_matrix * weights[:, None]).sum(axis=0)
    mse     = np.mean((g_hat - g_true) ** 2)
    bits    = int(d)
    return g_hat, mse, bits


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

M, D, MEAN_NORM = 20, 32, 10.0
N_SEEDS = 30

def avg_mse(fn, vecs_list, B, extra={}):
    return np.mean([fn(v, B, **extra)[1] for v in vecs_list])

# ── A: vary dissimilarity Δ ───────────────────────────────────────────────────
deltas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
mse_usq2_d, mse_usq16_d, mse_had_d = [], [], []

for delta in deltas:
    B      = MEAN_NORM + 4 * delta
    vecs_s = [generate_gaussian_vectors(M, D, MEAN_NORM, delta, s)[0] for s in range(N_SEEDS)]
    mse_usq2_d.append(avg_mse(usq_dme,     vecs_s, B, {"K": 2}))
    mse_usq16_d.append(avg_mse(usq_dme,    vecs_s, B, {"K": 16}))
    mse_had_d.append(avg_mse(hadamard_dme, vecs_s, B, {"seed": 99}))

# ── B: vary number of clients m ───────────────────────────────────────────────
m_list    = [4, 8, 12, 16, 20, 25, 30, 40, 50]
DELTA_LOW = 0.02
mse_usq2_m, mse_usq16_m, mse_had_m = [], [], []

for m in m_list:
    B      = MEAN_NORM + 4 * DELTA_LOW
    vecs_s = [generate_gaussian_vectors(m, D, MEAN_NORM, DELTA_LOW, s)[0] for s in range(N_SEEDS)]
    mse_usq2_m.append(avg_mse(usq_dme,     vecs_s, B, {"K": 2}))
    mse_usq16_m.append(avg_mse(usq_dme,    vecs_s, B, {"K": 16}))
    mse_had_m.append(avg_mse(hadamard_dme, vecs_s, B, {"seed": 99}))

# ── C: single run for reconstruction plot ─────────────────────────────────────
vecs_vis, g_true_vis = generate_gaussian_vectors(M, D, MEAN_NORM, delta=0.01, seed=7)
B_vis = MEAN_NORM + 4 * 0.01
g_usq2_vis,  mse_u2,  b2  = usq_dme(vecs_vis, B_vis, K=2)
g_usq16_vis, mse_u16, b16 = usq_dme(vecs_vis, B_vis, K=16)
g_had_vis,   mse_h,   bh  = hadamard_dme(vecs_vis, B_vis, seed=7)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRINT MSE TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 72)
print("  MSE Comparison: USQ K=2 vs USQ K=16 vs Hadamard")
print(f"  m={M}, d={D}, mean_norm={MEAN_NORM}, averaged over {N_SEEDS} seeds")
print("=" * 72)
print(f"\n  Bits/client:  USQ K=2={b2}  |  USQ K=16={b16}  |  Hadamard={bh}")
print(f"\n{'Δ':>8}  {'USQ K=2':>12}  {'USQ K=16':>12}  {'Hadamard':>12}  {'Winner':>10}")
print("-" * 62)
for i, d in enumerate(deltas):
    vals   = [mse_usq2_d[i], mse_usq16_d[i], mse_had_d[i]]
    names  = ["USQ K=2", "USQ K=16", "Hadamard"]
    winner = names[int(np.argmin(vals))]
    print(f"{d:>8.3f}  {mse_usq2_d[i]:>12.5f}  {mse_usq16_d[i]:>12.5f}"
          f"  {mse_had_d[i]:>12.5f}  {winner:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

C2   = "#E67E22"   # orange  — USQ K=2
C16  = "#E74C3C"   # red     — USQ K=16
CHAD = "#2980B9"   # blue    — Hadamard
CTRUE= "#27AE60"   # green   — true mean
BG   = "#F8F9FA"

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "USQ K=2  vs  USQ K=16  vs  HadamardMultiDim  —  Gaussian DME\n"
    f"m={M}, d={D}, MSE averaged over {N_SEEDS} seeds",
    fontsize=12, fontweight="bold", y=1.02
)

def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=9, framealpha=0.9)

# ── Plot 1: MSE vs Dissimilarity Δ ────────────────────────────────────────────
ax = axes[0]
ax.loglog(deltas, mse_usq2_d,  "o-",  color=C2,   lw=2, ms=7, label=f"USQ K=2  ({b2} bits)")
ax.loglog(deltas, mse_usq16_d, "^-",  color=C16,  lw=2, ms=7, label=f"USQ K=16 ({b16} bits)")
ax.loglog(deltas, mse_had_d,   "s--", color=CHAD, lw=2, ms=7, label=f"Hadamard ({bh} bits)")
style(ax, "Dissimilarity  Δ", "MSE", "MSE vs Dissimilarity")

# Annotate crossover between Hadamard and USQ K=16
cross16 = next((i for i in range(len(deltas)) if mse_had_d[i] > mse_usq16_d[i]), None)
if cross16:
    ax.axvline(deltas[cross16], color=C16, lw=1.2, ls=":", alpha=0.7)
    ax.text(deltas[cross16]*1.2, max(mse_usq2_d)*0.05,
            f"Had > USQ16\nΔ≈{deltas[cross16]}", fontsize=7.5, color=C16)

# ── Plot 2: MSE vs Clients m ──────────────────────────────────────────────────
ax = axes[1]
ax.semilogy(m_list, mse_usq2_m,  "o-",  color=C2,   lw=2, ms=7, label=f"USQ K=2  ({b2} bits)")
ax.semilogy(m_list, mse_usq16_m, "^-",  color=C16,  lw=2, ms=7, label=f"USQ K=16 ({b16} bits)")
ax.semilogy(m_list, mse_had_m,   "s--", color=CHAD, lw=2, ms=7, label=f"Hadamard ({bh} bits)")
style(ax, "Number of clients  m", "MSE",
      f"MSE vs Clients  (Δ={DELTA_LOW})")

# ── Plot 3: Reconstruction (first 20 coords) ──────────────────────────────────
ax = axes[2]
nc = 20
x  = np.arange(nc)
ax.plot(x, g_true_vis[:nc],  color=CTRUE, lw=2.2, label="True mean g",              zorder=6)
ax.plot(x, g_usq2_vis[:nc],  color=C2,    lw=1.6, ls=":",  label=f"USQ K=2  MSE={mse_u2:.3f}",  zorder=3)
ax.plot(x, g_usq16_vis[:nc], color=C16,   lw=1.6, ls="--", label=f"USQ K=16 MSE={mse_u16:.3f}", zorder=4)
ax.plot(x, g_had_vis[:nc],   color=CHAD,  lw=1.6, ls="-.", label=f"Hadamard MSE={mse_h:.5f}",   zorder=7)
ax.fill_between(x, g_usq2_vis[:nc],  g_true_vis[:nc], alpha=0.08, color=C2)
ax.fill_between(x, g_usq16_vis[:nc], g_true_vis[:nc], alpha=0.08, color=C16)
ax.fill_between(x, g_had_vis[:nc],   g_true_vis[:nc], alpha=0.08, color=CHAD)
style(ax, "Coordinate index  j", "Value",
      "Reconstruction  (Δ=0.01, first 20 dims)")

plt.tight_layout()
out = "all_three_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✓ Plot saved: all_three_comparison.png")
