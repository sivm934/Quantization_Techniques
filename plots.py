"""
plots.py
Generate all figures from Section 5 of the paper (Figs 2a–2i and Fig 3).

Usage:
    # Run experiments first (or use cached results):
    python experiments.py
    # Then generate plots:
    python plots.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs("figures", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "NoisySign":        "#e41a1c",
    "HadamardMultiDim": "#377eb8",
    "SparseReg":        "#4daf4a",
    "OneBit":           "#984ea3",
    "RandK":            "#ff7f00",
    "SRQ":              "#a65628",
    "Drive":            "#f781bf",
    "PermK":            "#999999",
    "CorrelatedSRQ":    "#17becf",
    "RandKSpatial":     "#bcbd22",
    "RandKSpatialProj": "#1f77b4",
    "Kashin":           "#8c564b",
}
LINESTYLES = {k: ("-" if k in ("NoisySign","HadamardMultiDim","SparseReg","OneBit")
                  else "--")
              for k in COLORS}
PROPOSED = {"NoisySign", "HadamardMultiDim", "SparseReg", "OneBit"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "figure.dpi": 120,
})


# ── Helper ────────────────────────────────────────────────────────────────────

def _load(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def _plot_mean_std(ax, x, data_dict, keys, xlabel, ylabel, title,
                   xlog=False, ylog=False, n_std=2):
    """Plot mean ± n_std * std for each compressor."""
    for key in keys:
        if key not in data_dict:
            continue
        arr = data_dict[key]            # (N_seeds, len(x))
        mu = arr.mean(0)
        sd = arr.std(0)
        lw = 2 if key in PROPOSED else 1
        ax.plot(x, mu, label=key, color=COLORS.get(key, "black"),
                linestyle=LINESTYLES.get(key, "-"), lw=lw)
        ax.fill_between(x, mu - n_std * sd, mu + n_std * sd,
                        color=COLORS.get(key, "black"), alpha=0.12)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="best", framealpha=0.7)
    ax.grid(True, which="both", alpha=0.3)


def _plot_iter(ax, data_dict, keys, ylabel, title, n_std=1):
    """Plot curves over iterations."""
    for key in keys:
        if key not in data_dict:
            continue
        arr = data_dict[key]
        T = arr.shape[1]
        iters = np.arange(1, T + 1)
        mu = arr.mean(0); sd = arr.std(0)
        lw = 2 if key in PROPOSED else 1
        ax.plot(iters, mu, label=key, color=COLORS.get(key, "black"),
                linestyle=LINESTYLES.get(key, "-"), lw=lw)
        ax.fill_between(iters, mu - n_std * sd, mu + n_std * sd,
                        color=COLORS.get(key, "black"), alpha=0.12)
    ax.set_xlabel("Iterations"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="best", framealpha=0.7)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2a: DME ℓ₂ error
# ══════════════════════════════════════════════════════════════════════════════

def plot_dme_l2():
    d = _load("results/dme_l2.npz")
    if d is None:
        print("  [skip] results/dme_l2.npz not found")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ["RandK", "SRQ", "Drive", "PermK", "CorrelatedSRQ", "SparseReg"]
    _plot_mean_std(ax, d["delta2_vals"], d, keys,
                  xlabel=r"Dissimilarity ($\Delta_2$)",
                  ylabel=r"$\ell_2$ error",
                  title=r"DME $\ell_2$ error vs $\Delta_2$ (Gaussian vectors)",
                  xlog=True, ylog=True, n_std=2)
    fig.tight_layout()
    fig.savefig("figures/fig2a_dme_l2.pdf")
    fig.savefig("figures/fig2a_dme_l2.png")
    plt.close(fig)
    print("  Saved figures/fig2a_dme_l2.*")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2b: DME ℓ∞ error
# ══════════════════════════════════════════════════════════════════════════════

def plot_dme_linf():
    d = _load("results/dme_linf.npz")
    if d is None:
        print("  [skip] results/dme_linf.npz not found")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ["NoisySign", "HadamardMultiDim", "CorrelatedSRQ", "SRQ", "Drive"]
    _plot_mean_std(ax, d["delta_inf_vals"], d, keys,
                  xlabel=r"Dissimilarity ($\Delta_\infty$)",
                  ylabel=r"$\ell_\infty$ error",
                  title=r"DME $\ell_\infty$ error vs $\Delta_\infty$ (Hypercube)",
                  xlog=True, ylog=True, n_std=2)
    fig.tight_layout()
    fig.savefig("figures/fig2b_dme_linf.pdf")
    fig.savefig("figures/fig2b_dme_linf.png")
    plt.close(fig)
    print("  Saved figures/fig2b_dme_linf.*")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2c: DME cosine distance
# ══════════════════════════════════════════════════════════════════════════════

def plot_dme_cosine():
    d = _load("results/dme_cosine.npz")
    if d is None:
        print("  [skip] results/dme_cosine.npz not found")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ["OneBit", "SparseReg", "SRQ", "RandKSpatialProj"]
    _plot_mean_std(ax, d["delta_corr_vals"], d, keys,
                  xlabel=r"Dissimilarity ($\Delta_{\rm corr}$)",
                  ylabel="Cosine distance",
                  title=r"DME cosine distance vs $\Delta_{\rm corr}$ (Unit vectors)",
                  xlog=False, ylog=False, n_std=2)
    fig.tight_layout()
    fig.savefig("figures/fig2c_dme_cosine.pdf")
    fig.savefig("figures/fig2c_dme_cosine.png")
    plt.close(fig)
    print("  Saved figures/fig2c_dme_cosine.*")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2f: Linear Regression – UJIndoorLoc
# ══════════════════════════════════════════════════════════════════════════════

def plot_linreg_ujindoorloc():
    d = _load("results/linreg_ujindoorloc.npz")
    if d is None:
        print("  [skip] results/linreg_ujindoorloc.npz not found")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ["NoisySign", "HadamardMultiDim", "SparseReg",
            "SRQ", "RandKSpatialProj", "OneBit", "CorrelatedSRQ"]
    _plot_iter(ax, d, keys,
               ylabel="Test MSE",
               title="Linear Regression on UJIndoorLoc", n_std=1)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig("figures/fig2f_linreg_ujindoorloc.pdf")
    fig.savefig("figures/fig2f_linreg_ujindoorloc.png")
    plt.close(fig)
    print("  Saved figures/fig2f_linreg_ujindoorloc.*")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2i: Linear Regression – Synthetic
# ══════════════════════════════════════════════════════════════════════════════

def plot_linreg_synthetic():
    d = _load("results/linreg_synthetic.npz")
    if d is None:
        print("  [skip] results/linreg_synthetic.npz not found")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ["HadamardMultiDim", "SparseReg", "SRQ",
            "RandKSpatialProj", "OneBit"]
    _plot_iter(ax, d, keys,
               ylabel="Test MSE",
               title="Linear Regression on Synthetic Dataset", n_std=1)
    fig.tight_layout()
    fig.savefig("figures/fig2i_linreg_synthetic.pdf")
    fig.savefig("figures/fig2i_linreg_synthetic.png")
    plt.close(fig)
    print("  Saved figures/fig2i_linreg_synthetic.*")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Logistic Regression – HAR
# ══════════════════════════════════════════════════════════════════════════════

def plot_logreg_har():
    d_train = _load("results/logreg_har_train.npz")
    d_test  = _load("results/logreg_har_test.npz")
    if d_train is None or d_test is None:
        print("  [skip] logreg_har results not found")
        return

    keys = ["HadamardMultiDim", "OneBit", "SparseReg", "RandKSpatialProj"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    _plot_iter(ax1, d_train, keys,
               ylabel="Training Logistic Loss",
               title="Logistic Regression on HAR (Training Loss)")
    _plot_iter(ax2, d_test, keys,
               ylabel="Test Accuracy",
               title="Logistic Regression on HAR (Test Accuracy)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    fig.savefig("figures/fig3_logreg_har.pdf")
    fig.savefig("figures/fig3_logreg_har.png")
    plt.close(fig)
    print("  Saved figures/fig3_logreg_har.*")


# ══════════════════════════════════════════════════════════════════════════════
# Combined summary figure (all DME sub-plots side by side, like paper Fig 2 top)
# ══════════════════════════════════════════════════════════════════════════════

def plot_dme_combined():
    d2 = _load("results/dme_l2.npz")
    di = _load("results/dme_linf.npz")
    dc = _load("results/dme_cosine.npz")
    if d2 is None and di is None and dc is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if d2 is not None:
        _plot_mean_std(axes[0], d2["delta2_vals"], d2,
                      ["RandK","SRQ","Drive","PermK","CorrelatedSRQ","SparseReg"],
                      r"$\Delta_2$", r"$\ell_2$ error",
                      r"(a) DME $\ell_2$ error", xlog=True, ylog=True)
    if di is not None:
        _plot_mean_std(axes[1], di["delta_inf_vals"], di,
                      ["NoisySign","HadamardMultiDim","CorrelatedSRQ","SRQ","Drive"],
                      r"$\Delta_\infty$", r"$\ell_\infty$ error",
                      r"(b) DME $\ell_\infty$ error", xlog=True, ylog=True)
    if dc is not None:
        _plot_mean_std(axes[2], dc["delta_corr_vals"], dc,
                      ["OneBit","SparseReg","SRQ","RandKSpatialProj"],
                      r"$\Delta_{\rm corr}$", "Cosine distance",
                      r"(c) DME cosine distance")

    fig.suptitle("Distributed Mean Estimation – Same Communication Budget",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/fig2_dme_combined.pdf")
    fig.savefig("figures/fig2_dme_combined.png")
    plt.close(fig)
    print("  Saved figures/fig2_dme_combined.*")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures ...")
    plot_dme_l2()
    plot_dme_linf()
    plot_dme_cosine()
    plot_dme_combined()
    plot_linreg_ujindoorloc()
    plot_linreg_synthetic()
    plot_logreg_har()
    print("\nDone – figures saved in ./figures/")
