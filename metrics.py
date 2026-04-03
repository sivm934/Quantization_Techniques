"""
metrics.py
The three error metrics studied in the paper:

  1. ℓ₂ error   : ‖g̃ − g‖₂²
  2. ℓ∞ error   : ‖g̃ − g‖∞
  3. Cosine dist : arccos(⟨g, g̃⟩) / π   (for unit vectors)

Also: downstream task metrics (KMeans cost, top eigenvalue, MSE, accuracy).
"""

import numpy as np


# ── DME error metrics ─────────────────────────────────────────────────────────

def l2_error(g_hat: np.ndarray, g: np.ndarray) -> float:
    """Squared ℓ₂ estimation error ‖g̃ − g‖₂²."""
    return float(np.sum((g_hat - g) ** 2))


def linf_error(g_hat: np.ndarray, g: np.ndarray) -> float:
    """ℓ∞ estimation error ‖g̃ − g‖∞."""
    return float(np.max(np.abs(g_hat - g)))


def cosine_distance(g_hat: np.ndarray, g: np.ndarray) -> float:
    """
    Cosine distance  arccos(⟨g, g̃⟩ / (‖g‖‖g̃‖)) / π  ∈ [0, 1].
    Both vectors are normalised internally.
    """
    g_norm = g / max(np.linalg.norm(g), 1e-10)
    h_norm = g_hat / max(np.linalg.norm(g_hat), 1e-10)
    dot = np.clip(float(g_norm @ h_norm), -1.0, 1.0)
    return float(np.arccos(dot) / np.pi)


# ── Theoretical dissimilarity measures ───────────────────────────────────────

def delta2(G: np.ndarray, g: np.ndarray) -> float:
    """Δ₂ = (1/m) Σ_i ‖g_i − g‖₂²  (ℓ₂ dissimilarity)."""
    return float(np.mean(np.sum((G - g[None, :]) ** 2, axis=1)))


def delta_inf(G: np.ndarray, g: np.ndarray) -> float:
    """Δ∞ = max_j (1/m) Σ_i |g_i^(j) − g^(j)|  (ℓ∞ dissimilarity)."""
    return float(np.max(np.mean(np.abs(G - g[None, :]), axis=0)))


def delta_corr(G: np.ndarray, g: np.ndarray) -> float:
    """
    Δ_corr = (1/(mπ)) Σ_i arccos(⟨g_i, g⟩)  (cosine dissimilarity for unit vecs).
    """
    g_n = g / max(np.linalg.norm(g), 1e-10)
    dots = G @ g_n / np.maximum(np.linalg.norm(G, axis=1), 1e-10)
    dots = np.clip(dots, -1.0, 1.0)
    return float(np.mean(np.arccos(dots)) / np.pi)


# ── Downstream task metrics ───────────────────────────────────────────────────

def kmeans_cost(X: np.ndarray, centers: np.ndarray) -> float:
    """KMeans objective: Σ_i min_k ‖x_i − c_k‖²."""
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def top_eigenvalue(X: np.ndarray, v: np.ndarray) -> float:
    """Rayleigh quotient ⟨v, Σ v⟩ / ‖v‖² (≈ top eigenvalue estimate)."""
    Sigma = X.T @ X / len(X)
    v_n = v / max(np.linalg.norm(v), 1e-10)
    return float(v_n @ Sigma @ v_n)


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_hat - y) ** 2))


def accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Binary classification accuracy."""
    return float(np.mean(np.sign(y_hat) == np.sign(y)))


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Logistic loss ℓ(w, (x,y)) = log(1 + exp(−⟨w,x⟩·y))."""
    margin = (X @ w) * y
    return float(np.mean(np.log1p(np.exp(-margin))))
