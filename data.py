from typing import Tuple, List
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings("ignore")

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic DME datasets
# ══════════════════════════════════════════════════════════════════════════════

def make_gaussian_vectors(m: int, d: int, delta2: float, g_norm: float = 100.0,
                          rng=None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = RNG
    g = rng.standard_normal(d)
    g = g / np.linalg.norm(g) * g_norm
    noise = rng.normal(0, delta2, size=(m, d))
    G = g[None, :] + noise
    return G, g


def make_hypercube_vectors(m: int, d: int, delta_inf: float, B: float = 100.0,
                           rng=None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = RNG
    g = rng.uniform(-B, B, d)
    noise = rng.uniform(-delta_inf, delta_inf, size=(m, d))
    G = np.clip(g[None, :] + noise, -B, B)
    return G, g


def make_unit_vectors(m: int, d: int, delta_corr: float,
                      rng=None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = RNG
    g = rng.standard_normal(d)
    g /= np.linalg.norm(g)

    angle = np.pi * delta_corr
    G = np.zeros((m, d))
    for i in range(m):
        perp = rng.standard_normal(d)
        perp -= perp.dot(g) * g
        norm = np.linalg.norm(perp)
        if norm < 1e-10:
            G[i] = g
            continue
        perp /= norm
        G[i] = np.cos(angle) * g + np.sin(angle) * perp
    return G, g


# ══════════════════════════════════════════════════════════════════════════════
# 2. Real datasets
# ══════════════════════════════════════════════════════════════════════════════

def _pca_reduce(X: np.ndarray, d: int = 512) -> np.ndarray:
    n_comp = min(d, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    X_r = pca.fit_transform(X)
    if n_comp < d:
        X_r = np.hstack([X_r, np.zeros((X_r.shape[0], d - n_comp))])
    return X_r


def load_mnist(m: int = 50, d: int = 512,
               rng=None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = RNG
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data.astype(float), mnist.target.astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = _pca_reduce(X, d)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    return X, y


def load_har(m: int = 20, d: int = 512,
             rng=None) -> Tuple[List[np.ndarray], List[np.ndarray],
                                np.ndarray, np.ndarray]:
    if rng is None:
        rng = RNG
    try:
        har = fetch_openml("har", version=1, as_frame=False, parser="auto")
    except Exception:
        print("[data] HAR not available; using synthetic substitute.")
        return _synthetic_har(m, d, rng)

    X, y = har.data.astype(float), har.target.astype(int)
    classes = np.unique(y)
    mask = (y == classes[-2]) | (y == classes[-1])
    X, y = X[mask], y[mask]
    y = np.where(y == classes[-2], -1, 1).astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = _pca_reduce(X, d)

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    chunk = len(train_X) // m
    client_X = [train_X[i*chunk:(i+1)*chunk] for i in range(m)]
    client_y = [train_y[i*chunk:(i+1)*chunk] for i in range(m)]
    return client_X, client_y, test_X, test_y


def _synthetic_har(m, d, rng):
    n = 2000
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    w_true /= np.linalg.norm(w_true)
    y = np.sign(X @ w_true + rng.standard_normal(n) * 0.1)
    split = int(0.8 * n)
    client_X = np.array_split(X[:split], m)
    client_y = np.array_split(y[:split], m)
    return client_X, client_y, X[split:], y[split:]


def load_ujindoorloc(m: int = 20, d: int = 512,
                     rng=None) -> Tuple[List, List, np.ndarray, np.ndarray]:
    if rng is None:
        rng = RNG
    try:
        ds = fetch_openml("UJIIndoorLoc", version=1, as_frame=False, parser="auto")
        X = ds.data[:, :d].astype(float)
        y = ds.target.astype(float)
    except Exception:
        print("[data] UJIndoorLoc not available; using synthetic substitute.")
        n = 2000
        X = rng.standard_normal((n, d))
        y = rng.standard_normal(n) * 10

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    chunk = max(1, len(train_X) // m)
    client_X = [train_X[i*chunk:(i+1)*chunk] for i in range(m)]
    client_y = [train_y[i*chunk:(i+1)*chunk] for i in range(m)]
    return client_X, client_y, test_X, test_y


def make_synthetic_regression(m: int = 20, d: int = 512, n_per_client: int = 1000,
                               delta2: float = 4.0,
                               rng=None) -> Tuple[List, List, np.ndarray, np.ndarray]:
    if rng is None:
        rng = RNG
    g = rng.standard_normal(d)
    g /= np.linalg.norm(g)
    W = g[None, :] + rng.normal(0, delta2, (m, d))

    client_X, client_y = [], []
    for i in range(m):
        Xi = rng.standard_normal((n_per_client, d))
        yi = Xi @ W[i] + rng.normal(0, 0.1, n_per_client)
        client_X.append(Xi)
        client_y.append(yi)

    X_test = rng.standard_normal((500, d))
    w_avg = W.mean(0)
    y_test = X_test @ w_avg
    return client_X, client_y, X_test, y_test