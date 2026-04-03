from typing import List, Tuple
import numpy as np
from metrics import (l2_error, linf_error, cosine_distance,
                     kmeans_cost, top_eigenvalue, mse, accuracy, logistic_loss)


def _compress(compressor, G: np.ndarray, rng) -> np.ndarray:
    return compressor.compress(G, rng)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DME
# ══════════════════════════════════════════════════════════════════════════════

def run_dme(compressor, G: np.ndarray, g: np.ndarray,
            error_metric: str = 'l2', rng=None) -> float:

    if rng is None:
        rng = np.random.default_rng()

    g_hat = _compress(compressor, G, rng)

    if error_metric == 'l2':
        return l2_error(g_hat, g)
    elif error_metric == 'linf':
        return linf_error(g_hat, g)
    elif error_metric == 'cosine':
        return cosine_distance(g_hat, g)
    else:
        raise ValueError(f"Unknown error_metric '{error_metric}'")


# ══════════════════════════════════════════════════════════════════════════════
# 2. KMeans
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans(compressor, X: np.ndarray, y: np.ndarray,
               n_clusters: int = 10, T: int = 20,
               m: int = 50, rng=None) -> List[float]:

    if rng is None:
        rng = np.random.default_rng()

    n, d = X.shape
    idx = rng.choice(n, n_clusters, replace=False)
    centers = X[idx].copy()
    shards = np.array_split(X, m)

    costs = []

    for t in range(T):
        new_centers_sum = np.zeros_like(centers)
        counts = np.zeros(n_clusters)

        for shard in shards:
            dists = np.sum((shard[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    new_centers_sum[k] += shard[mask].sum(0)
                    counts[k] += mask.sum()

        for k in range(n_clusters):
            if counts[k] > 0:
                centers[k] = new_centers_sum[k] / max(counts[k], 1)

        costs.append(kmeans_cost(X, centers))

    return costs


# ══════════════════════════════════════════════════════════════════════════════
# 3. Power Iteration
# ══════════════════════════════════════════════════════════════════════════════

def run_power_iteration(compressor, X: np.ndarray,
                        T: int = 30, m: int = 50, rng=None) -> List[float]:

    if rng is None:
        rng = np.random.default_rng()

    n, d = X.shape
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)

    shards = np.array_split(X, m)
    eigenvalues = []

    for t in range(T):
        G = np.zeros((m, d))

        for i, shard in enumerate(shards):
            G[i] = shard.T @ (shard @ v) / n

        g_hat = _compress(compressor, G, rng)

        norm = np.linalg.norm(g_hat)
        if norm > 1e-10:
            v = g_hat / norm

        eigenvalues.append(top_eigenvalue(X, v))

    return eigenvalues


# ══════════════════════════════════════════════════════════════════════════════
# 4. Linear Regression
# ══════════════════════════════════════════════════════════════════════════════

def run_linear_regression(compressor,
                         client_X: List[np.ndarray],
                         client_y: List[np.ndarray],
                         test_X: np.ndarray,
                         test_y: np.ndarray,
                         T: int = 50, lr: float = 0.01,
                         rng=None) -> List[float]:

    if rng is None:
        rng = np.random.default_rng()

    m = len(client_X)
    d = client_X[0].shape[1]
    w = np.zeros(d)

    test_mse = []

    for t in range(T):
        G = np.zeros((m, d))

        for i, (Xi, yi) in enumerate(zip(client_X, client_y)):
            resid = Xi @ w - yi
            G[i] = Xi.T @ resid / len(yi)

        g_hat = _compress(compressor, G, rng)
        w -= lr * g_hat

        pred = test_X @ w
        test_mse.append(mse(pred, test_y))

    return test_mse


# ══════════════════════════════════════════════════════════════════════════════
# 5. Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════

def run_logistic_regression(compressor,
                           client_X: List[np.ndarray],
                           client_y: List[np.ndarray],
                           test_X: np.ndarray,
                           test_y: np.ndarray,
                           T: int = 200, lr: float = 0.001,
                           rng=None) -> Tuple[List[float], List[float]]:

    if rng is None:
        rng = np.random.default_rng()

    m = len(client_X)
    d = client_X[0].shape[1]
    w = np.zeros(d)

    train_losses, test_accs = [], []

    for t in range(T):
        G = np.zeros((m, d))

        for i, (Xi, yi) in enumerate(zip(client_X, client_y)):
            margin = (Xi @ w) * yi
            weight = -yi / (1 + np.exp(margin))
            G[i] = Xi.T @ weight / len(yi)

        g_hat = _compress(compressor, G, rng)
        w -= lr * g_hat

        total_loss = 0.0
        n_total = 0

        for Xi, yi in zip(client_X, client_y):
            total_loss += logistic_loss(w, Xi, yi) * len(yi)
            n_total += len(yi)

        train_losses.append(total_loss / n_total)

        pred = test_X @ w
        test_accs.append(accuracy(pred, test_y))

    return train_losses, test_accs