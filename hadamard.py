from typing import Optional
import numpy as np


class HadamardMultiDim:

    def __init__(self, B: float = 100.0):
        self.B = B
        self._rho = None

    # ── Algorithm 2 ──────────────────────────────────────────────────────────

    def _hadamard1d_enc(self, s: float, level: int) -> float:
        B = self.B
        K = level
        width = 2 * B / (2 ** K)
        idx = int((s + B) / width)
        idx = min(idx, 2 ** K - 1)
        return -1.0 if (idx % 2 == 0) else 1.0

    # ── Algorithm 3 ──────────────────────────────────────────────────────────

    def init(self, m: int, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        rho = rng.permutation(m) + 1
        self._rho = rho
        return rho

    def encode(self, g: np.ndarray, client_idx: int,
               rho: Optional[np.ndarray] = None) -> np.ndarray:

        if rho is None:
            rho = self._rho

        level = int(rho[client_idx])
        b = np.array([self._hadamard1d_enc(g[j], level) for j in range(len(g))])
        return b

    def decode(self, B_mat: np.ndarray,
               rho: Optional[np.ndarray] = None) -> np.ndarray:

        if rho is None:
            rho = self._rho

        m, d = B_mat.shape
        g_hat = np.zeros(d)

        for i in range(m):
            level = int(rho[i])
            weight = self.B / (2 ** (level - 1))
            g_hat += B_mat[i] * weight

        return g_hat

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        m = G.shape[0]
        rho = self.init(m, rng)
        B_mat = np.stack([self.encode(G[i], i, rho) for i in range(m)])
        return self.decode(B_mat, rho)

    @staticmethod
    def bits_per_client(d: int) -> int:
        return d


# ── standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, d, m = 100.0, 64, 50
    g = rng.uniform(-B, B, d)
    delta_inf = 1.0
    G = np.clip(g[None, :] + rng.uniform(-delta_inf, delta_inf, (m, d)), -B, B)

    had = HadamardMultiDim(B=B)
    g_hat = had.compress(G, rng)
    print(f"HadamardMultiDim ℓ∞ error: {np.max(np.abs(g_hat - g)):.4f}")