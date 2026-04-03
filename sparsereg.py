from typing import Optional, Tuple, List
import numpy as np


class SparseReg:

    def __init__(self, L: int = 8, B: float = 100.0):
        self.L = L
        self.B = B
        self._A = None
        self._rho = None
        self._m = None

    # ── coefficient ──────────────────────────────────────────────────────────

    def _coeff(self, k: int, m: int) -> float:
        L, d = self.L, self._A.shape[1]
        ratio = 2 * np.log(max(L, 2)) / d
        ratio = min(ratio, 0.999)
        return self.B * np.sqrt(ratio) * ((1 - ratio) ** ((k - 1) / 2))

    # ── Init ─────────────────────────────────────────────────────────────────

    def init(self, m: int, d: int, rng=None) -> Tuple[np.ndarray, np.ndarray]:

        if rng is None:
            rng = np.random.default_rng()

        self._m = m
        self._A = rng.standard_normal((m * self.L, d))
        self._rho = rng.permutation(m) + 1

        return self._A, self._rho

    # ── Encode ───────────────────────────────────────────────────────────────

    def encode(self, g: np.ndarray, client_idx: int,
               A: Optional[np.ndarray] = None,
               rho: Optional[np.ndarray] = None) -> int:

        if A is None:
            A = self._A
        if rho is None:
            rho = self._rho

        level = int(rho[client_idx])
        g_prime = g.copy()

        for j in range(1, level):
            c_j = self._coeff(j, self._m)
            start = (j - 1) * self.L
            block = A[start: start + self.L]
            best_r = int(np.argmax(block @ g_prime))
            g_prime = g_prime - c_j * block[best_r]

        start = (level - 1) * self.L
        block = A[start: start + self.L]
        best_r = int(np.argmax(block @ g_prime))

        return best_r

    # ── Decode ───────────────────────────────────────────────────────────────

    def decode(self, encodings: List[int],
               A: Optional[np.ndarray] = None,
               rho: Optional[np.ndarray] = None) -> np.ndarray:

        if A is None:
            A = self._A
        if rho is None:
            rho = self._rho

        m = len(encodings)
        d = A.shape[1]
        g_hat = np.zeros(d)

        for i, b in enumerate(encodings):
            level = int(rho[i])
            c = self._coeff(level, m)
            row_idx = (level - 1) * self.L + b
            g_hat += c * A[row_idx]

        return g_hat

    # ── Convenience ──────────────────────────────────────────────────────────

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        m, d = G.shape
        self.init(m, d, rng)

        encodings = [self.encode(G[i], i) for i in range(m)]
        return self.decode(encodings)

    def bits_per_client(self) -> float:
        return np.log2(max(self.L, 2))


# ── test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, m = 128, 50
    B = 100.0

    g = rng.standard_normal(d)
    g = g / np.linalg.norm(g) * B

    G = g[None, :] + rng.normal(0, 2.0, (m, d))

    sr = SparseReg(L=8, B=B)
    g_hat = sr.compress(G, rng)

    print(f"SparseReg ℓ₂² error: {np.linalg.norm(g_hat - g)**2:.4f}")
    print(f"Bits/client: {sr.bits_per_client():.2f}")