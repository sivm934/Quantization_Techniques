import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# RandK
# ══════════════════════════════════════════════════════════════════════════════

class RandK:

    def __init__(self, K: int = 4):
        self.K = K

    def encode(self, g: np.ndarray, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        d = len(g)
        idx = rng.choice(d, self.K, replace=False)
        vals = g[idx] * (d / self.K)
        return idx, vals

    def decode_single(self, d: int, idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
        v = np.zeros(d)
        v[idx] = vals
        return v

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        m, d = G.shape
        total = np.zeros(d)
        for i in range(m):
            idx, vals = self.encode(G[i], rng)
            total += self.decode_single(d, idx, vals)
        return total / m

    def bits_per_client(self, d: int) -> float:
        return 32 * self.K + self.K * np.log2(max(d, 2))


# ══════════════════════════════════════════════════════════════════════════════
# SRQ
# ══════════════════════════════════════════════════════════════════════════════

class SRQ:

    def __init__(self, K: int = 2):
        self.K = K
        self._R = None

    def init(self, d: int, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        diag = rng.choice([-1.0, 1.0], size=d)
        perm = rng.permutation(d)
        self._R = (diag, perm)

    def _rotate(self, g: np.ndarray) -> np.ndarray:
        diag, perm = self._R
        return (g * diag)[perm]

    def _unrotate(self, g: np.ndarray) -> np.ndarray:
        diag, perm = self._R
        inv_perm = np.argsort(perm)
        return g[inv_perm] * diag

    def _quantize(self, v: np.ndarray, B: float, rng) -> np.ndarray:
        if B < 1e-10:
            return np.zeros_like(v)

        step = 2 * B / self.K
        v_clip = np.clip(v, -B, B)

        floor_idx = np.floor((v_clip + B) / step).astype(int)
        floor_idx = np.clip(floor_idx, 0, self.K - 1)

        lo = -B + floor_idx * step
        prob = (v_clip - lo) / step

        bits = (rng.uniform(size=v.shape) < prob).astype(float)
        return lo + bits * step

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        m, d = G.shape
        if self._R is None:
            self.init(d, rng)

        total = np.zeros(d)
        B = np.max(np.abs(G))

        for i in range(m):
            r = self._rotate(G[i])
            q = self._quantize(r, B, rng)
            total += self._unrotate(q)

        return total / m

    def bits_per_client(self, d: int) -> float:
        return self.K * d


# ══════════════════════════════════════════════════════════════════════════════
# Drive
# ══════════════════════════════════════════════════════════════════════════════

class Drive:

    def __init__(self):
        self._R = None

    def init(self, d: int, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        diag = rng.choice([-1.0, 1.0], size=d)
        perm = rng.permutation(d)
        self._R = (diag, perm)

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        m, d = G.shape
        if self._R is None:
            self.init(d, rng)

        diag, perm = self._R
        total = np.zeros(d)
        norms = np.linalg.norm(G, axis=1)

        for i in range(m):
            r = (G[i] * diag)[perm]
            s = np.sign(r)

            inv_perm = np.argsort(perm)
            u = s[inv_perm] * diag

            total += u * norms[i] * np.sqrt(d) / d

        return total / m

    def bits_per_client(self, d: int) -> float:
        return 32 + d


# ══════════════════════════════════════════════════════════════════════════════
# PermK
# ══════════════════════════════════════════════════════════════════════════════

class PermK:

    def __init__(self, K: int = 4):
        self.K = K

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        m, d = G.shape
        perm = rng.permutation(d)
        total = np.zeros(d)

        for i in range(m):
            start = (i * self.K) % d
            idx = perm[start:start + self.K]

            if len(idx) < self.K:
                idx = np.concatenate([idx, perm[:(self.K - len(idx))]])

            v = np.zeros(d)
            v[idx] = G[i, idx] * (d / self.K)
            total += v

        return total / m

    def bits_per_client(self, d: int) -> float:
        return 32 * self.K + self.K * np.log2(max(d, 2))


# ══════════════════════════════════════════════════════════════════════════════
# CorrelatedSRQ
# ══════════════════════════════════════════════════════════════════════════════

class CorrelatedSRQ:

    def __init__(self, K: int = 2):
        self.K = K

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        m, d = G.shape
        dither = rng.uniform(0, 1, d)

        B = np.max(np.abs(G))
        if B < 1e-10:
            return np.zeros(d)

        step = 2 * B / self.K
        total = np.zeros(d)

        for i in range(m):
            v = G[i]
            floor_idx = np.floor((v + B) / step + dither).astype(int)
            floor_idx = np.clip(floor_idx, 0, self.K)
            q = -B + floor_idx * step
            total += q

        return total / m

    def bits_per_client(self, d: int) -> float:
        return 2 * d * np.log2(max(self.K, 2)) + self.K * np.log2(max(d, 2))


# ── test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, m = 128, 100
    g = rng.standard_normal(d) * 10
    G = g[None, :] + rng.normal(0, 1, (m, d))

    for name, cls in [("RandK", RandK(K=8)),
                      ("SRQ", SRQ(K=4)),
                      ("Drive", Drive()),
                      ("PermK", PermK(K=8)),
                      ("CorrelatedSRQ", CorrelatedSRQ(K=4))]:
        g_hat = cls.compress(G.copy(), rng)
        err = np.linalg.norm(g_hat - g) ** 2
        print(f"{name:20s}  ℓ2² error: {err:.4f}")