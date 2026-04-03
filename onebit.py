from typing import Optional
import numpy as np


class OneBit:

    def __init__(self, t: int = 1, technique: str = 'II'):
        assert technique in ('I', 'II'), "technique must be 'I' or 'II'"
        self.t = t
        self.technique = technique
        self._Z = None

    # ── Init ─────────────────────────────────────────────────────────────────

    def init(self, m: int, d: int, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        Z = rng.standard_normal((m * self.t, d))
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        self._Z = Z / np.maximum(norms, 1e-10)
        return self._Z

    # ── Encode ───────────────────────────────────────────────────────────────

    def encode(self, g: np.ndarray, client_idx: int,
               Z: Optional[np.ndarray] = None) -> np.ndarray:

        if Z is None:
            Z = self._Z
        rows = Z[client_idx * self.t: (client_idx + 1) * self.t]
        return np.sign(rows @ g)

    # ── Decode ───────────────────────────────────────────────────────────────

    def _decode_technique_ii(self, B_mat: np.ndarray,
                              Z: np.ndarray) -> np.ndarray:

        m = B_mat.shape[0]
        total = np.zeros(Z.shape[1])

        for i in range(m):
            for k in range(self.t):
                z_ik = Z[i * self.t + k]
                b_ik = B_mat[i, k]
                total += z_ik * b_ik

        norm = np.linalg.norm(total)
        return total / max(norm, 1e-10)

    def _decode_technique_i(self, B_mat: np.ndarray,
                             Z: np.ndarray) -> np.ndarray:

        m, _ = B_mat.shape
        d = Z.shape[1]

        zs = []
        bs = []
        for i in range(m):
            for k in range(self.t):
                zs.append(Z[i * self.t + k])
                bs.append(B_mat[i, k])

        zs = np.array(zs)
        bs = np.array(bs)

        g_hat = self._decode_technique_ii(B_mat, Z)

        for _ in range(5):
            preds = np.sign(zs @ g_hat)
            correct = (preds == bs).astype(float)
            weights = np.where(correct > 0, 1.0, 0.1)
            weights /= weights.sum()

            g_new = (zs * bs[:, None] * weights[:, None]).sum(0)
            norm = np.linalg.norm(g_new)

            if norm < 1e-10:
                break

            g_hat = g_new / norm

        return g_hat

    def decode(self, B_mat: np.ndarray,
               Z: Optional[np.ndarray] = None) -> np.ndarray:

        if Z is None:
            Z = self._Z

        if self.technique == 'II':
            return self._decode_technique_ii(B_mat, Z)
        else:
            return self._decode_technique_i(B_mat, Z)

    # ── Convenience ──────────────────────────────────────────────────────────

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        m, d = G.shape
        Z = self.init(m, d, rng)
        B_mat = np.stack([self.encode(G[i], i, Z) for i in range(m)])
        return self.decode(B_mat, Z)

    def bits_per_client(self) -> int:
        return self.t


# ── standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, m = 512, 100

    g = rng.standard_normal(d)
    g /= np.linalg.norm(g)

    delta_corr = 0.1
    angle = np.pi * delta_corr

    G = np.zeros((m, d))
    for i in range(m):
        perp = rng.standard_normal(d)
        perp -= perp @ g * g
        perp /= max(np.linalg.norm(perp), 1e-10)
        G[i] = np.cos(angle) * g + np.sin(angle) * perp

    for tech in ('II', 'I'):
        ob = OneBit(t=1, technique=tech)
        g_hat = ob.compress(G, rng)
        cos_dist = np.arccos(np.clip(g @ g_hat, -1, 1)) / np.pi
        print(f"OneBit Technique {tech}: cosine distance = {cos_dist:.4f}")