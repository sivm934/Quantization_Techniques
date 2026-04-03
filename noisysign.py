from scipy.special import erfinv, erf
import numpy as np


class NoisySign:

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _phi(t: np.ndarray, sigma: float) -> np.ndarray:
        """Φ_σ(t) = erf(t / (√2 σ))"""
        return np.clip(erf(t / (np.sqrt(2) * sigma)), -1 + 1e-9, 1 - 1e-9)

    @staticmethod
    def _phi_inv(u: np.ndarray, sigma: float) -> np.ndarray:
        u = np.clip(u, -1 + 1e-9, 1 - 1e-9)
        return np.sqrt(2) * sigma * erfinv(u)

    # ── public interface ──────────────────────────────────────────────────────

    def encode(self, g: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        xi = rng.normal(0, self.sigma, size=g.shape)
        return np.sign(g + xi).astype(np.float64)

    def decode(self, B: np.ndarray) -> np.ndarray:
        avg = B.mean(axis=0)
        return self._phi_inv(avg, self.sigma)

    def compress(self, G: np.ndarray, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        m = G.shape[0]
        B = np.stack([self.encode(G[i], rng) for i in range(m)])
        return self.decode(B)

    @staticmethod
    def bits_per_client(d: int) -> int:
        return d


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, m = 128, 100
    g = rng.standard_normal(d) * 5
    G = g[None, :] + rng.normal(0, 0.5, (m, d))
    ns = NoisySign(sigma=3.0)
    g_hat = ns.compress(G, rng)
    print(f"NoisySign ℓ∞ error: {np.max(np.abs(g_hat - g)):.4f}")