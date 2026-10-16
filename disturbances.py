"""Disturbance laws at a prescribed Gelbrich distance r = delta * rho_ref.

mu = sqrt(eta) r d, Sigma = a^2 Sigma_hat, a = 1 + sqrt(1-eta) r / sqrt(tr Sigma_hat),
so that G((mu, Sigma), (0, Sigma_hat)) = r exactly. Samples are w = mu + Sigma^{1/2} xi
with Cov(xi) = I guaranteed by the generating formula (no sample-moment correction).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def wind_direction(mdl, channels):
    """Unit d in the wind channels maximizing the steady mean error along wall+floor normals."""
    Minv = np.linalg.inv(np.eye(mdl.n) - mdl.A_K)
    h = mdl.H[0] / np.linalg.norm(mdl.H[0]) + mdl.H[3] / np.linalg.norm(mdl.H[3])
    g = h @ Minv[:, channels]
    d = np.zeros(mdl.n)
    d[channels] = g / np.linalg.norm(g)
    return d


def law_parameters(Sigma_hat, d, r, eta):
    mu = np.sqrt(eta) * r * d
    a = 1.0 + np.sqrt(1.0 - eta) * r / np.sqrt(np.trace(Sigma_hat))
    return mu, a * a * Sigma_hat


def gelbrich(mu1, S1, mu2, S2):
    S2h = np.real(sqrtm(S2))
    cross = np.real(sqrtm(S2h @ S1 @ S2h))
    return float(np.sqrt(max(np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * cross), 0.0)))


def sample_xi(rng, family, size, n, nu=5, alpha=0.85, d_mix=None):
    """Standardized noise with mean 0 and identity covariance; shape size + (n,)."""
    g = rng.standard_normal(size + (n,))
    if family == "gaussian":
        return g
    if family == "student_t":
        V = rng.chisquare(nu, size=size)[..., None]
        return np.sqrt((nu - 2.0) / nu) * g / np.sqrt(V / nu)
    if family == "laplace":
        W = rng.exponential(1.0, size=size)[..., None]
        return np.sqrt(W) * g
    if family == "mixture":
        dm = d_mix / np.linalg.norm(d_mix)
        s = rng.choice([-1.0, 1.0], size=size)[..., None]
        # g ~ N(0, I - alpha^2 dm dm^T): remove part of the variance along dm
        proj = (g @ dm)[..., None] * dm
        g_c = g - proj + np.sqrt(1.0 - alpha ** 2) * proj
        return s * alpha * dm + g_c
    raise ValueError(family)


def disturbance_array(rng, family, M, T, mu, Sigma, nu, alpha, d_mix):
    n = len(mu)
    xi = sample_xi(rng, family, (M, T), n, nu=nu, alpha=alpha, d_mix=d_mix)
    L = np.real(sqrtm(Sigma))
    return mu + xi @ L.T
