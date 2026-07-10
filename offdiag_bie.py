"""Close evaluation of the two-sphere off-diagonal coupling blocks.

The off-diagonal blocks -A12, -A21 of the two-sphere BIE system
(2spheres_spherical.ipynb) are Galerkin projections of the double-layer
potential of one sphere evaluated at points on the other sphere.  When the
gap between the spheres is small these evaluation points are close to the
integration surface and plain product-Gauss quadrature fails.

This module assembles those blocks with the close-evaluation technique of
Carvalho, Khatri & Kim used in sound_hard_bie.py / density_bie.py:

* the integration grid on the source sphere is rotated so that its north
  pole sits at the point y* closest to the target x (SphereRotation via
  ComputeSurface), and

* the plane wave dens(y*) * exp(-i k nu* . (x - y)) -- an exact Helmholtz
  solution matching the density at y* -- is subtracted from the density,
  its double-layer contribution being restored analytically through
  Green's identity  D[e^{-ik nu*.(x-.)}](x) = S[ik nu.nu* e^{-ik nu*.(x-.)}](x).

Densities are the normalized Legendre polynomials Ptilde_n(cos theta) =
sqrt((2n+1)/4pi) P_n(cos theta) = Y_n^0 about the source sphere's center
(the axisymmetric setting of 2spheres_spherical.ipynb).

For spheres the exact result is available through the multipole identity

    D[Ptilde_n](x) = i k^2 a^2 j_n'(k a) h_n^(1)(k |x-C|) Ptilde_n(cos gamma)

(dlp_column_exact), which serves as ground truth to validate the quadrature.
"""
import numpy as np
import scipy.special as sp

from essentials_bie import GaussLegendre, ComputeSurface


def ptilde(N, x):
    """Normalized Legendre polynomials: columns Ptilde_n(x), n = 0..N-1."""
    n = np.arange(N)
    return np.sqrt((2 * n + 1) / (4 * np.pi)) * np.stack(
        [sp.eval_legendre(nn, x) for nn in n], axis=-1)


def dlp_column_exact(N, k, a, C, x):
    """Exact D[Ptilde_n](x), n = 0..N-1, for a sphere of radius a centered at C."""
    d = np.asarray(x, dtype=float) - C
    r = np.linalg.norm(d)
    n = np.arange(N)
    jp = sp.spherical_jn(n, k * a, derivative=True)
    hn = sp.spherical_jn(n, k * r) + 1j * sp.spherical_yn(n, k * r)
    return 1j * k**2 * a**2 * jp * hn * ptilde(N, d[2] / r)


class SphereQuadrature:
    """Product Gauss quadrature data of order Nq reused across evaluations."""

    def __init__(self, Nq):
        self.Nq, self.Mq = Nq, 2 * Nq
        mu, _, _, _, _, ws = GaussLegendre(Nq)
        self.ws = ws                      # (pi/2)-scaled GL weights for s in (0, pi)
        self.wt = ws / (np.pi / 2)        # plain GL weights for mu = cos(theta)
        t = np.arange(-np.pi, np.pi, np.pi / Nq)
        self.S_lin = np.repeat(0.5 * np.pi * (mu + 1), self.Mq)   # s slow
        self.S_pgq = np.repeat(np.arccos(mu), self.Mq)
        self.T = np.tile(t, Nq)                                   # t fast

    def average_t(self, v):
        return np.sum(v.reshape(self.Nq, self.Mq), axis=1) / self.Mq


def dlp_column_naive(N, k, a, C, x, quad):
    """D[Ptilde_n](x) by plain product-Gauss quadrature on the source sphere."""
    th, ph = quad.S_pgq, quad.T
    Yu = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    y = C + a * Yu
    dens = ptilde(N, Yu[:, 2])
    yd = x - y
    r = np.linalg.norm(yd, axis=1)
    G = 0.5 * a**2 * np.exp(1j * k * r) / r          # 2pi (trapezoid) / 4pi (Green) = 0.5
    D = ((1 / r - 1j * k) * np.sum(Yu * yd, axis=1) / r) * G
    return quad.wt @ np.apply_along_axis(quad.average_t, 0, D[:, None] * dens)


def _dlp_column_ckk(N, k, a, C, x, quad, subtract):
    """Carvalho-Khatri-Kim quadrature for D[Ptilde_n](x): the grid is rotated
    so its north pole sits at the closest point y*, the polar angle uses the
    linear map s = pi (mu + 1) / 2 of the Gauss-Legendre nodes (NOT arccos),
    and the spherical Jacobian sin(s) appears explicitly in the integrand.
    With subtract=True the plane wave matching the density at y* is subtracted
    and restored analytically (the close-evaluation method)."""
    d = np.asarray(x, dtype=float) - C
    rx = np.linalg.norm(d)
    nustar = d / rx                                   # normal at the closest point y*
    theta0 = np.arctan2(np.hypot(nustar[0], nustar[1]), nustar[2])
    phi0 = np.arctan2(nustar[1], nustar[0])
    _, _, Yu, nu, J = ComputeSurface(theta0, phi0, quad.S_lin, quad.T)
    y = C + a * Yu
    dens = ptilde(N, Yu[:, 2])                        # density at grid points, (pts, N)
    yd = x - y
    r = np.linalg.norm(yd, axis=1)
    G = 0.5 * a**2 * J * np.exp(1j * k * r) / r * np.sin(quad.S_lin)
    D = ((1 / r - 1j * k) * np.sum(nu * yd, axis=1) / r) * G
    if subtract:
        denstar = ptilde(N, nustar[2])                # density at y*, (N,)
        pw = np.exp(-1j * k * (yd @ nustar))          # plane wave, value 1 at y*
        integrand = D[:, None] * (dens - pw[:, None] * denstar) \
            + (G * 1j * k * (nu @ nustar) * pw)[:, None] * denstar
    else:
        integrand = D[:, None] * dens
    return quad.ws @ np.apply_along_axis(quad.average_t, 0, integrand)


def dlp_column_rotated(N, k, a, C, x, quad):
    """D[Ptilde_n](x) by the Carvalho-Khatri-Kim quadrature (rotated grid,
    linear s-map, explicit sin(s) Jacobian) WITHOUT the plane-wave subtraction."""
    return _dlp_column_ckk(N, k, a, C, x, quad, subtract=False)


def dlp_column_subtract(N, k, a, C, x, quad):
    """D[Ptilde_n](x) by the Carvalho-Khatri-Kim quadrature WITH the
    plane-wave subtraction (the full close-evaluation method)."""
    return _dlp_column_ckk(N, k, a, C, x, quad, subtract=True)


def coupling_block(N, k, a_src, C_src, a_tgt, C_tgt, quad, method="subtract"):
    """Galerkin block <Ptilde_m, D_src[Ptilde_n]> over the target sphere.

    Entry (m, n) = 2 pi a_tgt^2 sum_i wt_i Ptilde_m(mu_i) D_src[Ptilde_n](x_i)
    with x_i the Gauss-Legendre ring representatives on the target sphere
    (the geometry is axisymmetric about the z-axis).
    """
    column = {"exact": dlp_column_exact,
              "naive": dlp_column_naive,
              "rotated": dlp_column_rotated,
              "subtract": dlp_column_subtract}[method]
    mu = np.cos(quad.S_pgq[::quad.Mq])                # GL nodes, mu-ordering of quad
    vals = np.zeros((len(mu), N), dtype=complex)
    for i, m in enumerate(mu):
        x = C_tgt + a_tgt * np.array([np.sqrt(1 - m**2), 0.0, m])
        if method == "exact":
            vals[i] = column(N, k, a_src, C_src, x)
        else:
            vals[i] = column(N, k, a_src, C_src, x, quad)
    Pm = ptilde(N, mu)                                # (nodes, N)
    return 2 * np.pi * a_tgt**2 * (Pm.T * quad.wt) @ vals
