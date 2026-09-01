"""Multiple sound-hard sphere scattering with spheres at ARBITRARY centers in R^3.

This module generalizes the axisymmetric (Y_n^0-only) machinery of
``offdiag_bie.py`` / ``Nspheres_sound_hard.ipynb`` to spheres whose centers are
not collinear.  Off the z-axis the scattered field of a sphere is no longer
axisymmetric about the global axis, so every trace must be expanded in the FULL
spherical-harmonic basis Y_n^m (n = 0..Nb-1, m = -n..n), giving Nb^2 unknowns
per sphere, ordered exactly as ``ComputeSphericalHarmonics``.

Total-field formulation, identical to the axisymmetric solver:

    u(x) = u_inc(x) + sum_i D_i[u_i](x),

    [ a_i^2 diag(lambda_n(a_i)) delta_ij  -  (1 - delta_ij) B_ij ] c = b,

    lambda_n(a) = -i (ka)^2 j_n(ka) h_n^(1)'(ka)      (m-independent),

    B_ij = < Y_p^q , D_j[Y_n^m] >_{partial B_i}        (Galerkin coupling block),

    b^i = a_i^2 * (trace of the incident plane wave, projected on Y_n^m).

The coupling blocks are assembled with the Carvalho-Khatri-Kim close-evaluation
quadrature (rotated grid + plane-wave subtraction) exactly as in offdiag_bie,
the only change being that the layer density is the full Y_n^m evaluated at the
ROTATED grid angles rather than the axisymmetric Ptilde_n = Y_n^0.

Ground truth for spheres is the multipole identity (verified numerically in the
companion notebook against naive quadrature for all m):

    D[Y_n^m about C](x) = i k^2 a^2 j_n'(ka) h_n^(1)(k|x-C|) Y_n^m(theta_x, phi_x)

with (theta_x, phi_x) the spherical angles of x - C.

Conventions inherited (and validated) from the axisymmetric code:
- essentials_bie.Djn/Dh1n include a factor k (chain rule).
- The 0.5 kernel prefactor = 2pi (trapezoid) / 4pi (Green's function).
- The CKK polar angle uses the LINEAR map s = pi(mu+1)/2 of the GL nodes, the
  spherical Jacobian sin(s) appears explicitly (folded into the weights here),
  and on the unit source sphere the surface Jacobian J == 1 so ComputeSurface is
  not needed for the source integration.
- The plane wave subtracted is anchored at the target x, pw = exp(-i k nu*.(x-y));
  its double-layer contribution is restored analytically and is RANK-ONE across
  the basis columns: out[:, col] = A_col + B * dens_col(y*).
"""
import numpy as np
import scipy.special as sp

from essentials_bie import (
    GaussLegendre,
    SphereRotation,
    ComputeSphericalHarmonics,
)

# SphereQuadrature and ptilde are reused unchanged from the axisymmetric module.
from offdiag_bie import SphereQuadrature, ptilde


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _angles(v):
    """Spherical polar/azimuth angles of the rows of v (..., 3)."""
    v = np.asarray(v, dtype=float)
    theta = np.arctan2(np.hypot(v[..., 0], v[..., 1]), v[..., 2])
    phi = np.arctan2(v[..., 1], v[..., 0])
    return theta, phi


def basis_index(Nb):
    """(nvec, mvec) for the Nb^2 columns, ordered as ComputeSphericalHarmonics."""
    _, nvec, mvec = ComputeSphericalHarmonics(Nb, np.array([0.0]), np.array([0.0]))
    return nvec, mvec


# ---------------------------------------------------------------------------
# double-layer columns D[Y_n^m](x) at arbitrary target(s)
# ---------------------------------------------------------------------------
def dlp_columns_ynm(Nb, k, a, C, x, quad=None, method="subtract", chunk=64):
    """D[Y_n^m](x) for all Nb^2 columns (n=0..Nb-1, m=-n..n) at target(s) x.

    Parameters
    ----------
    Nb : int          basis truncation (Nb^2 columns).
    k  : float        wavenumber.
    a  : float        source-sphere radius.
    C  : (3,) array   source-sphere center.
    x  : (3,) or (npts, 3) array   evaluation target(s).
    quad : SphereQuadrature       required for the quadrature methods.
    method : 'exact' | 'naive' | 'rotated' | 'subtract'.
    chunk : int       targets processed per numpy batch.

    Returns (npts, Nb^2), or (Nb^2,) if a single (3,) target was passed.
    """
    C = np.asarray(C, dtype=float)
    X = np.asarray(x, dtype=float)
    single = X.ndim == 1
    X = np.atleast_2d(X)
    nvec, mvec = basis_index(Nb)
    Nb2 = Nb * Nb

    if method == "exact":
        d = X - C
        r = np.linalg.norm(d, axis=1)
        tx, px = _angles(d)
        Ynm_x, _, _ = ComputeSphericalHarmonics(Nb, tx, px)          # (npts, Nb^2)
        jp = sp.spherical_jn(nvec, k * a, derivative=True)           # (Nb^2,)
        hn = sp.spherical_jn(nvec, k * r[:, None]) \
            + 1j * sp.spherical_yn(nvec, k * r[:, None])             # (npts, Nb^2)
        out = 1j * k**2 * a**2 * jp[None, :] * hn * Ynm_x
        return out[0] if single else out

    if quad is None:
        raise ValueError("quad (SphereQuadrature) is required for method=%r" % method)

    subtract = method == "subtract"
    if method == "naive":
        S, T, rotate = quad.S_pgq, quad.T, False
        wq = np.repeat(quad.wt, quad.Mq) / quad.Mq
    elif method in ("rotated", "subtract"):
        S, T, rotate = quad.S_lin, quad.T, True
        wq = np.repeat(quad.ws, quad.Mq) / quad.Mq * np.sin(quad.S_lin)
    else:
        raise ValueError("unknown method %r" % method)
    nquad = S.shape[0]

    out = np.zeros((len(X), Nb2), dtype=complex)
    for lo in range(0, len(X), chunk):
        xb = X[lo:lo + chunk]
        nc = len(xb)
        d = xb - C
        rx = np.linalg.norm(d, axis=1)
        nustar = d / rx[:, None]                                     # (nc, 3)

        if rotate:
            theta0, phi0 = _angles(nustar)
            th, ph = SphereRotation(S[None, :], T[None, :],
                                    theta0[:, None], phi0[:, None])  # (nc, nquad)
        else:
            th, ph = np.broadcast_arrays(S[None, :], T[None, :])
            th = np.broadcast_to(th, (nc, nquad))
            ph = np.broadcast_to(ph, (nc, nquad))

        # source-grid unit points = outward normals; source points y = C + a*nu
        nux = np.sin(th) * np.cos(ph)
        nuy = np.sin(th) * np.sin(ph)
        nuz = np.cos(th)
        ydx = xb[:, 0, None] - (C[0] + a * nux)
        ydy = xb[:, 1, None] - (C[1] + a * nuy)
        ydz = xb[:, 2, None] - (C[2] + a * nuz)
        r = np.sqrt(ydx**2 + ydy**2 + ydz**2)
        nu_yd = nux * ydx + nuy * ydy + nuz * ydz
        G = 0.5 * a**2 * np.exp(1j * k * r) / r        # 2pi/4pi folded into 0.5
        D = ((1 / r - 1j * k) * nu_yd / r) * G
        WD = wq * D                                    # (nc, nquad)

        # layer density Y_n^m at the (rotated) grid angles -> (nc, nquad, Nb^2)
        dens, _, _ = ComputeSphericalHarmonics(Nb, th.ravel(), ph.ravel())
        dens = dens.reshape(nc, nquad, Nb2)
        A = np.einsum("pq,pqc->pc", WD, dens)          # (nc, Nb^2)

        if subtract:
            nsd = (nustar[:, 0, None] * ydx + nustar[:, 1, None] * ydy
                   + nustar[:, 2, None] * ydz)
            nu_nustar = (nux * nustar[:, 0, None] + nuy * nustar[:, 1, None]
                         + nuz * nustar[:, 2, None])
            pw = np.exp(-1j * k * nsd)                  # phase anchored at x
            B = np.sum(wq * pw * (1j * k * nu_nustar * G - D), axis=1)   # (nc,)
            t0, p0 = _angles(nustar)
            denstar, _, _ = ComputeSphericalHarmonics(Nb, t0, p0)       # (nc, Nb^2)
            A = A + B[:, None] * denstar

        out[lo:lo + chunk] = A

    return out[0] if single else out


# ---------------------------------------------------------------------------
# Galerkin coupling block over the full target-sphere grid
# ---------------------------------------------------------------------------
def target_grid(quad):
    """Collocation grid on the unit sphere: theta-slow / phi-fast, with the
    Galerkin weights W_GL matching sound_hard_bie.  Returns THETA, PHI, W_GL."""
    THETA = quad.S_pgq                 # repeat(arccos(mu), Mq)  -> theta slow
    PHI = quad.T                       # tile(t, Nq)             -> phi fast
    W_GL = np.repeat(quad.ws, quad.Mq) * (4.0 / quad.Mq)
    return THETA, PHI, W_GL


def coupling_block_ynm(Nb, k, a_src, C_src, a_tgt, C_tgt, quad, method="subtract"):
    """Galerkin block  B[(p,q), (n,m)] = < Y_p^q , D_src[Y_n^m] >_{partial B_tgt}.

    The double-layer potential of the SOURCE sphere is evaluated at the full
    theta-slow/phi-fast collocation grid of the TARGET sphere, then projected
    onto conj(Y_p^q) with the surface measure a_tgt^2 W_GL.  Returns (Nb^2, Nb^2).
    """
    THETA, PHI, W_GL = target_grid(quad)
    unit = np.column_stack([np.sin(THETA) * np.cos(PHI),
                            np.sin(THETA) * np.sin(PHI),
                            np.cos(THETA)])
    xnodes = C_tgt + a_tgt * unit
    vals = dlp_columns_ynm(Nb, k, a_src, C_src, xnodes, quad, method=method)  # (npts, Nb^2)
    Ytgt, _, _ = ComputeSphericalHarmonics(Nb, THETA, PHI)                    # (npts, Nb^2)
    P = Ytgt.conj().T * (a_tgt**2 * W_GL)                                     # (Nb^2, npts)
    return P @ vals


# ---------------------------------------------------------------------------
# incident plane wave: expansion coefficients + numerical verification
# ---------------------------------------------------------------------------
def plane_wave_coeffs(Nb, k, a, C, d=(0.0, 0.0, 1.0)):
    """Y_n^m coefficients of the trace of exp(i k d.x) on the sphere (a, C).

    exp(i k d.x) = exp(i k d.C) * 4pi sum_{n,m} i^n j_n(ka) conj(Y_n^m(dhat)) Y_n^m(xhat),
    so the coefficient of Y_n^m is  exp(i k d.C) * 4pi i^n j_n(ka) conj(Y_n^m(dhat)).
    For d = zhat this reduces to  exp(i k C_z) i^n sqrt(4pi(2n+1)) j_n(ka) delta_{m0}.
    """
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    C = np.asarray(C, dtype=float)
    nvec, mvec = basis_index(Nb)
    td, pd = _angles(d)
    Yd, _, _ = ComputeSphericalHarmonics(Nb, np.array([td]), np.array([pd]))
    Yd = Yd[0]
    jn = sp.spherical_jn(nvec, k * a)
    return np.exp(1j * k * (d @ C)) * 4 * np.pi * (1j)**nvec * jn * np.conj(Yd)


def rhs_block(Nb, k, a, C, d=(0.0, 0.0, 1.0)):
    """a^2 times the plane-wave trace coefficients (the RHS block for sphere)."""
    return a**2 * plane_wave_coeffs(Nb, k, a, C, d)


# ---------------------------------------------------------------------------
# diagonal symbol
# ---------------------------------------------------------------------------
def lambda_n(Nb, k, a):
    """Symbol of (1/2 I - K) on Y_n^m for a hard sphere; length Nb^2 (m-repeated)."""
    nvec, _ = basis_index(Nb)
    jn = sp.spherical_jn(nvec, k * a)
    yn = sp.spherical_yn(nvec, k * a)
    djn = sp.spherical_jn(nvec, k * a, derivative=True)
    dyn = sp.spherical_yn(nvec, k * a, derivative=True)
    lam = -1j * (k * a)**2 * jn * (djn + 1j * dyn)
    lam_w = 1 - 1j * (k * a)**2 * djn * (jn + 1j * yn)
    assert np.max(np.abs(lam - lam_w)) < 1e-12, "Wronskian identity failed"
    return lam


# ---------------------------------------------------------------------------
# assemble / solve the N-sphere block system
# ---------------------------------------------------------------------------
def assemble(Nb, k, radii, centers, quad, method="subtract", d=(0.0, 0.0, 1.0)):
    """Block system A c = b for the total-field traces on all spheres.

    radii   : (Ns,) sphere radii.
    centers : (Ns, 3) sphere centers.
    d       : incident plane-wave direction (default z-hat).
    """
    radii = np.asarray(radii, dtype=float)
    centers = np.asarray(centers, dtype=float)
    Ns = len(radii)
    Nb2 = Nb * Nb
    A = np.zeros((Ns * Nb2, Ns * Nb2), dtype=complex)
    b = np.zeros(Ns * Nb2, dtype=complex)
    for i in range(Ns):
        ai, Ci = radii[i], centers[i]
        A[i*Nb2:(i+1)*Nb2, i*Nb2:(i+1)*Nb2] = ai**2 * np.diag(lambda_n(Nb, k, ai))
        b[i*Nb2:(i+1)*Nb2] = rhs_block(Nb, k, ai, Ci, d)
        for j in range(Ns):
            if j == i:
                continue
            aj, Cj = radii[j], centers[j]
            # coupling_block_ynm(source = sphere j, target = sphere i); includes a_i^2
            A[i*Nb2:(i+1)*Nb2, j*Nb2:(j+1)*Nb2] = \
                -coupling_block_ynm(Nb, k, aj, Cj, ai, Ci, quad, method=method)
    return A, b


def solve(Nb, k, radii, centers, quad, method="subtract", d=(0.0, 0.0, 1.0)):
    """Solve the N-sphere system; returns a list of (Nb^2,) coefficient vectors."""
    A, b = assemble(Nb, k, radii, centers, quad, method=method, d=d)
    c = np.linalg.solve(A, b)
    Nb2 = Nb * Nb
    return [c[i*Nb2:(i+1)*Nb2] for i in range(len(radii))]


# ---------------------------------------------------------------------------
# field evaluator
# ---------------------------------------------------------------------------
def total_field(pts, Nb, k, radii, centers, coeffs, quad,
                d=(0.0, 0.0, 1.0), band=0.5):
    """u(x) = exp(i k d.x) + sum_i D_i[u_i](x), masking sphere interiors.

    The exact multipole columns are used away from every surface; within
    band*a_i of sphere i's surface the close-evaluation ('subtract') columns are
    used for that sphere (where naive quadrature would fail).
    """
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    radii = np.asarray(radii, dtype=float)
    centers = np.asarray(centers, dtype=float)
    u = np.exp(1j * k * (pts @ d)).astype(complex)

    dists = np.column_stack([np.linalg.norm(pts - C, axis=1) - a
                             for a, C in zip(radii, centers)])   # (npts, Ns)
    inside = np.min(dists, axis=1) < 0
    u[inside] = np.nan

    for i, (a, C) in enumerate(zip(radii, centers)):
        near = (~inside) & (dists[:, i] < band * a)
        far = (~inside) & (~near)
        if np.any(far):
            u[far] += dlp_columns_ynm(Nb, k, a, C, pts[far], method="exact") @ coeffs[i]
        if np.any(near):
            u[near] += dlp_columns_ynm(Nb, k, a, C, pts[near], quad,
                                       method="subtract") @ coeffs[i]
    return u
