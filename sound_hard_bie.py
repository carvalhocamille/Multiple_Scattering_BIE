import numpy as np

from essentials_bie import (
    GaussLegendre,
    ComputeSphericalHarmonics,
    ComputeSurface,
    ComputeIncidentFunction,
)


def Compute_data(n_order, k):
    """Galerkin solution of the sound-hard scattering BIE on the unit sphere.

    Solves (1/2 I - K) u = S[d_n u_inc] for the trace u of the scattered field,
    expanded in spherical harmonics Y_n^m, n < n_order, for an incident plane
    wave exp(i k z).  Returns the N^2 expansion coefficients (columns ordered
    as in ComputeSphericalHarmonics: n = 0..N-1, m = -n..n).

    The integral operators are evaluated with the rotated-grid quadrature of
    Carvalho, Khatri & Kim: for each collocation point (theta0, phi0) the
    integration grid is rotated so that its north pole s = 0 coincides with
    the collocation point, where the kernels are nearly singular.
    """
    N = n_order
    M = 2 * N

    # Product quadrature rule: Gauss-Legendre in mu = cos(theta), periodic
    # trapezoid in phi.  ws = (pi/2) * GL weights, for the map s = pi(mu+1)/2.
    mu, _, _, _, _, ws = GaussLegendre(N)
    s = 0.5 * np.pi * (mu + 1)
    z = np.arccos(mu)
    t = np.arange(-np.pi, np.pi, np.pi / N)

    # Collocation grid on the sphere: theta slow, phi fast.
    THETA = np.repeat(z, M)
    PHI = np.tile(t, N)
    # Quadrature weights in the same ordering (GL weight = 2*ws/pi, times the
    # trapezoid weight 2*pi/M).
    W_GL = np.repeat(ws, M) * (4.0 / M)

    # Rotated integration grid parameters: s slow, t fast.
    S = np.repeat(s, M)
    T = np.tile(t, N)

    Y_GL, _, _ = ComputeSphericalHarmonics(n_order, THETA, PHI)

    Kmtx = np.zeros((N * M, n_order**2), dtype=complex)
    F = np.zeros(N * M, dtype=complex)

    for j in range(N * M):
        theta0, phi0 = THETA[j], PHI[j]
        _, _, ystar, nustar, _ = ComputeSurface(0, 0, theta0, phi0)

        # integration surface rotated so its north pole is at (theta0, phi0)
        varTHETA, varPHI, y, nu, J = ComputeSurface(theta0, phi0, S, T)

        # Neumann data d_n u_inc = i k nu_z exp(i k z) at the rotated points
        f = ComputeIncidentFunction(y[:, 0], y[:, 1], y[:, 2], k)
        dfn = 1j * k * nu[:, 2] * f

        yd = ystar - y
        ydist = np.linalg.norm(yd, axis=1)
        nu_x_y = np.sum(nu * yd, axis=1)

        # single/double layer kernels times Jacobian and quadrature factors
        # (the 1/(4 pi) of the Green's function and the 2 pi of the trapezoid
        # rule combine into the 0.5)
        SLP = 0.5 * J * np.exp(1j * k * ydist) / ydist * np.sin(S)
        DLP = ((1 / ydist - 1j * k) * nu_x_y / ydist) * SLP

        # trapezoid average over t (fast index), then GL in s
        F[j] = ws @ np.sum((SLP * dfn).reshape(N, M), axis=1) / M

        # DLP applied to every spherical harmonic at once
        Ynm, _, _ = ComputeSphericalHarmonics(n_order, varTHETA, varPHI)
        Kmtx[j, :] = ws @ np.sum((DLP[:, None] * Ynm).reshape(N, M, -1), axis=1) / M

    # Galerkin projection onto the spherical harmonics
    P = Y_GL.conj().T * W_GL
    RHS = P @ F
    K = P @ Kmtx

    return np.linalg.solve(0.5 * np.eye(n_order**2) - K, RHS)
