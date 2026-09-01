"""Boundary-integral solver and close evaluation for smooth star-shaped
axisymmetric bodies (sound-hard Helmholtz scattering).

A body is  y = C + Q @ ( rho(theta) * rhat(theta, phi) )  where rho depends
only on the polar angle theta (through mu = cos theta), C is the center and Q a
rotation (orientation) matrix.  All quadrature and geometry are done in the
body's LOCAL frame; an exterior target x is mapped in with
x_local = Q.T @ (x - C).  The generalisation of essentials_bie.ComputeSurface
follows its docstring ("scale Y by a radius function R(theta, phi) ... and
include its derivatives in Y_theta / Y_phi").

Geometry (axisymmetric, derived in the module tests)
----------------------------------------------------
With the orthonormal spherical basis
    rhat = (sin th cos ph, sin th sin ph, cos th)
    that = ( cos th cos ph,  cos th sin ph, -sin th)   (d rhat / d theta)
    phihat = (-sin ph, cos ph, 0)                      (d rhat / d phi = sin th * phihat)
and rho_theta = d rho / d theta = -sin th * drho_dmu (vanishes at the poles,
keeping the surface smooth there), the tangents are
    y_theta = rho_theta rhat + rho that ,   y_phi = rho sin th phihat ,
so
    y_theta x y_phi = rho sin th ( rho rhat - rho_theta that ) ,
    outward unit normal   nu = ( rho rhat - rho_theta that ) / sqrt(rho^2 + rho_theta^2) ,
    area density          W  = rho sqrt(rho^2 + rho_theta^2)   (dS = W dOmega).
Because SphereRotation is an isometry of the parameter sphere, on a rotated
grid dS = W sin(s) ds dt, i.e. the Jacobian J returned here equals W evaluated
at the rotated polar angle.  In terms of mu directly
    W(mu) = rho sqrt( rho^2 + (1 - mu^2) drho_dmu^2 ).

Conventions inherited from sound_hard_bie / offdiag_bie (respect exactly):
 * solver collocation grid is theta-slow / phi-fast, W_GL = repeat(ws, M)*4/M;
 * CKK rotated grid uses the LINEAR map s = pi(mu+1)/2 with explicit sin(s);
 * kernel prefactor 0.5 = 2pi (trapezoid) / 4pi (Green);
 * plane-wave subtraction anchored at the target x with nu* at the closest y*.
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize_scalar

from essentials_bie import (
    GaussLegendre,
    SphereRotation,
    ComputeSphericalHarmonics,
)


# ---------------------------------------------------------------------------
# Shapes: radius as a smooth function of mu = cos(theta), with drho/dmu.
# ---------------------------------------------------------------------------
class Shape:
    """Base class: subclasses provide rho(mu) and drho_dmu(mu)."""

    def rho(self, mu):
        raise NotImplementedError

    def drho_dmu(self, mu):
        raise NotImplementedError

    def rho_theta(self, mu):
        """d rho / d theta = -sin(theta) * drho/dmu (vanishes at the poles)."""
        return -np.sqrt(np.clip(1.0 - mu**2, 0.0, None)) * self.drho_dmu(mu)

    def W(self, mu):
        """Area density W(mu): dS = W dOmega,  W = rho sqrt(rho^2 + rho_theta^2)."""
        rho = self.rho(mu)
        d = self.drho_dmu(mu)
        return rho * np.sqrt(rho**2 + (1.0 - mu**2) * d**2)


class Sphere(Shape):
    def __init__(self, a=1.0):
        self.a = float(a)

    def rho(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), self.a)

    def drho_dmu(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))


class Peanut(Shape):
    """rho(mu) = 0.5 sqrt(3 mu^2 + 1): poles at radius 1, pinched to 0.5 at the
    equator (a smooth axisymmetric peanut / dumbbell)."""

    def rho(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 0.5 * np.sqrt(3.0 * mu**2 + 1.0)

    def drho_dmu(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 1.5 * mu / np.sqrt(3.0 * mu**2 + 1.0)


class MushroomCap(Shape):
    """A smooth star-shaped 'mushroom cap' profile.

    rho is a cubic in mu = cos(theta), designed so the widest point (the cap
    'rim') sits just below the equator, the top is a rounded dome and the
    underside is tucked toward the axis -- the silhouette of a mushroom cap.
    The cubic is fixed by rho(+1) = 0.85 (rounded apex), rho(-1) = 0.50 (tucked
    underside), and a maximum rho(-0.35) = 1.18 (the rim):

        rho(mu) = 1.0703 - 0.5392 mu - 0.3953 mu^2 + 0.7142 mu^3

    strictly positive with min rho = 0.50, max rho = 1.18.
    """

    c = np.array([1.0703, -0.5392, -0.3953, 0.7142])   # [c0..c3] powers of mu

    def rho(self, mu):
        mu = np.asarray(mu, dtype=float)
        c = self.c
        return c[0] + c[1] * mu + c[2] * mu**2 + c[3] * mu**3

    def drho_dmu(self, mu):
        mu = np.asarray(mu, dtype=float)
        c = self.c
        return c[1] + 2 * c[2] * mu + 3 * c[3] * mu**2


def sphere(a=1.0):
    return Sphere(a)


def peanut():
    return Peanut()


def mushroom_cap():
    return MushroomCap()


# ---------------------------------------------------------------------------
# Local-frame geometry.
# ---------------------------------------------------------------------------
def _geometry(shape, theta, phi):
    """Local surface points Y, outward unit normals NU and area density W
    (= Jacobian) at parameter angles (theta, phi).  Arrays; leading axis = pts."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    st, ct = np.sin(theta), np.cos(theta)
    sp_, cp = np.sin(phi), np.cos(phi)
    mu = ct

    rho = shape.rho(mu)
    rho_th = shape.rho_theta(mu)              # = -st * drho_dmu, zero at poles
    norm = np.sqrt(rho**2 + rho_th**2)

    rhat = np.column_stack((st * cp, st * sp_, ct))
    that = np.column_stack((ct * cp, ct * sp_, -st))

    Y = rho[:, None] * rhat
    NU = (rho[:, None] * rhat - rho_th[:, None] * that) / norm[:, None]
    W = rho * norm
    return Y, NU, W


def compute_surface(shape, theta0, phi0, s, t):
    """Star-shaped generalisation of essentials_bie.ComputeSurface.

    Rotate the parameter grid (SphereRotation) so its north pole s = 0 sits at
    (theta0, phi0); return the rotated angles (theta, phi), the LOCAL-frame
    surface points Y, outward unit normals NU, and the Jacobian J with
    dS = J sin(s) ds dt (J = area density W evaluated at the rotated angles).

    Scalar (s, t) returns Y with shape (3,) and NU with a leading singleton,
    matching the historical ComputeSurface interface.
    """
    theta, phi = SphereRotation(s, t, theta0, phi0)
    scalar = np.ndim(theta) == 0
    Y, NU, W = _geometry(shape, theta, phi)
    if scalar:
        return theta, phi, Y[0], NU, W
    return theta, phi, Y, NU, W


# ---------------------------------------------------------------------------
# Quadrature data (shared with the close evaluation).
# ---------------------------------------------------------------------------
class StarQuadrature:
    """Product Gauss-Legendre (mu) x periodic trapezoid (t) quadrature data."""

    def __init__(self, Nq):
        self.Nq, self.Mq = Nq, 2 * Nq
        mu, _, _, _, _, ws = GaussLegendre(Nq)
        self.mu = mu
        self.ws = ws                       # (pi/2)-scaled GL weights for s-map
        self.wt = ws / (np.pi / 2)         # plain GL weights for mu integration
        t = np.arange(-np.pi, np.pi, np.pi / Nq)
        self.S_lin = np.repeat(0.5 * np.pi * (mu + 1), self.Mq)   # linear s-map, s slow
        self.S_pgq = np.repeat(np.arccos(mu), self.Mq)            # arccos map (naive)
        self.T = np.tile(t, Nq)                                   # t fast

    def average_t(self, v):
        return np.sum(v.reshape(self.Nq, self.Mq), axis=1) / self.Mq


# ---------------------------------------------------------------------------
# Collocation grid for the Galerkin solve (theta-slow / phi-fast).
# ---------------------------------------------------------------------------
def _collocation(Nq):
    """Product Gauss-Legendre (mu) x trapezoid (t) collocation grid of order
    Nq (theta slow / phi fast), plus the matching rotated integration grid."""
    N = Nq
    M = 2 * N
    mu, _, _, _, _, ws = GaussLegendre(N)
    s = 0.5 * np.pi * (mu + 1)
    z = np.arccos(mu)
    t = np.arange(-np.pi, np.pi, np.pi / N)

    THETA = np.repeat(z, M)
    PHI = np.tile(t, N)
    W_GL = np.repeat(ws, M) * (4.0 / M)      # solid-angle quadrature weights
    S = np.repeat(s, M)                       # rotated integration grid, s slow
    T = np.tile(t, N)
    return N, M, THETA, PHI, W_GL, S, T, ws


def mass_matrix(shape, Nb, quad_order=None):
    """Surface mass matrix M_{pq,nm} = <Y_p^q, Y_n^m>_S (NOT the identity for
    non-spherical shapes).  quad_order defaults to Nb."""
    N, M, THETA, PHI, W_GL, S, T, ws = _collocation(quad_order or Nb)
    Y_GL, _, _ = ComputeSphericalHarmonics(Nb, THETA, PHI)
    W_col = shape.W(np.cos(THETA))
    P = Y_GL.conj().T * (W_GL * W_col)       # Galerkin projection with area density
    return P @ Y_GL


def self_operator(shape, Nb, k, quad_order=None, return_parts=False):
    """Assemble the sound-hard single-body operator (1/2 M - K).

    M is the surface mass matrix and K the Galerkin double-layer matrix,
    assembled exactly like sound_hard_bie.Compute_data but with the star
    geometry (rotated integration grid per collocation point).  The integration
    /collocation quadrature order defaults to Nb; use quad_order > Nb for higher
    accuracy (e.g. ~2 Nb reaches the quadrature floor).
    """
    N, M, THETA, PHI, W_GL, S, T, ws = _collocation(quad_order or Nb)
    Y_GL, _, _ = ComputeSphericalHarmonics(Nb, THETA, PHI)
    W_col = shape.W(np.cos(THETA))
    P = Y_GL.conj().T * (W_GL * W_col)

    Kmtx = np.zeros((N * M, Nb**2), dtype=complex)
    for j in range(N * M):
        theta0, phi0 = THETA[j], PHI[j]
        _, _, ystar, _, _ = compute_surface(shape, theta0, phi0, 0.0, 0.0)
        varTHETA, varPHI, y, nu, J = compute_surface(shape, theta0, phi0, S, T)

        yd = ystar - y
        ydist = np.linalg.norm(yd, axis=1)
        nu_x_y = np.sum(nu * yd, axis=1)

        SLP = 0.5 * J * np.exp(1j * k * ydist) / ydist * np.sin(S)
        DLP = ((1.0 / ydist - 1j * k) * nu_x_y / ydist) * SLP

        Ynm, _, _ = ComputeSphericalHarmonics(Nb, varTHETA, varPHI)
        Kmtx[j, :] = ws @ np.sum((DLP[:, None] * Ynm).reshape(N, M, -1), axis=1) / M

    Mmat = P @ Y_GL
    Kmat = P @ Kmtx
    A = 0.5 * Mmat - Kmat
    if return_parts:
        return A, Mmat, Kmat
    return A


def incident_trace_projection(shape, Nb, k, C=None, Q=None, direction=None,
                              quad_order=None):
    """Galerkin projection P[u_inc trace] of the plane wave exp(i k d.x)
    (default direction d = +z) sampled at the collocation surface points, which
    are mapped to the GLOBAL frame y_global = C + Q @ y_local before evaluating
    the incident wave."""
    C = np.zeros(3) if C is None else np.asarray(C, dtype=float)
    Q = np.eye(3) if Q is None else np.asarray(Q, dtype=float)
    d = np.array([0.0, 0.0, 1.0]) if direction is None else np.asarray(direction, float)

    N, M, THETA, PHI, W_GL, S, T, ws = _collocation(quad_order or Nb)
    Y_GL, _, _ = ComputeSphericalHarmonics(Nb, THETA, PHI)
    W_col = shape.W(np.cos(THETA))
    P = Y_GL.conj().T * (W_GL * W_col)

    y_local, _, _ = _geometry(shape, THETA, PHI)
    y_global = C + y_local @ Q.T
    uinc = np.exp(1j * k * (y_global @ d))
    return P @ uinc


def solve_trace(shape, Nb, k, C=None, Q=None, direction=None, quad_order=None):
    """Solve (1/2 M - K) c = P[u_inc trace] for the total-field trace
    coefficients (total-field formulation u = u_inc + D[u], single body)."""
    A = self_operator(shape, Nb, k, quad_order=quad_order)
    rhs = incident_trace_projection(shape, Nb, k, C=C, Q=Q, direction=direction,
                                    quad_order=quad_order)
    return np.linalg.solve(A, rhs)


# ---------------------------------------------------------------------------
# Closest surface point (axisymmetric: 1D in theta after fixing the azimuth).
# ---------------------------------------------------------------------------
def closest_point(shape, x_local, n_coarse=400):
    """Return (theta*, phi*, y*, nu*): the parameter angles and local point /
    normal of the surface point closest to x_local."""
    x_local = np.asarray(x_local, dtype=float)
    R = np.hypot(x_local[0], x_local[1])
    Z = x_local[2]
    phi_s = np.arctan2(x_local[1], x_local[0])

    def dist2(theta):
        mu = np.cos(theta)
        rho = shape.rho(mu)
        return (rho * np.sin(theta) - R)**2 + (rho * np.cos(theta) - Z)**2

    th = np.linspace(0.0, np.pi, n_coarse)
    d = dist2(th)
    i = int(np.argmin(d))
    lo = th[max(i - 1, 0)]
    hi = th[min(i + 1, n_coarse - 1)]
    res = minimize_scalar(dist2, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-13})
    theta_s = float(res.x)

    Y, NU, _ = _geometry(shape, theta_s, phi_s)
    return theta_s, phi_s, Y[0], NU[0]


# ---------------------------------------------------------------------------
# Close evaluation of the double-layer potential at exterior targets.
# ---------------------------------------------------------------------------
def _dlp_columns_one(shape, Nb, k, x_local, quad, method):
    """D[Y_n^m](x) for all Nb^2 columns at a single LOCAL target x_local."""
    theta_s, phi_s, ystar, nustar = closest_point(shape, x_local)

    if method == "naive":
        # unrotated product-Gauss grid (arccos map, integrate in mu, no sin)
        theta, phi, y, nu, J = compute_surface(shape, 0.0, 0.0, quad.S_pgq, quad.T)
        yd = x_local - y
        r = np.linalg.norm(yd, axis=1)
        G = 0.5 * J * np.exp(1j * k * r) / r
        D = ((1.0 / r - 1j * k) * np.sum(nu * yd, axis=1) / r) * G
        Ynm, _, _ = ComputeSphericalHarmonics(Nb, theta, phi)
        return quad.wt @ np.apply_along_axis(quad.average_t, 0, D[:, None] * Ynm)

    # rotated grid centered at the closest-point preimage (theta*, phi*)
    theta, phi, y, nu, J = compute_surface(shape, theta_s, phi_s, quad.S_lin, quad.T)
    yd = x_local - y
    r = np.linalg.norm(yd, axis=1)
    G = 0.5 * J * np.exp(1j * k * r) / r * np.sin(quad.S_lin)
    D = ((1.0 / r - 1j * k) * np.sum(nu * yd, axis=1) / r) * G
    Ynm, _, _ = ComputeSphericalHarmonics(Nb, theta, phi)

    if method == "subtract":
        # density value at y* uses Y_n^m(theta*, phi*)
        denstar, _, _ = ComputeSphericalHarmonics(Nb, theta_s, phi_s)
        denstar = denstar[0]                      # (Nb^2,)
        pw = np.exp(-1j * k * (yd @ nustar))      # phase anchored at x, value 1 nowhere special
        integrand = D[:, None] * (Ynm - pw[:, None] * denstar) \
            + (G * 1j * k * (nu @ nustar) * pw)[:, None] * denstar
    else:                                          # 'rotated'
        integrand = D[:, None] * Ynm
    return quad.ws @ np.apply_along_axis(quad.average_t, 0, integrand)


def dlp_ynm_targets(shape, Nb, k, X, quad_order, method="subtract",
                    C=None, Q=None):
    """D[Y_n^m density](x) for all Nb^2 columns at every target in X.

    Density is Y_n^m in the body's LOCAL parameter angles.  Targets X are
    GLOBAL points; they are mapped to the local frame (x_local = Q.T @ (x - C))
    before the closest point / rotated-grid quadrature.  method is one of
    'subtract' (close evaluation, plane-wave subtraction), 'rotated' (rotated
    grid only) or 'naive' (plain product Gauss).  Returns (len(X), Nb^2)."""
    C = np.zeros(3) if C is None else np.asarray(C, dtype=float)
    Q = np.eye(3) if Q is None else np.asarray(Q, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    quad = quad_order if isinstance(quad_order, StarQuadrature) else StarQuadrature(quad_order)

    out = np.zeros((len(X), Nb**2), dtype=complex)
    for i, x in enumerate(X):
        x_local = Q.T @ (x - C)
        out[i] = _dlp_columns_one(shape, Nb, k, x_local, quad, method)
    return out


# ---------------------------------------------------------------------------
# Close evaluation of the single-layer potential (used by the validation).
# ---------------------------------------------------------------------------
def slp_density_targets(shape, k, X, dens_func, quad_order, method="rotated",
                        C=None, Q=None):
    """S[g](x) at exterior targets X for a density g given by dens_func.

    dens_func(y_local, nu_local) -> g values (per grid point), evaluated in the
    LOCAL frame.  The single layer is weakly singular; the rotated grid alone
    (method='rotated') already regularises it, so no plane-wave subtraction is
    used.  method='naive' uses the plain product-Gauss grid."""
    C = np.zeros(3) if C is None else np.asarray(C, dtype=float)
    Q = np.eye(3) if Q is None else np.asarray(Q, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    quad = quad_order if isinstance(quad_order, StarQuadrature) else StarQuadrature(quad_order)

    out = np.zeros(len(X), dtype=complex)
    for i, x in enumerate(X):
        x_local = Q.T @ (x - C)
        theta_s, phi_s, _, _ = closest_point(shape, x_local)
        if method == "naive":
            _, _, y, nu, J = compute_surface(shape, 0.0, 0.0, quad.S_pgq, quad.T)
            yd = x_local - y
            r = np.linalg.norm(yd, axis=1)
            G = 0.5 * J * np.exp(1j * k * r) / r
            g = dens_func(y, nu)
            out[i] = quad.wt @ quad.average_t(G * g)
        else:
            _, _, y, nu, J = compute_surface(shape, theta_s, phi_s, quad.S_lin, quad.T)
            yd = x_local - y
            r = np.linalg.norm(yd, axis=1)
            G = 0.5 * J * np.exp(1j * k * r) / r * np.sin(quad.S_lin)
            g = dens_func(y, nu)
            out[i] = quad.ws @ quad.average_t(G * g)
    return out
