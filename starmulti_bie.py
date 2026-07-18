"""Multiple smooth star-shaped scatterers at arbitrary positions/orientations
(sound-hard Helmholtz scattering).

This module couples several bodies of the kind built in ``starshape_bie.py``
(each  y = C_i + Q_i @ ( rho_i(cos theta) rhat )  with center C_i and rotation
Q_i) into one Galerkin block system, reusing every verified single-body
ingredient of ``starshape_bie`` unchanged:

    u(x) = u_inc(x) + sum_i D_i[u_i](x),

    [ A_ii delta_ij  -  (1 - delta_ij) B_ij ] c = b,

with, exactly as in ``multisphere_bie``,

    A_ii = self_operator(shape_i)              (1/2 M_i - K_i, frame invariant),
    B_ij = < Y_p^q , D_j[Y_n^m] >_{S_i}        (Galerkin coupling block),
    b^i  = incident_trace_projection_i         (P[u_inc trace], carries C_i,Q_i).

The single body's ``self_operator`` is frame invariant: K depends only on the
shape because its kernel is a function of surface geometry evaluated in the
LOCAL frame (rotation/translation invariant), and M is a purely local surface
inner product.  Only the incident-wave RHS sees C_i, Q_i (through the global
surface points that the plane wave is sampled at).  This is asserted
numerically in the companion notebook.

The coupling block evaluates the SOURCE body's double layer at the TARGET
body's theta-slow/phi-fast collocation nodes (mapped to global coordinates) and
projects onto conj(Y_p^q) with the SAME Galerkin measure (W_GL * W_tgt) used by
``self_operator`` / ``incident_trace_projection`` / ``mass_matrix``, so the whole
block system is one consistent Galerkin discretisation.  Each target node picks
its close-evaluation method automatically: 'subtract' (plane-wave subtraction)
when the node lies within ``band`` of the source surface, else 'rotated'.
"""

import numpy as np

from essentials_bie import ComputeSphericalHarmonics, SphereRotation
from starshape_bie import (
    _collocation,
    _geometry,
    StarQuadrature,
    self_operator,
    incident_trace_projection,
    closest_point,
)


# ---------------------------------------------------------------------------
# vectorised (batched-over-targets) close evaluation of D[Y_n^m] for a single
# star-shaped body, LOCAL-frame targets.  Mirrors starshape._dlp_columns_one but
# processes many targets per numpy call (the per-point loop is far too slow for
# the coupling grids and the finale field grid).
# ---------------------------------------------------------------------------
def _closest_theta_scan(shape, R, Z, n_theta=512):
    """Grid-scan the meridional closest-point polar angle theta* for targets
    given by cylindrical (R, Z) (vectorised).  Sub-grid accuracy is unnecessary:
    theta* only recentres the rotated quadrature grid."""
    th = np.linspace(0.0, np.pi, n_theta)
    rho = shape.rho(np.cos(th))
    sx = rho * np.sin(th)
    sz = rho * np.cos(th)
    d2 = (sx[None, :] - R[:, None]) ** 2 + (sz[None, :] - Z[:, None]) ** 2
    return th[np.argmin(d2, axis=1)]


def _batch_dlp_local(shape, Nb, k, Xloc, quad, method, chunk=128):
    """D[Y_n^m](x) columns for LOCAL-frame targets Xloc (nc, 3).  method is
    'naive', 'rotated' or 'subtract'.  Returns (nc, Nb^2).  Chunked to bound the
    (chunk, nquad, Nb^2) working set; the reductions are BLAS matmuls."""
    Xloc = np.atleast_2d(np.asarray(Xloc, dtype=float))
    nc = len(Xloc)
    Nb2 = Nb * Nb
    out = np.zeros((nc, Nb2), dtype=complex)

    if method == "naive":
        # one fixed product-Gauss grid, geometry + harmonics evaluated once
        th, ph, y, nu, J = _compute_surface_grid(shape, 0.0, 0.0, quad.S_pgq, quad.T)
        Ynm, _, _ = ComputeSphericalHarmonics(Nb, th, ph)          # (nq, Nb^2)
        wq = np.repeat(quad.wt, quad.Mq) / quad.Mq                  # (nq,)
        for lo in range(0, nc, chunk):
            xb = Xloc[lo:lo + chunk]
            yd = xb[:, None, :] - y[None, :, :]                    # (nb, nq, 3)
            r = np.linalg.norm(yd, axis=2)
            G = 0.5 * J[None, :] * np.exp(1j * k * r) / r
            D = ((1.0 / r - 1j * k) * np.einsum("qd,bqd->bq", nu, yd) / r) * G
            out[lo:lo + chunk] = (D * wq) @ Ynm                    # (nb, Nb^2)
        return out

    # rotated / subtract: grid recentred at each target's closest-point preimage
    R = np.hypot(Xloc[:, 0], Xloc[:, 1])
    Z = Xloc[:, 2]
    phi_s = np.arctan2(Xloc[:, 1], Xloc[:, 0])
    theta_s = _closest_theta_scan(shape, R, Z)
    wq = np.repeat(quad.ws, quad.Mq) / quad.Mq                     # (nq,)
    sinS = np.sin(quad.S_lin)                                       # (nq,)

    subtract = method == "subtract"
    for lo in range(0, nc, chunk):
        xb = Xloc[lo:lo + chunk]
        ts = theta_s[lo:lo + chunk]
        ps = phi_s[lo:lo + chunk]
        nb = len(xb)
        th, ph = SphereRotation(quad.S_lin[None, :], quad.T[None, :],
                                ts[:, None], ps[:, None])          # (nb, nq)
        y, nu, W = _geometry(shape, th.ravel(), ph.ravel())
        y = y.reshape(nb, -1, 3)
        nu = nu.reshape(nb, -1, 3)
        J = W.reshape(nb, -1)
        yd = xb[:, None, :] - y                                     # (nb, nq, 3)
        r = np.linalg.norm(yd, axis=2)
        G = 0.5 * J * np.exp(1j * k * r) / r * sinS[None, :]
        nu_yd = np.einsum("bqd,bqd->bq", nu, yd)
        D = ((1.0 / r - 1j * k) * nu_yd / r) * G
        Ynm, _, _ = ComputeSphericalHarmonics(Nb, th.ravel(), ph.ravel())
        Ynm = Ynm.reshape(nb, -1, Nb2)

        if subtract:
            # accurate closest-point normal + density anchor per target
            nustar = np.empty((nb, 3))
            denstar = np.empty((nb, Nb2), dtype=complex)
            for i in range(nb):
                th_i, ph_i, _, nu_i = closest_point(shape, xb[i])
                nustar[i] = nu_i
                di, _, _ = ComputeSphericalHarmonics(Nb, th_i, ph_i)
                denstar[i] = di[0]
            pw = np.exp(-1j * k * np.einsum("bqd,bd->bq", yd, nustar))
            nu_nustar = np.einsum("bqd,bd->bq", nu, nustar)
            # <D*wq, Ynm> - <D*wq, pw>*denstar + <G*ik*nu.nustar*pw*wq,1>*denstar
            DW = D * wq[None, :]                                    # (nb, nq)
            main = np.einsum("bq,bqk->bk", DW, Ynm)                 # (nb, Nb^2)
            coef = (np.sum(DW * pw, axis=1)
                    - np.sum(G * 1j * k * nu_nustar * pw * wq[None, :], axis=1))
            out[lo:lo + chunk] = main - coef[:, None] * denstar
        else:
            DW = D * wq[None, :]
            out[lo:lo + chunk] = np.einsum("bq,bqk->bk", DW, Ynm)
    return out


def _compute_surface_grid(shape, theta0, phi0, s, t):
    """Rotated-grid local geometry (theta, phi, y, nu, J) as flat arrays."""
    from starshape_bie import compute_surface
    theta, phi = SphereRotation(s, t, theta0, phi0)
    y, nu, W = _geometry(shape, theta, phi)
    return theta, phi, y, nu, W


# ---------------------------------------------------------------------------
# cheap vectorised distance from targets to an axisymmetric source surface
# ---------------------------------------------------------------------------
def surface_distance(shape, X_local, n_theta=512):
    """Approximate distance from each LOCAL-frame target (rows of X_local) to
    the surface of the axisymmetric ``shape``.

    Because the body is a surface of revolution the closest surface point lies
    in the target's meridional half-plane, so the distance depends only on the
    target's cylindrical radius R = hypot(x, y) and height Z = z.  A dense polar
    scan (n_theta nodes) gives a value that is always >= the true distance, so
    using it against a slightly padded band never *under*-triggers the close
    evaluation.  Fully vectorised; used only to choose the quadrature method.
    """
    X_local = np.atleast_2d(np.asarray(X_local, dtype=float))
    th = np.linspace(0.0, np.pi, n_theta)
    mu = np.cos(th)
    rho = shape.rho(mu)
    sx = rho * np.sin(th)                     # surface (R, Z) profile
    sz = rho * mu
    R = np.hypot(X_local[:, 0], X_local[:, 1])
    Z = X_local[:, 2]
    d2 = (sx[None, :] - R[:, None]) ** 2 + (sz[None, :] - Z[:, None]) ** 2
    return np.sqrt(d2.min(axis=1))


def _to_local(C, Q, X):
    """x_local = Q.T @ (x - C) for rows of X."""
    return (np.asarray(X, dtype=float) - np.asarray(C, dtype=float)) @ np.asarray(Q, dtype=float)


# ---------------------------------------------------------------------------
# batched close evaluation of D_src[Y_n^m] at global targets, per-target method
# ---------------------------------------------------------------------------
def dlp_bodies(src, Nb, k, X, quad, band=0.3, far_method="rotated"):
    """D_src[Y_n^m](x) columns at GLOBAL targets X for source body descriptor
    ``src = (shape, C, Q)`` (vectorised over targets).

    Targets within ``band`` of the source surface use the plane-wave-subtraction
    close evaluation; the rest use ``far_method`` ('rotated', accurate at every
    exterior distance, or 'naive', cheapest -- both are at the quadrature floor
    beyond ~0.3 of the surface, see the notebook's distance sweep).  Returns
    (len(X), Nb^2)."""
    shape, C, Q = src
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if not isinstance(quad, StarQuadrature):
        quad = StarQuadrature(quad)
    Xloc = _to_local(C, Q, X)
    dist = surface_distance(shape, Xloc)
    near = dist < band
    out = np.zeros((len(X), Nb * Nb), dtype=complex)
    if np.any(near):
        out[near] = _batch_dlp_local(shape, Nb, k, Xloc[near], quad, "subtract")
    far = ~near
    if np.any(far):
        out[far] = _batch_dlp_local(shape, Nb, k, Xloc[far], quad, far_method)
    return out


# ---------------------------------------------------------------------------
# Galerkin coupling block  < Y_p^q , D_src[Y_n^m] >_{S_tgt}
# ---------------------------------------------------------------------------
def coupling_block(tgt, src, Nb, k, quad_order, src_quad=None, band=0.3,
                   far_method="naive", return_nodes=False):
    """Coupling block B[(p,q),(n,m)] = < Y_p^q , D_src[Y_n^m] >_{S_tgt}.

    ``tgt`` and ``src`` are body descriptors (shape, C, Q).  The source body's
    double layer of every Y_n^m column is evaluated at the target body's
    theta-slow/phi-fast collocation nodes (global points), then projected onto
    conj(Y_p^q) with the Galerkin surface measure  W_GL * W_tgt(theta)  -- the
    same measure ``self_operator`` uses -- so the assembled system is one
    consistent Galerkin discretisation.  ``quad_order`` sets the target
    collocation order; ``src_quad`` (default = quad_order) the source
    close-evaluation order.  Returns (Nb^2, Nb^2)."""
    shape_t, C_t, Q_t = tgt
    N, M, THETA, PHI, W_GL, S, T, ws = _collocation(quad_order)
    y_local, _, _ = _geometry(shape_t, THETA, PHI)
    Xnodes = np.asarray(C_t, dtype=float) + y_local @ np.asarray(Q_t, dtype=float).T
    W_col = shape_t.W(np.cos(THETA))
    Ytgt, _, _ = ComputeSphericalHarmonics(Nb, THETA, PHI)
    P = Ytgt.conj().T * (W_GL * W_col)                    # (Nb^2, n_nodes)

    quad = StarQuadrature(src_quad or quad_order)
    vals = dlp_bodies(src, Nb, k, Xnodes, quad, band=band,
                      far_method=far_method)                 # (n_nodes, Nb^2)
    block = P @ vals
    if return_nodes:
        return block, Xnodes
    return block


# ---------------------------------------------------------------------------
# assemble / solve the multi-body block system
# ---------------------------------------------------------------------------
def assemble(bodies, Nb, k, direction=None, quad_order=None, src_quad=None,
             band=0.3):
    """Assemble (A, b) for a list of body descriptors ``bodies = [(shape,C,Q),..]``.

    Diagonal blocks are ``self_operator(shape_i)`` (frame invariant, cached per
    distinct shape object), off-diagonal blocks are ``-coupling_block(i<-j)``,
    and the RHS block is ``incident_trace_projection`` carrying (C_i, Q_i) for
    the incident plane wave (default direction +z).  ``quad_order`` defaults to
    Nb."""
    qo = quad_order or Nb
    nb = len(bodies)
    Nb2 = Nb * Nb
    A = np.zeros((nb * Nb2, nb * Nb2), dtype=complex)
    b = np.zeros(nb * Nb2, dtype=complex)

    self_cache = {}
    for i, (shape, C, Q) in enumerate(bodies):
        key = id(shape)
        if key not in self_cache:
            self_cache[key] = self_operator(shape, Nb, k, quad_order=qo)
        A[i * Nb2:(i + 1) * Nb2, i * Nb2:(i + 1) * Nb2] = self_cache[key]
        b[i * Nb2:(i + 1) * Nb2] = incident_trace_projection(
            shape, Nb, k, C=C, Q=Q, direction=direction, quad_order=qo)
        for j, src in enumerate(bodies):
            if j == i:
                continue
            A[i * Nb2:(i + 1) * Nb2, j * Nb2:(j + 1) * Nb2] = \
                -coupling_block(bodies[i], src, Nb, k, qo,
                                src_quad=src_quad, band=band)
    return A, b


def solve(bodies, Nb, k, direction=None, quad_order=None, src_quad=None,
          band=0.3):
    """Solve the multi-body system; return a list of (Nb^2,) trace-coefficient
    vectors, one per body."""
    A, b = assemble(bodies, Nb, k, direction=direction, quad_order=quad_order,
                    src_quad=src_quad, band=band)
    c = np.linalg.solve(A, b)
    Nb2 = Nb * Nb
    return [c[i * Nb2:(i + 1) * Nb2] for i in range(len(bodies))]


# ---------------------------------------------------------------------------
# total field evaluator
# ---------------------------------------------------------------------------
def interior_mask(bodies, pts):
    """Boolean mask: True where a point is inside any body (|x_local| < rho)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    inside = np.zeros(len(pts), dtype=bool)
    for shape, C, Q in bodies:
        xl = _to_local(C, Q, pts)
        r = np.linalg.norm(xl, axis=1)
        theta = np.arctan2(np.hypot(xl[:, 0], xl[:, 1]), xl[:, 2])
        inside |= r < shape.rho(np.cos(theta))
    return inside


def total_field(pts, bodies, coeffs, Nb, k, direction=None, quad_order=None,
                src_quad=None, band=0.3):
    """u(x) = exp(i k d.x) + sum_i D_i[u_i](x), interiors masked with NaN.

    Points within ``band`` of body i's surface use the plane-wave-subtraction
    close evaluation for that body's contribution; elsewhere the rotated grid.
    ``direction`` defaults to +z."""
    d = np.array([0.0, 0.0, 1.0]) if direction is None else np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    quad = StarQuadrature(src_quad or quad_order or Nb)

    u = np.exp(1j * k * (pts @ d)).astype(complex)
    inside = interior_mask(bodies, pts)
    u[inside] = np.nan
    ext = ~inside

    for i, (shape, C, Q) in enumerate(bodies):
        if not np.any(ext):
            break
        idx = np.where(ext)[0]
        vals = dlp_bodies((shape, C, Q), Nb, k, pts[idx], quad, band=band,
                          far_method="naive")
        u[idx] += vals @ coeffs[i]
    return u
