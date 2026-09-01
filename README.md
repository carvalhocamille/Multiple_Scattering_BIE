This is code treating the close evaluation problem for boundary integral equations.

## Files

- `essentials_bie.py` — shared machinery: quadrature rules, rotated sphere
  parametrization (Carvalho–Khatri–Kim), spherical harmonics, analytic solutions.
- `density_bie.py` — Galerkin BIE solve for the interior Dirichlet problem
  (manufactured point-source solution), used by `Multiple_scattering_3D.ipynb`.
- `sound_hard_bie.py` — Galerkin BIE solve for sound-hard plane-wave scattering,
  used by `sound_hard_scattering.ipynb`.
- `offdiag_bie.py` — close-evaluation assembly of the two-sphere off-diagonal
  coupling blocks, demonstrated in `2spheres_offdiag_close_eval.ipynb`.
- `2spheres_spherical.ipynb` — two-sphere block system with analytic
  (multipole) off-diagonal blocks.
- `theyellowhope.py` — misc utilities (colors, analytic solutions, an earlier
  spectral two-sphere solver).

## Status of the original TODOs

- ~~sound_hard_scattering.ipynb doesn't provide expected results.~~ **Fixed**:
  `approx1` (rotated grid + plane-wave subtraction) now shows the expected
  O(ε) convergence down to the quadrature floor. The causes were an extra
  factor of k in `ComputeExpansionFunction` (`Djn` already contains one), a
  transposed s/t reduction and mismatched Galerkin weights in
  `sound_hard_bie.py`, Neumann data evaluated on the wrong grid, and a
  scrambled "exact" reference / crossed error pairing in the notebook.
- ~~sound_hard_bie.py and essentials_bie.py are not optimal Python code.~~
  **Done**: vectorized, reshape branches removed, dead code and debug prints
  dropped; `density_bie.py` got the same weight/projection fixes.
- Ultimate goal — treat the off-diagonal blocks −A21, −A12 with the
  close-evaluation techniques: **prototype in place**. `offdiag_bie.py` +
  `2spheres_offdiag_close_eval.ipynb` assemble the coupling blocks by
  rotated-grid + subtraction quadrature and validate them against the exact
  multipole identity (machine precision at gap 1e−8 where naive quadrature has
  O(1) error). Remaining: swap these blocks into `2spheres_spherical.ipynb`
  (the exact scaling relation is documented in the demo notebook) and extend
  `ComputeSurface` to non-spherical shapes.

## Multiple scatterers

The close-evaluation coupling has since been carried through to full
multi-body solvers. `multisphere_bie.py` + `Nspheres_offaxis.ipynb` handle
$N$ sound-hard spheres at **arbitrary** (off-axis) centers in the full
$Y_n^m$ basis, with the coupling blocks assembled by rotated-grid +
plane-wave subtraction and validated against the exact multipole identity and
rotation covariance. `starshape_bie.py` extends `ComputeSurface` to smooth
star-shaped axisymmetric bodies (sphere / peanut / mushroom cap) with a
per-body center and orientation, providing the single-body operator, close
evaluation of both layer potentials, and closest-point search.
`starmulti_bie.py` + `nonspherical_scattering.ipynb` then couple several such
star-shaped bodies at arbitrary positions/orientations into one Galerkin
system (frame-invariant diagonal blocks, close-evaluation coupling blocks,
automatic subtraction-vs-far-quadrature selection per target), validated in
the sphere limit against `multisphere_bie`, by a multi-body Green's
representation with point sources inside each body (including targets $10^{-3}$
off each surface), and by self-convergence of the scattered field.

Note: the code now requires numpy 2.x-compatible scipy; a compatibility shim
in `essentials_bie.py` restores `scipy.special.sph_harm` on scipy ≥ 1.17.
