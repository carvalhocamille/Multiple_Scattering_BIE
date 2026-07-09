This is code treating the close evaluation problem for boundary integral equations. What needs to be adjusted:

- sound_hard_scattering.ipynb doesn't provide expected results. The computed "approx1" should yield linear convergence.
- sound_hard_bie.py and essentials_bie.py are not optimal Python code (lots of reshape to match size). Could be improved.
- ultimate goal : take the 2_spherical.ipynb and treat the off-diagonal blocks -A21, -A12 using the techniques in the above files
