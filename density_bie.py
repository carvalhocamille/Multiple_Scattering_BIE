import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.special import legendre, factorial
import collections.abc
from essentials_bie import *

def Computemu(n_order, kw):
    # QUADRATURE RULES
    N = n_order
    M = 2 * N

    # Gauss-Legendre/trapezoid product rule for theta (z = cos(theta))
    μ, ϕ, mu, phi, WTS, ws = GaussLegendre( N )
    # transform z quadrature rule for s quadrature rule
    s = 0.5 * np.pi * (μ + 1)
    # trapezoid rules for t
    dt = np.pi / N
    t = np.arange(-np.pi, np.pi, dt)

    # COMPOSITE ANGLE ARRAYS
    PHI = np.tile(t, N)
    THETA = np.repeat(np.arccos(μ), M)
    # weights matching the theta-slow/phi-fast ordering of THETA/PHI above
    # (WTS from GaussLegendre is ordered mu-fast and would be mismatched)
    W_GL = np.repeat(ws, M) * (4.0 / M)
    T = np.tile(t, N)
    S = np.repeat(s, M)
    #print('phi', phi, 'THETA', THETA)
    # SPHERICAL HARMONICS MATRIX
    Y_GL, _, _ = ComputeSphericalHarmonics(n_order, THETA, PHI)
    #Y_GL = np.zeros((len(THETA), N**2), dtype = complex)  
    # initialize the counter
    #j = 0    
    #for n in range(n_order):
    #    for m in range(-n, n + 1):
    #        Y_GL[:, j] = sc.special.sph_harm(m, n, PHI, THETA)
    #        j += 1
    
    # COMPUTE THE DIRICHLET BOUNDARY DATA
    _, _, x, _, _ = ComputeSurface(0, 0, THETA, PHI)
    f, _, _, _ = ComputeFunction(x[:, 0], x[:, 1], x[:, 2], kw)

    # COMPUTE SPHERICAL HARMONICS EXPANSION COEFFICIENTS
    f_coeffs = (Y_GL.conj().T @ np.diag(W_GL)) @ f 
    #print('f', f, 'f_coeffs',f_coeffs, 'YGLT', Y_GL.T)
    # ALLOCATE MEMORY FOR MATRICES
    Kmtx1 = np.zeros((N * M, n_order**2), dtype = complex)
    Kmtx2 = np.zeros((N * M, n_order**2), dtype = complex)
    Kmtx = np.zeros((N * M, n_order**2), dtype = complex)
    # COMPUTE THE FIRST INNER PRODUCT: BIE OPERATOR APPLIED TO SPHERICAL HARMONICS
    for j in range(N * M):
        # set the values of theta0 and phi0
        theta0 = THETA[j]
        phi0 = PHI[j]
        #print('theta0', theta0, 'phi0', phi0)
        # compute ystar
        _, _, ystar, nustar, _ = ComputeSurface(0, 0, theta0, phi0)

        # compute the surface
        varTHETA, varPHI, y, nu, J = ComputeSurface(theta0, phi0, S, T)
        #print('varTHETA', varTHETA, 'varPHI', varPHI)
        yd = ystar - y
        ydist = np.linalg.norm(yd, axis=1)
        nu_nustar = np.sum(nustar * nu, axis=1)
        nu_x_y = np.sum(nu * yd, axis=1)
        nustar_x_y = np.sum(nustar * yd, axis=1)

        SLP = 0.5 * J * np.exp(1j * kw * ydist) / ydist
        DLP = ((1 / ydist - 1j * kw) * nu_x_y / ydist) * SLP
        kernel = (DLP - 1j * kw * SLP)
        Kpw = DLP - 1j * kw * nu_nustar * SLP
        # compute plane wave
        PW = np.exp(-1j * kw * nustar_x_y)
        #print('PW', PW)

        # compute the integral operation
        k = 0
        for n in range(n_order):
            for m in range(-n, n + 1):
                ktemp01 = (sc.special.sph_harm( m, n, varPHI, varTHETA) -sc.special.sph_harm( m, n, phi0, theta0 )* PW)* (Kpw) * np.sin(S)
                ktemp02 = 1j * kw * (1 - nu_nustar) * sc.special.sph_harm( m, n, varPHI, varTHETA) * SLP * np.sin(S)
                ktemp1 = sc.special.sph_harm( m, n, varPHI, varTHETA) * kernel * np.sin(S)
                ktemp11 = np.sum(ktemp01.reshape(N, M).T, axis=0) / M
                ktemp12 = np.sum(ktemp02.reshape(N, M).T, axis=0) / M
                ktemp2 = np.sum(ktemp1.reshape(N, M).T, axis=0) / M
        
                Kmtx1[j, k] = np.dot(ktemp11, ws)
                Kmtx2[j, k] = np.dot(ktemp12, ws)
                Kmtx[j, k] = np.dot(ktemp2, ws)
                k += 1
                
    # COMPUTE THE SECOND INNER PRODUCT
    K1 = (Y_GL.conj().T @ np.diag(W_GL)) @ Kmtx1
    #print('K1',K1)
    K2 = (Y_GL.conj().T @ np.diag(W_GL)) @ Kmtx2
    K = (Y_GL.conj().T @ np.diag(W_GL)) @ Kmtx

    # SOLVE THE GALERKIN SYSTEM OF EQUATIONS
    c_coeffs1 = np.linalg.solve(K1 - K2, f_coeffs)
    c_coeffs2 = np.linalg.solve(0.5 * np.eye(n_order**2) + K, f_coeffs)

    return c_coeffs1, c_coeffs2