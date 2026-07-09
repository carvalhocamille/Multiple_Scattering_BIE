import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.special import legendre, factorial
import collections.abc
from essentials_bie import *

def Compute_data(n_order, k):
    # QUADRATURE RULES
    N = n_order
    M = 2 * N

    # Gauss-Legendre/trapezoid product rule for theta (z = cos(theta))
    μ, ϕ, mu, phi, WTS, ws = GaussLegendre( N )
    # transform z quadrature rule for s quadrature rule
    s = 0.5 * np.pi * (μ + 1)
    z = np.arccos(μ)
    # trapezoid rules for t
    dt = np.pi / N
    t = np.arange(-np.pi, np.pi, dt)

    # COMPOSITE ANGLE ARRAYS
    PHI = np.tile(t, N)
    THETA = np.repeat(z, M)
    W_GL = WTS
    T = np.tile(t, N)
    S = np.repeat(s, M)
    # SPHERICAL HARMONICS MATRIX
    Y_GL, _, _ = ComputeSphericalHarmonics(n_order, THETA, PHI)
    
    # COMPUTE THE NEUMANN BOUNDARY DATA
    _, _, x, _, _ = ComputeSurface(0, 0, THETA, PHI)
    f = ComputeIncidentFunction(x[:, 0], x[:, 1], x[:, 2], k)

    #Comparison Spherical expansion of f   
    dist  = np.sqrt(x[0, 0]**2+x[0, 1]**2+x[0, 2]**2)  
    f_coeffs = np.zeros((len(THETA), n_order**2), dtype = complex)
    # compute the expansion coefficients
    j = 0
    for nn in np.arange(n_order):
        for m in range(-nn, nn + 1):
            if m == 0:
                B_n = np.exp(1j * nn * np.pi / 2.0 ) * ( 2 * nn + 1 ) 
            else:
                B_n = 0
            print('B_n', B_n, 'jn', jn(k, dist, nn), 'dist', dist)
            f_coeffs[:,j] = B_n * jn(k, dist, nn )
            j += 1
    #fc = np.tile(f_coeffs, n_order)
    #fc= fc.flatten()
    print('f', f, 'fc', f_coeffs, 'Y_GL', Y_GL)
    FSH = (Y_GL * f_coeffs)
    print('f', f, 'FSH', FSH) #,'f-FSH', f-FSH)
    df_coeffs = B_n * Djn(k, dist, n_order)
    # ALLOCATE MEMORY FOR MATRICES
    Kmtx = np.zeros((N * M, n_order**2), dtype = complex)
    F = np.zeros((N*M, 1), dtype = complex)

    # COMPUTE THE FIRST INNER PRODUCT: BIE OPERATOR APPLIED TO SPHERICAL HARMONICS
    for j in range(N * M):
        # set the values of theta0 and phi0
        theta0 = THETA[j]
        phi0 = PHI[j]
        # compute ystar
        _, _, ystar, nustar, _ = ComputeSurface(0, 0, theta0, phi0)
        # compute the surface
        varTHETA, varPHI, y, nu, J = ComputeSurface(theta0, phi0, S, T)
        
        dfn = 1j*k* (nu[:,2]*f)
        yd = ystar - y
        ydist = np.linalg.norm(yd, axis=1)
        nu_nustar = np.sum(nustar * nu, axis=1)
        nu_x_y = np.sum(nu * yd, axis=1)
        nustar_x_y = np.sum(nustar * yd, axis=1)

        SLP = 0.5 * J * np.exp(1j * k * ydist) / ydist * np.sin(S)
        DLP = ((1 / ydist - 1j * k) * nu_x_y / ydist) * SLP

        F[j]= np.dot((np.sum((SLP*dfn).reshape(M,N), axis=0) / M) ,ws)
        
        # compute the integral operation
        kk = 0
        for n in range(n_order):
            for m in range(-n, n + 1):
                # DLP
                ktemp1 = sc.special.sph_harm( m, n, varPHI, varTHETA) * DLP 
                ktemp2 = np.sum(ktemp1.reshape(M, N), axis=0) / M
                Kmtx[j, kk] = np.dot(ktemp2, ws)             
                kk += 1

    #############################
    #Projection onto spherical harmonics
    RHS =  (Y_GL.T @ np.diag(W_GL)) @ F  
    K = (Y_GL.T @ np.diag(W_GL)) @ Kmtx 
    #print('RHS', RHS, 'K', K)

    # SOLVE THE GALERKIN SYSTEM OF EQUATIONS
    c_coeffs1 = np.linalg.solve(0.5 * np.eye(n_order**2) - K, RHS)

    return c_coeffs1