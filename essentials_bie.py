import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.special import legendre, factorial
import collections.abc

# scipy >= 1.17 removed sph_harm; restore it (old signature: sph_harm(m, n, azimuth, polar))
if not hasattr(sc.special, 'sph_harm'):
    def _sph_harm(m, n, theta, phi):
        return sc.special.sph_harm_y(np.asarray(n, dtype=int), np.asarray(m, dtype=int), phi, theta)
    sc.special.sph_harm = _sph_harm


def jn(k,a,n):
    return sc.special.spherical_jn( n, k * a, derivative = False )    

def h1n(k,a,n):
    return sc.special.spherical_jn( n, k * a, derivative = False ) \
    + 1j * sc.special.spherical_yn( n, k * a, derivative = False )    

def Djn(k,a,n):
    return k * sc.special.spherical_jn( n, k * a, derivative = True )   

def Dh1n(k,a,n):
    return k * sc.special.spherical_jn( n, k * a, derivative = True ) \
        + k * 1j * sc.special.spherical_yn( n, k * a, derivative = True )  

#######################################
def ComputeExpansionFunction(x, y, z, kw, N):
    dist  = 1 #np.sqrt(x**2+y**2+z**2)
    n = np.arange(N)    
    # compute the expansion coefficients
    B_n = np.exp( 1j * n * np.pi / 2.0 ) * ( 2 * n + 1 ) 

    # NOTE: Djn already contains a factor k (Djn = k * j_n'), so the BIE symbol
    # 1 - i (k a)^2 j_n'(ka) h_n(ka) is written with a single extra factor kw*dist.
    # c1/c2 then reduces (via the Wronskian) to -j_n'(ka)/h_n'(ka), the Mie coefficient.
    c1 = 1j*kw*dist**2 * Djn(kw, dist, n)*jn(kw, dist, n)
    c2 = 1 - 1j*kw*dist**2 * Djn(kw, dist, n) * h1n(kw, dist, n)
    
    u_coeffs = np.sqrt(4 * np.pi/(2*n+1)) * B_n * c1/c2
    return u_coeffs
    

#######################################
def ComputeIncidentFunction(x, y, z, kw):
    # compute the incident field
    f = np.exp(1j * kw * z) 
    #dist  = 1 #np.sqrt(x**2+y**2+z**2)
    #n = np.arange(N)    
    # compute the expansion coefficients
    #B_n = np.exp( 1j * n * np.pi / 2.0 ) * ( 2 * n + 1 ) 
    #f_coeffs = B_n * jn(kw, dist, n)
    #df_coeffs = B_n * Djn(kw, dist, n)
    return f #, f_coeffs, df_coeffs
    
#######################################
def ComputeFunction(x, y, z, kw):
    x0, y0, z0 = 0.1, 0.2, 0.3

    # compute the harmonic function
    ulen = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    coeff = 0.25 * np.pi
    u = coeff * np.exp(1j * kw * ulen) / ulen

    # compute the components of its gradient
    gradu_x = -((1 / ulen - 1j * kw) * (x - x0) / ulen) * u
    gradu_y = -((1 / ulen - 1j * kw) * (y - y0) / ulen) * u
    gradu_z = -((1 / ulen - 1j * kw) * (z - z0) / ulen) * u

    return u, gradu_x, gradu_y, gradu_z
#######################################
def ComputeHarmonicFunction(x, y, z):
    x0, y0, z0 = 0.0, 0.0, 0.0

    # compute the harmonic function
    ulen = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    u = 1 / ulen

    # compute the components of its gradient
    gradu_x = - (x - x0) / ulen**3
    gradu_y = - (y - y0) / ulen**3
    gradu_z = - (z - z0) / ulen**3

    return u, gradu_x, gradu_y, gradu_z
#######################################
def SphereRotation(s, t, theta0, phi0):
    # compute the auxilliary variables
    xi = np.cos(theta0) * np.cos(phi0) * np.sin(s) * np.cos(t) - np.sin(phi0) * np.sin(s) * np.sin(t) + np.sin(theta0) * np.cos(phi0) * np.cos(s)
    eta = np.cos(theta0) * np.sin(phi0) * np.sin(s) * np.cos(t) + np.cos(phi0) * np.sin(s) * np.sin(t) + np.sin(theta0) * np.sin(phi0) * np.cos(s)
    zeta = -np.sin(theta0) * np.sin(s) * np.cos(t) + np.cos(theta0) * np.cos(s)

    # compute the rotated angles
    theta = np.arctan2(np.sqrt(xi**2 + eta**2), zeta)
    phi = np.arctan2(eta, xi)

    return theta, phi
#######################################
def GaussLegendre( N ):
    """
    This function computes the product Gauss quadrature rule for numerical integration on the unit  sphere. It is the tensor product of the Gauss-Legendre quadrature rule in the cosine of the polar angle, μ = cos θ, and the periodic trapezoid rule in the azimuthal angle, ϕ. The user inputs the order of the quadrature rule N. The output is "stretched" vectors of μ, ϕ and the quadrature weights.
    """
    # compute the Gauss-Legendre quadrature rule
    μ, wt = np.polynomial.legendre.leggauss( N )
    wwt = wt * np.pi/2
    # compute the periodic trapezoid rule
    ϕ = -np.pi + np.arange(2*N) * np.pi / N

    # stretched quadrature points and weights: μ varies FAST, ϕ varies slow.
    # CAUTION: this is the opposite ordering of grids built with
    # np.repeat(θ, 2N) / np.tile(ϕ, N); build matching weights with
    # np.repeat(wt, 2N) * np.pi / N in that case.
    MU  = np.tile( μ, 2*N )
    PHI = np.repeat( ϕ, N )
    WTS = np.tile( wt, 2*N ) * np.pi / N

    # return original μ and ϕ vectors along with stretched vectors MU and PHI and quadrature weight vector WTS ( μ = cos(𝜃), 𝜃 corresponding to the polar angle)
    return μ, ϕ, MU, PHI, WTS, wwt
#######################################
def ComputeSphericalHarmonics( N, THETA, PHI ):
    """
    This function computes a matrix whose columns correspond to spherical harmonics ordered by their order and degree in a specific way. The rows of the matrix correspond to the evaluation of each spherical harmonic over a set of cosine of polar angles and azimuthal angles. order: n degree: m
    """    
    # allocate memory for the vectors and matrices
    nvec = np.zeros( N*N, dtype = np.int32 )
    mvec = np.zeros( N*N, dtype = np.int32 )
    npts = 1 if np.isscalar(THETA) else len(np.atleast_1d(THETA))
    Ynm  = np.zeros( ( npts, N*N ), dtype = complex )
    # compute the columns of the spherical harmonics matrix
    j = 0
    for n in range(N):
        for m in range(-n,n+1):
            nvec[j] = n
            mvec[j] = m
            Ynm[:,j] = sc.special.sph_harm( m, n, PHI, THETA )
            j += 1

    return Ynm, nvec, mvec
#######################################
def ComputeSurface(theta0, phi0, s, t):
    """
    Parametrization of the unit sphere on the grid (s, t) rotated so that its
    north pole s = 0 sits at (theta0, phi0).  Returns the rotated angles
    (theta, phi), the surface points Y, the outward unit normals NU and the
    Jacobian J = |Y_s x Y_t| / sin(s)  (the sin(s) factor is applied by the
    caller together with the quadrature weights).

    For scalar (s, t) the points Y have shape (3,) and NU, J keep a leading
    singleton dimension, matching the historical interface.

    To generalize to a non-spherical, star-shaped surface, scale Y by a
    radius function R(theta, phi) here and include its derivatives in
    Y_theta / Y_phi below.
    """
    theta, phi = SphereRotation(s, t, theta0, phi0)
    scalar = np.ndim(theta) == 0
    th, ph = np.atleast_1d(theta), np.atleast_1d(phi)

    # surface point and its tangents with respect to the rotated angles
    Y = np.column_stack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
    Y_theta = np.column_stack((np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)))
    Y_phi = np.column_stack((-np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), np.zeros_like(th)))

    # unit normal and Jacobian; at the pole (theta == 0, Jvec == 0) the sphere
    # limit is NU = e_z, J = 1
    Jvec = np.cross(Y_theta, Y_phi)
    Jlen = np.linalg.norm(Jvec, axis=1)
    pole = th == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        NU = Jvec / Jlen[:, None]
        J = Jlen / np.sin(th)
    NU[pole] = [0.0, 0.0, 1.0]
    J[pole] = 1.0

    if scalar:
        return theta, phi, Y[0], NU, J
    return theta, phi, Y, NU, J

#######################################