import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.special import legendre, factorial
import collections.abc


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

    c1 = 1j*kw*dist * Djn(kw, dist, n)*jn(kw, dist, n) 
    c2 = 1 - 1j*(kw*dist)**2 * Djn(kw, dist, n) * h1n(kw, dist, n)
    
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
    # meshgrid of indices
    [ indx, jndx ] = np.meshgrid( np.arange(N), np.arange(2*N) )

    # stretch indices
    indx = indx.reshape(2*N*N)
    jndx = jndx.reshape(2*N*N)

    # compute the quadrature points and weights
    MU  = μ[indx]
    PHI = ϕ[jndx]
    WTS = wt[indx] * np.pi / N
    
    # return original μ and ϕ vectors along with stretched vectors MU and PHI and quadrature weight vector WTS ( μ = cos(𝜃), 𝜃 corresponding to the polar angle) 
    return μ, ϕ, MU, PHI, WTS, wwt
#######################################
def ComputeSphericalHarmonics( N, THETA, PHI ):
    """
    This function computes a matrix whose columns correspond to spherical harmonics ordered by their order and degree in a specific way. The rows of the matrix correspond to the evaluation of each spherical harmonic over a set of cosine of polar angles and azimuthal angles. order: n degree: m
    """    
    # allocate memory for the vectors and matrices
    nvec = np.full( N*N, '0', dtype = np.int32 )
    mvec = np.full( N*N, '0', dtype = np.int32 )
    if np.isscalar(THETA)==False:
        Ynm  = np.full( ( len(THETA), N*N ), 'nan', dtype = 'complex' )
    else:
        Ynm  = np.full( ( 1, N*N ), 'nan', dtype = 'complex' )   
    # compute the columns of the spherical harmonics matrix
    j = 0

    for n in range(N):
        for m in range(-n,n+1):
            # compute nvec and mvec entries
            nvec[j] = n
            mvec[j] = m
            # compute the column of Ynm
            Ynm[:,j] = sc.special.sph_harm( m, n, PHI, THETA  )
            j += 1
            
    return Ynm, nvec, mvec
#######################################
def ComputeSurface(theta0, phi0, s, t):
    # Call SphereRotation to get rotated angles
    theta, phi = SphereRotation(s, t, theta0, phi0)
    #print('theta', theta, 'phi', phi )
    # Compute the vector on the unit sphere and its partial derivatives
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    x_theta = np.cos(theta) * np.cos(phi)
    y_theta = np.cos(theta) * np.sin(phi)
    z_theta = -np.sin(theta)
    
    x_phi = -np.sin(theta) * np.sin(phi)
    y_phi = np.sin(theta) * np.cos(phi)
    z_phi = np.zeros_like(z)

    # Compute the radius function and its partial derivatives
    R = np.ones_like(x)
    R_x = np.zeros_like(R)
    R_y = np.zeros_like(R)
    R_z = np.zeros_like(R)

    # Compute the surface vector and its partial derivatives
    A, B, C = 1, 1, 1

    Y = np.array([A * R * x, B * R * y, C * R * z]).T
    Y_x = np.array([A * R + A* R_x *x,  B*R_z*y,  C*R_x*z]).T
    Y_y = np.array([ A* R_y *x, B * R + B* R_y *y, + C* R_y *z]).T
    Y_z = np.array([ A* R_z *x, + B* R_z *y, C * R + C* R_z *z]).T

    # ADJUST WHEN len  = 0 for scalar 
    if np.isscalar(theta)==False:
        xt = np.reshape(x_theta, (1, len(x_theta)))
        yt = np.reshape(y_theta, (1, len(y_theta)))
        zt = np.reshape(z_theta, (1, len(z_theta)))

        xp = np.reshape(x_phi, (1, len(x_phi)))
        yp = np.reshape(y_phi, (1, len(y_phi)))
        zp = np.reshape(z_phi, (1, len(z_phi)))

        at = xt * Y_x[:,0] + yt * Y_y[:,0] + zt* Y_z[:,0]
        bt = xt*Y_x[:,1] + yt * Y_y[:,1] + zt * Y_z[:,1]
        ct = xt * Y_x[:,2] + yt * Y_y[:,2] + zt * Y_z[:,2]
        Y_theta = np.vstack((at,bt,ct)).T

        ap = xp * Y_x[:,0] + yp * Y_y[:,0] + zp* Y_z[:,0]
        bp = xp*Y_x[:,1] + yp * Y_y[:,1] + zp * Y_z[:,1]
        cp = xp * Y_x[:,2] + yp * Y_y[:,2] + zp * Y_z[:,2]
        Y_phi = np.vstack((ap,bp,cp)).T

    else:
        xt = x_theta
        yt = y_theta
        zt = z_theta
        xp = x_phi
        yp = y_phi
        zp = z_phi

        at = xt * Y_x[0] + yt * Y_y[0] + zt* Y_z[0]
        bt = xt*Y_x[1] + yt * Y_y[1] + zt * Y_z[1]
        ct = xt * Y_x[2] + yt * Y_y[2] + zt * Y_z[2]
        Y_theta = np.vstack((at,bt,ct)).T

        ap = xp * Y_x[0] + yp * Y_y[0] + zp* Y_z[0]
        bp = xp*Y_x[1] + yp * Y_y[1] + zp * Y_z[1]
        cp = xp * Y_x[2] + yp * Y_y[2] + zp * Y_z[2]
        Y_phi = np.vstack((ap,bp,cp)).T

    # Compute the unit normal vectors
    Jvec = np.cross(Y_theta, Y_phi)
    Jlen = np.linalg.norm(Jvec, axis=1)
    NU = np.vstack((np.divide(Jvec[:,0], Jlen), np.divide(Jvec[:,1], Jlen), np.divide(Jvec[:,2], Jlen))).T
    # Compute the Jacobian
    J = Jlen / np.sin(theta)
    
    # Special case: When theta == 0
    indx = np.where(theta == 0)[0]

    if len(indx) > 0:
        NU[indx] = [0, 0, 1]
        J[indx] = np.sqrt(( -B * C * R[indx] * R_x[indx])**2 + ( -A * C * R[indx] * R_y[indx])**2 + (A * B * R[indx] * R[indx])**2)

    return theta, phi, Y, NU, J

#######################################