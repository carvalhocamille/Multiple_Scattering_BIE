import time
import math 
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import mpl_toolkits.mplot3d as plt3d
import matplotlib.colors as mcolors
from ipywidgets import*
from ipywidgets import interactive
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Colors

# Primary
LYB = '#002856'
CPG = '#DAA900'

# Secondary
SSB = '#0091B3'
GW  = '#FFBF3C'
DSB = '#005487'

HDSG = '#5B5B5B'

BMG  = '#E5E5E5'
BLG  = '#EFEFEF'

# Accent

accentGreen  = '#64A43A'
accentOrange = '#F18A00'
accentBlue   = '#99D9D9'

# Colormap Function

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

# Colormaps

c = mcolors.ColorConverter().to_rgb
UCM_Blues_Dark    = make_colormap([c(HDSG), c(SSB), c(DSB), c(LYB)])
UCM_Blues_Light   = make_colormap([c(BLG), c(BMG), c(accentBlue), c(SSB), c(DSB)])
UCM_Yellows_Dark  = make_colormap([c(HDSG), c(GW), c(CPG)])
UCM_Yellows_Light = make_colormap([c(BLG), c(BMG), c(GW)])
UCM_Primary       = make_colormap([c(CPG), c(BLG), c(BMG), c(LYB)])
UCM_Secondary     = make_colormap([c(CPG), c(BLG), c(BMG), c(LYB)])
UCM_Grays_Dark    = make_colormap([c(BMG), c(HDSG)])
UCM_Grays_Light   = make_colormap([c(BLG), c(BMG)])
UCM_Combo_1       = make_colormap([c(GW), c(accentOrange), c(accentOrange), c(LYB)])
UCM_Combo_2       = make_colormap([c(LYB), c(accentGreen), c(SSB)])

UC_Colormaps = [UCM_Primary, UCM_Blues_Dark, UCM_Yellows_Dark, UCM_Grays_Dark, UCM_Combo_1,
                UCM_Secondary, UCM_Blues_Light, UCM_Yellows_Light, UCM_Grays_Light, UCM_Combo_2]
UC_Colormaps_Names = ['Primary Colors', 'Dark Blues', 'Dark Yellows', 'Dark Grays', 'Combo 1',
                'Secondary Colors', 'Light Blues', 'Light Yellows', 'Light Grays', 'Combo 2']


def jn(k,a,n):
    return sp.spherical_jn( n, k * a, derivative = False )    

def h1n(k,a,n):
    return sp.spherical_jn( n, k * a, derivative = False ) \
    + 1j * sp.spherical_yn( n, k * a, derivative = False )    

def Djn(k,a,n):
    return k * sp.spherical_jn( n, k * a, derivative = True )   

def Dh1n(k,a,n):
    return k * sp.spherical_jn( n, k * a, derivative = True ) \
        + k * 1j * sp.spherical_yn( n, k * a, derivative = True )  
    

def ComputeIncidentHarmonicExpansionCoeffs( N ):
    """
    This function computes the expansion coefficients for the field 
    scattered by a sound-hard sphere due to counter-propagating plane
    waves incident on it.
    """
    # compute an array of indices from n = 0 to n = N    
    n = np.arange( N )    
    # compute the expansion coefficients
    B_n = np.exp( 1j * n * np.pi / 2.0 ) * ( 2 * n + 1 ) 
           
    return B_n

def ComputeScatteredHarmonicExpansionCoeffsAnalytic( k, a, N ):
    """
    This function computes the expansion coefficients for the field 
    scattered by a sound-hard sphere due to counter-propagating plane
    waves incident on it. The computation is based on matching coefficients with the boundary condition.
    """
    # compute the expansion coefficients for the incident field
    B_n = ComputeIncidentHarmonicExpansionCoeffs( N )  
    
    # compute an array of indices from n = 0 to n = N   
    n = np.arange( N )
    
    # compute the expansion coefficients
    C_n = - np.sqrt( 4.0 * np.pi / ( 2 * n + 1 ) ) * Djn(k,a,n) / Dh1n(k,a,n) * B_n
    
    return C_n
    
def ComputeScatteredHarmonicExpansionCoeffsBIE( k, a, N ):
    """
    This function computes the expansion coefficients for the field 
    scattered by a sound-hard sphere due to counter-propagating plane
    waves incident on it. The computation is based on a Boundary Integral Equation.
    """
    # compute the expansion coefficients for the incident field    
    B_n = ComputeIncidentHarmonicExpansionCoeffs( N )
    
    # compute an array of indices from n = 0 to n = N   
    n = np.arange( N )
    
    # compute the expansion coefficients
    D_n = (1j * k * a ** 2 * jn(k,a,n))* Djn(k,a,n) / (1- 1j * k * a ** 2 * Djn(k,a,n) * h1n(k,a,n)) * B_n *  np.sqrt( 4.0 * np.pi / ( 2 * n + 1 ) )
    
    return D_n

def ComputeLegendrePolynomials( μ, N ):
    """ 
    This function computes the Gauss-Legendre quadrature rule of order N and evaluates 
    the Legendre polynomials at the Gauss-Legendre quadrature points up to degree N-1.
    
    This function returns the Gauss-Legendre quadrature points mu, quadrature weights wt,
    and a matrix whose columns are the Legendre polynomials evaluated on mu.
    """
    
    # allocate memory for the matrix of Legendre polynomials
    Pn = np.full( ( μ.size, N ), 'nan', dtype = 'complex' )
    
    # compute the Legendre polynomial of degree 0    
    Pn[:,0] = 1
    
    # compute the Legendre polynomial of degree 1   
    Pn[:,1] = μ

    # compute the remaining Legendre polynomials using the recursion relation    
    for n in range( 1, N - 1 ):   
        Pn[:,n+1] = ( ( 2 * n + 1 ) * μ * Pn[:,n] - n * Pn[:,n-1] ) / ( n + 1 )
        
    return Pn

def ComputeLegendrePolynomialDerivatives( μ, N ):
    """ 
    This function computes the derivatives of the Legendre polynomials.
    """
    
    # allocate memory for the matrix of Legendre polynomials

    DPn = np.full( ( μ.size, N ), 'nan', dtype = 'complex' )
    
    # compute the derivatives of the Legendre polynomials
    
    DPn[:,0] = 0;
    DPn[:,1] = 1;
    
    # compute the current Legendre polynomial of degree n = 2
    
    P_nminus1 = μ
    P_n = 1.5 * μ ** 2 - 0.5
    
    for n in range( 2, N ):
        
        # compute the Legendre polynomial of degree n+1
        
        P_nplus1 = ( ( 2 * n + 1 ) * μ * P_n - n * P_nminus1 ) / ( n + 1 )
        
        # compute the derivative of the Legendre polynomial
        
        DPn[:,n] = ( n + 1 ) / ( μ ** 2 - 1.0 ) * ( P_nplus1 - μ * P_n )
        
        # update the Legendre polynomials
        
        P_nminus1 = P_n
        P_n       = P_nplus1
        
    return DPn;

def ComputeForce( k, a, ρ0, κ0, ω, N, B_n, C_n ):
    """
    This function computes the radiation force using the harmonic expansion for the
    scattered scalar potential φ1.
    """
    # compute the Gauss-Legendre quadrature rule

    μ, w = np.polynomial.legendre.leggauss( N )

    # compute the Legendre polynomials

    P_n = ComputeLegendrePolynomials( μ, N )

    # compute array of indices

    n = np.arange( N )

    # compute the fields on the sphere using the harmonic expansions

    φ1_inc = P_n @ ( jn(k,a,n) * B_n )
    φ1_s   = P_n @ ( np.sqrt( ( 2 * n + 1 ) / ( 4.0 * np.pi ) ) * h1n(k,a,n) * C_n )

    # compute the derivatives of the Legendre polynomials

    DP_n = ComputeLegendrePolynomialDerivatives( μ, N )

    # compute θ-derivative of the incident field

    φ1_inc_θ = -np.sqrt( 1.0 - μ ** 2 ) * ( DP_n @ ( jn(k,a,n) * B_n ) )

    # compute θ-derivative of the scattered field

    φ1_s_θ = -np.sqrt( 1.0 - μ ** 2 ) * ( DP_n @ ( np.sqrt( ( 2 * n + 1 ) / ( 4.0 * np.pi ) ) * h1n(k,a,n) * C_n ) )

    # compute the time-average of the pressure

    p1_ave = 0.5 * np.abs( 1j * ρ0 * ω * ( φ1_inc + φ1_s ) ) ** 2

    # compute the time-average of the velocity

    v1_ave = 0.5 * np.abs( φ1_inc_θ / a + φ1_s_θ / a ) ** 2

    # compute the function to be integrated

    Fintegrand = -( 0.5 * κ0 * p1_ave - 0.5 * ρ0 * v1_ave ) * μ
    
    # compute the z-component of the force

    Fz = 2.0 * np.pi * a ** 2 * np.sum( Fintegrand * w )
    
    return Fz;

# Sphere Functions

def Cart2Sphere(x0, y0, z0):
    
    r = np.sqrt(x0**2 + y0**2 + z0**2)
    𝜃 = np.arccos(z0 / r)                   # Polar angle: 0 ≤ 𝜃 ≤ 𝜋
    𝜑 = np.arctan2(y0, x0)                  # Azimuthal angle: 0 ≤ 𝜑 ≤ 2𝜋
    
    return r, 𝜃, 𝜑

def Sphere2Cart(r0, 𝜃0, 𝜑0):
  
    x = r0 * np.sin(𝜃0) * np.cos(𝜑0)
    y = r0 * np.sin(𝜃0) * np.sin(𝜑0)
    z = r0 * np.cos(𝜃0)
    
    return x, y, z  

def SphereRotation( s, t, theta0, phi0 ):
# This function computes the rotation from the (s,t) angles on the unit
# sphere to the (theta,phi) angles on the unit sphere. See Appendix A of
# Carvalho, Khatri, and Kim (2020) for the mathematical details.

# Following implementation in Matlab by A. D. Kim 

    #compute the auxilliary variables

    xi   = ( np.cos( theta0 ) * np.cos( phi0 ) * np.sin( s ) * np.cos( t ) 
            - np.sin( phi0 ) * np.sin( s ) * np.sin( t ) 
            + np.sin( theta0 ) * np.cos( phi0 ) * np.cos( s ) )

    eta  = ( np.cos( theta0 ) * np.sin( phi0 ) * np.sin( s ) * np.cos( t ) 
            + np.cos( phi0 ) * np.sin( s ) * np.sin( t ) 
            + np.sin( theta0 ) * np.sin( phi0 ) * np.cos( s ) )

    zeta = - np.sin( theta0 ) * np.sin( s ) * np.cos( t ) + np.cos( theta0 ) * np.cos( s )

    #compute the rotated angles

    theta = np.arctan2( np.sqrt( xi**2 + eta**2 ), zeta )
    phi   = np.arctan2( eta, xi );

    return theta, phi

def TranslateSphere( a, theta, phi, x1, y1, z1 ):
    """
    This function computes the set of points for a sphere that is translated from a sphere centered
    at the origin of radius 'a' to a sphere centered at (x1,y1,z1)
    
    """
    
    # transfrom spherical corrdinates to cartesian
    x, y, z = Sphere2Cart(a, theta, phi)
    
    # shift cartestian coordinates to new points
    xnew = x + x1
    ynew = y + y1
    znew = z + z1
    
    # transform back to spherical coordinates
    r, theta, phi = Cart2Sphere(xnew, ynew, znew)
    
    return r, theta, phi

# Integration Functions.


#function to compute Gauss-Legendre nodes and weights
# Compute the Gauss Legendre quadrature points and weights using
# the method given in Spectral Methods in MATLAB by L. N. Trefethen (2000).
#
# Following implementation in Matlab by A. D. Kim.

def GaussLegendre(N):
    #N: number of nodes 
    
    beta = np.zeros(N-1)
    
    for i in range(1, N):
        beta[i-1]  = 0.5 / np.sqrt( 1.0 - (1 / ( 2.0 * i )**2) );
        
    T = np.diag(beta, k=1) + np.diag(beta, k=-1) 
    x, v = np.linalg.eig(T)
    i = np.argsort(x)
    x = x[i]
    w = 2.0 * v[0,i]**2
    
    return x, w
    
def SphereSurfaceEvaluation(O_1, O_2, theta, phi, radius, Reference):
    """
    This function maps the center of a sphere centered at the point O_2 = (x2, y2, z2)
    to a point on the surface of another sphere centered at O_1 = (x1, y1, z1) by creating
    a "ghost" sphere using symmetric properties of spherical coordinates.
    
    """
    
    # Center of sphere 1 and 2
    X1 = np.array([O_1[0], O_1[1], O_1[2]])
    X2 = np.array([O_2[0], O_2[1], O_2[2]])
    
    # Evaluation point
    xE, yE, zE = Sphere2Cart(radius, theta, phi)
    XE = np.array([xE, yE, zE])
    
    # Maps origin of sphere 1 to points on sphere 2 for a given theta, phi
    if( Reference == 1 ):   

        MapCart = X2 - X1 + XE
        MapSph  = Cart2Sphere(MapCart[0], MapCart[1], MapCart[2])
        
    # Maps origin of sphere 2 to points on sphere 1 for given theta, phi
    elif( Reference == 2 ):

        MapCart = X1 - X2 + XE
        MapSph  = Cart2Sphere(MapCart[0], MapCart[1], MapCart[2])
                                     
    return MapCart, MapSph

# Wrapper for spherical Bessel function to include negative order

def jl(order, x):
    return np.sqrt(np.pi / (2. * x) ) * sp.jv(order+0.5, x)

def jlPrime(order,x):
    return jl(order-1,x) - ( (order+1)/x ) * jl(order,x)

def spherical_hankel(order, z):
    return sp.spherical_jn(order, z) + 1j * sp.spherical_yn(order, z) 


# function for finding nearest index

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx



















#### Functions added Summer 2022 ################################################################################################################################################################################

    # function for computing quadrature points 
def ComputeProductGaussQd( N ):
    """
    This function computes the product Gauss quadrature rule for numerical integration on the unit  sphere. It is the tensor product of the Gauss-Legendre quadrature rule in the cosine of the polar angle, μ = cos θ, and the periodic trapezoid rule in the azimuthal angle, ϕ. The user inputs the order of the quadrature rule N. The output is "stretched" vectors of μ, ϕ and the quadrature weights.
    """

    # compute the Gauss-Legendre quadrature rule

    μ, wt = np.polynomial.legendre.leggauss( N )

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
    return μ, ϕ, MU, PHI, WTS;


    # function for computing spherical harmonics matrix
def ComputeSphericalHarmonics( N, MU, PHI ):
    """
    This function computes a matrix whose columns correspond to spherical harmonics ordered by their order and degree in a specific way. The rows of the matrix correspond to the evaluation of each spherical harmonic over a set of cosine of polar angles and azimuthal angles. order: n degree: m
    """

    
    # allocate memory for the vectors and matrices

    nvec = np.full( N*N, '0', dtype = np.int32 )
    mvec = np.full( N*N, '0', dtype = np.int32 )
    Ynm  = np.full( ( len(MU), N*N ), 'nan', dtype = 'complex' )
    
    #Ynm = 1j * 0. * MU
    
    # compute the columns of the spherical harmonics matrix

    j = 0

    for n in range(N):

        for m in range(-n,n+1):

            # compute nvec and mvec entries

            nvec[j] = n
            mvec[j] = m

            # compute the column of Ynm

            Ynm[:,j] = sp.sph_harm( m, n, PHI, np.arccos( MU )  )

            j += 1
            
    return Ynm, nvec, mvec;

    # function for computing points on spheres
def ComputePointsOnSpheres( C1, C2, N, a1, a2):
    """
    This function computes Gaussian Product quadrature points on the surface of two spheres
    centered at C1 and C2.
    """
    # compute quadrature rule
    mu, phi, Mu, Phi, Wts = ComputeProductGaussQd( N )
    
    # compute normals for unit sphere
    nx = np.sqrt( 1.0 - Mu ** 2 ) * np.cos( Phi )
    ny = np.sqrt( 1.0 - Mu ** 2 ) * np.sin( Phi )
    nz = Mu
    
    # compute shift (sphere 1 to sphere 2)
    ux1 = C2[0] + a1 * nx - C1[0]
    uy1 = C2[1] + a1 * ny - C1[1]
    uz1 = C2[2] + a1 * nz - C1[2]
    
    # compute shift (sphere 2 to sphere 1)
    ux2 = C1[0] + a2 * nx - C2[0]
    uy2 = C1[1] + a2 * ny - C2[1]
    uz2 = C1[2] + a2 * nz - C2[2]
    
    # allocate storage for R, S, T, X, Y, Z
    RST12 = np.zeros( [ 3, 2 * N ** 2 ] )
    XYZ12 = np.zeros( [ 3, 2 * N ** 2 ] )
    
    RST21 = np.zeros( [ 3, 2 * N ** 2 ] )
    XYZ21 = np.zeros( [ 3, 2 * N ** 2 ] )
    
    
    # compute R, S, T (1 to 2)
    RST12[0] = np.sqrt( ux1 ** 2 + uy1 ** 2 + uz1 ** 2 )
    RST12[1] = uz1 / RST12[0]                            # This is the cosine of the polar angle 
    RST12[2] = np.arctan2( uy1, ux1)                     # This is the azimuthal angle
    
    # compute R, S, T (2 to 1)
    RST21[0] = np.sqrt( ux2 ** 2 + uy2 ** 2 + uz2 ** 2 )
    RST21[1] = uz2 / RST21[0]                            # This is the cosine of the polar angle 
    RST21[2] = np.arctan2( uy2, ux2)                     # This is the azimuthal angle
    
    # store cartesian coordinates for both
    XYZ12[0], XYZ12[1], XYZ12[2] = ux1, uy1, uz1
    XYZ21[0], XYZ21[1], XYZ21[2] = ux2, uy2, uz2  
    
    return XYZ12, RST12, XYZ21, RST21;    


    # function for computing double layer potentials
def ComputeDoubleLayerPotentials( C1, C2, N, k, a1, a2):
    """
    This function computes the double layer potentials for the off-diagonal blocks of
    the system (for the left hand side) given the centers of the spheres and the number 
    of quadrature points. 
    """
    # compute points on spheres
    Cart1, Sph1, Cart2, Sph2 = ComputePointsOnSpheres( C1, C2, N, a1, a2)
    
    # compute spherical harmonics matrices
    Y12, mVec, nVec = ComputeSphericalHarmonics( N, Sph1[1], Sph1[2] )
    Y21, mVec, nVec = ComputeSphericalHarmonics( N, Sph2[1], Sph2[2] )
    
    # allocate storage for spectrum matrices
    HMAT12 = np.zeros([ 2 * N **2, len(mVec) ], dtype = complex)
    HMAT21 = np.zeros([ 2 * N **2, len(mVec) ], dtype = complex)
    
    # compute spectrum for 1,2 block and 2, 1 block
    for i in range( len(mVec) ):
        
        HMAT12[:,i] =  1j * ( k * a1 ) ** 2 * Djn( k, a1, mVec[i] ) * spherical_hankel( mVec[i], Sph1[0] )
        HMAT21[:,i] =  1j * ( k * a2 ) ** 2 * Djn( k, a2, mVec[i] ) * spherical_hankel( mVec[i], Sph2[0] )
        
    
    # combine spectrum and spherical harmonic matrices component wise
    
    DLP12 = HMAT12 * Y12
    DLP21 = HMAT21 * Y21
    
    return DLP12, DLP21;

    # function to compute diagonal blocks of LHS
def ComputeDiagonalBlocks( C1, C2, N, k, a1, a2):
    """
    This function computes the diagonal blocks of the system given
    the centers of the spheres and the number of quadrature points. 
    """
    
    # compute quadrature points: μ, ϕ, MU, PHI, WTS
    QUAD = ComputeProductGaussQd( N )
    
    # compute spherical harmonic matrices: Ynm, nvec, mvec
    YMN = ComputeSphericalHarmonics( N, QUAD[2], QUAD[3] )
    
    LM1 = 1.0 - 1j * (k*a1)**2 * Djn( k, a1, YMN[1] ) * h1n( k, a1, YMN[1] )
    LM2 = 1.0 - 1j * (k*a2)**2 * Djn( k, a2, YMN[1] ) * h1n( k, a2, YMN[1] )
    
    return np.diag(LM1), np.diag(LM2);
        
    # function to compute off-diagonal blocks
def ComputeOffDiagonalBlocksLHS( C1, C2, N, k, a1, a2):
    """
    This function computes  the off-diagonal blocks of the system (for the left hand side)
    given the centers of the spheres and the number of quadrature points. 
    """
    
    # compute quadrature points: μ, ϕ, MU, PHI, WTS
    QUAD = ComputeProductGaussQd( N )
    
    # compute spherical harmonic matrices: Ynm, nvec, mvec
    YMN = ComputeSphericalHarmonics( N, QUAD[2], QUAD[3] )
    
    # compute double-layer potential matrices: DLP12, DLP21
    DLP = ComputeDoubleLayerPotentials( C1, C2, N, k, a1, a2)
    
    # compute projections
    #
    # NOTE: You label DLP12 as the DLP from sphere 1 to sphere 2 and DLP21 as the DLP from sphere 2 to sphere 1
    # Thus, the (1,2) block of the system is DLP21 and the (2,1) block of the system is DLP12.
    
    Lambda21 = - np.conj(YMN[0]).T @ np.diag(QUAD[4]) @ DLP[0]
    
    Lambda12 = - np.conj(YMN[0]).T @ np.diag(QUAD[4]) @ DLP[1]

    return Lambda12, Lambda21;
    
     # function to assemble LHS of system
def AssembleLHS( C1, C2, N, k, a1, a2):
    
    # compute diagonal blocks
    LDiag = ComputeDiagonalBlocks( C1, C2, N, k, a1, a2)
    
    # compute off-diagonal blocks: Lambda12, Lambda21
    LOffDiag = ComputeOffDiagonalBlocksLHS( C1, C2, N, k, a1, a2)
    
    return np.block([ [ LDiag[0], LOffDiag[0] ], [ LOffDiag[1], LDiag[1] ] ] )

   
    # function to compute RHS
def ComputeRHS( N, k, a1, a2): 
    # compute quadrature pts and weights: μ, ϕ, MU, PHI, WTS
    QUAD = ComputeProductGaussQd( N )
    
    # compute spherical harmonic matrices: Ynm, nvec, mvec
    YMN = ComputeSphericalHarmonics( N, QUAD[2], QUAD[3] )
    
    # compute 
    B1  = np.exp( 1j * k * a1 * QUAD[2] )
    B2  = np.exp( 1j * k * a2 * QUAD[2] )
    
    b1  = np.conj( YMN[0] ).T @ np.diag( QUAD[4] ) @ B1
    b2  = np.conj( YMN[0] ).T @ np.diag( QUAD[4] ) @ B2
    
    return np.block([  b1 ,  b2  ] );    


def VectorEvaluateRepresentationFormula( XYZMesh, cntr1, cntr2, N, k, a1, a2, Coeff1, Coeff2 ):
    """
    This function computes the representation formula at a point PEval in the domain exterior to spheres
    """
    
    X1p = XYZMesh[0] - cntr1[0]
    Y1p = XYZMesh[1] - cntr1[1]
    Z1p = XYZMesh[2] - cntr1[2]
    
    X2p = XYZMesh[0] - cntr2[0]
    Y2p = XYZMesh[1] - cntr2[1]
    Z2p = XYZMesh[2] - cntr2[2]

    X1pVec = np.reshape( X1p, ( len( X1p[:,0] ) * len( X1p[0,:] ), 1 )  )
    Y1pVec = np.reshape( Y1p, ( len( Y1p[:,0] ) * len( Y1p[0,:] ), 1 )  )
    Z1pVec = np.reshape( Z1p, ( len( Z1p[:,0] ) * len( Z1p[0,:] ), 1 )  )

    X2pVec = np.reshape( X2p, ( len( X2p[:,0] ) * len( X2p[0,:] ), 1 )  )
    Y2pVec = np.reshape( Y2p, ( len( Y2p[:,0] ) * len( Y2p[0,:] ), 1 )  )
    Z2pVec = np.reshape( Z2p, ( len( Z2p[:,0] ) * len( Z2p[0,:] ), 1 )  )


    RST1p = Cart2Sphere(X1pVec, Y1pVec, Z1pVec)
    RST2p = Cart2Sphere(X2pVec, Y2pVec, Z2pVec)


    HMAT12 = np.zeros([ len(RST1p[0]), N ** 2 ], dtype = complex)
    HMAT21 = np.zeros([ len(RST2p[0]), N ** 2 ], dtype = complex)

    YNM1p  = np.zeros([ len(RST1p[0]), N ** 2 ], dtype = complex)
    YNM2p  = np.zeros([ len(RST1p[0]), N ** 2 ], dtype = complex)

    j = 0

    for n in range(N):

        for m in range(-n,n+1):
            
            HMAT12[:,j] = 1j * ( k * a1 ) ** 2 * jn( k, a1, n ) * spherical_hankel( n, RST1p[0] ) . T
            HMAT21[:,j] = 1j * ( k * a2 ) ** 2 * jn( k, a2, n ) * spherical_hankel( n, RST2p[0] ) . T
            
            # compute the column of Ynm

            YNM1p[:,j] = sp.sph_harm( m, n, RST1p[2], RST1p[1]  ) . T
            YNM2p[:,j] = sp.sph_harm( m, n, RST2p[2], RST2p[1]  ) . T

            j += 1
    
    Sph1 = (HMAT12 * YNM1p) @ Coeff1
    Sph2 = (HMAT21 * YNM2p) @ Coeff2

    return np.reshape( Sph1 + Sph2 , ( len( XYZMesh[0][:,0] ), len( XYZMesh[0][0,:] ) ) )    


def ComputeSphericalHarmonicsDerivative( N, MU, PHI ):
    """
    This function computes a matrix for each of the partial derivaives of the spherical harmonic function.
    Tht matrix columns correspond to spherical harmonics ordered by their order and degree in a specific way. 
    The rows of the matrix correspond to the evaluation of each spherical harmonic over a set of cosine of 
    polar angles and azimuthal angles. order: n degree: m
    """

    
    # allocate memory for the vectors and matrices

    nvec        = np.full( N*N, '0', dtype = np.int32 )
    mvec        = np.full( N*N, '0', dtype = np.int32 )
    DPhi_Ynm    = np.full( ( len(MU), N*N ), 'nan', dtype = 'complex' )
    DTheta_Ynm  = np.full( ( len(MU), N*N ), 'nan', dtype = 'complex' )

    
    #Ynm = 1j * 0. * MU
    
    # compute the columns of the spherical harmonics matrix

    j = 0

    for n in range(N):

        for m in range(-n,n+1):

            # compute nvec and mvec entries

            nvec[j] = n
            mvec[j] = m

            # compute the column of Ynm

            DPhi_Ynm[:,j] = 1j * m * sp.sph_harm( m, n, PHI, np.arccos( MU )  )
        
            if(m+1 <= n):
                DTheta_Ynm[:,j] = m * sp.sph_harm( m, n, PHI, np.arccos( MU )  ) / np.tan( np.arccos( MU ) ) + np.sqrt( ( n - m) * (n + m + 1) ) * np.exp( - 1j * PHI ) * sp.sph_harm( m + 1, n, PHI, np.arccos( MU )  )
            else:
                DTheta_Ynm[:,j] = m * sp.sph_harm( m, n, PHI, np.arccos( MU )  ) / np.tan( np.arccos( MU ) ) 


            j += 1
            
    return DPhi_Ynm, DTheta_Ynm, nvec, mvec;