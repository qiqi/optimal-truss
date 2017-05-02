'''
This module implements FEM in 2D with a Cartesian array of square elements.
It computes the linear stiffness matrix and geometric stiffness matrix.
Every element is a square of size 2x2. The elastic modulus of the elements
is provided by an input array.

Example:
    from numpy import *
    from cartfem2d import FEM2D

    nx, ny = 10, 2
    fem = FEM2D(0.3, nx, ny)  # Poisson ratio and rectangular domain shape
    E = 1E6 * ones([ny, nx])    # elastic modulus array of shape [ny,nx]
    K = fem.K_matrix(E)         # linear stiffness matrix
    dxy = fem.xy * 1E-6         # deformation
    assert dxy.shape == (ny+1, nx+1, 2)
    G = fem.G_matrix(E, dxy)    # geometric stiffness matrix

Qiqi Wang, May 2017
'''

from __future__ import division

from scipy.sparse import *
import scipy.sparse.linalg as splinalg
from numpy import *
from numpy.polynomial import legendre

def _ddx(x, y):
    'x derivative of shape function'
    return array([-(1-y), (1-y), (1+y), -(1+y)]) / 4

def _ddy(x, y):
    'y derivative of shape function'
    return array([-(1-x), -(1+x), (1+x), (1-x)]) / 4

def _Axx(x, y):
    '''linear component in xx stress field.
    sigma_xx = dot(_Axx(x,y), U)
    where U=[u00 v00 u10 v10 u11 v11 u01 v01]
    '''
    return ravel(transpose([_ddx(x, y), zeros(4)]))

def _Ayy(x, y):
    'linear component in yy stress field.  See doc for _Axx'
    return ravel(transpose([zeros(4), _ddy(x, y)]))

def _Axy(x, y):
    'linear component in xy stress field.  See doc for _Axx'
    return ravel(transpose([_ddy(x, y), _ddx(x, y)]))

def _Bxx(x, y):
    '''Quadratic component in xx stress field.
    sigma_xx = dot(_Axx(x,y), U) + dot(dot(_Bxx(x,y), U), U)
    where U=[u00 v00 u10 v10 u11 v11 u01 v01]
    '''
    e = _ddx(x, y)
    z = zeros(4)
    eu = ravel(transpose([e, z]))
    ev = ravel(transpose([z, e]))
    return (outer(eu, eu) + outer(ev, ev)) / 2

def _Byy(x, y):
    'Quadratic component in yy stress field.  See doc for _Bxx'
    e = _ddy(x, y)
    z = zeros(4)
    eu = ravel(transpose([e, z]))
    ev = ravel(transpose([z, e]))
    return (outer(eu, eu) + outer(ev, ev)) / 2

def _Bxy(x, y):
    'Quadratic component in xy stress field.  See doc for _Bxx'
    e0 = _ddx(x, y)
    e1 = _ddy(x, y)
    z = zeros(4)
    e0u = ravel(transpose([e0, z]))
    e0v = ravel(transpose([z, e0]))
    e1u = ravel(transpose([e1, z]))
    e1v = ravel(transpose([z, e1]))
    return outer(e0u, e1u) + outer(e0v, e1v)

def element_matrices(nu):
    '''
    Quadratic and cubic components of the elastic energy. Example:
        Q, C = element_matrix(0.3) # Poisson ratio
        UE_times_2 = dot(dot(Q, U), U) + dot(dot(dot(C, U), U), U)
    where U=[u00 v00 u10 v10 u11 v11 u01 v01]
    There is a quatic component but not computed here.
    '''
    n = 4
    xg, wg = legendre.leggauss(n)
    xg, yg = meshgrid(xg, xg)
    wg = outer(wg, wg)
    Q0 = zeros([2*n, 2*n])
    Q1 = zeros([2*n, 2*n])
    C0 = zeros([2*n, 2*n, 2*n])
    C1 = zeros([2*n, 2*n, 2*n])
    for i in range(n):
        for j in range(n):
            axx = _Axx(xg[i,j], yg[i,j])
            ayy = _Ayy(xg[i,j], yg[i,j])
            axy = _Axy(xg[i,j], yg[i,j])
            Q0 += wg[i,j] * (outer(axx, axx) + outer(ayy, ayy)
                             + 1/2 * outer(axy, axy))
            Q1 += wg[i,j] * (outer(axx, ayy) + outer(ayy, axx)
                             - 1/2 * outer(axy, axy))
            bxx = _Bxx(xg[i,j], yg[i,j])
            byy = _Byy(xg[i,j], yg[i,j])
            bxy = _Bxy(xg[i,j], yg[i,j])
            C0 += wg[i,j] * (2 * axx * bxx[:,:,newaxis]
                    + 2 * ayy * byy[:,:,newaxis] + axy * bxy[:,:,newaxis])
            C1 += wg[i,j] * (2 * axx * byy[:,:,newaxis]
                    + 2 * ayy * bxx[:,:,newaxis] - axy * bxy[:,:,newaxis])
    premul = 1 / (1 - nu**2)
    return premul * (Q0 + nu * Q1), premul * (C0 + nu * C1)

class FEM2D:
    '''
    The main interface of this module. See module doc for usage.
    '''
    def __init__(self, nu, nelx, nely):
        '''
        nu: Poisson ratio
        shape: (nelx,nely) Note each direction has one less element than nodes
        '''
        self.Q, C = element_matrices(nu)
        self.C = (transpose(C, [0,1,2])
                + transpose(C, [0,2,1])
                + transpose(C, [1,2,0])
                + transpose(C, [1,0,2])
                + transpose(C, [2,0,1])
                + transpose(C, [2,1,0])) / 2
        # the following is a Python adaptation of what's in top88.m
        nodenrs = reshape(arange((1+nelx)*(1+nely)), [1+nely,1+nelx])
        edofVec = ravel(2 * nodenrs[:-1,:-1])
        self.edofMat = edofVec[:,newaxis] + \
                ravel(array([nelx+1, nelx+2, 1, 0])[:,newaxis] * 2 + arange(2))
        self.iK = ravel(kron(self.edofMat, ones([8,1])))
        self.jK = ravel(kron(self.edofMat, ones([1,8])))
        self.nelx, self.nely = nelx, nely

    @property
    def xy(self):
        'return (nx+1, ny+1, 2) array. xy[:,:,0] is x; xy[:,:,1] is y.'
        x, y = meshgrid(arange(self.nelx+1), nely - arange(self.nely+1))
        return 2 * transpose([x, y], [1,2,0])

    def K_matrix(self, E):
        '''
        Linear stiffness matrix, square of size 2(nelx+1)(nely+1)
        Matrix is singular. Slice it before inverting (rhs can be force).
        '''
        assert E.shape == (self.nely, self.nelx)
        K = csr_matrix((kron(ravel(E), ravel(self.Q)), (self.iK, self.jK)))
        return (K + K.T) / 2

    def G_matrix(self, E, U):
        '''
        Geometric stiffness matrix, square of size 2(nelx+1)(nely+1)
        '''
        assert E.shape == (self.nely, self.nelx)
        U = ravel(U)
        G = ravel(E)[:,newaxis,newaxis] * dot(U[self.edofMat], self.C)
        G = csr_matrix((ravel(G), (self.iK, self.jK)))
        return (G + G.T) / 2

# ------------------------ tests --------------------- #

def test_matrices():
    'Test integrity of matrices and their agreement with top88 code'
    Q0, C0 = element_matrices(0)
    Q1, C1 = element_matrices(0.5)
    assert(abs(Q0 * 24 - around(Q0 * 24)).sum() < 1E-12)
    assert(abs(Q1 * 36 - around(Q1 * 36)).sum() < 1E-12)
    assert(abs(C0 * 24 - around(C0 * 24)).sum() < 1E-12)
    assert(abs(C1 * 36 - around(C1 * 36)).sum() < 1E-12)
    # copied from top88
    top88 = '''12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12
               -6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6
               -4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4
                2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2'''
    top88 = array([[row.split() for row in line.split(';')]
                   for line in top88.splitlines()], float)
    top88_Q0 = vstack([hstack(top88[:2]), hstack(top88[1::-1])])
    top88_Q1 = vstack([hstack(top88[2:]), hstack(top88[:1:-1])])
    assert (around(Q0 * 24) == top88_Q0).all()
    assert (around(Q1 * 36 - Q0 * 48) == top88_Q1).all()

def test_rotation_under_compression(nu):
    'Test energy of a single element under compression and rotation'
    Q, C = element_matrices(nu)
    x = array([-1, +1, +1, -1])
    y = array([-1, -1, +1, +1])
    theta = 0.1
    UE, dUE = [], []
    thetas = linspace(-0.01, 0.01, 100)
    for theta in thetas:
        u =  y * theta - 1E-3 * x
        v = -x * theta
        U = ravel(transpose([u, v]))
        UE.append(dot(dot(Q, U), U))
        dUE.append(dot(dot(dot(C, U), U), U))
    UE, dUE = array([UE, dUE])
    # energy according to linear elasticity is constant
    assert(UE.max() - UE.min() < 1E-12)
    # energy accounting for geometric nonlienarity has negative curvature
    assert(dUE.max() - dUE[dUE.size//2] < 1E-12)
    assert(dUE.min() - dUE[0] > -1E-12)
    assert(dUE.min() - dUE[-1] > -1E-12)
    # import pylab
    # pylab.figure()
    # pylab.plot(thetas, UE)
    # pylab.plot(thetas, UE + dUE)
    # pylab.legend(['Linear', 'Geometrically nonlinear'])
    # pylab.title('Rotation under compression')
    # pylab.xlabel('Rotation')
    # pylab.ylabel('Energy')

if __name__ == '__main__':
    test_matrices()
    test_rotation_under_compression(0)
    test_rotation_under_compression(0.5)

    nu = 0.3
    nelx, nely = 400,16 
    E = ones([nely,nelx]) * 1E6
    fem = FEM2D(nu, nelx, nely)
    x, y = fem.xy.transpose([2,0,1])
    K = fem.K_matrix(E)

    F = zeros(2*(nely+1)*(nelx+1))
    F[(nely//2+1) * (nelx+1) * 2 - 2] = -1
    U = zeros((nely+1)*(nelx+1)*2)
    fixeddofs = set(arange(nely+1) * (nelx+1) * 2)
    fixeddofs.update([(nely//2+1) * (nelx+1) * 2 - 1])
    alldofs = set(arange(U.size))
    freedofs = array(sorted(set.difference(alldofs, fixeddofs)), int)

    U[freedofs] = splinalg.spsolve(K[freedofs,:][:,freedofs], F[freedofs])

    U = U.reshape([nely + 1, nelx + 1, 2])
    G = fem.G_matrix(E, U)
    dx, dy = U.transpose([2,0,1]) * 5E4
    subplot(2,1,1)
    plot(x, y, 'k')
    plot(x.T, y.T, 'k')
    plot(x+dx, y+dy, 'r')
    plot((x+dx).T, (y+dy).T, 'r')
    axis('scaled')
    xlim([-2, x.max()+2])
    ylim([-2, y.max()+2])

    V = zeros(F.size)
    L, V_free = splinalg.eigs(G[freedofs,:][:,freedofs], 1,
                              M=K[freedofs,:][:,freedofs], which='SR')
    V[freedofs] = ravel(V_free.real)
    Pcr = -1 / real(L)[0]

    V = V.reshape([nely + 1, nelx + 1, 2])
    dx, dy = V.transpose([2,0,1]) * 1E1
    subplot(2,1,2)
    plot(x, y, 'k')
    plot(x.T, y.T, 'k')
    plot(x+dx, y+dy, 'r')
    plot((x+dx).T, (y+dy).T, 'r')
    axis('scaled')
    xlim([-2, x.max()+2])
    ylim([-2, y.max()+2])
    title(-1/real(L)[0])

    L = nelx * 4
    I = nely**3 * 2 / 3
    K = 1
    Pcr_Euler = pi**2 * E[0,0] * I / (K * L)**2
    print(Pcr / Pcr_Euler - 1)
