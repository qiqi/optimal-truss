'''
This module implements FEM in 2D with a Cartesian array of square elements.
It computes the linear stiffness matrix and geometric stiffness matrix.
Every element is a square of size 2x2. The elastic modulus of the elements
is provided by an input array.

Example:
    from numpy import *
    from cartfem2d import FEM2D

    nx, ny = 10, 2
    fem = FEM2D(0.3, nx, ny)    # Poisson ratio and rectangular domain shape
    E = 1E6 * ones([ny, nx])    # elastic modulus array of shape [ny,nx]
    M = fem.M_matrix(E)         # mass matrix
    K = fem.K_matrix(E)         # linear stiffness matrix
    dxy = fem.xy * 1E-6         # deformation
    assert dxy.shape == (ny+1, nx+1, 2)
    G = fem.G_matrix(E, dxy)    # geometric stiffness matrix

Qiqi Wang, May 2017
'''

from __future__ import division, print_function

from scipy.sparse import *
import scipy.sparse.linalg as splinalg
from numpy import *
from numpy.polynomial import legendre

def _shpfun(x, y):
    'value of the shape function'
    return array([(1-x)*(1-y), (1+x)*(1-y), (1+x)*(1+y), (1-x)*(1+y)]) / 4

def _ddx(x, y):
    'x derivative of shape function'
    return array([-(1-y), (1-y), (1+y), -(1+y)]) / 4

def _ddy(x, y):
    'y derivative of shape function'
    return array([-(1-x), -(1+x), (1+x), (1-x)]) / 4

def _Axx(x, y):
    '''linear component in xx Cauchy-Green finite strain tensor field.
    sigma_xx = dot(_Axx(x,y), U)
    where U=[u00 v00 u10 v10 u11 v11 u01 v01]
    '''
    return ravel(transpose([_ddx(x, y), zeros(4)]))

def _Ayy(x, y):
    'linear component in yy Cauchy-Green strain tensor. See doc for _Axx'
    return ravel(transpose([zeros(4), _ddy(x, y)]))

def _Axy(x, y):
    'linear component in xy Cauchy-Green strain tensor. See doc for _Axx'
    return ravel(transpose([_ddy(x, y), _ddx(x, y)]))

def _Bxx(x, y):
    '''Quadratic component in xx Cauchy-Green finite strain tensor field.
    sigma_xx = dot(_Axx(x,y), U) + dot(dot(_Bxx(x,y), U), U)
    where U=[u00 v00 u10 v10 u11 v11 u01 v01]
    '''
    e = _ddx(x, y)
    z = zeros(4)
    eu = ravel(transpose([e, z]))
    ev = ravel(transpose([z, e]))
    return (outer(eu, eu) + outer(ev, ev)) / 2

def _Byy(x, y):
    'Quadratic component in yy Cauchy-Green strain tensor. See doc for _Bxx'
    e = _ddy(x, y)
    z = zeros(4)
    eu = ravel(transpose([e, z]))
    ev = ravel(transpose([z, e]))
    return (outer(eu, eu) + outer(ev, ev)) / 2

def _Bxy(x, y):
    'Quadratic component in xy Cauchy-Green strain tensor. See doc for _Bxx'
    e0 = _ddx(x, y)
    e1 = _ddy(x, y)
    z = zeros(4)
    e0u = ravel(transpose([e0, z]))
    e0v = ravel(transpose([z, e0]))
    e1u = ravel(transpose([e1, z]))
    e1v = ravel(transpose([z, e1]))
    return outer(e0u, e1u) + outer(e0v, e1v)

def mass_element_matrices():
    '''
    Quadratic component of the kinetic energy. Example:
        Q = element_matrix() # Poisson ratio
        UK_times_2 = dot(dot(Q, U_dot), U_dot)
    where U=[u00 v00 u10 v10 u11 v11 u01 v01]
    '''
    n = 4
    xg, wg = legendre.leggauss(n)
    xg, yg = meshgrid(xg, xg)
    wg = outer(wg, wg)
    Q = zeros([2*n, 2*n])
    for i in range(n):
        for j in range(n):
            a = _shpfun(xg[i,j], yg[i,j])
            Q[0::2,0::2] += wg[i,j] * outer(a, a)
            Q[1::2,1::2] += wg[i,j] * outer(a, a)
    return Q

def stiffness_element_matrices(nu):
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
        self.M = mass_element_matrices()
        self.Q, C = stiffness_element_matrices(nu)
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
        self.iK = ravel(kron(self.edofMat, ones([8,1], dtype=int)))
        self.jK = ravel(kron(self.edofMat, ones([1,8], dtype=int)))
        self.nelx, self.nely = nelx, nely

    @property
    def xy(self):
        'return (nx+1, ny+1, 2) array. xy[:,:,0] is x; xy[:,:,1] is y.'
        x, y = meshgrid(arange(self.nelx+1), self.nely - arange(self.nely+1))
        return 2 * transpose([x, y], [1,2,0])

    def M_matrix(self, rho):
        '''
        Mass matrix, square of size 2(nelx+1)(nely+1)
        '''
        assert rho.shape == (self.nely, self.nelx)
        M = csr_matrix((kron(ravel(rho), ravel(self.M)), (self.iK, self.jK)))
        return (M + M.T) / 2

    def K_matrix(self, E):
        '''
        Linear stiffness matrix, square of size 2(nelx+1)(nely+1)
        Matrix is singular. Slice it before inverting (rhs can be force).
        '''
        assert E.shape == (self.nely, self.nelx)
        K = csr_matrix((kron(ravel(E), ravel(self.Q)), (self.iK, self.jK)))
        return (K + K.T) / 2

    def dK_matrix_dE(self, UL, UR):
        '''
        derivative of UL^T * K_matrix(E) * UR with respect to E
        '''
        UL_UR = reshape(UL[self.iK] * UR[self.jK], [-1, self.Q.size])
        return dot(UL_UR, ravel(self.Q)).reshape([self.nely, self.nelx])

    def G_matrix(self, E, U):
        '''
        Geometric stiffness matrix, square of size 2(nelx+1)(nely+1)
        '''
        assert E.shape == (self.nely, self.nelx)
        U = ravel(U)
        G = ravel(E)[:,newaxis,newaxis] * dot(U[self.edofMat], self.C)
        G = csr_matrix((ravel(G), (self.iK, self.jK)))
        return (G + G.T) / 2

# ------------------------------ helper functions ---------------------------- #

def plot_deformation(fem, U):
    nelx, nely = fem.nelx, fem.nely
    x, y = fem.xy.transpose([2,0,1])
    dx, dy = U.transpose([2,0,1])
    plot(x, y, 'k', lw=.5)
    plot(x.T, y.T, 'k', lw=.5)
    plot(x+dx, y+dy, 'r', lw=.5)
    plot((x+dx).T, (y+dy).T, 'r', lw=.5)
    axis('scaled')
    xlim([-10, x.max()+10])
    ylim([-10, y.max()+10])

def node_index_x(i, j, nelx, nely):
    if i < 0: i = nelx + 1 + i
    if j < 0: j = nely + 1 + j
    return (j * (nelx + 1) + i) * 2

def node_index_y(i, j, nelx, nely):
    return node_index_x(i, j, nelx, nely) + 1

# --------------------------------- tests ------------------------------------ #

def test_M_matrix():
    M = mass_element_matrices()
    Udot = zeros(8)
    Udot[0::2] = 1 # translating in x
    assert abs(dot(Udot, dot(M, Udot)) - 4) < 1E-8
    Udot[1::2] = 1 # translating in x and y
    assert abs(dot(Udot, dot(M, Udot)) - 8) < 1E-8
    Udot[:] = [1,-1,1,1,-1,1,-1,-1] # rotating counterclockwise
    assert abs(dot(Udot, dot(M, Udot)) - 8/3) < 1E-8

def test_K_matrices():
    'Test integrity of matrices and their agreement with top88 code'
    Q0, C0 = stiffness_element_matrices(0)
    Q1, C1 = stiffness_element_matrices(0.5)
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
    Q, C = stiffness_element_matrices(nu)
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

def test_K_matrix_diff(nu, nelx, nely):
    fem = FEM2D(nu, nelx, nely)
    dE = random.rand(nely, nelx)
    dK = fem.K_matrix(dE)
    UL, UR = random.rand(2, dK.shape[0])
    dJ = dot(UL, dK * UR)
    assert abs(dJ - (fem.dK_matrix_dE(UL, UR) * dE).sum()) < 1E-8

def clamped_beam_vibration(E, rho, nu, nelx, nely):
    E = E * ones([nely,nelx])
    rho = rho * ones([nely,nelx])
    fixeddofs = set([node_index_x(0, j, nelx, nely) for j in arange(nely+1)])
    fixeddofs.update([node_index_y(0, nely // 2, nelx, nely)])
    alldofs = set(arange(2*(nelx+1)*(nely+1)))
    freedofs = array(sorted(set.difference(alldofs, fixeddofs)), int)

    fem = FEM2D(nu, nelx, nely)
    K = fem.K_matrix(E)
    M = fem.M_matrix(rho)
    w2, V_free = splinalg.eigs(K[freedofs,:][:,freedofs], 3,
                               M=M[freedofs,:][:,freedofs], which='SR')
    V = zeros([2*(nelx+1)*(nely+1), 3])
    #V = V.reshape([nely + 1, nelx + 1, 2, V.shape[1]])
    #subplot(4,1,1); plot_deformation(fem, V[:,:,:,0] * 200)
    #subplot(4,1,2); plot_deformation(fem, V[:,:,:,1] * 200)
    #subplot(4,1,3); plot_deformation(fem, V[:,:,:,2] * 200)
    #subplot(4,1,4); plot_deformation(fem, V[:,:,:,3] * 200)
    return w2, V

def test_clamped_beam_vibration(n_refinement):
    E = 1E6
    rho = 1E3
    nelx, nely = 25, 1
    for i in range(n_refinement):
        w2, U = clamped_beam_vibration(E, rho, 0.3, nelx, nely)
        w2.sort()
        I = (2 * nely)**3 / 12
        w2_analytic = array([1.8751, 4.6941, 7.8548, 10.996, 14.137])**4 \
                    * E * I / rho / (2 * nely) / ((2 * nelx)**4)
        err = abs(w2.real / w2_analytic[:3] - 1).max()
        #print(nely, err)
        assert err < 0.5 / nely**2
        nelx *= 2
        nely *= 2

def beam_buckling_strength(E, nu, nelx, nely):
    E = E * ones([nely,nelx])
    F = zeros(2*(nely+1)*(nelx+1))
    F[(nely//2+1) * (nelx+1) * 2 - 2] = -1

    fixeddofs = set([node_index_x(0, j, nelx, nely) for j in arange(nely+1)])
    fixeddofs.update([node_index_y(0, nely // 2, nelx, nely)])
    alldofs = set(arange(F.size))
    freedofs = array(sorted(set.difference(alldofs, fixeddofs)), int)

    fem = FEM2D(nu, nelx, nely)

    K = fem.K_matrix(E)
    U = zeros((nely+1)*(nelx+1)*2)
    U[freedofs] = splinalg.spsolve(K[freedofs,:][:,freedofs], F[freedofs])
    U = U.reshape([nely + 1, nelx + 1, 2])

    G = fem.G_matrix(E, U)
    V = zeros(F.size)
    L, V_free = splinalg.eigs(G[freedofs,:][:,freedofs], 1,
                              M=K[freedofs,:][:,freedofs], which='SR')
    V[freedofs] = ravel(V_free.real)
    V = V.reshape([nely + 1, nelx + 1, 2])
    # import pylab
    # pylab.figure()
    # pylab.subplot(2,1,1); plot_deformation(fem, U * 1E5)
    # pylab.subplot(2,1,2); plot_deformation(fem, V * 10)
    return -1 / real(L)[0]

def test_euler_beam_buckling(n_refinement):
    E = 1E6
    nu = 0.3
    nelx, nely = 50,2
    for i in range(n_refinement):
        Pcr = beam_buckling_strength(E, nu, nelx, nely)
        L = nelx * 4
        I = nely**3 * 2 / 3
        K = 1
        Pcr_Euler = pi**2 * E * I / (K * L)**2
        rel_error = Pcr / Pcr_Euler - 1
        print(nelx, nely, rel_error)
        assert rel_error * nely**2 < 1
        nelx *= 2
        nely *= 2

def I_beam_buckling_strength(fraction):
    E_total = 1E6
    nu = 0.3
    nelx, nely = 50,4
    E = E_total * fraction * ones([nely,nelx])
    E[1:3,:] = E_total * (1 - fraction)
    F = zeros(2*(nely+1)*(nelx+1))
    F[node_index_x(-1, 0, nelx, nely)] = -.25 * fraction
    F[node_index_x(-1, 1, nelx, nely)] = -.25
    F[node_index_x(-1, 2, nelx, nely)] = -.5 * (1 - fraction)
    F[node_index_x(-1, 3, nelx, nely)] = -.25
    F[node_index_x(-1, 4, nelx, nely)] = -.25 * fraction

    fixeddofs = set([node_index_x(0, j, nelx, nely) for j in arange(nely+1)])
    fixeddofs.update([node_index_y(0, nely // 2, nelx, nely)])
    alldofs = set(arange(F.size))
    freedofs = array(sorted(set.difference(alldofs, fixeddofs)), int)

    fem = FEM2D(nu, nelx, nely)

    K = fem.K_matrix(E)
    U = zeros((nely+1)*(nelx+1)*2)
    U[freedofs] = splinalg.spsolve(K[freedofs,:][:,freedofs], F[freedofs])
    U = U.reshape([nely + 1, nelx + 1, 2])

    G = fem.G_matrix(E, U)
    V = zeros(F.size)
    L, V_free = splinalg.eigs(G[freedofs,:][:,freedofs], 1,
                              M=K[freedofs,:][:,freedofs], which='SR')
    V[freedofs] = ravel(V_free.real)
    V = V.reshape([nely + 1, nelx + 1, 2])
    #import pylab
    #pylab.clf()
    #pylab.subplot(2,1,1); plot_deformation(fem, U * 2E5); title('deformation')
    #pylab.subplot(2,1,2); plot_deformation(fem, V * 25); title('bucking mode')
    return -1 / real(L)[0]

def test_I_beam_buckling():
    fraction = 0.8
    strength = I_beam_buckling_strength(fraction)
    EI = 1E6 * (fraction * 7 / 3 + (1 - fraction) / 3)
    strength_euler = EI * (pi / 50)**2
    assert abs(strength_euler / strength - 1) < 0.05

if __name__ == '__main__':
    # test_M_matrix()
    # test_K_matrices()
    # test_rotation_under_compression(0)
    # test_rotation_under_compression(0.5)
    # test_K_matrix_diff(0.3, 20, 30)
    # test_clamped_beam_vibration(2)
    # test_euler_beam_buckling(4)
    # test_I_beam_buckling()
    # print('All tests completed')
    nelx, nely = 10, 4
    E = ones([nely,nelx])
    fixeddofs = set([node_index_x(0, j, nelx, nely) for j in arange(nely+1)])
    fixeddofs.update([node_index_y(0, nely // 2, nelx, nely)])
    alldofs = set(arange(2*(nelx+1)*(nely+1)))
    freedofs = array(sorted(set.difference(alldofs, fixeddofs)), int)

    fem = FEM2D(0.3, nelx, nely)
    K = fem.K_matrix(E)
