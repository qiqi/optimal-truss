from __future__ import division

from scipy.sparse import *
import scipy.sparse.linalg as splinalg
from numpy import *
from numpy.polynomial import legendre

def ddx(x, y):
    return array([-(1-y), (1-y), (1+y), -(1+y)]) / 4

def ddy(x, y):
    return array([-(1-x), -(1+x), (1+x), (1-x)]) / 4

def Axx(x, y):
    return ravel(transpose([ddx(x, y), zeros(4)]))

def Ayy(x, y):
    return ravel(transpose([zeros(4), ddy(x, y)]))

def Axy(x, y):
    return ravel(transpose([ddy(x, y), ddx(x, y)]))

def Bxx(x, y):
    e = ddx(x, y)
    z = zeros(4)
    eu = ravel(transpose([e, z]))
    ev = ravel(transpose([z, e]))
    return (outer(eu, eu) + outer(ev, ev)) / 2

def Byy(x, y):
    e = ddy(x, y)
    z = zeros(4)
    eu = ravel(transpose([e, z]))
    ev = ravel(transpose([z, e]))
    return (outer(eu, eu) + outer(ev, ev)) / 2

def Bxy(x, y):
    e0 = ddx(x, y)
    e1 = ddy(x, y)
    z = zeros(4)
    e0u = ravel(transpose([e0, z]))
    e0v = ravel(transpose([z, e0]))
    e1u = ravel(transpose([e1, z]))
    e1v = ravel(transpose([z, e1]))
    return outer(e0u, e1u) + outer(e0v, e1v)

def element_matrices(nu):
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
            axx = Axx(xg[i,j], yg[i,j])
            ayy = Ayy(xg[i,j], yg[i,j])
            axy = Axy(xg[i,j], yg[i,j])
            Q0 += wg[i,j] * (outer(axx, axx) + outer(ayy, ayy) + 1/2 * outer(axy, axy))
            Q1 += wg[i,j] * (outer(axx, ayy) + outer(ayy, axx) - 1/2 * outer(axy, axy))
            bxx = Bxx(xg[i,j], yg[i,j])
            byy = Byy(xg[i,j], yg[i,j])
            bxy = Bxy(xg[i,j], yg[i,j])
            C0 += wg[i,j] * (2 * axx * bxx[:,:,newaxis] + 2 * ayy * byy[:,:,newaxis] + axy * bxy[:,:,newaxis])
            C1 += wg[i,j] * (2 * axx * byy[:,:,newaxis] + 2 * ayy * bxx[:,:,newaxis] - axy * bxy[:,:,newaxis])

    premul = 1 / (1 - nu**2)
    return premul * (Q0 + nu * Q1), premul * (C0 + nu * C1)

class FEM:
    def __init__(self, nu, shape):
        self.Q, C = element_matrices(nu)
        self.C = C + rollaxis(C, 1) + rollaxis(C, 2)
        nelx, nely = shape
        nodenrs = reshape(arange((1+nelx)*(1+nely)), [1+nely,1+nelx])
        edofVec = ravel(2 * nodenrs[:-1,:-1])
        self.edofMat = edofVec[:,newaxis] + ravel(array([nelx+1, nelx+2, 1, 0])[:,newaxis] * 2 + arange(2))
        self.iK = ravel(kron(self.edofMat, ones([8,1])))
        self.jK = ravel(kron(self.edofMat, ones([1,8])))

    def K_matrix(self, E):
        K = csr_matrix((kron(ravel(E), ravel(self.Q)), (self.iK, self.jK)))
        return (K + K.T) / 2

    def G_matrix(self, E, U):
        U[self.edofMat]
        K = self.K_matrix(E)


def test_matrices():
    Q0, C0 = element_matrices(0)
    Q1, C1 = element_matrices(0.5)
    assert(abs(Q0 * 48 - around(Q0 * 48)).sum() < 1E-12)
    assert(abs(Q1 * 72 - around(Q1 * 72)).sum() < 1E-12)
    assert(abs(C0 * 48 - around(C0 * 48)).sum() < 1E-12)
    assert(abs(C1 * 72 - around(C1 * 72)).sum() < 1E-12)
    print('Q0')
    print(around(Q0 * 48))
    print('Q1')
    print(around(Q1 * 72 - Q0 * 96))
    # print('C0')
    # print(around(C0 * 48))
    # print('C1')
    # print(around(C1 * 72 - C0 * 96))

def test_rotation_under_compression(nu):
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

    import pylab
    pylab.figure()
    pylab.plot(thetas, UE)
    pylab.plot(thetas, UE + dUE)
    pylab.legend(['Linear', 'Geometrically nonlinear'])
    pylab.title('Rotation under compression')
    pylab.xlabel('Rotation')
    pylab.ylabel('Energy')

if __name__ == '__main__':
    nu = 0.3
    nelx, nely = 50, 2
    E = ones([nelx,nely]) * 1E6
    fem = FEM(nu, E.shape)
    K = fem.K_matrix(E)

    F = zeros(2*(nely+1)*(nelx+1))
    F[(nely//2+1) * (nelx+1) * 2 - 2] = -1
    U = zeros((nely+1)*(nelx+1)*2);
    fixeddofs = set(arange(nely+1) * (nelx+1) * 2)
    fixeddofs.update([(nely//2+1) * (nelx+1) * 2 - 1])
    alldofs = set(arange(U.size))
    freedofs = array(sorted(set.difference(alldofs, fixeddofs)), int)

    U[freedofs] = splinalg.spsolve(K[freedofs,:][:,freedofs], F[freedofs])

    self = fem
    G = ravel(E)[:,newaxis,newaxis] * dot(U[self.edofMat], self.C)
    G = csr_matrix((ravel(G), (self.iK, self.jK)))
    G = (G + G.T) / 2
    U = U.reshape([nely + 1, nelx + 1, 2])
    subplot(2,1,1)
    quiver(arange(nelx+1), nely-arange(nely+1), U[:,:,0], U[:,:,1])
    axis('scaled')
    xlim([-1, nelx+1])
    ylim([-1, nely+1])

    V = zeros_like(F)
    L, V[freedofs] = splinalg.eigs((K+50000*G)[freedofs,:][:,freedofs],1,which='SR')
    V = V.reshape([nely + 1, nelx + 1, 2])
    subplot(2,1,2)
    quiver(arange(nelx+1), nely-arange(nely+1), V[:,:,0], V[:,:,1])
    axis('scaled')
    xlim([-1, nelx+1])
    ylim([-1, nely+1])

    # test_matrices()
    # test_rotation_under_compression(0)
    # test_rotation_under_compression(0.5)
