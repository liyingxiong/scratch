'''
Created on 16.08.2016

@author: Yingxiong
'''
import numpy as np
from Pentadiagonal import solve
import matplotlib.pyplot as plt


def get_E(eps):
    return -(eps - 1) ** 2 + 1.5


def get_sig(eps):
    return -eps ** 3 / 3. + eps ** 2 + 0.5 * eps

# def get_E(eps):
#     return np.ones_like(eps)
# def get_sig(eps):
#     return 1. * eps


def get_K(eps):
    #     A = np.arange(7) + 1.
    A = np.ones(7)
    l = 10.

    e_dof_map = np.array([[i, i + 1] for i in np.arange(7)])

    k = get_E(eps) * A / l

    k_e_a = np.array([[[ke, -ke], [-ke, ke]] for ke in k])

    K = np.zeros((8, 8))

    for i, dof_map in enumerate(e_dof_map):
        K[dof_map[:, None], dof_map[None, :]] += k_e_a[i]

    d = np.diag(K).copy()
    a = np.diag(K, k=1).copy()
    b = np.diag(K, k=2).copy()

    return d, a, b


def apply_bc(d, a, b):

    # delete the fixed Dof
    d = np.delete(d, 0)
    a = np.delete(a, 0)
    b = np.delete(b, 0)

    # add the element for displacement control
    d = np.append(d, 0.)
    a = np.append(a, 1.)
    b = np.append(b, 0.)

    return d, a, b


def get_eps(U):
    l = 10.
    eps = np.diff(U) / l
    return eps


def get_Fint(eps):
    e_dof_map = np.array([[i, i + 1] for i in np.arange(7)])
#     A = np.arange(7) + 1.
    A = np.ones(7)
    sig_arr = get_sig(eps)
    F = sig_arr * A
    F_int_e = np.vstack((-F, F)).T
    F_int = np.zeros(8)
    for i, dof_map in enumerate(e_dof_map):
        F_int[dof_map] += F_int_e[i]
    F_int[0] = 0.
    return F_int


def eval():
    U = np.zeros(8)

    n = 0
    nmax = 14
    du8 = 2.
    du = np.zeros(8)
    du[-1] = du8
    k = 0
    kmax = 1000

    # the pull-out force
    P = np.zeros(8)

    while k <= kmax:
        k += 1
        print '====='
        print k
        eps = get_eps(U)
        d, a, c = get_K(eps)
        d, a, c = apply_bc(d, a, c)
        x = solve(8, d, a, c, a, c, du)
        dU = np.append(0, x[0:-1])
        U += dU

        print 'U', U
        F_int = get_Fint(get_eps(U))
        P[-1] = F_int[-1]
        R = P - F_int
        print 'R', R[1:-1]
        print np.linalg.norm(R[1:-1])
        if np.linalg.norm(R[1:-1]) < 1e-16:
            break
        du = np.hstack((R[1:-1], [0, 0]))

    return U, P

U, P = eval()
print 'P', P
eps = get_eps(U)
sig = get_sig(eps)
A = np.ones(7)
F = sig * A
print F
plt.plot(np.arange(8), U)
plt.show()
