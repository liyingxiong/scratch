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
    A = np.arange(7) + 1.
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


def fix_bc(d, a, b):
    # delete the fixed Dof
    d = np.delete(d, 0)
    a = np.delete(a, 0)
    b = np.delete(b, 0)

    return d, a, b


def get_eps(U):
    l = 10.
    eps = np.diff(U) / l
    return eps


def get_Fint(eps):
    e_dof_map = np.array([[i, i + 1] for i in np.arange(7)])
    A = np.arange(7) + 1.
    sig_arr = get_sig(eps)
    F = sig_arr * A
    F_int_e = np.vstack((-F, F)).T
    F_int = np.zeros(8)
    for i, dof_map in enumerate(e_dof_map):
        F_int[dof_map] += F_int_e[i]
    F_int[0] = 0.
    return F_int


U = np.zeros(8)

n = 0
nmax = 14
du8 = 2.
du = np.zeros(8)
du[-1] = du8


eps = get_eps(U)
d, a, c = get_K(eps)
d, a, c = apply_bc(d, a, c)
x = solve(8, d, a, c, a, c, du)

dU = np.append(0, x[0:-1])
p = np.zeros(8)
p[-1] = -x[-1]


A = np.arange(7) + 1.
# print get_sig(get_eps(dU)) * A


print '=================1=================='
# print dU
# print A * get_eps(dU)
# print get_Fint(get_eps(dU))

# print dfsfsfsdf
R = p - get_Fint(get_eps(dU))
print R
#
#
# print '====================2================'
U += dU
eps = get_eps(U)
d, a, b = get_K(eps)
d, a, b = apply_bc(d, a, b)
x = solve(8, d, a, c, a, c, np.hstack((R[1:-1], [0, 0])))
#
dU = np.append(0, x[0:-1])
# dp = np.zeros(8)
# dp[-1] = -x[-1]
# p += dp
#
# A = np.arange(7) + 1.
U += dU
# print get_Fint(get_eps(U))
print p
R = p - get_Fint(get_eps(U))
print R
#
#
# plt.plot(np.arange(8), dU)
# plt.show()
