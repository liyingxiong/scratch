'''
Created on 01.02.2017

@author: Yingxiong
'''
import numpy as np
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


def bar():
    E = 1.
    A = 1.
    l = 1.
    k = E * A / l

    Ke = k * np.array([[1, -1], [-1, 1]])

    Ke_array = np.array([Ke for i in range(5)])

    dof_map = np.array([[0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5]])
    return Ke_array, dof_map


def bond_el():
    Em = 10.
    Ef = 10.
    Gb = 0.1
    l = 1.
    km = Em / l
    kf = Ef / l
    kb = Gb / 2.

    Ke = np.array([[km + kb, -km, 0, -kb],
                   [-km, km + kb, -kb, 0],
                   [0, -kb, kb + kf, -kf],
                   [-kb, 0, -kf, kb + kf]])

    Ke_array = np.array([Ke for i in range(10000)])

    dof_map = np.array([2 * i + [0, 2, 3, 1] for i in np.arange(10000)])

    return Ke_array, dof_map


Ke_array, dof_map = bond_el()

a = np.array([1.])


def constraint(i, u_i, R):
    el_arr, row_arr = np.where(dof_map == i)
    for el, i_dof in zip(el_arr, row_arr):
        rows = dof_map[el]
        R[rows] += -u_i * Ke_array[el, :, i_dof]
        Ke_ii = Ke_array[el, i_dof, i_dof]
        Ke_array[el, i_dof, :] = 0.
        Ke_array[el, :, i_dof] = 0.
        Ke_array[el, i_dof, i_dof] = -Ke_ii


R = np.zeros(np.amax(dof_map) + 1)
# R[-1] = 100.

constraint(np.amax(dof_map) - 1, 0., R)
constraint(np.amax(dof_map), 10, R)
# constraint(4, 1., R)
import time as t
t0 = t.time()
# K = np.zeros((np.amax(dof_map) + 1, np.amax(dof_map) + 1))
K = lil_matrix((np.amax(dof_map) + 1, np.amax(dof_map) + 1))


for i, e_dof in enumerate(dof_map):
    col, row = np.meshgrid(e_dof, e_dof)
    K[row, col] += Ke_array[i]

print t.time() - t0

K = csc_matrix(K)

t1 = t.time()
U = cg(K, R, M=None)
print t.time() - t1

t2 = t.time()
U2 = spsolve(K, R)
print t.time() - t2


from conjugate_gradient import cg as cg1
t3 = t.time()
U3 = cg1(Ke_array, R, dof_map)
print t.time() - t3

# from conjugate_gradient_assemble import cg_assemble as cg2
# t4 = t.time()
# U4 = cg2(K.todense(), R, dof_map)
# print t.time() - t4

plt.plot(range(10001), U3[0::2])
plt.plot(range(10001), U3[1::2])
plt.show()
