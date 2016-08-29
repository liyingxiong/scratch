'''
Created on 30.06.2016

@author: Yingxiong
'''
import numpy as np
from Pentadiagonal import solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


E_m = 28484  # matrix_stiffness [MPa]
E_f = 170000  # reinforcement_stiffness [MPa]
E_b = 2.  # bond_interface_stiffness [MPa]

A_m = 1.
A_f = 1.
p = 10.  # perimeter of the bond interface

l = 1.  # length of the element

km = E_m * A_m / l
kf = E_f * A_f / l
kb = p * l * E_b / 2.

# the element stiffness matrix
# node order -- counter clockwise [ 0 1 2 3]
# ke = np.array([[km + kb, -km, 0, -kb], [-km, km + kb, -kb, 0],
#                [0, -kb, kf + kb, -kf], [-kb, 0, -kf, kf + kb]])

# node order -- counter clockwise [0 2 3 1]
ke = np.array([[km + kb, -kb, -km, 0], [-kb, kf + kb, 0, -kf],
               [-km, 0, km + kb, -kb], [0, -kf, -kb, kf + kb]])

print ke

n_element = 30  # number of elements
n_dof = 2. * (n_element + 1)
K = np.zeros((n_dof, n_dof))

# element_dof_map = np.zeros((n_element, 4), dtype=int)
#
# for i in np.arange(n_element):
#     element_dof_map[i, :] = 2 * i + np.array([0, 2, 3, 1])

element_dof_map = np.array([2 * i + [0, 1, 2, 3]
                            for i in np.arange(n_element)])

for dof_map in element_dof_map:
    K[dof_map[:, None], dof_map[None, :]] += ke

b = np.diag(K, k=2).copy()
# print b
# print dfsfs

# print np.linalg.matrix_rank(K)
# print K.shape
#
K = np.delete(K, 0, 0)
K = np.delete(K, 0, 1)
#
# print np.linalg.matrix_rank(K)
print K.shape

n_dof = 2 * (n_element + 1) - 1

y = np.zeros(n_dof)
y[-1] = 10.

import time as t

t1 = t.time()
for j in range(10):
    x1 = np.linalg.solve(K, y)
print 'numpy.solve', t.time() - t1

d = np.diag(K).copy()
a = np.diag(K, k=1).copy()
b = np.diag(K, k=2).copy()
c = np.diag(K, k=-1).copy()
e = np.diag(K, k=-2).copy()

print a == c
print b == e
# print c
# print e

t2 = t.time()
for j in range(10):
    x2 = solve(n_dof, d, a, b, c, e, y)
print 'solve', t.time() - t2

spK = csc_matrix(K)

t3 = t.time()
for j in range(10):
    x3 = spsolve(spK, y)
print 'scipy.spsolve', t.time() - t3
