'''
Created on 06.02.2017

@author: Yingxiong
A conjugate gradient solver for Ax=b, the matrix A is a stiffness matrix stored as a
stack of the element stiffness matrix in such a way that A[i] returns the stiffness 
matrix of the i-th element.
The numpy einsum method is used for the matrix vector product, the letters in the 
subscripts have the following meaning:
e -- element
d -- DOF 

'''
import numpy as np


def cg_assemble(A, b, dof_map, x0=None, max_iter=10e5, toler=1e-4):

    b = b.reshape(-1, 1)
    # no initial guess provided
    if x0 == None:
        x0 = np.zeros_like(b)
    i = 0
    r = (b - np.dot(A, x0))
    d = r.copy()
    delta_new = np.dot(r.T, r)
    delta0 = delta_new.copy()
    while i <= max_iter and delta_new > toler ** 2 * delta0:
        q = np.dot(A, d)
        alpha = float(delta_new / np.dot(d.T, q))
        x0 += alpha * d

        if i % 50 == 0:
            r = b - np.dot(A, x0)
        else:
            r -= alpha * q

        delta_old = delta_new.copy()
        delta_new = np.dot(r.T, r)
        beta = float(delta_new / delta_old)
        d = r + beta * d
        i += 1
    return x0
