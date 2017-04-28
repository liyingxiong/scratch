'''
Created on 06.02.2017

@author: Yingxiong
A conjugate gradient solver for Ax=b, the matrix A is a stiffness matrix stored as a
stack of the element stiffness matrix in such a way that A[i] returns the stiffness 
matrix of the i-th element.
'''
import numpy as np


def m_v(A, b, dof_map):
    '''stacked matrix - vector production, The numpy einsum method is 
    used for the matrix vector product, the letters in the 
    subscripts have the following meaning:
    e -- element
    d -- DOF 
    '''
    return np.bincount(dof_map.flatten(), weights=np.einsum(
        'edf, ef -> ed', A, b[dof_map]).flatten())


def cg(A, b, dof_map, x0=None, max_iter=10e5, toler=1e-4):

    # no initial guess provided
    if x0 == None:
        x0 = np.zeros_like(b)
    i = 0
    r = b - m_v(A, x0, dof_map)

    d = r.copy()
    delta_new = np.dot(r, r)
    delta0 = delta_new.copy()
    while delta_new > toler ** 2 * delta0:
        q = m_v(A, d, dof_map)
        alpha = delta_new / np.dot(d, q)
        x0 += alpha * d

        if i % 50 == 0:
            r = b - m_v(A, x0, dof_map)
        else:
            r -= alpha * q

        delta_old = delta_new.copy()
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d

        if i >= max_iter:
            print 'CG: convergence is not reached after %i iterations' % (max_iter)
            break

        i += 1

    return x0
