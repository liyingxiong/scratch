'''
Created on 26.04.2016

@author: Yingxiong
'''
from pysparse.direct import superlu
import numpy as np
from pysparse import spmatrix
A = spmatrix.ll_mat(5, 5)
for i in range(5):
    A[i, i] = i + 1
A = A.to_csr()
B = np.ones(5)
x = np.empty(5)
LU = superlu.factorize(A, diag_pivot_thresh=0.0)
LU.solve(B, x)

print np.array_str(x)
