'''
Created on 04.07.2016

@author: Yingxiong
'''
import numpy as np

def solve(n, d, a, b, c, e, y):
    
#     cdef int n, i

    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 2)
    z = np.zeros(n)
    gamma = np.zeros(n - 1)
    mu = np.zeros(n)

    # step 3
    mu[0] = d[0]
    alpha[0] = a[0] / mu[0]
    beta[0] = b[0] / mu[0]
    z[0] = y[0] / mu[0]

    # step 4
    gamma[0] = c[0]
    mu[1] = d[1] - alpha[0] * gamma[0]
    alpha[1] = (a[1] - beta[0] * gamma[0]) / mu[1]
    beta[1] = b[1] / mu[1]
    z[1] = (y[1] - z[0] * gamma[0]) / mu[1]

    # step 5
    for i in np.arange(2, n - 2):
        gamma[i - 1] = c[i - 1] - alpha[i - 2] * e[i - 2]
        mu[i] = d[i] - beta[i - 2] * e[i - 2] - alpha[i - 1] * gamma[i - 1]
        alpha[i] = (a[i] - beta[i - 1] * gamma[i - 1]) / mu[i]
        beta[i] = b[i] / mu[i]
        z[i] = (y[i] - z[i - 2] * e[i - 2] - z[i - 1] * gamma[i - 1]) / mu[i]

    gamma[n - 3] = c[n - 3] - alpha[n - 4] * e[n - 4]
    mu[n - 2] = d[n - 2] - beta[n - 4] * e[n - 4] - alpha[n - 3] * gamma[n - 3]
    alpha[n - 2] = (a[n - 2] - beta[n - 3] * gamma[n - 3]) / mu[n - 2]

    gamma[n - 2] = c[n - 2] - alpha[n - 3] * e[n - 3]
    mu[n - 1] = d[n - 1] - beta[n - 3] * e[n - 3] - alpha[n - 2] * gamma[n - 2]
    z[n - 2] = (y[n - 2] - z[n - 4] * e[n - 4] - z[n - 3]
                * gamma[n - 3]) / mu[n - 2]
    z[n - 1] = (y[n - 1] - z[n - 3] * e[n - 3] - z[n - 2]
                * gamma[n - 2]) / mu[n - 1]

    # step 6
    x = np.zeros(n)
    x[n - 1] = z[n - 1]
    x[n - 2] = z[n - 2] - alpha[n - 2] * x[n - 1]
    for i in np.arange(n - 3, -1, -1):
        x[i] = z[i] - alpha[i] * x[i + 1] - beta[i] * x[i + 2]
    return x

if __name__ == '__main__':

    import time as t
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve

    n = 10
    d = np.array([1, 2, 3, -4, 5, 6, 7, -1, 1, 8], dtype=np.float64)
    a = np.array([2, 2, 1, 5, -7, 3, -1, 4, 5], dtype=np.float64)
    b = np.array([1, 5, -2, 1, 5, 2, 4, -3], dtype=np.float64)
    c = np.array([3, 2, 1, 2, 1, 2, 1, -2, 4], dtype=np.float64)
    e = np.array([1, 3, 1, 5, 2, 2, 2, -1], dtype=np.float64)
    y = np.array([8, 33, 8, 24, 29, 98, 99, 17, 57, 108], dtype=np.float64)

    k = np.zeros((10, 10))

    k[0, 0:3] = [1, 2, 1]
    k[1, 0:4] = [3, 2, 2, 5]
    k[2, 0:5] = [1, 2, 3, 1, -2]
    k[3, 1:6] = [3, 1, -4, 5, 1]
    k[4, 2:7] = [1, 2, 5, -7, 5]
    k[5, 3:8] = [5, 1, 6, 3, 2]
    k[6, 4:9] = [2, 2, 7, -1, 4]
    k[7, 5:10] = [2, 1, -1, 4, -3]
    k[8, 6:10] = [2, -2, 1, 5]
    k[9, 7:10] = [-1, 4, 8]

    spk = csc_matrix(k)

    t1 = t.time()
    for j in range(10000):
        x1 = solve(n, d, a, b, c, e, y)
    print t.time() - t1

    t2 = t.time()

    for j in range(10000):
        x2 = np.linalg.solve(k, y)

    print t.time() - t2

    t3 = t.time()
    for j in range(10000):
        x3 = spsolve(spk, y)

    print t.time() - t3

    print x1
    print x2
    print x3
