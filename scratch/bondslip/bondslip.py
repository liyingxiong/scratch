'''
Created on Jul 23, 2015

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt

# stiffness of the nonlinear spring


def spring_k(s):
    k1 = 15
    k2 = 10
    k3 = -20
    b1 = 0.5
    b2 = 0.8
    if s < b1:
        return k1 * s
    elif b1 <= s and s <= b2:
        return k1 * b1 + k2 * (s - b1)
    else:
        return max(0, k1 * b1 + k2 * (b2 - b1) + k3 * (s - b2))
# spring_k1 = p.vectorize(spring_k)
#
# s = np.linspace(0, 2, 1000)
# tau = spring_k1(s)
#
# plt.plot(s, tau)
#
# plt.show()

# stiffness matrix of the bar


def bar_k(E, A, l):
    return E * A / l * np.array([[1., -1.],
                                 [-1., 1.]])
# number of nodes
n_nodes = 6

# length of the specimen
l = 1000.

# number of bars
n_bars = n_nodes / 2 - 1

# length of the bars
l_bar = l / n_bars

# Young's Modulus [Mpa]
E_bar = 2000

# cross sectional area [mm2]
A_bar = 20

k_b = bar_k(E_bar, A_bar, l_bar)

# global stiffness matrix
K = np.zeros((n_nodes / 2, n_nodes / 2))

# contribution of the bars to K
for i in np.arange(n_bars):
    #     K[i:i + 2, i:i + 2] += k_b
    K[i:i + 2, i:i + 2] += np.array()


print K

# stiffnesses of the nonlinear springs
k_s = np.zeros(n_nodes / 2)


# set constraints
K = np.delete(K, 3, 0)
K = np.delete(K, 3, 1)

print K

F = np.array([10, 0, 0])

u = np.linalg.solve(K, F)

# print u

# if __name__ == '__main__':
#     print bar_k(10, 10, 20)
