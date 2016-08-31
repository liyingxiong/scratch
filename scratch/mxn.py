'''
Created on 30.08.2016

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt


E = 100.
# material law - matrix
Sig = lambda eps: E * eps * (eps >= -0.1) * (eps <= 0.1)
# material law - reinforcement
Sig_reinf = lambda eps: 300 * eps

reinf_y_coord = np.array([6., 7.])
reinf_area = np.array([0.3, 0.5])

height = 10.
gravity_center = height / 2.
width = 2.

n_ip = 100
# discretization of the matrix
y_coord = np.linspace(0, height, n_ip)

# upper strain
eps_upper = -0.1
# lower strain
eps_lower = 0.05

# matrix strain array
strain_y = np.interp(y_coord, [0, height], [eps_lower, eps_upper])

# matirx stress array
stress_y = Sig(strain_y)

# reinf strain
strain_r = np.interp(reinf_y_coord, [0, height], [eps_lower, eps_upper])

# reinf stress
stress_r = Sig_reinf(strain_r)

# normal force - matrix
N_m = np.trapz(stress_y * width, y_coord)
# normal force - reinforcement
N_r = np.sum(stress_r * reinf_area)
N = N_m + N_r

# moment - matrix
M_m = np.trapz(stress_y * width * (y_coord - gravity_center), y_coord)
# moment - reinforcement
M_r = np.sum(stress_r * reinf_area * (reinf_y_coord - gravity_center))
M = M_m + M_r


plt.subplot(121)
plt.plot(strain_y, y_coord)
plt.plot((0, 0), (0, height))
for i, strain in enumerate(strain_r):
    plt.plot((strain, 0), (reinf_y_coord[i], reinf_y_coord[i]), '--')

plt.subplot(122)
plt.plot(stress_y, y_coord)
plt.plot((0, 0), (0, height))
for i, stress in enumerate(stress_r):
    plt.plot((stress, 0), (reinf_y_coord[i], reinf_y_coord[i]), '--')

plt.show()
