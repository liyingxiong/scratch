'''
Created on 03.06.2016

@author: Yingxiong
'''
import matplotlib.pyplot as plt
import numpy as np


nu = 0.3  # possion's ratio
eps11 = -1.
eps = np.array([[eps11, 0, ], [0, -nu * eps11]])
n_mp = 6  # number of microplanes

# the angles of the microplanes
alpha_arr = np.linspace(0., np.pi, n_mp)
# the normal vectors of the microplanes
MPN = np.vstack((np.cos(alpha_arr), np.sin(alpha_arr)))
# the strain on the microplane
e = np.einsum('ij,jk->ik', eps, MPN)
# magnitude of the normal strain vector for each microplane
e_N = np.einsum('ik,ik->k', e, MPN)
print e_N
# normal strain vector for each microplane
e_N_vec = np.einsum('i,ji->ji', e_N, MPN)
# tangential strain vector for each microplane
e_T_vec = e - e_N_vec
# magnitude of the normal strain vector for each microplane
e_T = np.sqrt(np.einsum('ik,ik->k', e_T_vec, e_T_vec))

# plot the figures
ax = plt.subplot(121, projection='polar')
ax.plot(alpha_arr, e_N, color='b')
# ax.plot(alpha_arr + np.pi, e_N, color='b')
ax.set_ylim(np.amin(e_N), np.amax(e_N))
ax.set_yticks(np.arange(-0.3, 1, 0.2))
# ax.plot(np.linspace(0, 2 * np.pi, 100, endpoint=True),
#         0. * np.ones(100), color='k', lw=2)
ax.set_title('normal stain')

ax2 = plt.subplot(122, projection='polar')
ax2.plot(alpha_arr, e_T, color='b')
# ax2.plot(alpha_arr + np.pi, e_T, color='b')
ax2.set_ylim(np.amin(e_T), np.amax(e_T))
# ax2.plot(np.linspace(0, 2 * np.pi, 100, endpoint=True), -
#          0.0 * np.ones(100), color='k', lw=2)
# ax2.set_yticks(np.arange(0, 0.3, 0.1))
ax2.set_title('shear stain')

plt.show()
