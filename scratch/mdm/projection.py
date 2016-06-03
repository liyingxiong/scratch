'''
Created on 03.06.2016

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt

nu = 0.3  # possion's ratio
eps11 = 1.
eps = np.array([[eps11, 0, ], [0, -nu * eps11]])
n_mp = 100  # number of microplanes

# the angles of the microplanes
alpha_arr = np.linspace(0., np.pi, n_mp)
# the normal vectors of the microplanes
MPN = np.vstack((np.cos(alpha_arr), np.sin(alpha_arr)))
# the strain on the microplane
e = np.dot(eps, MPN)
print e
ax = plt.subplot(121, projection='polar')
ax.plot(alpha_arr, np.abs(e[0]), color='b')
ax.plot(alpha_arr + np.pi, np.fabs(e[0]), color='b')
# ax.set_rmax(2.0)
ax.set_ylim(-1, 1)
ax.set_yticks(np.arange(0, 1, 0.2))
ax.plot(np.linspace(0, 2 * np.pi, 100, endpoint=True),
        0. * np.ones(100), color='k', lw=2)
ax.set_title('normal stain')

ax2 = plt.subplot(122, projection='polar')
ax2.plot(alpha_arr, np.fabs(e[1]), color='b')
ax2.plot(alpha_arr + np.pi, np.fabs(e[1]), color='b')
ax2.set_ylim(-0.2, 0.3)
ax2.plot(np.linspace(0, 2 * np.pi, 100, endpoint=True), -
         0.0 * np.ones(100), color='k', lw=2)
ax2.set_yticks(np.arange(0, 0.3, 0.1))
ax2.set_title('shear stain')

plt.show()
