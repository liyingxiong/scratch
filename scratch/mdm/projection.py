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


def projection(eps, n_mp):
    '''project the strain tensor on each microplane -- 2d case,
    return the normal strain e_N, 
    shear strain e_T,
    directions of the microplanes alpha_arr,
    normal vector of the microplanes MPN'''
    # the angles of the microplanes
    alpha_arr = np.linspace(0., np.pi, n_mp)
    # the normal vectors of the microplanes
    MPN = np.vstack((np.cos(alpha_arr), np.sin(alpha_arr)))
    # the tangential vectors of the microplanes
    MPT = np.vstack((np.sin(alpha_arr), -np.cos(alpha_arr)))
    # the strain on the microplane
    e = np.einsum('ij,jk->ik', eps, MPN)
    # magnitude of the normal strain vector for each microplane
    e_N = np.einsum('ik,ik->k', e, MPN)
    # magnitude of the shear strain vector for each microplane
    e_T = np.einsum('ik,ik->k', e, MPT)

    return e_N, e_T, alpha_arr, MPN

if __name__ == '__main__':
    nu = 0.3  # possion's ratio
    eps11 = 1.
    eps = np.array([[eps11, 0, ], [0, -nu * eps11]])
    n_mp = 100  # number of microplanes

    e_N, e_T, alpha_arr, MPN = projection(eps, n_mp)
    # plot the figures
    ax = plt.subplot(121, projection='polar')
    ax.plot(alpha_arr, e_N, color='b')
    # ax.plot(alpha_arr + np.pi, e_N, color='b')
    ax.set_ylim(-1.5, 1)
    ax.set_yticks(np.arange(np.amin(e_N), np.amax(e_N), 0.3))
    ax.plot(np.linspace(0, 2 * np.pi, 100, endpoint=True),
            0. * np.ones(100), 'r-')
    ax.plot(np.linspace(0, 2 * np.pi, 100, endpoint=True),
            -0.8 * np.ones(100), color='k', lw=2)
    ax.set_title('normal stain')

    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(alpha_arr, e_T, color='b')
    # ax2.plot(alpha_arr + np.pi, e_T, color='b')
    ax2.set_ylim(-1.7, 0.7)
    ax2.plot(np.linspace(0, 2 * np.pi, 100, endpoint=True), -
             1. * np.ones(100), 'k-', lw=2)

    ax2.plot(np.linspace(0, 2 * np.pi, 100, endpoint=True), -
             0.0 * np.ones(100), 'r-')
    ax2.set_yticks(np.arange(-0.7, 0.7, 0.3))
    ax2.set_title('shear stain')

    plt.show()
