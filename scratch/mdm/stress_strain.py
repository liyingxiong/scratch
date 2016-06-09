'''
Created on 07.06.2016

@author: Yingxiong
'''
from projection import projection
import numpy as np
import matplotlib.pyplot as plt

# stiffness tensor for 2d plane stress
E = 1000.  # Young's modulus
v = 0.3  # possion's ratio
D_e_2d_stress = np.zeros((2, 2, 2, 2))
D_e_2d_stress[0, 0, :, :] = E / (1 - v ** 2) * np.array([[1., 0.], [0., v]])
D_e_2d_stress[0, 1, :, :] = E / \
    (1 - v ** 2) * np.array([[0., 1. - v], [0., 0.]])
D_e_2d_stress[1, 0, :, :] = E / \
    (1 - v ** 2) * np.array([[0., 0.], [1. - v, 0.]])
D_e_2d_stress[1, 1, :, :] = E / (1 - v ** 2) * np.array([[v, 0.], [0., 1]])


def f_damage(e_max):
    '''the damage function'''
    e0 = 59e-6
    ef = 250e-6
    e_max = e_max + np.finfo(float).eps  # guarantee that e_max is non-zero
    return 1. * (e_max <= e0) + np.sqrt(e0 / e_max * np.exp((e0 - e_max) / (ef - e0))) * (e_max > e0)


def mdm_law(eps, n_mp, c):
    ''' eps -- strain tensor
        n_mp -- number of microplanes
        c -- influence of the shear strain
    '''

    # the weights of the microplanes
    MPW = np.ones(n_mp) / n_mp * 2

    # the maximum achieved equivalent strain
    e_max = np.zeros(n_mp)

    # projection the strain tensor to the microplanes
    e_N, e_T, alpha_arr, MPN = projection(eps, n_mp)
    # the equivalent microplane strain
    e_equi = np.sqrt(e_N ** 2 + c * e_T ** 2)
    # update maximum achieved equivalent strain
    e_max = e_max * (e_max >= e_equi) + e_equi * (e_max < e_equi)

    phi_arr = f_damage(e_max)

    phi_mtx = np.einsum('i,i,ji,ki->jk', phi_arr, MPW, MPN, MPN)

    delta = np.identity(2)

    # Eq. (21) in [Jir99]
    beta_tns = 0.25 * (np.einsum('ik,jl->ijkl', phi_mtx, delta) +
                       np.einsum('il,jk->ijkl', phi_mtx, delta) +
                       np.einsum('jk,il->ijkl', phi_mtx, delta) +
                       np.einsum('jl,ik->ijkl', phi_mtx, delta))

    # damaged stiffness tensor
    D = np.einsum('pqrs, ijpq, klrs->ijkl', D_e_2d_stress, beta_tns, beta_tns)

    sig = np.einsum('ijkl,kl->ij', D, eps)

    return sig


if __name__ == '__main__':

    c = 0.  # influence of the shear strain
    eps = np.array([[1., 0.], [0., 0.]])
    n_mp = 20  # number of microplanes
    sig11_lst = []
    eps11_lst = np.linspace(0., 1e-3, 200)
    for eps11 in eps11_lst:
        eps = np.array([[eps11, 0.], [0., -eps11]])
        sig11_lst.append(mdm_law(eps, n_mp, c)[0][0])

    plt.plot(np.linspace(0., 1e-3, 200), sig11_lst)
    plt.xlabel('strain')
    plt.ylabel('stress')

    plt.figure()
    angles = np.linspace(-0.95 * np.pi / 2., 0.95 *
                         np.pi / 2., 300, endpoint=True)
    ratios = np.tan(angles)
    sig_max = []
    for ratio in ratios:
        sig11_lst = []
        for eps11 in np.linspace(0, 5e-4, 100):
            eps = np.array([[eps11, 0.], [0., ratio * eps11]])
            sig11_lst.append(mdm_law(eps, n_mp, c)[0][0])
        sig_max.append(np.amax(sig11_lst))

    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, sig_max)
    ax.plot([0, np.pi], [0.1, 0.1])
    ax.plot([0.5 * np.pi, 1.5 * np.pi], [0.1, 0.1])

    plt.show()
