'''
Created on 12.01.2016

@author: Yingxiong
'''

from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
import numpy as np
from scipy.misc import derivative


class MATSEval(HasTraits):

    E_m = Float(28484, tooltip='Stiffness of the matrix [MPa]',
                auto_set=False, enter_set=False)

    E_f = Float(170000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(2.0, tooltip='Bond stiffness [MPa]')

    sigma_y = Float(1.05,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    K_bar = Float(0.08,  # 191e-6,
                  label="K",
                  desc="Plasticity modulus",
                  enter_set=True,
                  auto_set=False)

    H_bar = Float(0.0,  # 191e-6,
                  label="H",
                  desc="Hardening modulus",
                  enter_set=True,
                  auto_set=False)

    # bond damage law -- maximum slip
#     A = Float(0.05)
#     g = lambda self, k: np.exp(-self.A * k)

    alpha = Float(1.0)
    beta = Float(1.0)
    g = lambda self, k: 1. / (1 + np.exp(-self.alpha * k + 6.)) * self.beta

    # bond damage law -- accumulated slip
    r = Float(0.1)
    q = Float(1)
    cyc = lambda self, z: 1. / \
        (1 + self.r * z ** self.q + self.r ** 2. * z ** (2. * self.q))

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, alpha, q, kappa):
        #         g = lambda k: 0.8 - 0.8 * np.exp(-k)
        #         g = lambda k: 1. / (1 + np.exp(-2 * k + 6.))
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f
        sig_trial = sig[:, :, 1]/(1-self.g(kappa)) + self.E_b * d_eps[:,:, 1]
        xi_trial = sig_trial - q
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha)
        elas = f_trial <= 1e-8
        plas = f_trial > 1e-8
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig

        d_gamma = f_trial / (self.E_b + self.K_bar + self.H_bar) * plas
        alpha += d_gamma
        kappa += d_gamma
        q += d_gamma * self.H_bar * np.sign(xi_trial)
        w = self.g(kappa)

        sig_e = sig_trial - d_gamma * self.E_b * np.sign(xi_trial)
        sig[:, :, 1] = (1-w)*sig_e

        E_p = -self.E_b / (self.E_b + self.K_bar + self.H_bar) * derivative(self.g, kappa, dx=1e-6) * sig_e \
            + (1 - w) * self.E_b * (self.K_bar + self.H_bar) / \
            (self.E_b + self.K_bar + self.H_bar)

        D[:, :, 1, 1] = (1-w)*self.E_b*elas + E_p*plas

        return sig, D, alpha, q, kappa

    def get_bond_slip(self):
        '''for plotting the bond slip relationship
        '''
        un = np.linspace(6, 2, 50)
        re = np.linspace(2, 6, 50)
        s_arr = np.hstack((np.linspace(0, 6, 100), un, re, un, re, un, re))
        sig_e_arr = np.zeros_like(s_arr)
        sig_n_arr = np.zeros_like(s_arr)
        w_arr = np.zeros_like(s_arr)
        w_max_arr = np.zeros_like(s_arr)
        w_cyc_arr = np.zeros_like(s_arr)
        w_arr[0] = 1.
        w_max_arr[0] = 1.
        w_cyc_arr[0] = 1.

        sig_e = 0.
        eps = 0.
        eps_max = 0.
        zeta = 0.

        for i in range(1, len(s_arr)):
            d_eps = s_arr[i] - s_arr[i - 1]

            zeta += np.abs(d_eps)  # the accumulated slip
            eps += d_eps
            eps_max = max(eps_max, np.abs(eps))

            sig_e += self.E_b * d_eps

            w_max = self.g(eps_max)
#             print eps_max
#             print w_max
            w_cyc = self.cyc(zeta)
            w = w_max * w_cyc
            w_max_arr[i] = w_max
            w_cyc_arr[i] = w_cyc
            w_arr[i] = w
            sig_n_arr[i] = w * sig_e
            sig_e_arr[i] = sig_e

#         return s_arr, sig_n_arr, sig_e_arr, w_arr, w_max_arr, w_cyc_arr
        return s_arr, sig_n_arr, sig_e_arr, w_cyc_arr

    n_s = Constant(3)


class MATSEval_cyc(HasTraits):

    n_s = Constant(3)

    E_m = Float(28484, tooltip='Stiffness of the matrix [MPa]',
                auto_set=False, enter_set=False)

    E_f = Float(170000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(2., tooltip='Bond stiffness [MPa]')

    sigma_y = Float(1.05,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    K_bar = Float(0.08,  # 191e-6,
                  label="K",
                  desc="Plasticity modulus",
                  enter_set=True,
                  auto_set=False)

    H_bar = Float(0.0,  # 191e-6,
                  label="H",
                  desc="Hardening modulus",
                  enter_set=True,
                  auto_set=False)

    # bond damage law -- maximum slip
    A = Float(0.05)
    g = lambda self, k: np.exp(-self.A * k)

    # bond damage law -- accumulated slip
    r = Float(0.2)
    q = Float(1)
    cyc = lambda self, z: 1. / \
        (1 + self.r * z ** self.q + self.r ** 2. * z ** (2. * self.q))

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, alpha, q, kappa):
        #         g = lambda k: 0.8 - 0.8 * np.exp(-k)
        #         g = lambda k: 1. / (1 + np.exp(-2 * k + 6.))
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig
#
#         kappa = kappa * (kappa > eps[:,:, 1]) + eps[:,:, 1] * (kappa <= eps[:,:, 1])
        kappa += np.abs(d_eps[:, :, 1])
        w = self.cyc(kappa)

#         print w

        sig_e = self.E_b * eps[:, :, 1]

        sig[:, :, 1] = w * sig_e

#         print sig_e[-1, -1]
#         print kappa[-1, -1]
#         print sig[-1, -1, 1]
#         print '======'

        D[:, :, 1, 1] = w * self.E_b + derivative(self.cyc, kappa, dx=1e-6) * sig_e*np.sign(d_eps[:,:, 1])

        return sig, D, alpha, q, kappa

    def get_bond_slip(self):
        '''only for plotting the bond slip relationship
        '''
        un = np.linspace(6, 2, 50)
        re = np.linspace(2, 6, 50)
#         s_arr = np.hstack((np.linspace(0, 6, 100), un, re, un, re, un, re))
        s_arr = np.hstack((np.linspace(0, 6, 100), un, re, un, re))
        sig_e_arr = np.zeros_like(s_arr)
        sig_n_arr = np.zeros_like(s_arr)
        w_arr = np.zeros_like(s_arr)
        w_max_arr = np.zeros_like(s_arr)
        w_cyc_arr = np.zeros_like(s_arr)
        D_arr = np.zeros_like(s_arr)
        w_arr[0] = 1.
        w_max_arr[0] = 1.
        w_cyc_arr[0] = 1.

        sig_e = 0.
        eps = 0.
        eps_max = 0.
        zeta = 0.
        sig_n = 0.

        for i in range(1, len(s_arr)):
            d_eps = s_arr[i] - s_arr[i - 1]

            zeta += np.abs(d_eps)  # the accumulated slip
            eps += d_eps
            eps_max = max(eps_max, np.abs(eps))

            sig_e += self.E_b * d_eps

            w_max = self.g(eps_max)
            print eps_max
            print w_max
            w_cyc = self.cyc(zeta)
            w = w_max * w_cyc
            w_max_arr[i] = w_max
            w_cyc_arr[i] = w_cyc
#             w_arr[i] = w

            sig_n = w_cyc * sig_e
#             sig_n += w_cyc * self.E_b * d_eps
            sig_n_arr[i] = sig_n
            sig_e_arr[i] = sig_e
            D_arr[i] = w_cyc * self.E_b + \
                derivative(self.cyc, zeta, dx=1e-6) * sig_e * np.sign(d_eps)

# return s_arr, sig_n_arr, sig_e_arr, w_arr, w_max_arr, w_cyc_arr, D_arr
        return s_arr, sig_n_arr, sig_e_arr, w_cyc_arr


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    mat = MATSEval_cyc()

    slip, sig_n_arr, sig_e_arr, w_cyc_arr = mat.get_bond_slip()

    plt.subplot(121)
    plt.plot(np.arange(len(slip)), slip)
    plt.xlabel('load step')
    plt.ylabel('slip')
    plt.subplot(122)
    plt.plot(slip, sig_n_arr,)
#     plt.plot(slip, sig_e_arr, '--')
#     plt.plot(slip, w_arr, '--')
#     plt.plot(slip, w_max_arr, '--')
    plt.plot(slip, w_cyc_arr, '--')
#     plt.plot(slip, np.hstack((0, np.diff(sig_n_arr) / np.diff(slip))))
    plt.title('bond-slip law')
#     plt.plot(slip, D_arr, 'k--', lw=2)
    plt.show()
