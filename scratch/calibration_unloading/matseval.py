'''
Created on 12.01.2016

@author: Yingxiong
'''

from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
import numpy as np
from scipy.misc import derivative


class MATSEval(HasTraits):

    #     slip = np.array([0.0, 0.16620689655172414, 0.48448275862068972, 1.2327586206896552, 1.6810344827586208, 2.1293103448275863, 2.5775862068965516,
    # 3.0258620689655173, 3.4741379310344831, 3.9224137931034484,
    # 4.3706896551724137, 4.818965517241379, 5.2672413793103452,
    # 5.7155172413793105, 6.1637931034482758, 6.5])

    slip = np.array([0.0, 53.035468221615929 / 5000., 0.48448275862068972, 1.2327586206896552, 1.6810344827586208, 2.1293103448275863, 2.5775862068965516,
                     3.0258620689655173, 3.4741379310344831, 3.9224137931034484, 4.3706896551724137, 4.818965517241379, 5.2672413793103452, 5.7155172413793105, 6.1637931034482758, 6.5])

    bond = np.array([0.0, 53.035468221615929, 46.946106690853028, 65.101608913634891, 74.37265092804293, 82.344713623823054, 91.684424071878439, 101.9654741591593,
                     111.012646204703, 120.53795013437013, 126.24306086555609, 130.16179466319494, 133.44300562572602, 136.05871770760615, 138.04480490752258, 138.4520655715642])

    E_m = Float(28484, tooltip='Stiffness of the matrix [MPa]',
                auto_set=False, enter_set=False)

    E_f = Float(170000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(5000,
                tooltip='Bond stiffness [N/mm]')

    sigma_y = Float(53.0354,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    K_bar = Float(0.,  # 191e-6,
                  label="K",
                  desc="Plasticity modulus",
                  enter_set=True,
                  auto_set=False)

    H_bar = Float(40.,  # 191e-6,
                  label="H",
                  desc="Hardening modulus",
                  enter_set=True,
                  auto_set=False)
    # bond damage law

    def g(self, kappa):
        bond_diff = np.diff(self.slip)
        bond_diff[0] = 0.
        delta_sig_p = bond_diff * self.E_b * \
            (self.K_bar + self.H_bar) / (self.E_b + self.K_bar + self.H_bar)
        sig = self.sigma_y + np.cumsum(delta_sig_p)
        sig = np.hstack((0., sig))
        eps_p = self.slip - sig / self.E_b
        d = self.bond[1::] / sig[1::]
        d = np.hstack((1, d))
        return np.interp(kappa, eps_p, 1. - d)

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
        s_arr = np.hstack((np.linspace(0, 4, 200),
                           np.linspace(4., 3.95, 10),
                           np.linspace(3.95, 6.5, 100)))
#         s_arr = np.linspace(0, 6.5, 100)
        sig_e_arr = np.zeros_like(s_arr)
        sig_n_arr = np.zeros_like(s_arr)
        w_arr = np.zeros_like(s_arr)

        sig_e = 0.
        alpha = 0.
        q = 0.
        kappa = 0.

        for i in range(1, len(s_arr)):
            d_eps = s_arr[i] - s_arr[i - 1]
            sig_e_trial = sig_e + self.E_b * d_eps
            xi_trial = sig_e_trial - q
            f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha)
            if f_trial <= 1e-8:
                sig_e = sig_e_trial
            else:
                d_gamma = f_trial / (self.E_b + self.K_bar + self.H_bar)
                alpha += d_gamma
                kappa += d_gamma
                q += d_gamma * self.H_bar * np.sign(xi_trial)
                sig_e = sig_e_trial - d_gamma * self.E_b * np.sign(sig_e_trial)
            w = self.g(kappa)
            w_arr[i] = w
            sig_n_arr[i] = (1. - w) * sig_e
            sig_e_arr[i] = sig_e

        return s_arr, sig_n_arr, sig_e_arr, w_arr

    n_s = Constant(3)

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    mat = MATSEval()

    slip, sig_n_arr, sig_e_arr, w_arr = mat.get_bond_slip()
    fig, ax1 = plt.subplots()
    plt.plot(slip, sig_n_arr)
    plt.plot(slip, sig_e_arr, '--')
    ax2 = ax1.twinx()
    plt.plot(slip, w_arr, '--')
    plt.ylim(0, 1)
    plt.title('bond-slip law')
    plt.show()
