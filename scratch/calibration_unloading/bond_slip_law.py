'''
Created on 02.08.2016

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

slip = np.array([0.0, 53.035468221615929 / 5000., 0.48448275862068972, 1.2327586206896552, 1.6810344827586208, 2.1293103448275863, 2.5775862068965516,
                 3.0258620689655173, 3.4741379310344831, 3.9224137931034484, 4.3706896551724137, 4.818965517241379, 5.2672413793103452, 5.7155172413793105, 6.1637931034482758, 6.5])
bond = np.array([0.0, 53.035468221615929, 46.946106690853028, 65.101608913634891, 74.37265092804293, 82.344713623823054, 91.684424071878439, 101.9654741591593,
                 111.012646204703, 120.53795013437013, 126.24306086555609, 130.16179466319494, 133.44300562572602, 136.05871770760615, 138.04480490752258, 138.4520655715642])

E_b = 5000.
K_bar = 0.
H_bar = 40.
sig_y = 53.035

bond_diff = np.diff(slip)
bond_diff[0] = 0.
delta_sig_p = bond_diff * E_b * (K_bar + H_bar) / (E_b + K_bar + H_bar)
sig = sig_y + np.cumsum(delta_sig_p)
sig = np.hstack((0., sig))
eps_p = slip - sig / E_b

w = bond[1::] / sig[1::]
w = np.hstack((1, w))


fig, ax1 = plt.subplots()
plt.plot(slip, sig)
plt.plot(slip, bond, marker='.')
ax2 = ax1.twinx()
plt.plot(slip, w)
plt.ylim(0, 1)

plt.show()

plt.plot(eps_p, w)
plt.show()


g = interp1d(eps_p, 1. - w)


def get_bond_slip():
    '''for plotting the bond slip relationship
    '''
#     s_arr = np.hstack((np.linspace(0, 10, 200),
#                        np.linspace(10., 10. - sigma_y / E_b, 10)))
    s_arr = np.linspace(0, 6.5, 100)
    sig_e_arr = np.zeros_like(s_arr)
    sig_n_arr = np.zeros_like(s_arr)
    w_arr = np.zeros_like(s_arr)

    sig_e = 0.
    alpha = 0.
    q = 0.
    kappa = 0.

    for i in range(1, len(s_arr)):
        d_eps = s_arr[i] - s_arr[i - 1]
        sig_e_trial = sig_e + E_b * d_eps
        xi_trial = sig_e_trial - q
        f_trial = abs(xi_trial) - (sig_y + K_bar * alpha)
        if f_trial <= 1e-8:
            sig_e = sig_e_trial
        else:
            d_gamma = f_trial / (E_b + K_bar + H_bar)
            alpha += d_gamma
            kappa += d_gamma
            q += d_gamma * H_bar * np.sign(xi_trial)
            sig_e = sig_e_trial - d_gamma * E_b * np.sign(sig_e_trial)
        w = g(kappa)
        w_arr[i] = w
        sig_n_arr[i] = (1. - w) * sig_e
        sig_e_arr[i] = sig_e

    return s_arr, sig_n_arr, sig_e_arr, w_arr

s_arr, sig_n_arr, sig_e_arr, w_arr = get_bond_slip()

plt.plot(s_arr, sig_n_arr)
# plt.plot(slip, bond, marker='.')
plt.show()
# plot the experimental curve

# fpath = 'D:\\data\\pull_out\\all\\DPO-20cm-0-3300SBR-V3_R3_f.asc'
# d, f = np.loadtxt(fpath,  delimiter=';')
# plt.plot(x, np.interp(x, d / 2., f * 1000.),
# '--', color=color, label=label)
# plt.plot(d[d <= 11.] / 2., f[d <= 11.] * 1000.)
# #
# plt.show()
