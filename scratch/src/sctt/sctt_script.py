'''
Created on May 7, 2014

@author: rch
'''

import numpy as np
import pylab as p

T = 30.
E_m = 25e3
E_f = 180e3
v_f = 0.05
L_cb2 = 100.0  # [mm] half the length of a crack bridg
n_mp = 50  # number of material points
max_ll = 60.0  # [kN] ??
n_ll = 100  # number of load levels

ll_arr = np.linspace(0, max_ll, 100)

z_arr = np.linspace (0, L_cb2, n_mp)
E_c = E_m * (1 - v_f) + E_f * v_f

sig_m = T * v_f / (1 - v_f) * z_arr
sig_m = sig_m.clip(max=ll_arr[:, np.newaxis] / (E_m * (1 - v_f) + E_f * v_f) * E_m)

sig_mu_min = 0.5 * max_ll
sig_mu_max = 0.7 * max_ll
sig_mu = np.linspace(sig_mu_min, sig_mu_max, n_mp)

crack_arr = np.array([0.0], dtype='f')



print sig_m
p.plot(z_arr, sig_m[20])
p.show()
