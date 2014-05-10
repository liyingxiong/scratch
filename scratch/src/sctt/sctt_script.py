import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import brentq
from matplotlib import pyplot as plt

#===============================================================================
# Crack bridge model
#===============================================================================
T = 30.  # [MPa]
E_m = 25e+3  # [MPa]
E_f = 180e+3  # [MPa]
v_f = 0.05  # [-]
sig_fu = 1.8e+3  # [MPa]
eps_fu = sig_fu / E_f  # ultimate fiber strain
sig_cu = sig_fu * v_f  # ultimate composite stress

# local spatial coordinate
cb_z = np.linspace(0, 100, 500)
# define the range of load levels
cb_sig_c = np.linspace(0, 170, 200)
# define the matrix stress profile sig_m( lambda, z )
cb_sig_m = T * v_f / (1 - v_f) * cb_z
# crack bridge matrix stress profile
cb_sig_m = cb_sig_m.clip(max=cb_sig_c[:, np.newaxis] / (E_m * (1 - v_f) + E_f * v_f) * E_m)
# fiber strain ahead of the crack bridge
cb_eps_f = cb_sig_c[:, np.newaxis] / (E_f * v_f) - T * cb_z / E_f
# crack bridge reinforcement strain profile
cb_eps_f = cb_eps_f.clip(min=cb_sig_c[:, np.newaxis] / (E_m * (1 - v_f) + E_f * v_f))
# function for evaluating specimen matrix stress
get_sig_m = interp2d(cb_z, cb_sig_c, cb_sig_m, kind='linear')
# function for evaluating specimen reinforcement strain
get_eps_f = interp2d(cb_z, cb_sig_c, cb_eps_f, kind='linear')

#===============================================================================
# Tensile specimen
#===============================================================================
n_x = 1000  #
L = 1000.0  # length - mm
x = np.linspace(0, L, n_x)
sig_mu = np.linspace(30, 35, n_x)
max_sig_mu = np.max(sig_mu)
max_lambda = sig_cu

# the function for cracking load of a material point
def get_cr_sig_cacking(sig_mu, d):
    fun = lambda sig_c: sig_mu - get_sig_m(d, sig_c)
    try:
        return brentq(fun, 0, sig_cu)
    except:
        # solution found (saturated crack state) return the ultimate composite stress
        return sig_cu

get_cr_sig_cacking_vct = np.vectorize(get_cr_sig_cacking)

def get_next_crack(z):
    '''Determine the new crack position and lelvel of composite stress
    '''
    # for each material point find the load factor initiating a crack
    cr_sig_cacking_arr = get_cr_sig_cacking_vct(sig_mu, z)
    # get the minimum load factor
    c_idx = np.argmin(cr_sig_cacking_arr)
    sig_c = cr_sig_cacking_arr[c_idx]
    return c_idx, sig_c

def get_cracking_history():
    # crack position list
    cr_x_list = []
    # initial crack distance array (no crack - infinite)
    d = np.ones_like(x) * 1.0e+20
    # crack distances list for each cracking state
    cr_d_list = [d]
    # crack composite stress level list
    cr_sig_c_list = [0.0]
    while True:
        x_idx, sig_c = get_next_crack(d)
        if sig_c == sig_cu: break
        cr_x_list.append(x[x_idx])
        d_grid = np.abs(x[:, np.newaxis] - np.array(cr_x_list)[np.newaxis, :])
        d = np.amin(d_grid, axis=1)
        cr_d_list.append(d)
        cr_sig_c_list.append(sig_c)
    # append composite ultimate state
    cr_d_list.append(d)
    cr_sig_c_list.append(sig_cu)
    return np.array(cr_sig_c_list), np.array(cr_d_list)

def get_eps_c_history(cr_sig_c_list, cr_d_list):
    # get the effective composite strains
    eps_f = np.zeros_like(x)
    cr_eps_c_list = []
    for sig_c, d_arr in zip(cr_sig_c_list, cr_d_list):
        d_map = np.argsort(d_arr)
        eps_f[d_map] = get_eps_f(d_arr[d_map], sig_c)
        cr_eps_c_list.append(np.trapz(eps_f, x) / L)
    return np.array(cr_eps_c_list)

cr_sig_c, cr_d = get_cracking_history()
cr_eps_c = get_eps_c_history(cr_sig_c, cr_d)

plt.subplot(1, 2, 1)
plt.plot(cr_eps_c, cr_sig_c)
plt.plot([0.0, eps_fu], [0.0, sig_cu])
# plt.subplot(1, 2, 2)
# plt.plot(x, eps_f)
# plt.ylim(ymin=0.0)
plt.show()
