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
get_sig_m_z = interp2d(cb_z, cb_sig_c, cb_sig_m, kind='linear')
# function for evaluating specimen reinforcement strain
get_eps_f_z = interp2d(cb_z, cb_sig_c, cb_eps_f, kind='linear')

#===============================================================================
# Tensile specimen
#===============================================================================
n_x = 1000  #
L = 1000.0  # length - mm
x = np.linspace(0, L, n_x)
sig_mu_x = np.linspace(30, 35, n_x)
max_sig_mu = np.max(sig_mu_x)
max_lambda = sig_cu

# the function for cracking load of a material point
def get_cr_sig_cracking(sig_mu, z_x):
    fun = lambda sig_c: sig_mu - get_sig_m_z(z_x, sig_c)
    try:
        return brentq(fun, 0, sig_cu)
    except:
        # solution found (saturated crack state) return the ultimate composite stress
        return sig_cu

get_cr_sig_cracking_x = np.vectorize(get_cr_sig_cracking)

def get_next_crack(z_x):
    '''Determine the new crack position and lelvel of composite stress
    '''
    # for each material point find the load factor initiating a crack
    sig_c_cr_x = get_cr_sig_cracking_x(sig_mu_x, z_x)
    # get the position with the minimum load factor
    y_idx = np.argmin(sig_c_cr_x)
    sig_c_cr = sig_c_cr_x[y_idx]
    return y_idx, sig_c_cr

def get_cracking_history():
    # crack position list
    y_i = []
    # initial crack distance array (no crack - infinite)
    d_x = np.ones_like(x) * L
    # crack distances list for each cracking state
    d_x_i = [d_x]
    # crack composite stress level list
    sig_c_i = [0.0]
    while True:
        x_idx, sig_c = get_next_crack(d_x)
        if sig_c == sig_cu: break
        y_i.append(x[x_idx])
        d_grid = np.abs(x[:, np.newaxis] - np.array(y_i)[np.newaxis, :])
        d_x = np.amin(d_grid, axis=1)
        d_x_i.append(d_x)
        sig_c_i.append(sig_c)
    # append composite ultimate state
    d_x_i.append(d_x)
    sig_c_i.append(sig_cu)
    return np.array(sig_c_i), np.array(d_x_i)

def get_eps_c_i(cr_sig_c_list, cr_d_list):
    '''For each cracking level calculate the corresponding
    composite strain eps_c.
    '''
    return np.array([np.trapz(get_eps_f_x(sig_c_i, d_x), x) / L
                     for sig_c_i, d_x in zip(cr_sig_c_list, cr_d_list)
                     ])

def get_eps_f_x(sig_c, d_arr):
    eps_f = np.zeros_like(x)
    d_map = np.argsort(d_arr)
    eps_f[d_map] = get_eps_f_z(d_arr[d_map], sig_c)
    return eps_f

def get_sig_m_x(sig_c, d_arr):
    eps_m = np.zeros_like(x)
    d_map = np.argsort(d_arr)
    eps_m[d_map] = get_sig_m_z(d_arr[d_map], sig_c)
    return eps_m

sig_c_i, d_x_i = get_cracking_history()
eps_c_i = get_eps_c_i(sig_c_i, d_x_i)

plt.subplot(2, 2, 1)
plt.plot(x, sig_mu_x)
plt.plot([0.0, eps_fu], [0.0, sig_cu])

plt.subplot(2, 2, 3)
plt.plot(eps_c_i, sig_c_i)
plt.plot([0.0, eps_fu], [0.0, sig_cu])

plt.subplot(2, 2, 2)
plt.plot(x, get_eps_f_x(sig_c_i[-1], d_x_i[-1]))
plt.ylim(ymin=0.0)

plt.subplot(2, 2, 4)
plt.plot(x, get_sig_m_x(sig_c_i[-1], d_x_i[-1]))
plt.ylim(ymin=0.0)

plt.show()
