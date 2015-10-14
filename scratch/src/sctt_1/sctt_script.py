import numpy as np
from scipy.interpolate import interp2d, interp1d
from scipy.optimize import brentq, newton
from matplotlib import pyplot as plt
from stats.misc.random_field.random_field_1D import RandomField
random_field = RandomField(seed=False,
                           lacor=2.,
                           length=200,
                           nx=1000,
                           nsim=1,
                           loc=.0,
                           shape=10.,
                           scale=4.0,
                           distr_type='Weibull')

#=========================================================================
# Crack bridge model
#=========================================================================
T = 12.  # [MPa]
E_m = 25e+3  # [MPa]
E_f = 180e+3  # [MPa]
v_f = 0.01  # [-]
sig_fu = 1.0e+3  # [MPa]
eps_fu = sig_fu / E_f  # ultimate fiber strain
sig_cu = sig_fu * v_f  # ultimate composite stress

n_z = 1000
n_sig_c = 1000
# local spatial coordinate
cb_z = np.linspace(0, 100, n_z)
# define the range of load levels
cb_sig_c = np.linspace(0, 170, n_sig_c)
# define the matrix stress profile sig_m( lambda, z )
cb_sig_m = T * v_f / (1 - v_f) * cb_z
# crack bridge matrix stress profile
cb_sig_m = cb_sig_m.clip(
    max=cb_sig_c[:, np.newaxis] / (E_m * (1 - v_f) + E_f * v_f) * E_m)
# fiber strain ahead of the crack bridge
cb_eps_f = cb_sig_c[:, np.newaxis] / (E_f * v_f) - T * cb_z / E_f
# crack bridge reinforcement strain profile
cb_eps_f = cb_eps_f.clip(
    min=cb_sig_c[:, np.newaxis] / (E_m * (1 - v_f) + E_f * v_f))
# function for evaluating specimen matrix stress
get_sig_m_z = interp2d(cb_z, cb_sig_c, cb_sig_m, kind='linear')
# function for evaluating specimen reinforcement strain
get_eps_f_z = interp2d(cb_z, cb_sig_c, cb_eps_f, kind='linear')

#=========================================================================
# Fragmentation process
#=========================================================================
n_x = 1000  #
L = 400.0  # length - mm
x = np.linspace(0, L, n_x)
# sig_mu_x = np.linspace(3.0, 4.5, n_x)
sig_mu_x = random_field.random_field


def get_sig_c_z(sig_mu, z):
    '''Determine the composite remote stress initiating a crack at position z
    '''
    fun = lambda sig_c: sig_mu - get_sig_m_z(z, sig_c)
    try:
        # search the cracking stress level between zero and ultimate composite
        # stress
        return brentq(fun, 0, sig_cu)
    except:
        # solution not found (shielded zone) return the ultimate composite
        # stress
        return sig_cu

get_sig_c_x_i = np.vectorize(get_sig_c_z)


def get_sig_c_i(z_x):
    '''Determine the new crack position and level of composite stress
    '''
    # for each material point find the load factor initiating a crack
    sig_c_x_i = get_sig_c_x_i(sig_mu_x, z_x)
    # get the position with the minimum load factor
    y_idx = np.argmin(sig_c_x_i)
    y_i = x[y_idx]
    sig_c_i = sig_c_x_i[y_idx]
    return sig_c_i, y_i


def get_z_x(x, y_i):
    '''Calculate the array distances to the nearest crack y - local z coordinate
    '''
    y = np.array(y_i)
    z_grid = np.abs(x[:, np.newaxis] - y[np.newaxis, :])
    return np.amin(z_grid, axis=1)


def get_LL_x(x, y_lst):
    '''Derive the boundary conditions for each material point.
    '''
    y = np.array(y_lst)
    # todo: handle the sorting of the crack positions
    #
    # y_map = np.argsort(y)
    # y_sorted = y[y_map]
    d = (y[1:] - y[:-1]) / 2.0
    L_left = -np.hstack([y[0], d])
    L_right = np.hstack([d, L - y[-1]])
    z_grid = x[np.newaxis, :] - y[:, np.newaxis]
    left_ranges = np.logical_and(
        L_left[:, np.newaxis] <= z_grid, z_grid <= 0.0)
    right_ranges = np.logical_and(
        L_right[:, np.newaxis] >= z_grid, z_grid >= 0.0)
    ranges = np.logical_or(left_ranges, right_ranges)
    row_idx = np.where(ranges)[0]
    return np.vstack([-L_left[row_idx], L_right[row_idx]])


def get_L_x(x, y_lst):
    '''Derive the boundary conditions for each material point.
    '''
    y = np.sort(y_lst)
    d = (y[1:] - y[:-1]) / 2.0
    xp = np.sort(np.hstack([0, y[:-1] + d, y, x[-1]]))
    Lp = np.hstack([y[0], np.repeat(d, 2), L - y[-1], np.NAN])
    f = interp1d(xp, Lp, kind='zero')
    return f(x)


def get_cracking_history():
    '''Trace the response crack by crack.
    '''
    # crack position list
    y_lst = []
    # initial crack distance array (no crack - infinite)
    z_x = np.ones_like(x) * 2 * L
    # LL_x = np.ones_like(x) * 2 * L
    # crack distances list for each cracking state
    z_x_lst = [z_x]
    # crack composite stress level list
    sig_c_lst = []
    while True:
        sig_c_i, y_i = get_sig_c_i(z_x)
        if sig_c_i == sig_cu:
            break
        y_lst.append(y_i)
        z_x = get_z_x(x, y_lst)
        sig_c_lst.append(sig_c_i)
        z_x_lst.append(z_x)
    # append composite ultimate state
    z_x_lst.append(z_x)
#     sig_c_lst.append(sig_cu)
    return np.array(sig_c_lst), np.array(z_x_lst), np.array(y_lst)

#=========================================================================
# Postprocessing of the calculated response.
#=========================================================================


def get_eps_c_i(sig_c_i, z_x_i):
    '''For each cracking level calculate the corresponding
    composite strain eps_c.
    '''
    return np.array([np.trapz(get_eps_f_x(sig_c, z_x), x) / L
                     for sig_c, z_x in zip(sig_c_i, z_x_i)
                     ])


def get_eps_f_x(sig_c, z_x):
    eps_f = np.zeros_like(x)
    z_x_map = np.argsort(z_x)
    eps_f[z_x_map] = get_eps_f_z(z_x[z_x_map], sig_c)
    return eps_f


def get_sig_m_x(sig_c, z_x):
    sig_m = np.zeros_like(x)
    z_x_map = np.argsort(z_x)
    sig_m[z_x_map] = get_sig_m_z(z_x[z_x_map], sig_c)
    return sig_m

if True:
    sig_c_i, z_x_i, y_i = get_cracking_history()
    eps_c_i = get_eps_c_i(sig_c_i, z_x_i)
    load_arr = np.unique(np.hstack((np.linspace(0, sig_cu, 30), sig_c_i)))
    load_arr = np.unique(np.hstack((np.linspace(0, 3., 20), load_arr)))
    load_arr = np.unique(np.hstack((np.linspace(3., 4., 50), load_arr)))

    eps = np.zeros_like(load_arr)
    for i, load in enumerate(load_arr):
        idx = np.searchsorted(sig_c_i, load)
        z_x = z_x_i[idx]
        sig_m = get_sig_m_x(load, z_x)
        eps[i] = np.trapz(get_eps_f_x(load, z_x), x) / L

        plt.cla()
        plt.plot(x, sig_mu_x, 'g', lw=1.5, label='matrix strength')
        plt.plot(x, sig_m, 'k', lw=1, label='matrix stress')
        plt.ylim(0, 1.2 * np.amax(sig_mu_x))
        savepath = 'D:\cracking history\\1\\load_step' + str(i + 1) + '.png'
        plt.legend(loc='best', ncol=2)
        plt.savefig(savepath)

    for i, load in enumerate(load_arr):

        plt.cla()
        plt.plot(eps, load_arr)
        plt.plot(eps_c_i[sig_c_i <= load], sig_c_i[sig_c_i <= load], 'ko')
        plt.plot(eps[i], load, 'ro')
        savepath = 'D:\cracking history\\1\\global' + str(i + 1) + '.png'
        plt.savefig(savepath)

    plt.cla()
    plt.plot((0, L), (0, 0), 'k', lw=2)
    plt.plot((0, L), (80, 80), 'k', lw=2)
    plt.plot((0, 0), (0, 80), 'k', lw=2)
    plt.plot((L, L), (0, 80), 'k', lw=2)

    idx = 0
    for i, load in enumerate(load_arr):
        y = np.linspace(0, 80, 10)
        idx_old = idx
        idx = np.searchsorted(sig_c_i, load)
        if idx <> idx_old:
            x_co = y_i[idx - 1] + 3. * (np.random.random(10) - 0.5)
            plt.plot(x_co, y, lw=2, color='0.7')

        plt.xlim((-100, 500))
        plt.ylim((-100, 500))

        savepath = 'D:\cracking history\\1\\crack' + str(i + 1) + '.png'
        plt.savefig(savepath)


#     plt.subplot(2, 2, 1)
#     plt.plot(eps_c_i, sig_c_i)
#     plt.plot([0.0, eps_fu], [0.0, sig_cu])
#
#     plt.subplot(2, 2, 2)
#     for i in range(1, len(z_x_i)):
#         plt.plot(x, get_eps_f_x(sig_c_i[i], z_x_i[i]))
#         plt.plot(x, get_sig_m_x(sig_c_i[i], z_x_i[i]) / E_m)
#     plt.ylim(ymin=0.0)
#
#     plt.subplot(2, 2, 3)
#     plt.plot(x, z_x_i[-1])
#
#     plt.subplot(2, 2, 4)
#     plt.plot(x, sig_mu_x)
#     for i in range(1, len(z_x_i)):
#         plt.plot(x, get_sig_m_x(sig_c_i[i], z_x_i[i]))
#     plt.ylim(ymin=0.0)
#     plt.axis('off')

    plt.show()
