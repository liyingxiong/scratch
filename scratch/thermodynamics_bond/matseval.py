'''
Created on 27.08.2016

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt


def get_bond_slip():
    '''for plotting the bond slip relationship
    '''
    # slip array as input
    s_arr = np.hstack((np.linspace(0, 1e-3, 200),
                       np.linspace(1e-3, 7e-4, 100),
                       np.linspace(7e-4, 1e-3, 100)))
#     s_arr = np.linspace(0, 1e-3, 200)
    # arrays to store the values
    # nominal stress
    tau_arr = np.zeros_like(s_arr)
    # sliding stress
    tau_pi_arr = np.zeros_like(s_arr)
    # damage factor
    w_arr = np.zeros_like(s_arr)
    # sliding slip
    s_pi_arr = np.zeros_like(s_arr)

    # material parameters
    # shear modulus [MPa]
    G = 36000.
    # damage - brittleness [MPa^-1]
    Ad = 1e2
    Z = lambda z: 1. / Ad * (-z) / (1 + z)
    # damage - Threshold
    s0 = 2.5e-4
    Y0 = 0.5 * G * s0 ** 2
    # damage function
    f_damage = lambda Yw: 1 - 1. / (1 + Ad * (Yw - Y0))
    # Kinematic hardening modulus [MPa]
    gamma = 2e6
    X = lambda alpha: gamma * alpha
    # constant in the sliding threshold function
    tau_pi_bar = 0.0
    # parameter in the sliding potential
    a = 0.1

    # state variables
    tau = 0.
    tau_pi = 0.
    alpha = 0.
    z = 0.
    s = 0.  # total slip
    s_pi = 0.  # sliding slip
    w = 0.  # damage

    # value of sliding threshold function at previous step
    f_pi_old = -tau_pi_bar
    # value of sliding stress at previous step
    tau_pi_old = 0.

    for i in range(1, len(s_arr)):
        d_s = s_arr[i] - s_arr[i - 1]
        s += d_s
        Yw = 0.5 * G * s ** 2

        # damage threshold function
        fw = Yw - (Y0 + Z(z))
        # in case damage is activated
        if fw > 1e-8:
            w = f_damage(Yw)
            z = -w

        # trial sliding stress
        tau_pi_trial = w * G * (s - s_pi)
        # sliding threshold function
        f_pi = np.abs(tau_pi_trial - X(alpha)) - tau_pi_bar
        # in case sliding is activated
        if f_pi > 1e-8:
            d_lam_pi =  f_pi_old / \
                (w * G + gamma * -np.sign(tau_pi_old - X(alpha))
                 * (a * X(alpha) - np.sign(tau_pi_old - X(alpha))))

            # update sliding and alpha
            s_pi += d_lam_pi * np.sign(tau_pi_old - X(alpha))
            alpha += d_lam_pi * (a * X(alpha) - np.sign(tau_pi_old - X(alpha)))

        # update all the state variables
        tau = (1 - w) * G * s + w * G * (s - s_pi)
        tau_arr[i] = tau
        tau_pi = w * G * (s - s_pi)
        tau_pi_arr[i] = tau_pi
        w_arr[i] = w
        s_pi_arr[i] = s_pi

        tau_pi_old = tau_pi
        f_pi_old = np.abs(tau_pi - X(alpha)) - tau_pi_bar

    return s_arr, tau_arr, tau_pi_arr, w_arr, s_pi_arr

s_arr, tau_arr, tau_pi_arr, w_arr, s_pi_arr = get_bond_slip()
plt.subplot(221)
plt.plot(s_arr, tau_arr, label='stress')
plt.plot(s_arr, tau_pi_arr, label='sliding stress')
plt.xlabel('slip')
plt.ylabel('stress')
plt.legend()
plt.subplot(222)
plt.plot(s_arr, w_arr)
plt.ylim(0, 1)
plt.xlabel('slip')
plt.ylabel('damage')
plt.subplot(223)
plt.plot(s_arr, s_pi_arr)
plt.xlabel('slip')
plt.ylabel('sliding slip')
# plt.ylim(s_arr[0], s_arr[-1])
plt.show()
