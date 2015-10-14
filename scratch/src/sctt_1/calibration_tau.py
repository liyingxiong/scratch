'''
Created on Feb 19, 2015

@author: Li Yingxiong
'''
import numpy as np
from scipy.interpolate import interp1d
import os
from matplotlib import pyplot as plt
from spirrid.rv import RV
from math import pi
from sctt.calibration.tau_strength_dependence import interp_tau_shape, interp_tau_scale

w_arr = np.linspace(0., 1., 100)
sig_w = np.zeros_like(w_arr)
home_dir = 'D:\\Eclipse\\'
for i in np.array([1, 2, 3, 4, 5]):
    path = [home_dir, 'git',  # the path of the data file
            'rostar',
            'scratch',
            'diss_figs',
            'CB'+str(i)+'.txt']
    filepath = os.path.join(*path)
#     exp_data = np.zeros_like(w_arr)
    file1 = open(filepath, 'r')
    cb = np.loadtxt(file1, delimiter=';')
    test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
    test_ydata = cb[:, 1] 
    interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
    sig_w += 0.2*interp(w_arr)
    
plt.plot(w_arr, sig_w)
plt.figure()
# plt.plot(w_arr, np.sqrt(w_arr))
# plt.show()

s = 0.0090
mvalue = 6.0

def func1(w_arr, k, theta):
 
    tau = RV('gamma', shape=k, scale=theta, loc=0.)
    n_int = 500
    p_arr = np.linspace(0.5/n_int, 1 - 0.5/n_int, n_int)
    tau_arr = tau.ppf(p_arr) + 1e-10
    
    sV0 = s
    m = mvalue
    r = 3.5e-3
    E_f = 180e3
    lm =1000.
 
    def cdf(e, depsf, r, lm, m, sV0):
        '''weibull_fibers_cdf_mc'''
        s = ((depsf*(m+1.)*sV0**m)/(2.*pi*r**2.))**(1./(m+1.))
        a0 = (e+1e-15)/depsf
        expfree = (e/s) ** (m + 1)
        expfixed = a0 / (lm/2.0) * (e/s) ** (m + 1) * (1.-(1.-lm/2.0/a0)**(m+1.))
        mask = a0 < lm/2.0
        exp = expfree * mask + np.nan_to_num(expfixed * (mask == False))
        return 1. - np.exp(- exp)

        
    T = 2. * tau_arr / r + 1e-10
#     k = np.sqrt(T/E_f)
#     ef0cb = k*np.sqrt(w_arr)
   
    ef0cb = np.sqrt(w_arr[:, np.newaxis] * T[np.newaxis, :]  / E_f)
    ef0lin = w_arr[:, np.newaxis]/lm + T[np.newaxis, :]*lm/4./E_f
    depsf = T/E_f
    a0 = ef0cb/depsf
    mask = a0 < lm/2.0
    e = ef0cb * mask + ef0lin * (mask == False)
    Gxi = cdf(e, depsf, r, lm, m, sV0)
    mu_int = e * (1.-Gxi)
#     plt.plot(w_arr, np.average(Gxi, axis=1))
    sigma = mu_int*E_f
     
    return np.sum(sigma, axis=1) / n_int * (11. * 0.445) / 1000

shape = 0.057273453501868139
scale = 1.6285211551178209
sig=func1(w_arr, shape, scale)

plt.figure()
plt.plot(w_arr, sig)
plt.show()



