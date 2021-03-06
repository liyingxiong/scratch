# from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from traits.api import HasTraits, Array, List, Float, Property, \
    cached_property
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gammainc, gamma
from math import pi
from scipy.optimize import basinhopping, fmin, fmin_l_bfgs_b, fmin_cobyla, brute, nnls
from scipy.interpolate import interp1d
from numpy.linalg import solve, lstsq


class Calibration(HasTraits):

    data = Array

    tau_arr = Array(input=True)

    r = Float(0.0035, auto_set=False, enter_set=True, input=True,
              distr=['uniform', 'norm'], desc='fiber radius')

    E_f = Float(200e3, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    m = Float(7., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    sV0 = Float(0.0042, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    V_f = Float(0.01, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])

    w_arr = Array

    responses = Property(depends_on='r, E_f, m, sV0, V_f')
    @cached_property
    def _get_responses(self):
        T = 2. * self.tau_arr / self.r
        # scale parameter with respect to a reference volume
        s = ((T * (self.m + 1) * self.sV0 ** self.m) / (2. * self.E_f * pi * self.r ** 2)) ** (1. / (self.m + 1))
        ef0 = np.sqrt(self.w_arr[:, np.newaxis] * T[np.newaxis, :] / self.E_f)
        Gxi = 1 - np.exp(-(ef0 / s) ** (self.m + 1))
        mu_int = ef0 * (1 - Gxi)
        I = s * gamma(1 + 1. / (self.m + 1)) * gammainc(1 + 1. / (self.m + 1), (ef0 / s) ** (self.m + 1))
        mu_broken = I / (self.m + 1)
        sigma = (mu_int + mu_broken) * self.E_f
        return sigma

    def sumed_response(self, weights_arr):
        sigma = self.responses
        return np.sum(weights_arr * sigma, axis=1)

    def residuum(self, x_arr):
        sumed_response = self.sumed_response(x_arr)
        return np.sum((self.data - sumed_response) ** 2)

#     def optimize(self):
#         b = []
#         for i in range(100):
#             b.append((0., None))
#         return fmin_l_bfgs_b(self.residuum, np.repeat(0, 100), bounds=b, approx_grad=True)

if __name__ == '__main__':

    w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2
#     w_arr = np.linspace(0.0, 8, 401)
#
#     from etsproxy.util.home_directory import \
#         get_home_directory
# 
#     import os.path
# 
#     home_dir = get_home_directory()
    
#     print home_dir
#     path = [home_dir, 'git',
#             'simvisage',
#             'quaducom',
#             'quaducom',
#             'meso',
#             'homogenized_crack_bridge',
#             'rigid_matrix',
#             'DATA', 'PO01_RYP.ASC']
    
#     path = [home_dir, 'git',
#             'rostar',
#             'scratch',
#             'diss_figs',
#             'CB1.txt']

#     filepath = os.path.join(*path)
    
    data = np.zeros_like(w_arr)
    
#     for i in range(5):
    filepath = 'F:\\Eclipse\\git\\rostar\\scratch\\diss_figs\\CB1.txt'
    file1 = open(filepath, 'r')
    cb = np.loadtxt(file1, delimiter=';')
    test_xdata = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    test_ydata = cb[:,1] / (11. * 0.445) * 1000
    interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
    data = interp(w_arr)
            
#         data = data + 0.2*data1
    
#     plt.plot(test_xdata, test_ydata)
#     plt.plot(w_arr, data)
#     plt.show()

#     cb = CBClampedRandXi()
#     data1 = []
#     for wi in w_arr:
#         data1.append(cb(wi, 6., 200e3, 0.01, 0.0035, 2.0, 0.0026))
#     data = []
#     for wi in w_arr:
#         data.append(cb(wi, 20., 200e3, 0.01, 0.0035, 2.0, 0.0026))
#     data = (0.4*np.array(data1) + 0.6*np.array(data2)) / 0.0035**2
    
#     for m in np.linspace(2., 30., 29):
#         
#         print m
    
    cali = Calibration(
                       data=data,
                       w_arr=w_arr,
                       V_f=0.00122375,
                       tau_arr=np.logspace(np.log10(1e-5), np.log10(1), 200))

#     weights = np.ones_like(cali.tau_arr)/float(len(cali.tau_arr))


# 
    def residuum(arr):
        cali.sV0 = float(arr)
        sigma = cali.responses
        sigma[0] = 1e6*np.ones_like(cali.tau_arr)
        data[0] = 1e6
        residual = nnls(sigma, data)[1]
        return residual
# 
    sV0 = brute(residuum, ((0.0001, 0.01),), Ns=20)
# 
#         m = cali.m
#         print 'shape', m, 'scale', sV0
    
#     T = 2. * cali.tau_arr / cali.r
#     s = ((T * (m + 1) * sV0 ** m) / (2. * cali.E_f * pi * cali.r ** 2)) ** (1. / (m + 1))
#     avg_eps = s*gamma(1 + 1/(m + 1))
    
#     print 'average breaking strain', avg_eps


    sigma = cali.responses
#
#     print sigma


    sigma[0] = 1e5*np.ones_like(cali.tau_arr)
    data[0] = 1e5
    
    x, y = nnls(sigma, data)
    
    sigma[0] = np.zeros_like(cali.tau_arr)
    data[0] = 0


#         print 'residual', y
    
    print np.sum(x)
#     x = cali.optimize()[0]

    
    sigma_avg = cali.sumed_response(x)
    
    plt.clf()
    plt.subplot(221)
    plt.plot(cali.w_arr, sigma_avg, '--', linewidth=2)
    plt.plot(cali.w_arr, data)
#     plt.text(0.5, 0.5, 'm='+str(m)+', sV0='+str(float(sV0))[:7])
    plt.subplot(222)
    plt.bar(np.log10(cali.tau_arr), x, width=0.02)
#     plt.plot(cali.tau_arr, x)
    plt.subplot(223)
    plt.plot(cali.w_arr, sigma)
    plt.subplot(224)
    for i, sigmai in enumerate(sigma.T):
        plt.plot(cali.w_arr, sigmai, color='0', lw='1.5', alpha=x[i]/np.max(x))
    plt.show()
    
#         savepath = 'F:\\parametric study\\cb_avg\\m='+str(m)+' sV0='+str(float(sV0))[:7]+'.png'
#         plt.savefig(savepath)





