from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from enthought.traits.api import HasTraits, Array, List, Float, Property, \
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

    m = Float(2., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    sV0 = Float(0.0026, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    V_f = Float(0.01, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])
    
    w_arr = Array
    
    
    responses = Property(depends_on='m, sV0')
    @cached_property
    def _get_responses(self):
        T = 2. * self.tau_arr / self.r
        #scale parameter with respect to a reference volume
        s = ((T * (self.m+1) * self.sV0**self.m)/(2. * self.E_f * pi * self.r ** 2))**(1./(self.m+1))
        ef0 = np.sqrt(self.w_arr[:, np.newaxis]*T[np.newaxis, :]/self.E_f)
        Gxi = 1 - np.exp(-(ef0/s)**(self.m+1))
        mu_int = ef0 * (1-Gxi)
        I = s * gamma(1 + 1./(self.m+1)) * gammainc(1 + 1./(self.m+1), (ef0/s)**(self.m+1))
        mu_broken = I / (self.m+1)
        sigma = (mu_int + mu_broken) * self.E_f * self.V_f * self.r**2 / 0.0035**2
        return sigma
    
    def sumed_response(self, weights_arr):
        sigma = self.responses
        return np.sum(weights_arr*sigma, axis=1)
    
    def residuum(self, x_arr):
        sumed_response = self.sumed_response(x_arr)
        return np.sum((self.data - sumed_response)**2)
    
#     def fprime(self, x_arr):
        
        
    
    def optimize(self):
        b = []
        for i in range(100):
            b.append((0.,None))
        return fmin_l_bfgs_b(self.residuum, np.repeat(0, 100), bounds=b, approx_grad=True)
        


    
if __name__ == '__main__':
    
    w_arr=np.linspace(0.0, np.sqrt(8.), 401)**2
#     
    filepath = 'C:/Users/user/git/simvisage/quaducom/quaducom/meso/homogenized_crack_bridge/rigid_matrix/DATA/PO01_RYP.ASC'
    file1 = open(filepath, 'r')
    test_xdata = - np.loadtxt(file1, delimiter=';')[:,3]
    test_xdata = test_xdata - test_xdata[0]
    file2 = open(filepath, 'r')
    test_ydata = (np.loadtxt(file2, delimiter=';')[:,1] + 0.035)/0.45 * 1000
    interp = interp1d(test_xdata, test_ydata)
    data = interp(w_arr)
#     
    



#     cb = CBClampedRandXi()
#     data1 = []
#     for wi in w_arr:
#         data1.append(cb(wi, 6., 200e3, 0.01, 0.0035, 2.0, 0.0026))
#     data = []
#     for wi in w_arr:
#         data.append(cb(wi, 20., 200e3, 0.01, 0.0035, 2.0, 0.0026))
#     data = (0.4*np.array(data1) + 0.6*np.array(data2)) / 0.0035**2
        
    cali = Calibration(data=data,
                       w_arr=w_arr,
                       tau_arr = np.linspace(0.001, 10, 200))
    
#     weights = np.ones_like(cali.tau_arr)/float(len(cali.tau_arr))
        

    
    def residuum(arr):
        cali.m = arr[0]
        cali.sV0 = arr[1]
        sigma = cali.responses
        residual = nnls(sigma[1:,], data[1:])[1]
#         sigma_avg = cali.sumed_response(x)
        return residual
      
    m, s = brute(residuum, ((0,10),(0.0001, 0.01)), Ns=60)
   
    print m,s
    
    sigma = cali.responses
#   
#     print sigma
    
    x, y = nnls(sigma[1:], data[1:])
    
    print y

#     x = cali.optimize()[0]

  
    sigma_avg = cali.sumed_response(x)
#     
    plt.subplot(221)
    plt.plot(cali.w_arr, sigma_avg,'--', linewidth=2)
    plt.plot(cali.w_arr,data)
    plt.subplot(222)
    plt.bar(cali.tau_arr, x, width=0.4)
    plt.subplot(223)
    plt.plot(cali.w_arr, sigma)
    plt.subplot(224)
    plt.plot(cali.w_arr, data-sigma_avg)


    
    plt.show()
    
                
                
                

            