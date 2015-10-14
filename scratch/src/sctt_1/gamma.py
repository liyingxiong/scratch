'''
Created on Feb 5, 2015

@author: Li Yingxiong
'''
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
from spirrid.rv import RV


from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC

tau_shape = [0.079392235619918011, 0.070557619484416842, 0.063299300020833421, 0.057273453501868139, 0.052218314012133477, 0.04793148098051675]
tau_scale = [0.85377504364710732, 1.0754895775375046, 1.3336402595345596, 1.6285211551178209, 1.9598281655654299, 2.3273933214348754]

# p = np.linspace(1e-5, 1., 1000)
x = np.linspace(0., 0.5, 1000)


for i, k in enumerate(tau_shape):
    tau = RV('gamma', shape=k, scale=tau_scale[i], loc=0.)
#     x = tau.ppf(p)
    plt.plot(x, tau.pdf(x), label=str(i))
plt.legend()
plt.show()
    
    


# rv1 = fibers_MC(m = 7.0, sV0=0.0095)
# 
# 
# pdf = rv.pdf(x)
# 
# plt.plot(x, pdf, lw=2)
# 
# plt.xlabel('bond strength [MPa]')
# 
# plt.show()