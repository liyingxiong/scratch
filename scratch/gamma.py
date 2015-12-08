'''
Created on 15.10.2015

@author: Yingxiong
'''
from spirrid.rv import RV
import numpy as np
import matplotlib.pyplot as plt

tau = RV('gamma', shape=0.539, scale=1.44, loc=0.00126)

x = np.linspace(0, 1, 1000)

y = tau.pdf(x)

plt.plot(x, y)
plt.xlabel('bond strength [Mpa]')
plt.title('probability distribution function')

plt.show()