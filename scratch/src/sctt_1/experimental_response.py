'''
Created on Jan 27, 2015

@author: Li Yingxiong
'''
import os
import numpy as np
from matplotlib import pyplot as plt


# home_dir = 'D:\\eclipse\\'
# path = [home_dir, 'git',  # the path of the data file
# 'rostar',
# 'scratch',
# 'diss_figs',
# 'TT-6C-0'+str(1)+'.txt']
# filepath2 = filepath = os.path.join(*path)
# data = np.loadtxt(filepath2, delimiter=';')
# plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, color='0.5', label='vf=1.5%')

home_dir = 'D:\\eclipse\\'
path = [home_dir, 'git',  # the path of the data file
'rostar',
'scratch',
'diss_figs',
'TT-4C-0'+str(1)+'.txt']
filepath2 = filepath = os.path.join(*path)
data = np.loadtxt(filepath2, delimiter=';')
plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., '--', lw=2, label='vf=1.0%')

plt.xlabel('strain')
plt.ylabel('stress [MPa]')

# plt.legend()
plt.show()

