'''
Created on 22.02.2016

@author: Yingxiong
'''

import numpy as np
import StringIO
import matplotlib.pyplot as plt

bond_file = 'D:\\data\\dresden\\bond_law.txt'
s = open(bond_file).read().replace(',', '.')
bond_data = np.loadtxt(StringIO.StringIO(s), usecols=(5, 6, 8, 9, 11, 12))

x = np.linspace(0, 1.5, 1000)

for i in range(12):
    w_arr = np.array([0, bond_data[i, 1], bond_data[i, 3], bond_data[i, 5]])
    T_arr = np.array([0, bond_data[i, 0], bond_data[i, 2], bond_data[i, 4]])
    plt.plot(x, np.interp(x, w_arr, T_arr), label='V4-TA-' + str(i + 1))

plt.legend(ncol=3)
plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.show()
