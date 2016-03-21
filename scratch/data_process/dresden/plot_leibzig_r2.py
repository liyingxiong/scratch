'''
Created on 19.02.2016

@author: Yingxiong
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import StringIO
from os import listdir
from os.path import join

fpath = 'D:\\data\\leibzig'
print listdir(fpath)
for fname in listdir(fpath):
    data = np.loadtxt(
        join(fpath, fname), delimiter=';', skiprows=1, usecols=(1, 2, 3))
    l = fname.replace('.asc', '')
    data = data.T
    plt.plot((data[1] + data[2]) / 2, data[0] * 1000., label=l)

    plt.show()

plt.xlim(0, 3)
plt.ylim(0, 400)
plt.xlabel('crack opening [mm]')
plt.ylabel('force [N]')
plt.legend(ncol=3, loc='best')
plt.show()
