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

delta_f_arr = np.array(
    [13.9, 14.0, 13.9, 13.9, 13.7, 13.7, 13.6, 13.6, 13.4, 13.5])
delta_f = np.mean(delta_f_arr)
print delta_f

fpath = 'D:\\data\\dresden_r2'
print listdir(fpath)
for fname in listdir(fpath):
    s = open(join(fpath, fname)).read().replace(',', '.')
    data = np.loadtxt(StringIO.StringIO(s), comments='*', skiprows=39)
    l = fname.replace('2AZ2-', '')
    l = l.replace('.asc', '')
    data = data.T
    plt.plot((data[1] + data[2]) / 2,
             (data[3] - delta_f * data[3] / np.amax(data[3])), label=l)

plt.xlim(0, 3)
plt.ylim(0, 250)
plt.xlabel('crack opening [mm]')
plt.ylabel('force [N]')
plt.legend(ncol=3, loc='best')
plt.show()
