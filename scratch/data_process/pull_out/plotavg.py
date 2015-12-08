'''
Created on 22.10.2015

@author: Yingxiong
'''
from os import listdir
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.interpolate import interp1d
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)
x = np.linspace(0, 25, 1000)
color = ['k', 'r', 'b', 'g', 'y', 'c']
for i, length in enumerate(['30', '40', '50', '60', '70', '80']):
    fpath = 'D:\\data\\pull_out\\' + length + 'cm'
    d_arr = np.zeros((1000, len(listdir(fpath))))
    for j, fname in enumerate(listdir(fpath)):
        data = np.loadtxt(os.path.join(fpath, fname),  delimiter=';')
#         flabel = fname.replace('-0-3300SBR', '')
#         flabel = flabel.replace('DPO-', '')
#         flabel = flabel.replace('.txt', '')
        interp = interp1d(
            data[0], data[1], bounds_error=False, fill_value=0.)
        d_arr[:, j] = interp(x)
#         plt.plot(data[0][data[0] < 25], data[1][data[0] < 25],
#                  lw=1.5, color=color[i], label=flabel)
    avg = np.mean(d_arr, axis=1)
    plt.plot(x, avg, linecycler.next(), label=length + 'cm')
plt.xlim((0, 25))
plt.ylim((0, 35))
plt.legend(ncol=2)
plt.show()
