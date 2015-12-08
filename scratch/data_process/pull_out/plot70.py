'''
Created on 22.10.2015

@author: Yingxiong
'''
from os import listdir
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)

fpath = 'D:\\data\\pull_out\\70cm'
for fname in listdir(fpath):
    data = np.loadtxt(os.path.join(fpath, fname),  delimiter=';')
    flabel = fname.replace('-0-3300SBR', '')
    flabel = flabel.replace('DPO-', '')
    flabel = flabel.replace('.txt', '')
    plt.plot(data[0][data[0] < 25], data[1][data[0] < 25],
             next(linecycler), lw=1.5, label=flabel)
plt.xlim((0, 25))
plt.ylim((0, 35))
plt.legend()
plt.show()
