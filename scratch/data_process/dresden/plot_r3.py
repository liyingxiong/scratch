'''
Created on 19.02.2016

@author: Yingxiong
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import StringIO

delta_f = np.array(
    [14.0, 13.9, 13.9, 13.8, 13.7, 13.7, 13.7, 13.5, 13.5, 13.6, 13.5, 13.4])

for i in range(12):
    if i >= 9:
        fpath = 'D:\\data\\dresden\\V4-TA-' + str(i + 1) + '.asc'
        l = 'V4-TA-' + str(i + 1)
    else:
        fpath = 'D:\\data\\dresden\\V4-TA-0' + str(i + 1) + '.asc'
        l = 'V4-TA-0' + str(i + 1)

    s = open(fpath).read().replace(',', '.')
    data = np.loadtxt(StringIO.StringIO(s), comments='*')

    data = data.T
    plt.plot((data[1] + data[2]) / 2, (data[3] - delta_f[i]), label=l)

plt.xlim(0, 3)
plt.ylim(0, 600)
plt.xlabel('crack opening [mm]')
plt.ylabel('force [N]')
plt.legend(ncol=3)
plt.show()
