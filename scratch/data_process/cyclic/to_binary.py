'''
Created on Aug 5, 2015

@author: Yingxiong
'''
import numpy as np
import pandas as pd
import time as t
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmax


def avg(a):
    n = 50
    length = len(a) / int(n) * int(n)
    a = a[:length]
    return np.mean(a.reshape(-1, n), axis=1)


# file_ = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-04-20_TTb-2C-1cm-0-800SBR_cyc\\TTb-2C-1cm-0-800SBR-V1_Aramis2d.csv'
directory = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-18_TTb-2C-3cm-0-3300EP_cyc-Aramis-2d\\'
f = 'TTb-2C-3cm-0-3300EP-V5_cyc-Aramis2d-force.csv'
file_ = directory + f

# file_ = 'D:\data\\temp\\TTb-2C-1cm-0-800SBR-V5_cyc-Aramis2d-force.csv'
data = pd.read_csv(file_, header=None, names=['force'])
f_arr = data.values

# f_slice = f_arr[:700000]

f_slice = f_arr

idx = argrelmax(f_slice, order=10)[0]

print len(idx)

f_lm = f_slice[idx]

thr = (np.mean(f_lm) + np.amax(f_lm)) * 0.5

filter = np.where(f_lm < 40)

idx = np.delete(idx, filter)

plt.plot(np.arange(len(idx)), idx, '-r.')

diff = np.diff(idx)

cycles = idx[np.where((diff > 5000) & (diff < 25000))[0]]

print cycles

disp_data = pd.read_csv(
    file_.replace('force', 'disp'), header=None, names=['disp'])
d_arr = disp_data.values

for i in cycles:
    plt.figure()
    plt.plot(avg(d_arr[i:i + 22000]), avg(f_arr[i:i + 22000]))
    plt.title(str(i))
plt.show()


# x = np.linspace(0, 3000, 10000000)
# y = np.sin(x)
#
# data = pd.DataFrame(y)
#
#
# plt.plot(x, y)
# plt.plot(x[idx], y[idx], 'ro')
# plt.show()

# diff = data.diff()

# c = diff[1:-1] * diff[2:]


# print diff
#
# print diff[1:-1]
#
# print diff[2:]
#
# print diff.iloc[1:-1].values * diff.iloc[2:].values

# plt.plot(x, data.diff())
# plt.show()

# def avg(a):
#     n = 50
#     return np.mean(a.reshape(-1, n), axis=1)
#
# file_f = 'D:\data\\temp\\TTb-2C-1cm-0-800SBR-V5_cyc-Aramis2d-force.csv'
#
# directory = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-03_TTb-2C-14mm-0-3300SBR_cyc-Aramis2d\\'
# force_file = 'TTb-2C-14mm-0-3300SBR-V1_cyc-Aramis2d-force.csv'
#
# file_f = directory + force_file
#
# force = pd.read_csv(file_f, header=None, names=['force'])
#
# f_arr = force.values
#
# length = len(f_arr) / int(50) * int(50)
#
# file_d = file_f.replace('force', 'disp')
#
# disp = pd.read_csv(file_d, header=None, names=['disp'])
#
# d_arr = disp.values
#
# plt.plot(avg(d_arr[:length]), avg(f_arr[:length]))
#
# plt.show()


#
# b = np.diff(a.T)[0]
#
# plt.plot(np.arange(len(b)), b, '-ro')
#
# plt.show()

# lmidx = argrelextrema(a, np.greater)
#
#
# plt.plot(np.arange(len(lmidx[0])), lmidx[0], '-ro')

# plt.show()

# plt.plot(np.arange(10000), force[10000000: 10010000], 'ro')
# plt.show()
#
# t1 = t.time()
#
# a = pd.read_hdf(file_.replace('.csv', 'data'), 'data')
# print len(a)
#
# print t.time() - t1

#
# fout = file_.replace('.csv', '') + '-binary'
#
# a.tofile(fout)
#
#
# print len(a)
#
# b = np.fromfile(fout)
#
# print len(b)
