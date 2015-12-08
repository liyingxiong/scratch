'''
Created on 10.11.2015

@author: Yingxiong
'''
import numpy as np
import pandas as pd
import time as t
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmax


def shift(d, f):
    idx = np.where((np.abs(np.diff(d)) > 2) & (np.abs(np.diff(f)) < 1.0))[0]
    print idx
    for i in idx:
        d[i + 1::] += d[i] - d[i + 1]
    return d


def avg(a, n):
    #     n = 100
    length = len(a) / int(n) * int(n)
    a = a[:length]
    return np.mean(a.reshape(-1, n), axis=1)

directory = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-18_TTb-2C-3cm-0-3300EP_cyc-Aramis-2d\\'
f = 'TTb-2C-3cm-0-3300EP-V5_cyc-Aramis2d-force.csv'
file_ = directory + f

data = pd.read_csv(file_, header=None, names=['force'])
f_arr = data.values
f_avg = avg(f_arr, 100)
disp_data = pd.read_csv(
    file_.replace('force', 'disp'), header=None, names=['disp'])
d_arr = disp_data.values
d_avg = avg(d_arr, 100)


idx = argrelmax(f_avg, order=10)[0]

f_lm = f_avg[idx]

thr = (np.mean(f_lm) + np.amax(f_lm)) * 0.5

filter = np.where(f_lm < 27)

idx = np.delete(idx, filter)

# print idx

# plt.plot(np.arange(len(avg(f_arr))), avg(f_arr))

# TTb-2C-3cm-0-3300EP-V5_cyc-Aramis2d
# plt.figure()
plt.plot(d_avg[0:1682], f_avg[0:1682], 'k')
plt.plot(d_avg[2591:2803], f_avg[2591:2803], 'k')
plt.plot(d_avg[3387:3598], f_avg[3387:3598], 'k')
plt.plot(d_avg[4580:4791], f_avg[4580:4791], 'k')
plt.plot(d_avg[53073:53286], f_avg[53073:53286],
         'k', label=f.replace('-force.csv', ''))

# TTb-2C-14mm-0-3300SBR-V1_cyc-Aramis2d
# plt.figure()
# plt.plot(np.hstack((d_avg[0:978], d_avg[1172:1340])),
#          np.hstack((f_avg[0:978], f_avg[1172:1340])), 'k')
# plt.plot(d_avg[1997:2164], f_avg[1997:2164], 'k')
# plt.plot(d_avg[3064:3232], f_avg[3064:3232], 'k')
# plt.plot(d_avg[3909:4077], f_avg[3909:4077], 'k')
# plt.plot(d_avg[40270:40438], f_avg[40270:40438],
#          'k', label=f.replace('-force.csv', ''))


f_mono = 'TTb-2C-3cm-0-3300EP-V4_Aramis2d.csv'
data_m = pd.read_csv(directory + f_mono, sep=';',
                     skiprows=[1],  decimal=',',  usecols=[1, 4, 5, 6])
force_m = data_m.Kraft.values
disp_m = -0.5 * \
    (data_m.WA1_hinten + 0.5 * (data_m.WA2_links + data_m.WA3_rechts))

f_m_avg = avg(force_m, 10)
d_m_avg = shift(avg(disp_m, 10), avg(force_m, 10))

plt.plot(d_m_avg[0:np.argmax(f_m_avg)], f_m_avg[
         0:np.argmax(f_m_avg)], label=f_mono.replace('.csv', ''))
processed = directory.replace('simdb', 'processed')
plt.xlabel('displacement [mm]')
plt.ylabel('force [kN]')
plt.legend(loc='best')
# plt.ylim((0, 35))


# plt.show()

cyc_f = np.hstack(
    (f_avg[0:1682], f_avg[2591:2803], f_avg[3387:3598], f_avg[4580:4791], f_avg[53073:53286]))
cyc_d = np.hstack(
    (d_avg[0:1682], d_avg[2591:2803], d_avg[3387:3598], d_avg[4580:4791], d_avg[53073:53286]))
cyc_data = np.vstack((cyc_f, cyc_d))
np.savetxt(
    processed + f, cyc_data, header='1st row force, 2nd row displacement')
mono_data = np.vstack(
    (f_m_avg[0:np.argmax(f_m_avg)], d_m_avg[0:np.argmax(f_m_avg)]))
np.savetxt(processed + f_mono, mono_data,
           header='1st row force, 2nd row displacement')
