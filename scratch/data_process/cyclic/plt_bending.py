'''
Created on 19.11.2015

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


def shift(d, f):
    idx = np.where((np.abs(np.diff(d)) > 5) & (np.abs(np.diff(f)) < 0.1))[0]
    print idx
    for i in idx:
        d[i + 1::] += d[i] - d[i + 1]
    return d


directory = 'D:\data\\simdb\\exdata\\bending_tests\\four_point\\2015-09-02_BT-1C-55mm-0-3300SBR_cyc-Aramis2d\\'
f = 'BT-1C-55mm-0-3300EP-V2_S4P2(13)_cyc-Aramis2d.csv'


data = pd.read_csv(directory + f, sep=';',
                   skiprows=[1],  decimal=',',  usecols=[1, 5, 8])


force = -data.Kraft.values
d_middle = -data['WA Mitte'].values
d_bottom = -data['WA unten'].values

f_avg = avg(force)
d_m_avg = avg(d_middle)
d_b_avg = avg(d_bottom)

d_m_avg = shift(d_m_avg, f_avg)

# plt.plot(np.arange(len(d_m_avg)), d_m_avg, '-r.')


# BT-1C-55mm-0-3300EP-V2_S4P2(13)_cyc-Aramis2d f_threshold=0.1
plt.plot(d_m_avg[0:368 * 2], f_avg[0:368 * 2], 'k')
plt.plot(d_m_avg[510 * 2:552 * 2], f_avg[510 * 2:552 * 2], 'k')
plt.plot(d_m_avg[1118 * 2:1159 * 2], f_avg[1118 * 2:1159 * 2], 'k')
plt.plot(d_m_avg[2158 * 2:2201 * 2], f_avg[2158 * 2:2201 * 2], 'k')
plt.plot(d_m_avg[5379 * 2:5421 * 2], f_avg[5379 * 2:5421 * 2], 'k')
plt.plot(d_m_avg[35634 * 2:35677 * 2], f_avg[35634 * 2:35677 * 2], 'k')
plt.plot(d_m_avg[35830 * 2:35938 * 2],
         f_avg[35830 * 2:35938 * 2], 'k', label=f.replace('.csv', ''))

# plt.plot(d_b_avg[0:368 * 2], f_avg[0:368 * 2], 'k')
# plt.plot(d_b_avg[510 * 2:552 * 2], f_avg[510 * 2:552 * 2], 'k')
# plt.plot(d_b_avg[1118 * 2:1159 * 2], f_avg[1118 * 2:1159 * 2], 'k')
# plt.plot(d_b_avg[2158 * 2:2201 * 2], f_avg[2158 * 2:2201 * 2], 'k')
# plt.plot(d_b_avg[5379 * 2:5421 * 2], f_avg[5379 * 2:5421 * 2], 'k')
# plt.plot(d_b_avg[35634 * 2:35677 * 2], f_avg[35634 * 2:35677 * 2], 'k')
# plt.plot(d_b_avg[35830 * 2:35938 * 2],
#          f_avg[35830 * 2:35938 * 2], 'k', label=f.replace('.csv', ''))


# BT-1C-55mm-0-3300SBR-V3_S2P1(12)-cyc-Aramis2d f_threshold=0.3
# plt.plot(d_m_avg[0:470], f_avg[0:470], 'k')
# plt.plot(d_m_avg[764:849], f_avg[764:849], 'k')
# plt.plot(d_m_avg[1957:2042], f_avg[1957:2042], 'k')
# plt.plot(d_m_avg[4041:4125], f_avg[4041:4125], 'k')
# plt.plot(d_m_avg[22342:22425], f_avg[22342:22425], 'k')
# plt.plot(d_m_avg[98525:98611], f_avg[98525:98611], 'k')
# plt.plot(d_m_avg[98980:99207], f_avg[98980:99207],
#          'k', label=f.replace('.csv', ''))

# plt.plot(d_b_avg[0:470], f_avg[0:470], 'k')
# plt.plot(d_b_avg[764:849], f_avg[764:849], 'k')
# plt.plot(d_b_avg[1957:2042], f_avg[1957:2042], 'k')
# plt.plot(d_b_avg[4041:4125], f_avg[4041:4125], 'k')
# plt.plot(d_b_avg[22342:22425], f_avg[22342:22425], 'k')
# plt.plot(d_b_avg[98525:98611], f_avg[98525:98611], 'k')
# plt.plot(d_b_avg[98980:99207], f_avg[98980:99207],
#          'k', label=f.replace('.csv', ''))


f_monotonic = 'BT-1C-55mm-0-3300EP-V2_S3P2(11)-Aramis2d.csv'

data_m = pd.read_csv(directory + f_monotonic, sep=';',
                     skiprows=[1],  decimal=',',  usecols=[1, 5, 8])

# plt.plot(
#     shift(avg(-data_m['WA Mitte']), avg(-data_m['Kraft'])),
#     avg(-data_m['Kraft']), label=f_monotonic.replace('.csv', ''))

# plt.plot(-data_m['WA unten'], -data_m['Kraft'],
#          label=f_monotonic.replace('.csv', ''))

processed = directory.replace('simdb', 'processed')
# plt.xlabel('vertical displacement at center [mm]')
plt.xlabel('horizontal displacement at bottom [mm]')
plt.ylabel('force [kN]')

plt.legend(loc='best')
# plt.show()

cyc_f = np.hstack((f_avg[0:368 * 2], f_avg[510 * 2:552 * 2], f_avg[1118 * 2:1159 * 2], f_avg[
                  2158 * 2:2201 * 2], f_avg[5379 * 2:5421 * 2], f_avg[35634 * 2:35677 * 2], f_avg[35830 * 2:35938 * 2]))
cyc_d_m = np.hstack((d_m_avg[0:368 * 2], d_m_avg[510 * 2:552 * 2], d_m_avg[1118 * 2:1159 * 2], d_m_avg[
    2158 * 2:2201 * 2], d_m_avg[5379 * 2:5421 * 2], d_m_avg[35634 * 2:35677 * 2], d_m_avg[35830 * 2:35938 * 2]))
cyc_d_b = np.hstack((d_b_avg[0:368 * 2], d_b_avg[510 * 2:552 * 2], d_b_avg[1118 * 2:1159 * 2], d_b_avg[
    2158 * 2:2201 * 2], d_b_avg[5379 * 2:5421 * 2], d_b_avg[35634 * 2:35677 * 2], d_b_avg[35830 * 2:35938 * 2]))
cyc_data = np.vstack((cyc_f, cyc_d_m, cyc_d_b))
np.savetxt(processed + f, cyc_data,
           header='1st row force, 2nd row middle vertical displacement, 3rd row bottom displacement')
mono_data = np.vstack(
    (avg(-data_m['Kraft']), shift(avg(-data_m['WA Mitte']), avg(-data_m['Kraft'])), avg(-data_m['WA unten'])))
np.savetxt(processed + f_monotonic, mono_data,
           header='1st row force, 2nd row middle vertical displacement, 3rd rowbottom displacement')
