'''
Created on 17.11.2015

@author: Yingxiong
'''

import pandas as pd
from datetime import datetime
import time as t
import numpy as np
from matplotlib import pyplot as plt


def shift(d, f):
    idx = np.where((np.abs(np.diff(d)) > 0.2) & (np.abs(np.diff(f)) < 0.2))[0]
    print idx
    for i in idx:
        d[i + 1::] += d[i] - d[i + 1]
    return d


def avg(a):
    n = 1
    length = len(a) / int(n) * int(n)
    a = a[:length]
    return np.mean(a.reshape(-1, n), axis=1)


directory = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-10_TTb-1C-3cm-0-3300EP_cyc-Aramis2d\\'
for f in ['TTb-1C-3cm-0-3300EP-V4_cyc-Aramis2d-sideview.csv', 'TTb-1C-3cm-0-3300EP-V5_cyc-Aramis2d.csv', 'TTb-1C-3cm-0-3300EP-V6_cyc-Aramis2d-sideview-notched.csv']:
    # for f in ['TTb-2C-14mm-0-3300EP-V2_cyc-Aramis2d.csv']:

    data = pd.read_csv(directory + f, sep=';',
                       skiprows=[1],  decimal=',',  usecols=[1, 4, 5, 6])

    print data

    force = data.Kraft.values

    if f == 'TTb-1C-3cm-0-3300EP-V4_cyc-Aramis2d-sideview.csv':
        disp = -data.WA3_rechts.values
    elif f == 'TTb-1C-3cm-0-3300EP-V5_cyc-Aramis2d.csv':
        disp = -data.WA1_hinten.values
    else:
        disp = -0.5 * (data.WA2_links + data.WA3_rechts)

    f_avg = avg(force)
#     d_avg = shift(avg(disp), avg(force))
    d_avg = avg(disp)

    # plt.plot(np.arange(len(f_avg)), f_avg)

    # TTb-2C-1cm-0-800SBR-V4_cyc-Aramis2d.csv
    # plt.plot(d_avg[0:1141], f_avg[0:1141], 'k')
    # plt.plot(d_avg[2033:2201], f_avg[2033:2201], 'k')
    # plt.plot(d_avg[5774:5944], f_avg[5774:5944], 'k')
    # plt.plot(d_avg[6505:6908], f_avg[6505:6908], 'k', label=f.replace('.csv', ''))

    # TTb-2C-14mm-0-3300EP-V2_cyc-Aramis2d.csv
#     plt.plot(d_avg[0:155], f_avg[0:155], 'k')
#     plt.plot(d_avg[217:281], f_avg[217:281], 'k', label=f.replace('.csv', ''))

    # TTb-1C-3cm-0-3300EP-V5_cyc-Aramis2d.csv
    # plt.plot(d_avg[0:118], f_avg[0:118], 'k', label=f.replace('.csv', ''))

#     plt.figure()
# #
#     plt.plot(-data.WA1_hinten, f_avg, label=f.replace('.csv', '-hinten'))
#     plt.plot(-data.WA2_links, f_avg, label=f.replace('.csv', '-links'))
#     plt.plot(-data.WA3_rechts, f_avg, label=f.replace('.csv', '-rechts'))
    plt.plot(d_avg, f_avg, label=f.replace('.csv', ''))

    processed = directory.replace('simdb', 'processed')

    cyc_data = np.vstack((f_avg, d_avg))
    np.savetxt(
        processed + f, cyc_data, header='1st row force, 2nd row displacement')


plt.xlabel('displacement [mm]')
plt.ylabel('force [kN]')


for f_mono in ['TTb-1C-3cm-0-3300EP-V2_Aramis2d.csv']:
    data_m = pd.read_csv(directory + f_mono, sep=';',
                         skiprows=[1],  decimal=',',  usecols=[1, 4, 5, 6])
    force_m = data_m.Kraft.values
    disp_m = -0.5 * \
        (data_m.WA1_hinten + 0.5 * (data_m.WA2_links + data_m.WA3_rechts))

    f_m_avg = avg(force_m)
    d_m_avg = shift(avg(disp_m), avg(force_m))

    plt.plot(d_m_avg[0:np.argmax(f_m_avg)], f_m_avg[
             0:np.argmax(f_m_avg)], label=f_mono.replace('.csv', ''))
plt.legend(loc='best')
mono_data = np.vstack((f_m_avg, d_m_avg))
np.savetxt(processed + f_mono, mono_data,
           header='1st row force, 2nd row displacement')

plt.ylim((0, 35))
plt.show()

# cyc_f = np.hstack(
#     (f_avg[0:155], f_avg[217:281]))
# cyc_d = np.hstack(
#     (d_avg[0:155], d_avg[217:281]))
# cyc_data = np.vstack((cyc_f, cyc_d))
# np.savetxt(
#     processed + f, cyc_data, header='1st row force, 2nd row displacement')
# mono_data = np.vstack((f_m_avg, d_m_avg))
# np.savetxt(processed + f_mono, mono_data,
#            header='1st row force, 2nd row displacement')
