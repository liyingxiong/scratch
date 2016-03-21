'''
Created on 25.11.2015

@author: Yingxiong
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time as t
import csv
import os


def shift(d, f):
    'restore the jump in displacement caused by resetting of the gauge'
    idx = np.where(
        (np.abs(np.diff(d)) > 1.) & (np.abs(np.diff(f)) < 0.05))[0]
    print idx
    for i in idx:
        d[i + 1::] += d[i] - d[i + 1]
    return d


directory = 'D:\data\\'
fname = 'BT-3PT-1s-20cm-d8mm-RF2_2C-cyc-Aramis2d.h5'

# directory = 'D:\data\\unprocessed\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-18_TTb-2C-3cm-0-3300EP_cyc-Aramis-2d\\'
# fname = 'TTb-2C-3cm-0-3300EP-V2_cyc-Aramis2d-sideview.h5'

# directory = 'D:\data\\unprocessed\\exdata\\bending_tests\\four_point\\2015-09-02_BT-1C-55mm-0-3300SBR_cyc-Aramis2d\\'
# fname = 'BT-1C-55mm-0-3300EP-V3b_S1P1(7)_cyc-Aramis2d.h5'


reduced = pd.DataFrame()
data = pd.read_hdf(directory + fname, 'data')
print data

# data = pd.read_csv(directory + fname, sep=';', decimal='.',
#                    parse_dates=[0], header=[0, 1], index_col=[0])

# plt.plot_date(data.index, data['Kraft'], '-', marker='.')
#
# plt.show()
#
# restore the jumps
# disp_center = -data['WA Mitte'].values.flatten()
# force = -data['Kraft'].values.flatten()
# data['WA Mitte'] = - shift(disp_center, force)

# disp_v = -data['WA1_hinten'].values.flatten()
# disp_l = -data['WA2_links'].values.flatten()
# disp_r = -data['WA3_rechts'].values.flatten()
# force = data['Kraft'].values.flatten()
# data['WA1_vorne'] = - shift(disp_v, force)
# data['WA2_links'] = - shift(disp_l, force)
# data['WA3_rechts'] = - shift(disp_r, force)

# BT-1C-55mm-0-3300EP-V1_S3P2(10)-cyc-Aramis2d
# reduced = reduced.append(data.loc['2015-09-22 10:28:33':'2015-09-22 10:42:52'])

# TTb-2C-3cm-0-3300EP-V2_cyc-Aramis2d-sideview
# reduced = reduced.append(data.loc['2015-08-18 11:23:45':'2015-08-18 11:34:09'])
# reduced = reduced.append(data.loc['2015-08-18 11:38:14':'2015-08-18 11:42:35'])

# BT-4PT-1s-20cm-d8mm-RF2_2C-cyc-Aramis2d_processed
# reduced = reduced.append(data.loc['2016-01-19 09:10:19':'2016-01-19 09:31:17'])
# reduced = reduced.append(data.loc['2016-01-19 09:37:43':'2016-01-19 09:39:07'])
# reduced = reduced.append(data.loc['2016-01-19 09:58:34':'2016-01-19 09:59:59'])
# reduced = reduced.append(data.loc['2016-01-19 10:19:24':'2016-01-19 10:20:49'])
# reduced = reduced.append(data.loc['2016-01-19 10:55:03':'2016-01-19 10:56:27'])
# reduced = reduced.append(data.loc['2016-01-19 15:27:03':'2016-01-19 15:28:28'])

# BT-3PT-1s-20cm-d8mm-RF2_2C-cyc-Aramis2d
reduced = reduced.append(data.loc['2016-01-25 10:48:21':'2016-01-25 11:36:23'])
reduced = reduced.append(data.loc['2016-01-25 11:41:40':'2016-01-25 11:43:05'])
reduced = reduced.append(data.loc['2016-01-25 11:52:17':'2016-01-25 11:53:41'])
reduced = reduced.append(data.loc['2016-01-25 12:02:04':'2016-01-25 12:03:28'])
reduced = reduced.append(data.loc['2016-01-25 13:03:04':'2016-01-25 13:04:28'])
reduced = reduced.append(data.loc['2016-01-25 17:22:38':'2016-01-25 17:24:01'])
reduced = reduced.append(data.loc['2016-01-25 17:41:00':'2016-01-25 17:42:25'])
reduced = reduced.append(data.loc['2016-01-26 07:31:54':'2016-01-26 07:33:18'])
reduced = reduced.append(data.loc['2016-01-27 09:09:22':'2016-01-27 09:10:47'])
reduced = reduced.append(data.loc['2016-01-27 09:16:45':'2016-01-27 09:25:54'])


# reduced = data

plt.plot(-reduced['Weg'], reduced['Kraft'])
# plt.plot(-reduced['WA_M2'], reduced['Kraft'])
# plt.plot(-reduced['Schub1'], reduced['Kraft'])

# plt.plot(-reduced['WA Mitte'].values, -reduced['Kraft'].values)

plt.show()
#
save = 1

if save:
    d_save = directory.replace('unprocessed', 'reduced')
    f_save = fname.replace('h5', 'csv')
    temp = fname.replace('.h5', '-temp.csv')

    reduced.to_csv(d_save + temp, sep=';', decimal=',', float_format='%.4f',
                   date_format='%Y.%m.%d %H:%M:%S')

    # to delete the blank row caused by the 2-rows header
    input = open(d_save + temp, 'rb')
    output = open(d_save + f_save, 'wb')
    writer = csv.writer(output)
    for i, row in enumerate(csv.reader(input)):
        if i != 2:
            writer.writerow(row)
    input.close()
    output.close()
    os.remove(d_save + temp)
