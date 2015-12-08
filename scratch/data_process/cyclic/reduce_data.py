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
        (np.abs(np.diff(d)) > 0.02) & (np.abs(np.diff(f)) < 0.02))[0]
    print idx
    for i in idx:
        d[i + 1::] += d[i] - d[i + 1]
    return d


directory = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-03_TTb-2C-14mm-0-3300SBR_cyc-Aramis2d\\'
fname = 'TTb-2C-14mm-0-3300SBR-V1_cyc-Aramis2d.h5'

reduced = pd.DataFrame()

data = pd.read_hdf(directory + fname, 'data')

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

# BT-1C-55mm-0-3300SBR-V3_S2P1(12)-cyc-Aramis2d
# reduced = reduced.append(data.loc['2015-09-23 08:38:39':'2015-09-23 08:46:24'])
# reduced = reduced.append(data.loc['2015-09-23 08:51:20':'2015-09-23 08:52:47'])
# reduced = reduced.append(data.loc['2015-09-23 09:11:15':'2015-09-23 09:12:41'])
# reduced = reduced.append(data.loc['2015-09-23 09:45:56':'2015-09-23 09:47:22'])
# reduced = reduced.append(data.loc['2015-09-23 14:50:58':'2015-09-23 14:52:25'])
# reduced = reduced.append(data.loc['2015-09-24 12:00:46':'2015-09-24 12:02:13'])
# reduced = reduced.append(data.loc['2015-09-24 12:08:28':'2015-09-24 12:12:19'])

# BT-1C-55mm-0-3300EP-V2_S4P2(13)_cyc-Aramis2d
# reduced = reduced.append(data.loc['2015-09-24 12:58:31':'2015-09-24 13:10:46'])
# reduced = reduced.append(data.loc['2015-09-24 13:15:29':'2015-09-24 13:16:55'])
# reduced = reduced.append(data.loc['2015-09-24 13:35:44':'2015-09-24 13:37:10'])
# reduced = reduced.append(data.loc['2015-09-24 14:10:29':'2015-09-24 14:11:54'])
# reduced = reduced.append(data.loc['2015-09-24 15:57:50':'2015-09-24 15:59:15'])
# reduced = reduced.append(data.loc['2015-09-25 08:46:29':'2015-09-25 08:47:55'])
# reduced = reduced.append(data.loc['2015-09-25 08:52:43':'2015-09-25 08:56:38'])

# TTb-2C-14mm-0-3300SBR-V1_cyc-Aramis2d
reduced = reduced.append(data.loc['2015-04-08 10:18:19':'2015-04-08 10:29:29'])
reduced = reduced.append(data.loc['2015-04-08 10:34:59':'2015-04-08 10:36:23'])
reduced = reduced.append(data.loc['2015-04-08 10:43:55':'2015-04-08 10:45:19'])
reduced = reduced.append(data.loc['2015-04-08 10:50:58':'2015-04-08 10:52:21'])
reduced = reduced.append(data.loc['2015-04-08 11:21:11':'2015-04-08 11:22:35'])
reduced = reduced.append(data.loc['2015-04-08 15:54:06':'2015-04-08 15:55:30'])

# TTb-2C-3cm-0-3300EP-V5_cyc-Aramis2d
# reduced = reduced.append(data.loc['2015-08-19 08:23:13':'2015-08-19 08:34:27'])
# reduced = reduced.append(data.loc['2015-08-19 08:40:28':'2015-08-19 08:41:55'])
# reduced = reduced.append(data.loc['2015-08-19 08:45:59':'2015-08-19 08:47:13'])
# reduced = reduced.append(data.loc['2015-08-19 08:53:46':'2015-08-19 08:55:13'])
# reduced = reduced.append(data.loc['2015-08-19 09:25:30':'2015-08-19 09:26:55'])
# reduced = reduced.append(data.loc['2015-08-19 14:17:07':'2015-08-19 14:18:34'])

# TTb-2C-1cm-0-800SBR-V4_cyc-Aramis2d
# reduced = reduced.append(data.loc['2015-04-22 08:08:35':'2015-04-22 08:18:08'])
# reduced = reduced.append(data.loc['2015-04-22 08:25:33':'2015-04-22 08:26:59'])
# reduced = reduced.append(data.loc['2015-04-22 08:56:44':'2015-04-22 08:58:10'])
# reduced = reduced.append(data.loc['2015-04-22 09:02:50':'2015-04-22 09:06:09'])

print reduced
print reduced.dropna(how='all')

# disp_center = -reduced['WA Mitte'].values.flatten()
#
# force = -reduced['Kraft'].values.flatten()
#
# print np.diff(disp_center)
#
# print np.where((np.abs(np.diff(disp_center)) > 5) &
# (np.abs(np.diff(force)) < 0.1))[0]

plt.plot(-reduced['WA1_hinten'], reduced['Kraft'])
plt.plot(-reduced['WA2_links'], reduced['Kraft'])
plt.plot(-reduced['WA3_rechts'], reduced['Kraft'])

plt.show()

d_save = directory.replace('simdb', 'reduced')
f_save = fname.replace('h5', 'csv')
temp = fname.replace('.h5', '-temp.csv')

reduced.to_csv(d_save + temp, sep=';', float_format='%.4f',
               date_format='%Y.%m.%d %H:%M:%S', decmal=',')

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
