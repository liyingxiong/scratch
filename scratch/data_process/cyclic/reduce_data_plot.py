'''
Created on 24.11.2015

@author: Yingxiong
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time as t
import csv
import os

directory = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-11_TTb-2C-14mm-0-3300EP_cyc-Aramis2d\\'
fname = 'TTb-2C-14mm-0-3300EP-V2_cyc-Aramis2d.csv'

n_rows = sum(1 for row in open(directory + fname))

reduced = pd.read_csv(
    directory + fname, nrows=1, sep=';', decimal=',', index_col=[0], header=[0, 1], parse_dates=[0])

print reduced

c_size = 10e4

n_chunks = n_rows / c_size + 1

i = 1

for chunk in pd.read_csv(directory + fname, chunksize=c_size, sep=';', decimal=',', parse_dates=[0], header=[0, 1], index_col=[0]):

    print '%d of %d chunks loaded' % (i, n_chunks)

    i += 1

    reduced = reduced.append(chunk.resample('1S'))

hdfname = fname.replace('csv', 'h5')

# reduced.to_hdf(directory + hdfname, 'data', mode='w')

# plt.plot_date(reduced.index.values, reduced['Kraft'], '-', marker='.')

plt.plot(-reduced['WA1_hinten'], reduced['Kraft'], marker='.')
plt.plot(-reduced['WA2_links'], reduced['Kraft'], marker='.')
plt.plot(-reduced['WA3_rechts'], reduced['Kraft'], marker='.')

plt.show()

d_save = directory.replace('simdb', 'reduced')
temp = fname.replace('.csv', '-temp.csv')

reduced.to_csv(d_save + temp, sep=';', float_format='%.4f',
               date_format='%Y.%m.%d %H:%M:%S', decmal=',')

# to delete the blank row caused by the 2-rows header
input = open(d_save + temp, 'rb')
output = open(d_save + fname, 'wb')
writer = csv.writer(output)
for i, row in enumerate(csv.reader(input)):
    if i != 2:
        writer.writerow(row)
input.close()
output.close()
os.remove(d_save + temp)