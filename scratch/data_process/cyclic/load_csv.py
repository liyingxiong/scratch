'''
Created on 10.11.2015

@author: Yingxiong
'''
import pandas as pd
from datetime import datetime
import time as t

directory = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-18_TTb-2C-3cm-0-3300EP_cyc-Aramis-2d\\'
f = 'TTb-2C-3cm-0-3300EP-V5_cyc-Aramis2d.csv'

dtypes = [datetime, float, float, float, float, float, float]

chunksize = 10 ** 5
i = 0
t1 = t.time()
for chunk in pd.read_csv(directory + f, sep=';', skiprows=[1], parse_dates=[0], decimal=',',
                         index_col=0, usecols=[0, 1, 4, 5, 6], chunksize=chunksize):
    print chunk
print 'loading time:', t.time() - t1

# data = data.set_index('Datum/Uhrzeit')
# print data.resample('1S')
