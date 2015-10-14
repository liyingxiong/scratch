'''
Created on Aug 5, 2015

@author: Yingxiong
'''
import numpy as np

file_ = 'D:\data\\cyclic_loading\\2015-08-03_TTb-2C-14mm-0-3300SBR-V6_cyc-Aramis2d-force.csv'

a = np.loadtxt(file_)

fout = 'D:\data\\cyclic_loading\\2015-08-03_TTb-2C-14mm-0-3300SBR-V6_cyc-Aramis2d-force-binary'

a.tofile(fout)


print len(a)

b = np.fromfile(fout)

print len(b)
