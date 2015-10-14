'''
Created on Aug 5, 2015

@author: Yingxiong
'''

file_ = 'D:\data\\cyclic_loading\\2015-08-03_TTb-2C-14mm-0-3300SBR-V6_cyc-Aramis2d'

with open(file_ + '.csv', 'r') as fin:
    with open(file_ + '-dot.csv', 'w') as fout:
        for line in fin:
            fout.write(line.replace(',', '.'))
