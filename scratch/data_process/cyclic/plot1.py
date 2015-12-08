'''
Created on 30.10.2015

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt


def avg(a):
    n = 50
    return np.mean(a.reshape(-1, n), axis=1)


# v1=========================

# d_file = 'D:\data\cyclic_loading\TTb-2C-1cm-0-800SBR-V1_cyc-Aramis2d-disp.csv'
# f_file = 'D:\data\cyclic_loading\TTb-2C-1cm-0-800SBR-V1_cyc-Aramis2d-force.csv'
# force = np.loadtxt(f_file)
# disp = np.loadtxt(d_file)
# plt.plot(avg(-disp[0:95000]), avg(force[0:95000]))
# plt.show()

# v2 ============================
# f_file = 'D:\data\\cyclic_loading\\TTb-2C-1cm-0-800SBR-V2_cyc-Aramis2d-force-binary'
# d_file = 'D:\data\\cyclic_loading\\TTb-2C-1cm-0-800SBR-V2_cyc-Aramis2d-disp-binary'
# force = np.fromfile(f_file)
# disp = np.fromfile(d_file)
# plt.plot(avg(-disp[0: 47500 + 2800]), avg(force[0: 47500 + 2800]))
# plt.plot(avg(-disp[130600:141000 + 11000]), avg(force[130600:141000 + 11000]))
# plt.show()

# v5 ============================
f_file = 'D:\data\\cyclic_loading\\TTb-2C-1cm-0-800SBR-V5_cyc-Aramis2d-force-binary'
d_file = 'D:\data\\cyclic_loading\\TTb-2C-1cm-0-800SBR-V5_cyc-Aramis2d-disp-binary'
force = np.fromfile(f_file)
disp = np.fromfile(d_file) / 25.
plt.plot(avg(-disp[0: 90450]), avg(force[0:90450]), 'k')
plt.plot(avg(-disp[90450: 97850 + 9500]),
         avg(force[90450: 97850 + 9500]), label='10e0')
# plt.plot((avg(-disp[90450: 97850 + 9500])[-1], avg(-disp[33884050: 33891400 + 9500])[0]),
#          (avg(force[90450: 97850 + 9500])[-1], avg(force[190500: 197850 + 9500])[0]), 'k--')
plt.plot(avg(-disp[190500: 197850 + 9500]),
         avg(force[190500: 197850 + 9500]), label='10e3')
plt.plot(avg(-disp[582700: 590150 + 9500]),
         avg(force[582700: 590150 + 9500]), label='10e4')
plt.plot(avg(-disp[3721400: 3728850 + 9500]),
         avg(force[3721400: 3728850 + 9500]), label='10e5')
plt.plot(avg(-disp[33884050: 33891400 + 9500]),
         avg(force[33884050: 33891400 + 9500]), label='10e6')
plt.plot(avg(-disp[-26601: -3001]),
         avg(force[-26601: -3001]), 'k')

plt.legend(loc='best')
plt.show()
