'''
Created on 14.10.2015

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt

fpath31 = 'D:\\data\\pull_out\\DPO-30cm-0-3300SBR-V1.txt'
fpath32 = 'D:\\data\\pull_out\\DPO-30cm-0-3300SBR-V2.txt'

fpath33 = 'D:\\data\\pull_out\\2015-10-14_DPO-30cm-0-3300SBR_R2\\DPO-30cm-0-3300SBR-V1_R2.txt'
fpath34 = 'D:\\data\\pull_out\\2015-10-14_DPO-30cm-0-3300SBR_R2\\DPO-30cm-0-3300SBR-V3_R2.txt'
fpath35 = 'D:\\data\\pull_out\\2015-10-14_DPO-30cm-0-3300SBR_R2\\DPO-30cm-0-3300SBR-V5g_R2.txt'


d31 = np.loadtxt(fpath31, delimiter=';')
d32 = np.loadtxt(fpath32, delimiter=';')
d33 = np.loadtxt(fpath33, delimiter=';')
d34 = np.loadtxt(fpath34, delimiter=';')
d35 = np.loadtxt(fpath35, delimiter=';')

plt.plot(d31[0], d31[1] * 9. / 8., label='30cm-v1, rubber, F_c=30kN')
plt.plot(d32[0], d32[1], label='30cm-v2, rubber, F_c=50kN')
plt.plot(d34[0][d34[0] < 20], d34[1][d34[0] < 20],
         label='30cm-V3_R2, sandpaper, F_c=50kN')
plt.plot(d33[0][d33[0] < 20], d33[1][d33[0] < 20],
         label='30cm-V1_R2, sandpaper, F_c=50kN')
plt.plot(d35[0][d35[0] < 20], d35[1][d35[0] < 20], label='30cm-V5g_R2, glued')

plt.xlabel('crack opening [mm]')
plt.ylabel('force [kN]')
plt.ylim((0, 20))
plt.xlim((0, 25))
plt.legend(loc='best', ncol=2)
plt.show()
