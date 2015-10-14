'''
Created on Jul 30, 2015

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt

fpath31 = 'D:\\data\\pull_out\\DPO-30cm-0-3300SBR-V1.txt'
fpath32 = 'D:\\data\\pull_out\\DPO-30cm-0-3300SBR-V2.txt'

fpath41 = 'D:\\data\\pull_out\\DPO-40cm-0-3300SBR-V2.txt'
fpath42 = 'D:\\data\\pull_out\\DPO-40cm-0-3300SBR-V3.txt'

fpath51 = 'D:\\data\\pull_out\\DPO-50cm-0-3300SBR-V1.txt'
fpath52 = 'D:\\data\\pull_out\\DPO-50cm-0-3300SBR-V2.txt'

fpath61 = 'D:\\data\\pull_out\\DPO-60cm-0-3300SBR-V1.txt'
fpath62 = 'D:\\data\\pull_out\\DPO-60cm-0-3300SBR-V2.txt'

d31 = np.loadtxt(fpath31, delimiter=';')
d32 = np.loadtxt(fpath32, delimiter=';')

d41 = np.loadtxt(fpath41, delimiter=';')
d42 = np.loadtxt(fpath42, delimiter=';')

d51 = np.loadtxt(fpath51, delimiter=';')
d52 = np.loadtxt(fpath52, delimiter=';')

d61 = np.loadtxt(fpath61, delimiter=';')
d62 = np.loadtxt(fpath62, delimiter=';')

plt.plot(d61[0], d61[1], 'g--', label='60cm-v1, F_c=50kN')
plt.plot(d62[0], d62[1], 'g', label='60cm-v2, F_c=60kN')

plt.plot(d51[0], d51[1], 'k--', label='50cm-v1, F_c=50kN')
plt.plot(d52[0], d52[1], 'k', label='50cm-v2, F_c=50kN')

plt.plot(d41[0][d41[0] <= 22.], d41[1][d41[0] <= 22.],
         'b--', label='40cm-v2, F_c=50kN')
plt.plot(d42[0], d42[1], 'b', label='40cm-v3, F_c=60kN')

plt.plot(d31[0], d31[1] * 9. / 8., 'r--', label='30cm-v1, F_c=30kN')
plt.plot(d32[0], d32[1], 'r', label='30cm-v2, F_c=50kN')

plt.xlabel('crack opening [mm]')
plt.ylabel('force [kN]')
plt.ylim((0, 35))
plt.xlim((0, 25))
plt.legend(loc='best')
plt.show()
