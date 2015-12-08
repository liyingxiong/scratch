'''
Created on 19.10.2015

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt


fpath31 = 'D:\\data\\pull_out\\DPO-30cm-0-3300SBR-V1.txt'
fpath32 = 'D:\\data\\pull_out\\DPO-30cm-0-3300SBR-V2.txt'

# fpath41 = 'D:\\data\\pull_out\\DPO-40cm-0-3300SBR-V2.txt'
fpath42 = 'D:\\data\\pull_out\\DPO-40cm-0-3300SBR-V3.txt'

fpath51 = 'D:\\data\\pull_out\\DPO-50cm-0-3300SBR-V1.txt'
fpath52 = 'D:\\data\\pull_out\\DPO-50cm-0-3300SBR-V2.txt'

fpath61 = 'D:\\data\\pull_out\\DPO-60cm-0-3300SBR-V1.txt'
fpath62 = 'D:\\data\\pull_out\\DPO-60cm-0-3300SBR-V2.txt'

d31 = np.loadtxt(fpath31, delimiter=';')
d32 = np.loadtxt(fpath32, delimiter=';')

# d41 = np.loadtxt(fpath41, delimiter=';')
d42 = np.loadtxt(fpath42, delimiter=';')

d51 = np.loadtxt(fpath51, delimiter=';')
d52 = np.loadtxt(fpath52, delimiter=';')

d61 = np.loadtxt(fpath61, delimiter=';')
d62 = np.loadtxt(fpath62, delimiter=';')

# plt.plot(d61[0], d61[1], 'k', label='60cm-v1, F_c=50kN')
# plt.plot(d62[0], d62[1], 'k', label='60cm-v2, F_c=60kN')
#
# plt.plot(d51[0], d51[1], 'k', label='50cm-v1, F_c=50kN')
# plt.plot(d52[0], d52[1], 'k', label='50cm-v2, F_c=50kN')
#
# plt.plot(d41[0][d41[0] <= 22.], d41[1][d41[0] <= 22.],
# 'b--', label='40cm-v2, F_c=50kN')
# plt.plot(d42[0], d42[1], 'k', label='40cm-v3, F_c=60kN')

# plt.plot(d31[0], d31[1] * 9. / 8., 'k', label='30cm-v1, F_c=30kN')
# plt.plot(d32[0], d32[1], 'k', label='30cm-v2, F_c=50kN')


fpath33 = 'D:\\data\\pull_out\\2015-10-14_DPO-30cm-0-3300SBR_R2\\DPO-30cm-0-3300SBR-V1_R2.txt'
fpath34 = 'D:\\data\\pull_out\\2015-10-14_DPO-30cm-0-3300SBR_R2\\DPO-30cm-0-3300SBR-V3_R2.txt'
fpath35 = 'D:\\data\\pull_out\\2015-10-14_DPO-30cm-0-3300SBR_R2\\DPO-30cm-0-3300SBR-V5g_R2.txt'


d33 = np.loadtxt(fpath33, delimiter=';')
d34 = np.loadtxt(fpath34, delimiter=';')
d35 = np.loadtxt(fpath35, delimiter=';')

# plt.plot(d34[0][d34[0] < 20], d34[1][d34[0] < 20],
#          label='30cm-V3_R2, sandpaper, F_c=50kN')
# plt.plot(d33[0][d33[0] < 20], d33[1][d33[0] < 20],
#          label='30cm-V1_R2, sandpaper, F_c=50kN')
# plt.plot(d35[0][d35[0] < 20], d35[1][d35[0] < 20], label='30cm-V5g_R2, glued')


# 2015-10-16_DPO-3300SBR_R2
# fpath43 = 'D:\\data\\pull_out\\2015-10-16_DPO-3300SBR_R2\\DPO-40cm-V1_R2.txt'
# d43 = np.loadtxt(fpath43, delimiter=';')
# plt.plot(d43[0][d43[0] < 20], d43[1][d43[0] < 20],
#          label='40cm-V1_R2, sandpaper, F_c=50kN')
#
# fpath44 = 'D:\\data\\pull_out\\2015-10-16_DPO-3300SBR_R2\\DPO-40cm-V4_R2.txt'
# d44 = np.loadtxt(fpath44, delimiter=';')
# plt.plot(d44[0][d44[0] < 20], d44[1][d44[0] < 20],
#          label='40cm-V4_R2, sandpaper, F_c=50kN')
#
# fpath44 = 'D:\\data\\pull_out\\2015-10-16_DPO-3300SBR_R2\\DPO-50cm-V1_R2.txt'
# d44 = np.loadtxt(fpath44, delimiter=';')
# plt.plot(d44[0][d44[0] < 20], d44[1][d44[0] < 20],
#          label='50cm-V1_R2, sandpaper, F_c=50kN')
#
# fpath44 = 'D:\\data\\pull_out\\2015-10-16_DPO-3300SBR_R2\\DPO-50cm-V2_R2.txt'
# d44 = np.loadtxt(fpath44, delimiter=';')
# plt.plot(d44[0][d44[0] < 20], d44[1][d44[0] < 20],
#          label='50cm-V2_R2, sandpaper, F_c=50kN')
#
# fpath44 = 'D:\\data\\pull_out\\2015-10-16_DPO-3300SBR_R2\\DPO-60cm-V1_R2.txt'
# d44 = np.loadtxt(fpath44, delimiter=';')
# plt.plot(d44[0][d44[0] < 20], d44[1][d44[0] < 20],
#          label='60cm-V1_R2, sandpaper, F_c=50kN')
#
# fpath44 = 'D:\\data\\pull_out\\2015-10-16_DPO-3300SBR_R2\\DPO-60cm-V2_R2.txt'
# d44 = np.loadtxt(fpath44, delimiter=';')
# plt.plot(d44[0][d44[0] < 20], d44[1][d44[0] < 20],
#          label='60cm-V2_R2, sandpaper, F_c=50kN')
# #
# fpath44 = 'D:\\data\\pull_out\\2015-10-16_DPO-3300SBR_R2\\DPO-60cm-V3_R2.txt'
# d44 = np.loadtxt(fpath44, delimiter=';')
# plt.plot(d44[0][d44[0] < 20], d44[1][d44[0] < 20],
#          label='60cm-V3_R2, sandpaper, F_c=50kN')
#
fpath44 = 'D:\\data\\pull_out\\2015-10-16_DPO-3300SBR_R2\\DPO-70cm-V1_R2.txt'
d44 = np.loadtxt(fpath44, delimiter=';')
plt.plot(d44[0][d44[0] < 20], d44[1][d44[0] < 20], marker=None, color=None,
         label='70cm-V1_R2, sandpaper, F_c=50kN')


plt.xlabel('crack opening [mm]')
plt.ylabel('force [kN]')
plt.ylim((0, 30))
plt.xlim((0, 25))
plt.legend(loc='best', ncol=2)
plt.show()
