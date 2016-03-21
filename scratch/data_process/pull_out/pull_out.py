'''
Created on Jul 29, 2015

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt

data = 'D:\\data\\pull_out_r4\\DPOR4103.asc'

scaling = 0

a = np.loadtxt(data, delimiter=';')


f = a[:, 1]

if scaling:  # only 8 yarns
    f = f * 9. / 8.

wli = -a[:, 2]
wre = -a[:, 3]
wvo = -a[:, 4]

# plt.plot(w1, f)
# plt.plot(w2, f)
# plt.plot(w3, f)

# plt.figure()
# plt.plot(np.arange(len(w1)), w1, 'ro', label='1')
# plt.figure()
# plt.plot(np.arange(len(w2)), w2, 'ro', label='2')
# plt.figure()
# plt.plot(np.arange(len(w3)), w3, 'ro', label='3')


def shift(w):
    idx = np.where(np.abs(np.diff(w)) > 0.3)[0]
    for i in np.arange(len(idx) - 1):
        if idx[i + 1] == idx[i] + 1:
            w[idx[i + 1]] = w[idx[i]]
        else:
            pass
#     w[idx[1::]] = w[idx[0]]
#     plt.figure()
#     plt.plot(np.arange(len(w)), w, 'ro')
#     plt.show()
#     w[np.argmin(np.diff(w)) + 1::] += w[np.argmin(np.diff(w))] - \
#         w[np.argmin(np.diff(w)) + 1]
#     w[np.argmin(np.diff(w)) + 1::] += w[np.argmin(np.diff(w))] - \
#         w[np.argmin(np.diff(w)) + 1]
    idx1 = np.where(np.abs(np.diff(w)) > 0.3)[0]
    print idx1
    for i in idx1:
        w[i + 1::] += w[i] - w[i + 1]
    return w

#
# w1 = shift(w1)
# w2 = shift(w2)
# w3 = shift(w3)
#
# ==================================

plt.plot(wli, f, label='li')
plt.plot(wre, f, label='re')
plt.plot(wvo, f, label='vo')
plt.legend()
plt.show()

# plt.figure()
# plt.plot(np.arange(len(w1)), w1, label='1')
# plt.plot(np.arange(len(w2)), w2, label='2')
# plt.plot(np.arange(len(w3)), w3, label='3')
# plt.legend()

#========================================================

w_avg = ((wli + wre) / 2 + wvo) / 2

# w_avg = w2

# w_avg = (w1 + w2) / 2
#
# plt.figure()
# plt.plot(w_avg, f)
save = 1
if save:
    fpath = data.replace('.ASC', '.txt')
    fpath = fpath.replace('r4', 'r4_avg')
    print fpath
    np.savetxt(fpath, np.vstack((w_avg[w_avg < 23], f[w_avg < 23])), fmt='%.8f', delimiter=';',
               header='first line crack opening; second line force')
    d = np.loadtxt(fpath, delimiter=';')
    plt.figure()
    plt.plot(d[0], d[1])

    plt.show()

# plt.plot(w)

# w2, f2 = shift(w2, f)
# w3, f3 = shift(w3, f)


#=================================================================


# plt.plot(w1, f1, label='1')
# plt.plot(w2, f2, label='2')
# plt.plot(w3, f3, label='3')
# plt.legend()
# plt.show()

#=================================================================

# w_max = np.amin([np.amax(w1), np.amax(w2), np.amax(w3)])
# w_min = np.amax([np.amin(w1), np.amin(w2), np.amin(w3)])
# w_arr = np.linspace(w_min, w_max, 2000)
#
# interp1 = interp1d(w1, f1)
# interp2 = interp1d(w2, f2)
# interp3 = interp1d(w3, f3)
#
# f_avg = ((interp1(w_arr) + interp3(w_arr)) / 2 + interp2(w_arr)) / 2
#
# plt.plot(w_arr, f_avg)
#
#
#
# plt.show()
