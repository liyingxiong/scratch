'''
Created on 22.08.2014

@author: Li Yingxiong
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gamma
from scipy.interpolate import interp1d
import os.path


w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2
exp_data = np.zeros_like(w_arr)
home_dir = 'D:\\Eclipse\\'
for i in np.array([1, 2, 3, 4, 5]):
    path = [home_dir, 'git',  # the path of the data file
            'rostar',
            'scratch',
            'diss_figs',
            'CB'+str(i)+'.txt']
    filepath = os.path.join(*path)
#     exp_data = np.zeros_like(w_arr)
    file1 = open(filepath, 'r')
    cb = np.loadtxt(file1, delimiter=';')
    test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
    test_ydata = cb[:, 1] / (11. * 0.445) * 1000
    interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
    plt.plot(w_arr, interp(w_arr), label='test'+str(i))
    exp_data += 0.2*interp(w_arr)
plt.plot(w_arr, exp_data, '--', lw=2, label='average')
plt.legend()
plt.ylim((0,800))
plt.xlabel('crack opening [mm]')
plt.ylabel('fiber stress [Mpa]')
plt.show()
