'''
Created on 23.10.2015

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from os import listdir
import os
from itertools import cycle


o30 = []
n30 = []

o40 = []
n40 = []

o50 = []
n50 = []

o60 = []
n60 = []

n70 = []
n80 = []


fpath = 'D:\\data\\pull_out\\all'
for fname in listdir(fpath):
    data = np.loadtxt(os.path.join(fpath, fname),  delimiter=';')
    flabel = fname.replace('-0-3300SBR', '')
    flabel = flabel.replace('DPO-', '')
    flabel = flabel.replace('.txt', '')
    flabel = flabel.replace('cm', '')

    if not 'R2' in flabel and '30' in flabel:
        o30.append(np.amax(data[1]))
    elif not 'R2' in flabel and '40' in flabel:
        o40.append(np.amax(data[1]))
    elif not 'R2' in flabel and '50' in flabel:
        o50.append(np.amax(data[1]))
    elif not 'R2' in flabel and '60' in flabel:
        o60.append(np.amax(data[1]))
    elif '30' in flabel:
        n30.append(np.amax(data[1]))
    elif '40' in flabel:
        n40.append(np.amax(data[1]))
    elif '50' in flabel:
        n50.append(np.amax(data[1]))
    elif '60' in flabel:
        n60.append(np.amax(data[1]))
    elif '70' in flabel:
        n70.append(np.amax(data[1]))
    elif '80' in flabel:
        n80.append(np.amax(data[1]))


plt.plot([30, 40, 50, 60], [np.mean(o30), np.mean(o40), np.mean(
    o50), np.mean(o60)], '-bo', label='Leibzig')
plt.plot(30 * np.ones_like(o30), o30, 'bo', fillstyle='none')
plt.plot(40 * np.ones_like(o40), o40, 'bo', fillstyle='none')
plt.plot(50 * np.ones_like(o50), o50, 'bo', fillstyle='none')
plt.plot(60 * np.ones_like(o60), o60, 'bo', fillstyle='none')


plt.plot([30, 40, 50, 70, 80], [np.mean(n30), np.mean(n40), np.mean(
    n50), np.mean(n70), np.mean(n80)], '-ks', label='Dresden')
plt.plot(30 * np.ones_like(n30), n30, 'ks', fillstyle='none')
plt.plot(40 * np.ones_like(n40), n40, 'ks', fillstyle='none')
plt.plot(50 * np.ones_like(n50), n50, 'ks', fillstyle='none')
plt.plot(60 * np.ones_like(n60), n60, 'ks', fillstyle='none')
plt.plot(70 * np.ones_like(n70), n70, 'ks', fillstyle='none')
plt.plot(80 * np.ones_like(n80), n80, 'ks', fillstyle='none')


plt.legend(loc='best')
plt.xlim((20, 90))
plt.ylim((0, 35))
plt.show()
