'''
Created on 07.07.2016

@author: Yingxiong
'''
from cbfe.fets1d52ulrh import FETS1D52ULRH
from ibvpy.api import BCDof
from matseval import MATSEval_cyc, MATSEval
from tloop import TLoop
from tstepper import TStepper
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

ts = TStepper()
n_dofs = ts.domain.n_dofs
d_array = np.array([0., 500., 200., 500., 200., 500., 200., 500.])
dd_arr = np.abs(np.diff(d_array))
x = np.hstack((0, np.cumsum(dd_arr) / sum(dd_arr)))
tf = interp1d(x, d_array)

# a = np.linspace(0, 1, 10000)
#
# plt.plot(a, tf(a))
# plt.show()


ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
              BCDof(var='f', dof=n_dofs - 1, value=1., time_function=tf)]

tl = TLoop(ts=ts, d_t=0.001)

U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()
# print 'U_record', U_record

plt.plot(U_record[:, n_dofs - 1], F_record[:, n_dofs - 1])
plt.xlabel('displacement')
plt.ylabel('force')
plt.show()
