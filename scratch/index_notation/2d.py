'''
Created on 24.04.2017

@author: Yingxiong
'''
import numpy as np
import sympy as sp
xi_ = sp.symbols('xi')
eta_ = sp.symbols('eta')

Ni = sp.Matrix([(1. - xi_) * (1. - eta_) / 4.0,
                (1. + xi_) * (1. - eta_) / 4.0,
                (1. + xi_) * (1. + eta_) / 4.0,
                (1. - xi_) * (1. + eta_) / 4.0], dtype=np.float_)

print np.array(Ni.subs({'xi': 1, 'eta': 1}))

#
# Nii = sp.Matrix([[xi_, -xi_], [eta_, -eta_]])
#
# print np.array(Nii.subs({'xi': 1, 'eta': 1}))
#
#
# dNi = sp.Matrix([Ni.diff('xi'), Ni.diff('eta')])
# #
# print np.array(dNi.subs({'xi': 1, 'eta': 1}))
