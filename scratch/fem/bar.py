'''
Created on 14.08.2015

@author: Yingxiong
'''
import numpy as np

# Young's modulus
E = np.array([1000.])

# Gauss integration points
ip = np.array([-np.sqrt(0.6), 0., np.sqrt(0.6)])
weight = np.array([0.55555555555555, 0.88888888888888, 0.55555555555555])

# B matrix
B = np.array([[-0.5, 0.5],
              [-0.5, 0.5],
              [-0.5, 0.5]])
