'''
Created on 05.02.2017

@author: Yingxiong
'''
from numpy import array, zeros, arange, array_equal, hstack, dot
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly


shape = 5

el_mtx = array([[10, -10],
                [-10, 10]], dtype='float_')

mtx_arr = array([el_mtx for i in range(shape)], dtype=float)
dof_map = array([arange(shape),
                 arange(shape) + 1], dtype=int).transpose()

K = SysMtxAssembly()

K.add_mtx_array(dof_map_arr=dof_map, mtx_arr=mtx_arr)
K.register_constraint(a=0,  u_a=0.)  # clamped end
K.register_constraint(a=4, u_a=2.)
K.register_constraint(a=2,  u_a=1.)

R = zeros(K.n_dofs)
R[-1] = 10.
print 'u =',  K.solve(R)

print R
print K.sys_mtx_arrays[0].mtx_arr
