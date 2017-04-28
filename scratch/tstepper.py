'''
Created on 12.01.2016

@author: Yingxiong
'''
import numpy as np
from debontrix.debontrix.fem.matseval import MATSEval
from debontrix.fem.element import FETS1D52ULRH


class TStepper(object):

    '''Time stepper object for non-linear Newton-Raphson solver.
    '''

    def __init__(self, **kwargs):
        # material
        self.mats_eval = MATSEval()
        # element
        self.fets_eval = FETS1D52ULRH()
        # the cross-sectional area of the matrix
        self.A_m = 1.
        # the cross-sectional area of the reinforcement
        self.A_f = 1.
        # Number of elements
        self.n_e_x = 50
        # total specimen length
        self.L_x = 600.0
        # set the keyword parameters (to replace the default values)
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

        # the cross-sectional area of matrix, bond interface and reinforcement
        self.A = np.array([self.A_m, 1., self.A_f])
        # the total number of DoFs
        self.n_dofs = 2 * (self.n_e_x + 1)
        # meshing, return the array containing the coordinates of each element
        self.elem_x_map = self.get_elem_x_map()
        # the DoFs of each element on each node
        self.elem_dof_map = np.array([2 * i + [[0, 1], [2, 3]]
                                      for i in np.arange(self.n_e_x)])
        # Array of Jacobian matrices
        self.J_mtx = self.get_J_mtx()
        # Array of Jacobi determinants
        self.J_det = self.get_J_det()
        # the B (displacement-strain) matrix
        self.B = self.get_B()

    def get_elem_x_map(self):
        # array containing the x coordinates of each node
        x_arr = np.linspace(0, self.L_x, self.n_e_x + 1, endpoint=True)

        # reshape the x_arr in a 3D array, this can be applied to any general elements
        # 1st-dimension: element [e]
        # 2nd-dimension: node [n]
        # 3rd-dimension: spatial dimensions [d]
        # example, elem_x_map[0,0,0] - first element, first node, x coordinate
        # elem_x_map[1,1,1] - second element, second node, y coordinate

        # the nodes of each element
        elem_node = np.array([[i, i + 1] for i in np.arange(self.n_e_x)])
        # map the x coordinates [e, n, d]
        elem_x_map = x_arr[elem_node[:, :, np.newaxis]]
        return elem_x_map

    def get_J_mtx(self):
        '''
        i -- integration point

        d -- spatial dimension
        f -- alternative for spatial dimension 
        n -- element node
        e -- element
        '''
        fets_eval = self.fets_eval
        # [ i, d ]
        ip_coords = fets_eval.ip_coords
        # [ i, d, n]
        dNr_geo = np.array([fets_eval.get_dNr_geo_mtx(ip) for ip in ip_coords])
        # [ e, i, d, f]
        J_mtx = np.einsum('idn,enf->eidf', dNr_geo, self.elem_x_map)
        return J_mtx

    def get_J_det(self):
        return np.linalg.det(self.J_mtx)

    def get_B(self):
        '''
        i -- integration point
        d -- spatial dimension
        f -- alternative for spatial dimension 
        n -- element node
        e -- element
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval

        n_s = mats_eval.n_s

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_ip = fets_eval.n_gp
        n_e = self.n_e_x
        #[ i, d]
        r_ip = fets_eval.ip_coords
        #[ n, d ]
        geo_r = fets_eval.geo_r

        J_inv = np.linalg.inv(self.J_mtx)

        # shape function and its derivatives for the unknowns with respect to the
        # local coordinates r
        #[ i, d, n]
        Nr = np.array([fets_eval.get_N_mtx(ip) for ip in r_ip])
        dNr = np.array([fets_eval.get_dNr_mtx(ip) for ip in r_ip])

        # [ i, n, d ]
        Nx = np.einsum('idn->ind', Nr)
        #[ e, i, n, d ]
        dNx = np.einsum('eidf,ifn->eind', J_inv, dNr)

        # put the elements of B to the corresponding position
        B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:, :,:, B_N_n_rows, B_N_n_cols] = (B_factors[None, None,:] *
                                              Nx[:, :, N_idx])
        B[:, :,:, B_dN_n_rows, B_dN_n_cols] = dNx[:,:,:, dN_idx]
        return B

    def get_K(self, Ke):

        # get the diagonals of K
        d_Ke = np.einsum('eii->ei', Ke)
        d_Ke = d_Ke.reshape(-1, 2)
        d_Ke_middle = d_Ke[1:-1]
        d_middle = d_Ke_middle[::2] + d_Ke_middle[1::2]
        d = np.vstack((d_Ke[0], d_middle, d_Ke[-1])).flatten()

        # get the diagonal with offset 1
        Ke_12 = Ke[:, 0, 1]
        Ke_34 = Ke[:, 2, 3]
        a_nonzero = np.hstack((Ke_12[0], Ke_12[1::] + Ke_34[0:-1], Ke_34[-1]))
        a = np.zeros(len(d) - 1)
        a[0::2] = a_nonzero

        # get the diagonal with offset 2
        Ke_13 = Ke[:, 0, 2]
        Ke_24 = Ke[:, 1, 3]
        b = np.vstack((Ke_13, Ke_24)).flatten(order='F')

        return d, a, b

    @staticmethod
    def apply_bc(d, a, b):

        # delete the fixed Dof
        d = np.delete(d, 0)
        a = np.delete(a, 0)
        b = np.delete(b, 0)

        # add the element for displacement control
        d = np.append(d, 0.)
        a = np.append(a, 1.)
        b = np.append(b, 0.)

        return d, a, b

    def get_corr_pred(self, U, d_U, eps, sig, alpha, q, kappa):
        '''Function calculating the residuum and tangent operator.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        elem_dof_map = self.elem_dof_map

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs
        n_el_dofs = n_dof_r * n_nodal_dofs
        # [ i ]
        w_ip = fets_eval.ip_weights

        # [n_e, n_dof_r, n_dim_dof]
        d_u_n = d_U[elem_dof_map]

        # [n_e, n_ip, n_s]
        d_eps = np.einsum('einsd,end->eis', self.B, d_u_n)

#         print 'd_eps', d_eps
        # update strain
        eps += d_eps

        # material response state variables at integration point
        sig, D, alpha, q, kappa = mats_eval.get_corr_pred(
            eps, d_eps, sig, alpha, q, kappa)

        # system matrix
        Ke = np.einsum('i,s,einsd,eist,eimtf,ei->endmf',
                       w_ip, self.A, self.B, D, self.B, self.J_det)
        Ke = Ke.reshape(-1, n_el_dofs, n_el_dofs)
        d, a, b = self.get_K(Ke)
        # apply bc
        d, a, b = self.apply_bc(d, a, b)

        # internal forces
        # [n_e, n_n, n_dim_dof]
        Fe_int = np.einsum('i,s,eis,einsd,ei->end',
                           w_ip, self.A, sig, self.B, self.J_det)
#         print 'Fe_int', Fe_int
        F_int = np.bincount(elem_dof_map.flatten(), weights=Fe_int.flatten())

        return F_int, d, a, b, eps, sig, alpha, q, kappa

if __name__ == '__main__':
    t = TStepper()
    print t.get_B().shape
