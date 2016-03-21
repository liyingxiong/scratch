'''
Created on 19.01.2016

@author: Yingxiong
'''
from envisage.ui.workbench.api import WorkbenchApplication
from mayavi.sources.api import VTKDataSource, VTKFileReader
from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
from ibvpy.api import BCDof
from ibvpy.fets.fets_eval import FETSEval, IFETSEval
from ibvpy.mats.mats1D import MATS1DElastic
from ibvpy.mats.mats1D5.mats1D5_bond import MATS1D5Bond
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import interp1d
from scipy.optimize import newton, brentq, bisect, minimize_scalar


class MATSEval(HasTraits):

    E_m = Float(28484, tooltip='Stiffness of the matrix',
                auto_set=False, enter_set=False)

    E_f = Float(170000, tooltip='Stiffness of the fiber',
                auto_set=False, enter_set=False)

    slip = List([0.])

    bond = List([0.])

#     b_s_law = Property
#
#     def _get_b_s_law(self):
#         return interp1d(self.slip, self.bond)
# return np.interp(self.slip, self.bond)
    def b_s_law(self, x):
        return np.interp(x, self.slip, self.bond)

    def G(self, x):
        d = np.diff(self.bond) / np.diff(self.slip)
        d = np.append(d, d[-1])
        G = interp1d(np.array(self.slip), d, kind='zero')
        y = np.zeros_like(x)
        y[x < self.slip[0]] = d[0]
        y[x > self.slip[-1]] = d[-1]
        x[x < self.slip[0]] = self.slip[-1] + 10000.
        y[x <= self.slip[-1]] = G(x[x <= self.slip[-1]])
        return y
#     G = Property
#
#     def _get_G(self):
#         d = np.diff(self.bond) / np.diff(self.slip)
#         d = np.append(d, np.nan)
#         return interp1d(self.slip, d, kind='zero')

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1):
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:,:, 0, 0] = self.E_m
        D[:,:, 2, 2] = self.E_f

#         d = np.diff(self.bond) / np.diff(self.slip)
#         d = np.append(d, np.nan)
#
#         G = interp1d(np.array(self.slip) * (1. + 1e-8), d, kind='zero')
#         print d
#         print G(np.array([[0.035, 0.0035], [0.0035, 0.0035]]))
#         print self.slip
#         a = eps[:,:, 1]
#         print np.amax(a)
        try:
            D[:,:, 1, 1] = self.G(eps[:,:, 1])
        except:
            print np.array(self.slip)
            print eps[:,:, 1]
            sys.exit()
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig

        sig[:,:, 1] = self.b_s_law(eps[:,:, 1])
        return sig, D

    n_s = Constant(3)


class FETS1D52ULRH(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

    implements(IFETSEval)

    debug_on = True

    A_m = Float(40.5 * 7.5 - 1.85, desc='matrix area [mm2]')
    A_f = Float(1.85, desc='reinforcement area [mm2]')
    L_b = Float(1., desc='perimeter of the bond interface [mm]')

    # Dimensional mapping
    dim_slice = slice(0, 1)

    n_nodal_dofs = Int(2)

    dof_r = Array(value=[[-1], [1]])
    geo_r = Array(value=[[-1], [1]])
    vtk_r = Array(value=[[-1.], [1.]])
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

    n_dof_r = Property
    '''Number of node positions associated with degrees of freedom. 
    '''
    @cached_property
    def _get_n_dof_r(self):
        return len(self.dof_r)

    n_e_dofs = Property
    '''Number of element degrees
    '''
    @cached_property
    def _get_n_dofs(self):
        return self.n_nodal_dofs * self.n_dof_r

    def _get_ip_coords(self):
        offset = 1e-6
        return np.array([[-1 + offset, 0., 0.], [1 - offset, 0., 0.]])

    def _get_ip_weights(self):
        return np.array([1., 1.], dtype=float)

    # Integration parameters
    ngp_r = 2

    def get_N_geo_mtx(self, r_pnt):
        '''
        Return geometric shape functions
        @param r_pnt:
        '''
        r = r_pnt[0]
        N_mtx = np.array([[0.5 - r / 2., 0.5 + r / 2.]])
        return N_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.
        '''
        return np.array([[-1. / 2, 1. / 2]])

    def get_N_mtx(self, r_pnt):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        return self.get_N_geo_mtx(r_pnt)

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx(r_pnt)


class TStepper(HasTraits):

    '''Time stepper object for non-linear Newton-Raphson solver.
    '''

    mats_eval = Property(Instance(MATSEval))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_mats_eval(self):
        return MATSEval()

    fets_eval = Property(Instance(FETS1D52ULRH))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRH()

    A = Property
    '''array containing the A_m, L_b, A_f
    '''
    @cached_property
    def _get_A(self):
        return np.array([self.fets_eval.A_m, self.fets_eval.L_b, self.fets_eval.A_f])

    # specimen length
    L_x = Float(18.4)

    # Number of elements
    n_e_x = Int(10)

    domain = Property(Instance(FEGrid), depends_on='L_x, n_e_x')
    '''Diescretization object.
    '''
    @cached_property
    def _get_domain(self):
        # Element definition
        domain = FEGrid(coord_max=(self.L_x,),
                        shape=(self.n_e_x,),
                        fets_eval=self.fets_eval)
        return domain

    bc_list = List(Instance(BCDof))

    J_mtx = Property(depends_on='L_x, n_e_x')
    '''Array of Jacobian matrices.
    '''
    @cached_property
    def _get_J_mtx(self):
        fets_eval = self.fets_eval
        domain = self.domain
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:,:, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)
        # [ n_e, n_geo_r, n_dim_geo ]
        elem_x_map = domain.elem_X_map
        # [ n_e, n_ip, n_dim_geo, n_dim_geo ]
        J_mtx = np.einsum('ind,enf->eidf', dNr_geo, elem_x_map)
        return J_mtx

    J_det = Property(depends_on='L_x, n_e_x')
    '''Array of Jacobi determinants.
    '''
    @cached_property
    def _get_J_det(self):
        return np.linalg.det(self.J_mtx)

    B = Property(depends_on='L_x, n_e_x')
    '''The B matrix
    '''
    @cached_property
    def _get_B(self):
        '''Calculate and assemble the system stiffness matrix.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain

        n_s = mats_eval.n_s

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_ip = fets_eval.n_gp
        n_e = domain.n_active_elems
        #[ d, i]
        r_ip = fets_eval.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:,:, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

        J_inv = np.linalg.inv(self.J_mtx)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:,:, None] * r_ip[None,:])
        dNr = 0.5 * geo_r[:,:, None] * np.array([1, 1])

        # [ i, n, d ]
        Nr = np.einsum('dni->ind', Nr)
        dNr = np.einsum('dni->ind', dNr)
        Nx = Nr
        # [ n_e, n_ip, n_dof_r, n_dim_dof ]
        dNx = np.einsum('eidf,inf->eind', J_inv, dNr)

        B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:,:,:, B_N_n_rows, B_N_n_cols] = (B_factors[None, None,:] *
                                              Nx[:,:, N_idx])
        B[:,:,:, B_dN_n_rows, B_dN_n_cols] = dNx[:,:,:, dN_idx]

        return B

    def apply_essential_bc(self):
        '''Insert initial boundary conditions at the start up of the calculation.. 
        '''
        self.K = SysMtxAssembly()
        for bc in self.bc_list:
            bc.apply_essential(self.K)

    def apply_bc(self, step_flag, K_mtx, F_ext, t_n, t_n1):
        '''Apply boundary conditions for the current load increement
        '''
        for bc in self.bc_list:
            bc.apply(step_flag, None, K_mtx, F_ext, t_n, t_n1)

    def get_corr_pred(self, step_flag, d_U, eps, sig, t_n, t_n1):
        '''Function calculationg the residuum and tangent operator.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain
        elem_dof_map = domain.elem_dof_map
        n_e = domain.n_active_elems
        n_dof_r, n_dim_dof = self.fets_eval.dof_r.shape
        n_nodal_dofs = self.fets_eval.n_nodal_dofs
        n_el_dofs = n_dof_r * n_nodal_dofs
        # [ i ]
        w_ip = fets_eval.ip_weights

        d_u_e = d_U[elem_dof_map]
        #[n_e, n_dof_r, n_dim_dof]
        d_u_n = d_u_e.reshape(n_e, n_dof_r, n_nodal_dofs)
        #[n_e, n_ip, n_s]
        d_eps = np.einsum('einsd,end->eis', self.B, d_u_n)

        # update strain
        eps += d_eps
#         if np.any(sig) == np.nan:
#             sys.exit()

        # material response state variables at integration point
        sig, D = mats_eval.get_corr_pred(eps, d_eps, sig, t_n, t_n1)

        # system matrix
        self.K.reset_mtx()
        Ke = np.einsum('i,s,einsd,eist,eimtf,ei->endmf',
                       w_ip, self.A, self.B, D, self.B, self.J_det)

        self.K.add_mtx_array(
            Ke.reshape(-1, n_el_dofs, n_el_dofs), elem_dof_map)

        # internal forces
        # [n_e, n_n, n_dim_dof]
        Fe_int = np.einsum('i,s,eis,einsd,ei->end',
                           w_ip, self.A, sig, self.B, self.J_det)
        F_int = -np.bincount(elem_dof_map.flatten(), weights=Fe_int.flatten())
        self.apply_bc(step_flag, self.K, F_int, t_n, t_n1)
        return F_int, self.K, eps, sig


class TLoop(HasTraits):

    ts = Instance(TStepper)
    d_t = Float(0.01)
    t_max = Float(1.0)
    k_max = Int(2000)
    tolerance = Float(1e-6)
    w_arr = Array
    pf_arr = Array

    def pf(self, tau_i, w_i, eps, sig):
        eps_temp = np.copy(eps)
        sig_temp = np.copy(sig)
        dw = w_i - self.ts.mats_eval.slip[-2]
        n = 10.
        d_t = dw / n
        self.ts.mats_eval.bond[-1] = tau_i
#         print d_t, 'dt'
        for _ in range(int(n)):
            step_flag = 'predictor'
            d_U_k = np.zeros(n_dofs)
            k = 0
            while k < self.k_max:
                R, K, eps_temp, sig_temp = ts.get_corr_pred(
                    step_flag, d_U_k, eps_temp, sig_temp, 0., d_t)
                F_ext = -R
                K.apply_constraints(R)
                d_U_k = K.solve()
                k += 1
                if k == self.k_max:
                    print tau_i
                    print np.linalg.norm(R)
                    raise Exception('Non convergence')
                step_flag = 'corrector'
                if np.linalg.norm(R) < self.tolerance:
                    #                     print F_ext[-1]
                    break
        return F_ext[-1]

    def update_eps_sig(self, w_i, eps, sig):

        eps_temp = np.copy(eps)
        sig_temp = np.copy(sig)
        dw = w_i - self.ts.mats_eval.slip[-2]
        n = 20.
        d_t = dw / n

        for _ in range(int(n)):
            step_flag = 'predictor'
            d_U_k = np.zeros(n_dofs)
            k = 0
            while k < self.k_max:
                R, K, eps_temp, sig_temp = ts.get_corr_pred(
                    step_flag, d_U_k, eps_temp, sig_temp, 0., d_t)
                F_ext = -R
                K.apply_constraints(R)
                d_U_k = K.solve()
                k += 1
                if k == self.k_max:
                    print np.linalg.norm(R)
                    raise Exception('Non convergence')
                step_flag = 'corrector'
                if np.linalg.norm(R) < self.tolerance:
                    break
        return eps_temp, sig_temp

    def pf_w(self, w):
        return np.interp(w, self.w_arr, self.pf_arr)

    def regularization(self, eps, sig, i):
        '''regularization
        '''
        print i
        eps_temp = np.copy(eps)
        sig_temp = np.copy(sig)
        d_w = self.w_arr[i] - self.w_arr[i - 1]
        w_arr = [self.w_arr[i] - 0.5 * d_w,
                 self.w_arr[i], self.w_arr[i] + 0.5 * d_w]
        tau_i = 0.
        for j in range(3):
            print 'j', j
            self.ts.mats_eval.slip.append(w_arr[j])
            self.ts.mats_eval.bond.append(0.)
#             print eps_temp, sig_temp
            tau = lambda tau_i: self.pf(
                tau_i, w_arr[j], eps_temp, sig_temp) - self.pf_w(w_arr[j])
#             print eps_temp, sig_temp
#             try:
            tau_j = brentq(tau, 10., 200., xtol=1e-16)
#             except:
#                 tau_j = 0.
            print tau_j
            tau_i += tau_j
            eps_temp, sig_temp = self.update_eps_sig(
                w_arr[j], eps_temp, sig_temp)

        del self.ts.mats_eval.slip[-3:]
        del self.ts.mats_eval.bond[-3:]

        return tau_i / 3.

    def eval(self):

        ts.apply_essential_bc()
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = self.ts.mats_eval.n_s

        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))

        i = 0

        eps1 = np.copy(eps)
        sig1 = np.copy(sig)

        while i < len(self.w_arr) - 1:
            i += 1

#             tau_i = self.regularization(eps, sig, i)
#             print tau_i
#             self.ts.mats_eval.slip.append(self.w_arr[i])
#             self.ts.mats_eval.bond.append(tau_i)
#             eps, sig = self.update_eps_sig(self.w_arr[i], eps, sig)

            self.ts.mats_eval.slip.append(self.w_arr[i])
            self.ts.mats_eval.bond.append(0.)
            print self.w_arr[i]
            tau = lambda tau_i: self.pf(
                tau_i, self.w_arr[i], eps1, sig1) - self.pf_arr[i]
            try:
                tau_i = brentq(tau, 0.1, 200., xtol=1e-16)
            except:
                print tau(0.1)
                print tau(200)
                plt.plot(self.ts.mats_eval.slip, self.ts.mats_eval.bond)
                plt.xlabel('slip [mm]')
                plt.ylabel('bond [N/mm]')
                plt.show()
            print tau_i
            print '============='
            self.ts.mats_eval.bond[-1] = tau_i
#
            eps1, sig1 = self.update_eps_sig(self.w_arr[i], eps1, sig1)

#             if i % 2.0 == 0.:
#                 b_avg = np.mean(self.ts.mats_eval.bond[-2:])
#                 s_avg = np.mean(self.ts.mats_eval.slip[-2:])
#                 del self.ts.mats_eval.bond[-2:]
#                 del self.ts.mats_eval.slip[-2:]
#                 self.ts.mats_eval.bond.append(b_avg)
#                 self.ts.mats_eval.slip.append(s_avg)
#
#                 eps, sig = self.update_eps_sig(s_avg, eps, sig)
#                 eps1 = np.copy(eps)
#                 sig1 = np.copy(sig)

        return self.ts.mats_eval.slip, self.ts.mats_eval.bond

if __name__ == '__main__':

    #=========================================================================
    # nonlinear solver
    #=========================================================================
    # initialization

    length = {'G1056': 16.1, 'G1057': 16.6, 'G1058': 17.3, 'G1059': 16.2, 'G1060': 16.4, 'G1061': 15.4,
              'G1062': 16.0, 'G1063': 16.5, 'G1064': 16.9, 'G1065': 15.5, 'G1066': 17.9}

    import StringIO
    from os import listdir
    from os.path import join
    fpath = 'D:\\data\\leibzig'
    bond_lst = []
    for fname in listdir(fpath):
        print fname
        s = open(join(fpath, fname)).read().replace(',', '.')
        data = np.loadtxt(
            join(fpath, fname), delimiter=';', skiprows=1, usecols=(1, 2, 3))
        l = fname.replace('.asc', '')

        ts = TStepper()

        n_dofs = ts.domain.n_dofs

        ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
                      BCDof(var='u', dof=n_dofs - 1, value=1.0)]

        ts.L_x = length[l]

        data = data.T

        x = (data[1] + data[2]) / 2.
        y = data[0] * 1000

        x[0] = 0.
        y[0] = 0.

        interp = interp1d(x, y)

        x = np.hstack((0, np.linspace(0.04, 2.5, 1000)))

        interp1 = interp1d(x, interp(x))

#         w_arr = np.hstack(
#             (np.linspace(0, 0.109, 2), np.linspace(0.13, 0.30, 10), np.linspace(0.35, 1.5, 8)))

        w_arr = np.hstack(
            (np.linspace(0, 0.15, 12), np.linspace(0.2, 2.5, 20)))

        pf_arr = interp1(w_arr)

        tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr)

        slip, bond = tl.eval()

        bond_lst.append(np.copy(bond))

        plt.plot(slip, bond, label='TA' + l)

    bond_avg = np.average(np.array(bond_lst), axis=0)
    plt.plot(w_arr, bond_avg, 'k--', lw=2, label='average')
    print w_arr
    print [bond_avg]

    plt.legend(ncol=3, loc='best')
    plt.xlabel('slip [mm]')
    plt.ylabel('bond [N/mm]')
    plt.show()
