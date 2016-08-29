'''
Created on 12.01.2016

@author: Yingxiong
'''
import numpy as np
from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
from tstepper import TStepper


class TLoop(HasTraits):

    ts = Instance(TStepper)
    d_t = Float(0.01)
    t_max = Float(1.0)
    k_max = Int(100)
    tolerance = Float(1e-6)

    def eval(self):

        self.ts.apply_essential_bc()

        t_n = 0.
        t_n1 = t_n
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = self.ts.mats_eval.n_s
        U_k = np.zeros(n_dofs)
        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))
        alpha = np.zeros((n_e, n_ip))
        q = np.zeros((n_e, n_ip))
        kappa = np.zeros((n_e, n_ip))

        U_record = np.zeros(n_dofs)
        F_record = np.zeros(n_dofs)
        sf_record = np.zeros(2 * n_e)
        t_record = [t_n]
        eps_record = [np.zeros_like(eps)]
        sig_record = [np.zeros_like(sig)]

        while t_n1 <= self.t_max - self.d_t:
            print "current time step:", t_n1
            t_n1 = t_n + self.d_t
            k = 0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
            while k <= self.k_max:
                if k == self.k_max:
                    print 'non-convergence'
                    import sys
                    sys.exit()

                R, K, eps, sig, alpha, q, kappa, F_ext_record = self.ts.get_corr_pred(
                    step_flag, U_k, d_U_k, eps, sig, t_n, t_n1, alpha, q, kappa)
                F_ext = -R
                K.apply_constraints(R)
                d_U_k = K.solve()
                d_U += d_U_k
                if np.linalg.norm(R) < self.tolerance:
                    F_record = np.vstack((F_record, F_ext_record))
                    U_k += d_U
                    U_record = np.vstack((U_record, U_k))
                    sf_record = np.vstack((sf_record, sig[:, :, 1].flatten()))
                    eps_record.append(np.copy(eps))
                    sig_record.append(np.copy(sig))
                    t_record.append(t_n1)
                    break
                k += 1
                step_flag = 'corrector'

            t_n = t_n1
        return U_record, F_record, sf_record, np.array(t_record), eps_record, sig_record

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from ibvpy.api import BCDof
    import time as t

    ts = TStepper()

    n_dofs = ts.domain.n_dofs

    ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=5.)]

    tl = TLoop(ts=ts)

    t1 = t.time()
    U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()

    print 'time consumed', t.time() - t1

    plt.plot(U_record[:, n_dofs - 1],
             F_record[:, n_dofs - 1], marker='.')
    plt.xlabel('displacement')
    plt.ylabel('force')
#     plt.figure()
#     plt.plot(np.arange(len(U_record[-1, :]) / 2.), U_record[-1, 0::2])
#     plt.plot(np.arange(len(U_record[-1, :]) / 2.), U_record[-1, 1::2])

    plt.show()
