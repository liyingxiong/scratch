'''
Created on Jun 8, 2016

@author: rch
'''

if __name__ == '__main__':
    from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import \
        MATS2DMicroplaneDamage

    from ibvpy.mats.mats2D.mats2D_elastic import MATS2DElastic

    import numpy as np

    from ibvpy.mats.mats_explore import MATSExplore, MATS2DExplore
    from ibvpy.mats.matsXD.matsXD_cmdm import \
        PhiFnStrainSoftening

    import matplotlib.pyplot as plt

#     phi_fn = PhiFnStrainSoftening(Epp=1e-4, Efp=2e-4, h=0.001)
#     mats_eval = MATS2DMicroplaneDamage(nu=0.3,
#                                        n_mp=30, phi_fn=phi_fn)
    mats_eval = MATS2DElastic(E=30e+3, nu=0.3, stress_state='plane_stress')
    stress_max = []
#     alpha_rad = np.pi / 4
    for alpha_rad in [0., np.pi / 4]:

        explorer = MATSExplore(dim=MATS2DExplore(mats_eval=mats_eval))
        bc_proportional = explorer.tloop.tstepper.bcond_mngr.bcond_list[0]
        bc_proportional.alpha_rad = alpha_rad

        u = explorer.tloop.eval()

        print 'u', u

        stress_strain = explorer.rtrace_mngr.rtrace_bound_list[0].trace
        strain = stress_strain.xdata
        stress = stress_strain.ydata
        plt.plot(strain, stress, marker='.', label='rad=' + str(alpha_rad))
        plt.xlabel('strain')
        plt.ylabel('stress')
        plt.legend(loc='best')

        if alpha_rad == np.pi / 4:
            plt.figure()
            eps_eps = explorer.rtrace_mngr.rtrace_bound_list[1].trace
            eps11 = eps_eps.xdata
            eps22 = eps_eps.ydata
            plt.plot(eps11, eps22, label='eps11-eps22')
            plt.legend(loc='best')

            plt.figure()
            sig_sig = explorer.rtrace_mngr.rtrace_bound_list[2].trace
            sig11 = sig_sig.xdata
            sig22 = sig_sig.ydata
            plt.plot(sig11, sig22, label='sig11-sig22')

        D = explorer.tloop.tstepper.tse_integ.D_el

        print np.dot(D, u)

    plt.legend(loc='best')
    plt.show()


#     stress_argmax = np.argmax(stress)
#
#     max_stress = stress[stress_argmax]
#     print max_stress
#     stress_max.append(max_stress)
