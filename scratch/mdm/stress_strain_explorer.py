'''
Created on Jun 8, 2016

@author: rch
'''

if __name__ == '__main__':
    from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import \
        MATS2DMicroplaneDamage

    import numpy as np

    from ibvpy.mats.mats_explore import MATSExplore, MATS2DExplore
    from ibvpy.mats.matsXD.matsXD_cmdm import \
        PhiFnStrainSoftening

    phi_fn = PhiFnStrainSoftening(Epp=1e-4, Efp=2e-4, h=0.001)
    mats_eval = MATS2DMicroplaneDamage(nu=0.3,
                                       n_mp=30, phi_fn=phi_fn)

    stress_max = []
    for alpha_rad in [0.0, np.pi / 8.0, np.pi / 4.0]:

        explorer = MATSExplore(dim=MATS2DExplore(mats_eval=mats_eval),
                               n_steps=10)

        bc_proportional = explorer.tloop.tstepper.bcond_mngr.bcond_list[0]

        bc_proportional.alpha_rad = alpha_rad

        explorer.tloop.eval()

        stress_strain = explorer.rtrace_mngr.rtrace_bound_list[0].trace
        strain = stress_strain.xdata
        stress = stress_strain.ydata
        print 'strain', strain.shape
        print strain
        print 'stress', stress.shape
        print stress
        stress_argmax = np.argmax(stress)

        max_stress = stress[stress_argmax]
        print max_stress
        stress_max.append(max_stress)
