'''
Created on Jun 8, 2016

@author: rch
'''

if __name__ == '__main__':
    from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import \
        MATS2DMicroplaneDamage

    from ibvpy.mats.mats_explore import MATSExplore, MATS2DExplore
    from ibvpy.mats.matsXD.matsXD_cmdm import \
        PhiFnStrainHardeningLinear, PhiFnStrainSoftening, \
        PhiFnStrainHardening

#     from ibvpy.mats.mats2D5.mats2D5_cmdm.mats2D5_cmdm import \
#         MATS2D5MicroplaneDamage
#
#     from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import \
#         MATS3DElastic
#
#     from ibvpy.mats.matsXD.matsXD_cmdm.matsXD_cmdm_phi_fn import \
#         PhiFnStrainHardeningLinear
#
#     phi_fn = PhiFnStrainHardeningLinear(alpha=0.5, beta=0.7)
#     explorer = MATSExplore(
#         dim=MATS3DExplore(mats_eval=MATS3DElastic(E=30000., nu=0.2)))

#     phi_fn = PhiFnStrainHardeningLinear(alpha=0.5, beta=0.7)
#     phi_fn = PhiFnStrainHardening(Epp=1e-4, Efp=2e-4, Dfp=0.1, Elimit=8e-2)
    phi_fn = PhiFnStrainSoftening(Epp=1e-4, Efp=2e-4, h=0.001)
    mats_eval = MATS2DMicroplaneDamage(nu=0.3,
                                       n_mp=30, phi_fn=phi_fn)

    explorer = MATSExplore(dim=MATS2DExplore(mats_eval=mats_eval),
                           n_steps=10)

    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp(ibv_resource=explorer)
    ibvpy_app.main()
