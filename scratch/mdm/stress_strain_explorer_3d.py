'''
Created on Jun 8, 2016

@author: rch
'''

if __name__ == '__main__':
    from ibvpy.mats.mats3D.mats3D_cmdm.mats3D_cmdm import \
        MATS3DMicroplaneDamage

    from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import \
        MATS3DElastic

    import numpy as np

    from ibvpy.mats.mats_explore import MATSExplore, MATS3DExplore
    from ibvpy.mats.matsXD.matsXD_cmdm import \
        PhiFnStrainSoftening

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    phi_fn = PhiFnStrainSoftening(Epp=1e-4, Efp=2e-4, h=0.001)
    mats_eval = MATS3DMicroplaneDamage(
        nu=0.3, phi_fn=phi_fn, regularization=False)
    mats_eval.regularization = False

#     mats_eval = MATS3DElastic(E=30e+3, nu=0.3)
    phi = np.pi / 4
    theta = np.pi / 4

    n = 32
    i_arr = np.arange(-n / 2 + 3, n - 1, 2)
    alpha_arr = i_arr * np.pi / n

#     fig = plt.figure()
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax2 = fig.add_subplot(122, projection='3d')

    sig1_max = []
    sig2_max = []
    sig3_max = []

    from mayavi import mlab
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    for phi in alpha_arr:
        for theta in alpha_arr:

            explorer = MATSExplore(dim=MATS3DExplore(mats_eval=mats_eval))
            bc_proportional = explorer.tloop.tstepper.bcond_mngr.bcond_list[0]
            bc_proportional.phi = phi
            bc_proportional.theta = theta

            u = explorer.tloop.eval()

            sig1_eps1 = explorer.rtrace_mngr.rtrace_bound_list[6].trace
            sig1 = sig1_eps1.xdata
            eps1 = sig1_eps1.ydata

            sig2_sig3 = explorer.rtrace_mngr.rtrace_bound_list[7].trace
            sig2 = sig2_sig3.xdata
            sig3 = sig2_sig3.ydata

            eps2_eps3 = explorer.rtrace_mngr.rtrace_bound_list[8].trace
            eps2 = eps2_eps3.xdata
            eps3 = eps2_eps3.ydata

            idx = np.argmax(np.abs(sig1))
            sig1_max.append(sig1[idx])
            sig2_max.append(sig2[idx])
            sig3_max.append(sig3[idx])

            mlab.plot3d(sig1, sig2, sig3)

    l = len(alpha_arr)
    sig1_max_arr = np.reshape(sig1_max, (l, l))
    sig2_max_arr = np.reshape(sig2_max, (l, l))
    sig3_max_arr = np.reshape(sig3_max, (l, l))

#     pts = mlab.points3d(sig1_max, sig2_max, sig3_max, np.sqrt(
#         np.array(sig1_max) ** 2 + np.array(sig2_max) ** 2 + np.array(sig3_max) ** 2), opacity=0)
#     mesh = mlab.pipeline.delaunay2d(pts)
#     surf = mlab.pipeline.surface(mesh)
    s = mlab.mesh(sig1_max_arr, sig2_max_arr, sig3_max_arr)

    mlab.axes(s)

    mlab.show()
