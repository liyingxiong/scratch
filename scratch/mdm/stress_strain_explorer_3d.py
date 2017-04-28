'''
Created on Jun 8, 2016

@author: rch
'''

from ibvpy.mats.mats3D.mats3D_cmdm.mats3D_cmdm import \
    MATS3DMicroplaneDamage

from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import \
    MATS3DElastic

import numpy as np
from ibvpy.api import RTraceGraph
from ibvpy.mats.mats_explore import MATSExplore, MATS3DExplore
from ibvpy.mats.matsXD.matsXD_cmdm import \
    PhiFnStrainSoftening

from Haigh_Westergaard_Cartesian_Spherical import haigh_westergaard_to_cartesian, cartesian_to_spherical


def get_envelope(xi, theta, phi, Epp, Efp, h, G_f, f_t=2.8968, md=0., max_strain=0.005):
    phi_fn = PhiFnStrainSoftening(
        Epp=Epp, Efp=Efp, h=h, f_t=f_t, G_f=G_f, md=md)
    mats_eval = MATS3DMicroplaneDamage(
        nu=0.3, phi_fn=phi_fn, regularization=False)
    mats_eval.regularization = False

    sig1_max = []
    sig2_max = []
    sig3_max = []

    eps1_max = []
    eps2_max = []
    eps3_max = []

    frac_max = []

    sig1_record = np.array([])
    sig2_record = np.array([])
    sig3_record = np.array([])
    frac_record = np.array([])

    for i in np.arange(len(theta)):

        explorer = MATSExplore(dim=MATS3DExplore(mats_eval=mats_eval))
        explorer.tloop.tolerance = 1e-5
        bc_proportional = explorer.tloop.tstepper.bcond_mngr.bcond_list[0]
        bc_proportional.phi = phi[i]
        bc_proportional.theta = theta[i]
        bc_proportional.max_strain = max_strain

        rt = RTraceGraph(name='time - fracture energy',
                         var_x='time', idx_x=0,
                         var_y='fracture_energy', idx_y=0,
                         record_on='update')
        explorer.rtrace_mngr.rtrace_list.append(rt)

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

        time_frac = explorer.rtrace_mngr.rtrace_bound_list[9].trace
        frac = time_frac.ydata

#         try:
        idx = np.argmax(np.abs(sig1))
        sig1_max.append(sig1[idx])
        sig2_max.append(sig2[idx])
        sig3_max.append(sig3[idx])

        eps1_max.append(eps1[idx])
        eps2_max.append(eps2[idx])
        eps3_max.append(eps3[idx])

        frac_max.append(frac[idx])

#         mlab.plot3d(sig1, sig2, sig3)

        sig1_record = np.hstack((sig1_record, sig1[0:idx + 1]))
        sig2_record = np.hstack((sig2_record, sig2[0:idx + 1]))
        sig3_record = np.hstack((sig3_record, sig3[0:idx + 1]))
        frac_record = np.hstack((frac_record, frac[0:idx + 1]))

    sig1_max_arr = np.reshape(sig1_max, xi.shape)
    sig2_max_arr = np.reshape(sig2_max, xi.shape)
    sig3_max_arr = np.reshape(sig3_max, xi.shape)

    eps1_max_arr = np.reshape(eps1_max, xi.shape)
    eps2_max_arr = np.reshape(eps2_max, xi.shape)
    eps3_max_arr = np.reshape(eps3_max, xi.shape)
    frac_max_arr = np.reshape(frac_max, xi.shape)

    return sig1_max_arr, sig2_max_arr, sig3_max_arr, eps1_max_arr, eps2_max_arr, eps3_max_arr, frac_max_arr, sig1_record, sig2_record, sig3_record, frac_record

if __name__ == '__main__':

    # discretization with Haigh-Westergaard coordinates
    xi, theta1 = np.mgrid[-0.50:1:8j, 0:2 * np.pi:7j]
    theta1 += 0.01
#     xi, theta = np.mgrid[-0.5:1:8j, -np.pi / 6.:np.pi / 6.:5j]
#     xi, theta1 = np.mgrid[-0.5:1:8j, 0:np.pi / 3. - 0.01:8j]
    rho = np.sqrt(1 - xi ** 2)
    x, y, z = haigh_westergaard_to_cartesian(xi, rho, theta1)
    r, theta, phi = cartesian_to_spherical(x, y, z)
    theta = theta.flatten()
    phi = phi.flatten()

    sig1_max_arr, sig2_max_arr, sig3_max_arr, eps1_max_arr, eps2_max_arr, eps3_max_arr = get_envelope(
        theta, phi, Epp=5e-3, Efp=250e-6, h=0.01, G_f=0.001117)
    sig1_max_arr2, sig2_max_arr2, sig3_max_arr2, eps1_max_arr2, eps2_max_arr2, eps3_max_arr2 = get_envelope(
        theta, phi, Epp=5e-3, Efp=250e-6, h=0.01, G_f=0.001117, md=0.1)

    from mayavi import mlab
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    s = mlab.mesh(sig1_max_arr, sig2_max_arr, sig3_max_arr, scalars=sig1_max_arr *
                  np.sqrt(3.) / 3. + sig2_max_arr * np.sqrt(3.) / 3. + sig3_max_arr * np.sqrt(3.) / 3.)

    s2 = mlab.mesh(sig1_max_arr2, sig2_max_arr2, sig3_max_arr2, scalars=sig1_max_arr2 *
                   np.sqrt(3.) / 3. + sig2_max_arr2 * np.sqrt(3.) / 3. + sig3_max_arr2 * np.sqrt(3.) / 3.)

    mlab.axes(s)

    mlab.figure(2, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    e = mlab.mesh(eps1_max_arr, eps2_max_arr, eps3_max_arr)
    e2 = mlab.mesh(eps1_max_arr2, eps2_max_arr2, eps3_max_arr2)

    mlab.axes(e)
    mlab.show()
