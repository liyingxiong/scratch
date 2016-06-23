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

    phi_fn = PhiFnStrainSoftening(Epp=1e-4, Efp=2e-4, h=0.001)
    mats_eval = MATS2DMicroplaneDamage(nu=0.3,
                                       n_mp=30, phi_fn=phi_fn)
#     mats_eval = MATS2DElastic(E=30e+3, nu=0.3, stress_state='plane_stress')

    stress_max = []
    n = 16
    i_arr = np.arange(-n / 2 + 3, n - 1, 2)
    alpha_arr = i_arr * np.pi / n

    fig = plt.figure()
#     ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122)
#     ax4 = fig.add_subplot(224)

    lines_ax1 = []
    lines_ax2 = []
    lines_ax3 = []

    sig11_max_lst = []
    sig22_max_lst = []

    eps11_max_lst = []
    eps22_max_lst = []

    for i, alpha_rad in zip(i_arr, alpha_arr):

        print i

        explorer = MATSExplore(dim=MATS2DExplore(mats_eval=mats_eval))
        bc_proportional = explorer.tloop.tstepper.bcond_mngr.bcond_list[0]
        bc_proportional.alpha_rad = alpha_rad

        u = explorer.tloop.eval()

#         stress_strain = explorer.rtrace_mngr.rtrace_bound_list[3].trace
#         stress = stress_strain.xdata
#         strain = stress_strain.ydata
#         lines_ax1.append(
# ax1.plot(strain, stress, marker='.', label='%ipi/' % i + str(n))[0])

        eps_eps = explorer.rtrace_mngr.rtrace_bound_list[4].trace
        eps11 = eps_eps.xdata
        eps22 = eps_eps.ydata
        lines_ax2.append(
            ax2.plot(eps11, eps22, label='%ipi/' % i + str(n), marker='.')[0])

        sig_sig = explorer.rtrace_mngr.rtrace_bound_list[5].trace
        sig11 = sig_sig.xdata
        sig22 = sig_sig.ydata
        lines_ax3.append(
            ax3.plot(sig11, sig22, label='%ipi/' % i + str(n), marker='.')[0])

        sig_max_idx = np.argmax(np.abs(sig11))
        sig11_max_lst.append(sig11[sig_max_idx])
        sig22_max_lst.append(sig22[sig_max_idx])
        eps11_max_lst.append(eps11[sig_max_idx])
        eps22_max_lst.append(eps22[sig_max_idx])


#     ax1.set_xlabel('eps11')
#     ax1.set_ylabel('sig11')
#     leg1 = ax1.legend(loc='best')
#     ax1.set_title('eps11-sig11')

    ax2.plot(eps11_max_lst, eps22_max_lst, 'k')
    ax2.set_xlabel('eps11')
    ax2.set_ylabel('eps22')
    leg2 = ax2.legend(loc='best', ncol=2)
    ax2.set_title('eps11-eps22')
    ax2.set_aspect('equal')
    ax2.ticklabel_format(style='sci', scilimits=(0, 0))
    ax2.axhline(0, color='black')
    ax2.axvline(0, color='black')

    ax3.plot(sig11_max_lst, sig22_max_lst, 'k')
    ax3.set_xlabel('sig11')
    ax3.set_ylabel('sig22')
    leg3 = ax3.legend(loc='best', ncol=2)
    ax3.set_title('sig11-sig22')
    ax3.set_aspect('equal')
    ax3.axhline(0, color='black')
    ax3.axvline(0, color='black')

    lined = dict()
#     for legline, origline in zip(leg1.get_lines(), lines_ax1):
# legline.set_picker(5)  # 5 pts tolerance
#         lined[legline] = origline

    for legline, origline in zip(leg2.get_lines(), lines_ax2):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    for legline, origline in zip(leg3.get_lines(), lines_ax3):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()
