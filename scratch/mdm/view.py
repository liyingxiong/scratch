'''
Created on 08.07.2016

@author: Yingxiong
'''
from cbfe.view import Mainwindow
from ibvpy.api import BCDof
from tloop import TLoop
from tstepper import TStepper
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from traits.api import Range
from matseval import MATSEval
from tstepper import TStepper
from tloop import TLoop
from matplotlib import pyplot as plt
from ibvpy.api import BCDof
from traits.api import HasTraits, Property, Instance, cached_property, Str, Button, Range, on_trait_change, Array, List, Any, Float, Button
from matplotlib.figure import Figure
from cbfe.scratch.mpl_figure_editor import MPLFigureEditor
from traitsui.api import View, Item, Group, HSplit, Handler, InstanceEditor, UItem, VGroup
import numpy as np


class Mainwindow(HasTraits):

    #     panel = Instance(ControlPanel)
    mats_eval = Any

    time_stepper = Any

    time_loop = Any

    t_record = Array
    U_record = Array
    F_record = Array
    sf_record = Array
    eps_record = List
    sig_record = List

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure()
        return figure

    update = Button()

    def _update_fired(self):
        self.time.value = 1.0
        self.draw()
        self.figure.canvas.draw()

    sigma_y = Range(0.5, 1.5, value=1.05)
    E_b = Range(1.0, 30, value=11.)
    K_bar = Float(2.)
    alpha = Range(0.50, 2.50, value=1.0)
    beta = Range(0.00, 1.00, value=1.0)
    r = Float(0.3)
    q = Float(1.)

    @on_trait_change('E_b, r, q')
    def plot(self):
        self.mats_eval.E_b = self.E_b
        self.mats_eval.r = self.r
        self.mats_eval.q = self.q
#         self.draw()
#         self.figure.canvas.draw()

    L_x = Float(600)

    @on_trait_change('L_x')
    def plot1(self):
        self.time_stepper.L_x = self.L_x
#         self.draw()
#         self.figure.canvas.draw()

    ax1 = Property()

    @cached_property
    def _get_ax1(self):
        return self.figure.add_subplot(231)

    ax2 = Property()

    @cached_property
    def _get_ax2(self):
        return self.figure.add_subplot(232)

    ax3 = Property()

    @cached_property
    def _get_ax3(self):
        return self.figure.add_subplot(234)

    ax4 = Property()

    @cached_property
    def _get_ax4(self):
        return self.figure.add_subplot(235)

    ax5 = Property()

    @cached_property
    def _get_ax5(self):
        return self.figure.add_subplot(233)

    ax6 = Property()

    @cached_property
    def _get_ax6(self):
        return self.figure.add_subplot(236)

    def draw(self):
        self.U_record, self.F_record, self.sf_record, self.t_record, self.eps_record, self.sig_record = self.time_loop.eval()
        n_dof = 2 * self.time_stepper.domain.n_active_elems + 1

        slip, sig_n_arr, sig_e_arr, w_arr = self.time_stepper.mats_eval.get_bond_slip()
        self.ax1.cla()
        l_bs, = self.ax1.plot(slip, sig_n_arr)
#         self.ax1.plot(slip, sig_e_arr, '--')
        self.ax1.plot(slip, w_arr, '--')
        self.ax1.set_title('bond-slip law')

        self.ax2.cla()
        l_po, = self.ax2.plot(self.U_record[:, n_dof], self.F_record[:, n_dof])
        marker_po, = self.ax2.plot(
            self.U_record[-1, n_dof], self.F_record[-1, n_dof], 'ro')
        self.ax2.set_title('pull-out force-displacement curve')

        self.ax3.cla()
        X = np.linspace(
            0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        l_sf, = self.ax3.plot(X_ip, self.sf_record[-1, :])
        self.ax3.set_title('shear flow in the bond interface')

        self.ax4.cla()
        U = np.reshape(self.U_record[-1, :], (-1, 2)).T
        l_u0, = self.ax4.plot(X, U[0])
        l_u1, = self.ax4.plot(X, U[1])
        l_us, = self.ax4.plot(X, U[1] - U[0])
        self.ax4.set_title('displacement and slip')

        self.ax5.cla()
        l_eps0, = self.ax5.plot(X_ip, self.eps_record[-1][:, :, 0].flatten())
        l_eps1, = self.ax5.plot(X_ip, self.eps_record[-1][:, :, 2].flatten())
        self.ax5.set_title('strain')

        self.ax6.cla()
        l_sig0, = self.ax6.plot(X_ip, self.sig_record[-1][:, :, 0].flatten())
        l_sig1, = self.ax6.plot(X_ip, self.sig_record[-1][:, :, 2].flatten())
        self.ax6.set_title('stress')

        self.ax3.set_ylim(np.amin(self.sf_record), np.amax(self.sf_record))
        self.ax4.set_ylim(np.amin(self.U_record), np.amax(self.U_record))
#         self.ax5.set_ylim(
# np.amin(self.eps_record[:, :, 0::2]), np.amax(self.eps_record[:, :,
# 0::2]))
        self.ax6.set_ylim(np.amin(self.sig_record), np.amax(self.sig_record))

    time = Range(0.00, 1.00, value=1.00)

    @on_trait_change('time')
    def draw_t(self):
        idx = (np.abs(self.time - self.t_record)).argmin()
        n_dof = 2 * self.time_stepper.domain.n_active_elems + 1

        self.ax2.cla()
        l_po, = self.ax2.plot(self.U_record[:, n_dof], self.F_record[:, n_dof])
        marker_po, = self.ax2.plot(
            self.U_record[idx, n_dof], self.F_record[idx, n_dof], 'ro')
        self.ax2.set_title('pull-out force-displacement curve')

        self.ax3.cla()
        X = np.linspace(
            0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        l_sf, = self.ax3.plot(X_ip, self.sf_record[idx, :])
        self.ax3.set_title('shear flow in the bond interface')

        self.ax4.cla()
        U = np.reshape(self.U_record[idx, :], (-1, 2)).T
        l_u0, = self.ax4.plot(X, U[0])
        l_u1, = self.ax4.plot(X, U[1])
        l_us, = self.ax4.plot(X, U[1] - U[0])
        self.ax4.set_title('displacement and slip')

        self.ax5.cla()
        l_eps0, = self.ax5.plot(X_ip, self.eps_record[idx][:, :, 0].flatten())
        l_eps1, = self.ax5.plot(X_ip, self.eps_record[idx][:, :, 2].flatten())
        self.ax5.set_title('strain')

        self.ax6.cla()
        l_sig0, = self.ax6.plot(X_ip, self.sig_record[idx][:, :, 0].flatten())
        l_sig1, = self.ax6.plot(X_ip, self.sig_record[idx][:, :, 2].flatten())
        self.ax6.set_title('stress')

        self.ax3.set_ylim(np.amin(self.sf_record), np.amax(self.sf_record))
        self.ax4.set_ylim(np.amin(self.U_record), np.amax(self.U_record))
#         self.ax5.set_ylim(
# np.amin(self.eps_record[:, :, 0::2]), np.amax(self.eps_record[:, :,
# 0::2]))
        self.ax6.set_ylim(np.amin(self.sig_record), np.amax(self.sig_record))

        self.figure.canvas.draw()

    view = View(HSplit(Item('figure', editor=MPLFigureEditor(),
                            dock='vertical', width=0.7, height=0.9),
                       VGroup(Group(Item('E_b'),
                                    Item('r'),
                                    Item('q'),
                                    label='bond-slip law', show_labels=True, show_border=True),
                              Item('L_x'),
                              Item('time'),
                              Group(Item('mats_eval'),
                                    # Item('fets_eval'),
                                    Item('time_stepper'),
                                    Item('time_loop'),
                                    show_border=True),
                              Item('update', show_label=False),
                              ),
                       show_labels=False),
                resizable=True,
                height=0.9, width=1.0,
                )

if __name__ == '__main__':

    ts = TStepper()
    n_dofs = ts.domain.n_dofs
    d_array = np.array(
        [0., 500., 200., 500., 200., 500., 200., 500., 200., 500., 200., 500., 200., 500.])
    dd_arr = np.abs(np.diff(d_array))
    x = np.hstack((0, np.cumsum(dd_arr) / sum(dd_arr)))
    tf = interp1d(x, d_array)

    ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
                  BCDof(var='f', dof=n_dofs - 1, value=1., time_function=tf)]

    tl = TLoop(ts=ts, d_t=0.002)

    window = Mainwindow(mats_eval=ts.mats_eval,
                        time_stepper=ts,
                        time_loop=tl)
    window.draw()

    window.configure_traits()
