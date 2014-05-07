'''
'''
from etsproxy.traits.api import \
    HasStrictTraits, Instance, Float, List, Property, \
    cached_property
import numpy as np
from random_field_1D import RandomField
from scipy.optimize import brentq
import copy


class CompositeTensileTest(HasStrictTraits):

    rfield = Instance(RandomField)

    xgrid = Property(depends_on='rfield')
    @cached_property
    def _get_xgrid(self):
        return self.rfield.xgrid

    mstrength = Property(depends_on='rfield')
    @cached_property
    def _get_mstrength(self):
        return self.rfield.random_field

    crack_list = List
    t = Float
    maxload = Float
    E_r = Float(1000.)
    E_m = Float(100.)
    v_r = Float(0.1)

    @staticmethod
    def heaviside(x):
        return x >= 0.0

    def matrix_stress(self, x_coord, load):
        '''evaluate the stress of the point at x_coord'''
        stress = load
        if self.crack_list:
            nearest_crack = min(self.crack_list, key=lambda x: abs(x - x_coord))
            a = self.heaviside(x_coord - nearest_crack + load / self.t) * \
                self.heaviside(nearest_crack + load / self.t - x_coord)
            stress = load - self.t * (load / self.t - abs(x_coord - nearest_crack)) * a
        return stress

    def matrix_stress_field(self, load):
        field = np.ones_like(self.xgrid) * load
        if self.crack_list:
            i = 0
            for x in self.xgrid:
                field[i] = self.matrix_stress(x, load)
                i += 1
        return field

    def matrix_strain_field(self, mstress):
        return mstress / self.E_m

    def reinf_strain_field(self, load, mstress):
        rstress = np.ones_like(self.xgrid) * self.E_r / self.E_m * load + \
            (np.ones_like(self.xgrid) * load - mstress) * (1 - self.v_r) / self.v_r
        rstrain = rstress / self.E_r
        return rstrain

    def next_load(self, load):
        '''determine the next crack load level and crack position'''
        maxstress = self.matrix_stress_field(self.maxload)
        possible = np.where(self.mstrength <= maxstress)
        lam_min = self.maxload
        #
        # x_arr 
        # get_lambda( x_x_arr
        # la = np.vectorize( x )
        # np.min(la)
        for x in self.xgrid[possible]:
            fun = lambda load: self.matrix_stress(x, load) - \
                    self.mstrength[np.where(self.xgrid == x)[0][0]]
            lam = brentq(fun, load, self.maxload)
            if lam < lam_min:
                lam_min = lam
                crack = x
        self.crack_list.append(crack)
        return lam_min

    def evaluate(self):
        load = 0
        load_list = []
        strain_list = []
        while True:
            m_stress = self.matrix_stress_field(load)
            r_strain = self.reinf_strain_field(load, m_stress)
            avg_strain = np.trapz(r_strain, self.xgrid) / self.xgrid[-1]
            load_list.append(load)
            strain_list.append(avg_strain)
            try:
                load = self.next_load(load)
            except:
                break
        return load_list, strain_list, m_stress

    def evaluate_loadsteps(self):
        loadsteps = np.linspace(0., self.maxload, 101)
        load_record = copy.copy(loadsteps)
        strain_record = np.zeros_like(loadsteps)
        i = 0
        for load in loadsteps:
            m_stress = self.matrix_stress_field(load)
            if np.any(m_stress >= self.mstrength):
                crack_load = self.next_load(load / 2)
                m_stress = self.matrix_stress_field(crack_load)
                load = crack_load
                load_record[i] = crack_load
            r_strain = self.reinf_strain_field(load, m_stress)
            avg_strain = np.trapz(r_strain, self.xgrid) / self.xgrid[-1]
            strain_record[i] = avg_strain
            i += 1
        return load_record, strain_record, m_stress, r_strain

    def crack_width(self, reinf_strain, m_stress):
        '''evaluate the crack with according the reinforcement strain'''
        m_strain = self.matrix_strain_field(m_stress)
        i = 0
        j = 0
        width_arr = np.zeros_like(self.crack_list)
        present = min(self.crack_list)
        width = 0
        for x_coord in self.xgrid:
            crack = min(self.crack_list, key=lambda x: abs(x - x_coord))
            if crack != present:
                width_arr[j] = width
                width = 0
                j += 1
            width += (reinf_strain[i] - m_strain[i]) * self.rfield.interval
            present = crack
            i += 1
        width_arr[-1] = width
        return width_arr


if __name__ == '__main__':

    r_f = RandomField(lacor=1,
                     mean=0.5,
                     stdev=.05,
                     length=100.,
                     n_p=1001)

    ctt = CompositeTensileTest(rfield=r_f,
                               t=0.1,
                               crack_list=[],
                               maxload=0.7)

    result = ctt.evaluate_loadsteps()
    strain = ctt.reinf_strain_field(ctt.maxload, result[2])
    crack_arr = ctt.crack_width(strain, result[2])

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(9, 10))
    l_d = fig.add_subplot(311, xlabel='strain', ylabel='stress')
    l_d.plot(result[1], result[0], linewidth=2)
    m_stress = fig.add_subplot(312, xlabel='length', ylabel='stress')
    m_stress.plot(ctt.xgrid, ctt.mstrength, \
             color='black', label='Strength Field')
    m_stress.plot(ctt.xgrid, result[2], \
             color='blue', label='Stress Field')
    cra = fig.add_subplot(313, xlabel='Crack Width', ylabel='Number')
    cra.hist(crack_arr, bins=20)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.5)
    plt.show()
