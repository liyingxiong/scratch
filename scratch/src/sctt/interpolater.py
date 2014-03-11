'''
'''
import numpy as np
from etsproxy.traits.api import \
    HasStrictTraits, Instance, Float, List, Property, \
    cached_property, Array
from scipy.interpolate import interp2d, RectBivariateSpline, interp1d
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
    import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from spirrid.rv import RV


class CrackBridge(HasStrictTraits):
    '''obtain the data for interpolation from the instance of
    CompositeCrackBridge
    '''
    
#     t = Float(0.1)
#     E_r = Float(1000.)
#     E_m = Float(100.)
#     v_r = Float(0.1)
#     load_arr = np.linspace(0., 1., 41)
#     distance_arr = np.linspace(0, 15., 121)
#       
#     @staticmethod
#     def heaviside(x):
#         return x >= 0.0
#        
#     def matrix_stress(self):
# #         for load in self.load_arr:
#         stress = self.load_arr[:, None] - \
#             (self.load_arr[:, None]/self.t - self.distance_arr[None, :])* \
#             self.t*self.heaviside(self.load_arr[:, None]/self.t - \
#                                   self.distance_arr[None, :])
#         return stress, self.load_arr, self.distance_arr
    
    ccb = Instance(CompositeCrackBridge)
    
    matrix_stress = Property(denpends_on='ccb')
    @cached_property 
    def  _get_matrix_stress(self):
        sigma_m_list = []
        damage_list = []
        epsm_list = []
        E_f_mean = 0.
        for reinf in self.ccb.reinforcement_lst:
            E_f_mean += reinf.E_f*reinf.V_f/self.ccb.V_f_tot
            n_int = reinf.n_int
        w = 1e-15
        step = 0.0001
        i = 1.
        while True:
            self.ccb.w = w
            self.ccb.damage
            sigma_c = np.sum(self.ccb._epsf0_arr*self.ccb.sorted_stats_weights \
                          *self.ccb.sorted_V_f*self.ccb.sorted_nu_r \
                          *self.ccb.sorted_E_f*(1. - self.ccb.damage))
            damage = np.sum(self.ccb.sorted_stats_weights*self.ccb.damage)
            sigma_m = sigma_c/(1 - self.ccb.V_f_tot + \
                               E_f_mean/self.ccb.E_m*self.ccb.V_f_tot) 
            if sigma_m_list:
                if sigma_m < sigma_m_list[-1]:
                    break
            sigma_m_list.append(sigma_m)
            damage_list.append(damage)
            epsm_list.append(self.ccb._epsm_arr[n_int+1::])
            w += i**2*step
            i += 1.
        epsm_arr = np.array(epsm_list)
        sigm_arr = epsm_arr * self.ccb.E_m
        sigma_m_arr = np.array(sigma_m_list)
        damage_arr = np.array(damage_list)
        return sigm_arr, sigma_m_arr, self.ccb._x_arr[n_int+1::], damage_arr



class Interpolater(HasStrictTraits):
    ''' evaluate the stress field of the matrix according the distances and
    load level, evaluate the portion of damaged filaments according the load
    level
    '''
    
    cb = Instance(CrackBridge)
    
    f_stress = Property(depends_on='cb')
    @cached_property
    def _get_f_stress(self):
        stress = self.cb.matrix_stress
        X = stress[2]
        Y = stress[1]
        f_stress = interp2d(X, Y, stress[0])
        return f_stress
    
    f_damage = Property(depends_on='cb')
    @cached_property
    def _get_f_damage(self):
        stress = self.cb.matrix_stress
        f_damage = interp1d(stress[1], stress[3], \
                            bounds_error=False,  fill_value=0.)
        return f_damage
        
    def interpolate(self, distance_arr, load):
        '''calculate the matrix stress field'''
#         stress = self.cb.matrix_stress
#         X = stress[2]
#         Y = stress[1]
#         f_stress = interp2d(X, Y, stress[0])
        order = np.argsort(distance_arr)
        mat_stress = self.f_stress(distance_arr[order], load)
        return mat_stress[np.argsort(order)]

#         X = stress[1]
#         Y = stress[2]
#         f_stress = RectBivariateSpline(X, Y, stress[0])
#         mat_stress = f_stress( load, distance_arr)
#         return mat_stress.reshape((-1,len(distance_arr)))[0]

    def interp_damage(self, load):
        '''calculate the damaged portion of filaments'''
        return self.f_damage(load)
        
if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import time as t
    
    reinf = ContinuousFibers(r=0.0035,
                          tau=RV('weibull_min', loc=0.0, shape=1., scale=4),
                          V_f=0.01,
                          E_f=180e3,
                          xi=fibers_MC(m=2, sV0=0.003),
                          label='carbon',
                          n_int=100)
     
    model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 Ll=100.,
                                 Lr=100.,
                                 )
    CB = CrackBridge(ccb=model)
    
    stress = CB.matrix_stress
#     print CB.distance_arr.shape
#     print stress.shape
    interp = Interpolater(cb = CB)
#     distance = np.array([1, 0, 1, 2, 1, 0, 1, 2])
    distance = np.linspace(0, 20, 100)
    ip1 = t.clock()
    f = interp.interpolate(distance,30)
    print t.clock()-ip1
    
    d2 = np.linspace(0, 20, 1000)
    ip2 = t.clock()
    g = interp.interpolate(d2, 35.)
    print t.clock()-ip2
    
    ip3 = t.clock()
    h = interp.interpolate(distance, 0.5)
    print t.clock()-ip3
    
    fig = plt.figure(figsize=(6,10))
    ax1 = fig.add_subplot(211, projection='3d')
    X, Y = np.meshgrid(stress[2], stress[1])
    ax1.plot_wireframe(X, Y, stress[0], rstride=1, cstride=2)
    ax2 = fig.add_subplot(212)
    ax2.plot(np.linspace(0, 20, 100), f)
    plt.show()
    
