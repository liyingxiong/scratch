'''
'''
import numpy as np
from etsproxy.traits.api import \
    HasStrictTraits, Instance, Float, List, Property, \
    cached_property, Array
from scipy.interpolate import interp2d, RectBivariateSpline, interp1d
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
    import CompositeCrackBridge
from crack_bridge_constant_bond import CrackBridgeConstantBond
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from util.traits.either_type import EitherType
from spirrid.rv import RV


class CrackBridge(HasStrictTraits):
    '''
    Obtain the data for interpolation from the instance of
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
    
    ccb = EitherType(klasses=[CompositeCrackBridge, \
                              CrackBridgeConstantBond])
#     cbcb = Instance(CrackBridgeConstantBond)
    
    ccb_data = Property(denpends_on='ccb')
    @cached_property 
    def  _get_ccb_data(self):
        '''
        Prepare the date for interpolation, the data is obtain form the 
        CompositeCrackBridge model.
        parameters:
        epsf_arr - the array containing the reinforcement strain profile
        sigm_arr - the array containing matrix stress profile
        load_arr - the array containing the load levels
        '''
        if isinstance(self.ccb, CompositeCrackBridge):
            '''
            crack bridge with random bond, random reinforcement breaking 
            strain, and random reinforcement radius
            ''' 
            load_list = []
    #         damage_list = []
            epsf_list = []
            epsm_list = []
            E_f_mean = 0.
            for reinf in self.ccb.reinforcement_lst:
                E_f_mean += reinf.E_f*reinf.V_f/self.ccb.V_f_tot
                n_int = reinf.n_int
            w = 1e-15
            step = 0.0001
            j = 1.
            while True:
                self.ccb.w = w
                self.ccb.damage
                epsf_x = np.zeros_like(self.ccb._x_arr[n_int+1::])
                
                for i, depsf in enumerate(self.ccb.sorted_depsf):
                    epsf_x += np.maximum(self.ccb._epsf0_arr[i] - depsf * \
                        np.abs(self.ccb._x_arr[n_int+1::]), \
                        self.ccb._epsm_arr[n_int+1::])* \
                        self.ccb.sorted_stats_weights[i]
                    
                sigma_c = np.sum(self.ccb._epsf0_arr \
                                 *self.ccb.sorted_stats_weights \
                              *self.ccb.sorted_V_f*self.ccb.sorted_nu_r \
                              *self.ccb.sorted_E_f*(1. - self.ccb.damage))
    #             damage = np.sum(self.ccb.sorted_stats_weights*self.ccb.damage)
                sigma_m = sigma_c/(1 - self.ccb.V_f_tot + \
                                   E_f_mean/self.ccb.E_m*self.ccb.V_f_tot) 
                if load_list:
                    if sigma_m < load_list[-1]:
                        break
                epsf_list.append(epsf_x)                
                load_list.append(sigma_m)
    #             damage_list.append(damage)
                epsm_list.append(self.ccb._epsm_arr[n_int+1::])
                w += j**2*step
                j += 1.
            epsf_arr = np.array(epsf_list)    
            epsm_arr = np.array(epsm_list)
            sigm_arr = epsm_arr * self.ccb.E_m
            load_arr = np.array(load_list)
            xgrid = self.ccb._x_arr[n_int+1::]
#         damage_arr = np.array(damage_list)

        elif isinstance(self.ccb, CrackBridgeConstantBond):
            '''crack bridge with constant bond'''
            load_arr = np.linspace(0, 60, 100)
            sigm_arr = np.empty([len(load_arr), len(self.ccb.xgrid)])
            epsf_arr = np.empty([len(load_arr), len(self.ccb.xgrid)])
                
            for i,load in enumerate(load_arr):
                sigm_arr[i,] = self.ccb.matrix_stress(load)
                epsf_arr[i,] = self.ccb.reinf_strain(load)
                
            xgrid = self.ccb.xgrid
            
        return sigm_arr, load_arr, xgrid, epsf_arr
    



class Interpolater(HasStrictTraits):
    ''' 
    Evaluate the stress field of the matrix according the distances and
    load level, evaluate the strain field in the intact reinforcement according
    the load level
    '''
    
    cb = Instance(CrackBridge)
    
    f_stress = Property(depends_on='cb')
    '''the interpolation function for matrix stress field'''
    @cached_property
    def _get_f_stress(self):
        data = self.cb.ccb_data
        X = data[2]
        Y = data[1]
        f_stress = interp2d(X, Y, data[0])
        return f_stress
    
#     f_damage = Property(depends_on='cb')
#     @cached_property
#     def _get_f_damage(self):
#         stress = self.cb.matrix_stress
#         f_damage = interp1d(stress[1], stress[3], \
#                             bounds_error=False,  fill_value=0.)
#         return f_damage
    
    f_fstrain = Property(depends_on='cb')
    '''the interpolation function for reinforcement strain field'''
    @cached_property
    def _get_f_fstrain(self):
        data = self.cb.ccb_data
        X = data[2]
        Y = data[1]
        f_fstrain = interp2d(X, Y, data[3])
        return f_fstrain
    
    def interpolate_single(self, distance, load):
        '''calculate the matirx stress on a single matrial point'''
        return self.f_stress(distance, load)
    
    def interpolate_m_stress(self, distance_arr, load):
        '''calculate the matrix stress field'''
#         stress = self.cb.matrix_stress
#         X = stress[2]
#         Y = stress[1]
#         f_stress = interp2d(X, Y, stress[0])
        order = np.argsort(distance_arr)
        mat_stress = self.f_stress(distance_arr[order], load)
#         if len(order) != len(mat_stress):
#             print len(distance_arr[order]), len(order), len(mat_stress)
#             check = self.f_stress(distance_arr[order],load)
#             print len(check)
#             print len(load)
#             from matplotlib import pyplot as plt
#             plt.plot(np.linspace(0, 100, 1001), distance_arr)
#             plt.plot(np.linspace(0, 100, 1001), distance_arr[order])
#             plt.show()
        return mat_stress[np.argsort(order)]
    
    def interpolate_reinf_strain(self, distance_arr, load):
        '''calculate the reinforcement strain field'''
        order = np.argsort(distance_arr)
        reinf_strain = self.f_fstrain(distance_arr[order], load)
        return reinf_strain[np.argsort(order)]


#         X = stress[1]
#         Y = stress[2]
#         f_stress = RectBivariateSpline(X, Y, stress[0])
#         mat_stress = f_stress( load, distance_arr)
#         return mat_stress.reshape((-1,len(distance_arr)))[0]

#     def interp_damage(self, load):
#         '''calculate the damaged portion of filaments'''
#         return self.f_damage(load)
        
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
     
    cb1 = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 Ll=100.,
                                 Lr=100.,
                                 )
    
    cb2 = CrackBridgeConstantBond(T = 30.,
                                 E_m = 25e3,
                                 E_r = 180e3,
                                 v_r = 0.05)

    
    CB = CrackBridge(ccb=cb2)
    
    stress = CB.ccb_data
#     print CB.distance_arr.shape
#     print stress.shape
    interp = Interpolater(cb = CB)
#     distance = np.array([1, 0, 1, 2, 1, 0, 1, 2])
    distance = np.linspace(0, 100, 1001)
    m_stress = interp.interpolate_m_stress(distance,120)
    r_strain = interp.interpolate_reinf_strain(distance,120)
#     d2 = np.linspace(0, 20, 50)
#     ip2 = t.clock()
#     g = interp.interpolate(d2, np.array([35., 36., 37.]))
#     print g
#     print t.clock()-ip2
    
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221, projection='3d')
    X, Y = np.meshgrid(stress[2], stress[1])
    ax1.plot_wireframe(X, Y, stress[0], rstride=1, cstride=2)
    ax2 = fig.add_subplot(222)
    ax2.plot(np.linspace(0, 100, 1001), m_stress)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_wireframe(X, Y, stress[3], rstride=1, cstride=2)
    ax4 = fig.add_subplot(224)
    ax4.plot(np.linspace(0, 100, 1001), r_strain)
    plt.show()
    
