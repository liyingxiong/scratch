'''
'''
from etsproxy.traits.api import \
    HasStrictTraits, Instance, Float, List, Array, Property, \
    cached_property
import numpy as np
from random_field_1D import RandomField
from scipy.optimize import brentq, brenth, root, newton_krylov
import copy
from matplotlib import pyplot as plt
from interpolater import CrackBridge, Interpolater
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
    import CompositeCrackBridge
from crack_bridge_constant_bond import CrackBridgeConstantBond
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from spirrid.rv import RV


class CompositeTensileTest(HasStrictTraits):
    
#     rfield = Instance(RandomField)
    interp = Instance(Interpolater)
    
#     xgrid = Property(depends_on='rfield')
#     @cached_property
#     def _get_xgrid(self):
#         return self.rfield.xgrid
#     
#     mstrength = Property(depends_on='rfield')
#     '''the matrix strength field'''
#     @cached_property
#     def _get_mstrength(self):
#         return self.rfield.random_field
    
    xgrid = Array
    mstrength = Array
    
    crack_list = List
    maxload = Float
    
    E_r = Property(depends_on='interp')
    @cached_property
    def _get_E_r(self):
        E_r = 0
        for reinf in self.interp.cb.ccb.reinforcement_lst:
            E_r += reinf.E_f*reinf.V_f/self.interp.cb.ccb.V_f_tot
        return E_r
    
    E_m = Property(depends_on='interp')
    @cached_property
    def _get_E_m(self):
        return self.interp.cb.ccb.E_m
    
    v_r = Property(depends_on='interp')
    @cached_property
    def _get_v_r(self):
        return self.interp.cb.ccb.V_f_tot
    
    min_distance = Property(depends_on='crack_list')  
    @cached_property
    def _get_min_distance(self):
        distance = abs(self.xgrid[:, None] - \
                    np.array(self.crack_list)[None, :])
        min_distance = np.amin(distance, axis=1)
        return min_distance

    
    @staticmethod
    def heaviside(x):
        return x >= 0.0
       
    def matrix_stress_field(self, load):
        '''
        evaluate the matrix stress field by interpolation, the data for 
        interpolation is obtained from a pre-solved crack bridge model 
        parameters: load - the current load level
                    min_distance - the distance form a point to its nearest 
                                   crack plane
        '''
        if self.crack_list:
            field = self.interp.interpolate_m_stress(self.min_distance, load)
        else:
            field = np.ones_like(self.xgrid)*load
        return field
    
    def matrix_strain_field(self, mstress):
        '''evaluate the matrix strain field according to the stress field'''
        return mstress / self.E_m
    
    def reinf_strain_field(self, load, mstress):
        '''
        evaluate the reinforcement strain field by interpolation
        parameters: load - the current load level
                    min_distance - the distance form a point to its nearest 
                                   crack plane
        '''
#         damage = self.interp.interp_damage(load)
#         load = np.ones_like(self.xgrid)*load
# #         rstress = np.ones_like(self.xgrid)*self.E_r/self.E_m*load + \
# #             (np.ones_like(self.xgrid)*load - mstress)*(1 -self.v_r)/self.v_r
#         rstress = load*self.E_r/self.E_m + \
#             (load - mstress)*(1-self.v_r)/(self.v_r*(1-damage)) + \
#             (load - mstress)*self.E_r*damage/(self.E_m*(1 - damage))
#             
#         rstrain = rstress/self.E_r
        if self.crack_list:
            rstrain = self.interp.interpolate_reinf_strain(self.min_distance, load)
        else:
            rstrain = np.ones_like(self.xgrid)*load/self.E_m
        return rstrain
        
    def next_load(self,load):
#     def next_load(self):

        '''
        determine the next crack load level under which a new crack emerges
        and the new crack position, the load level is obtained by solving the
        equation minimum(matrix_strength - matrix_stress(load))=0
        parameters: load - the initial guess
                    maxstress - the stress field in the matrix corresponding
                                to the maximum load level and current crack
                                distribution
                    possible - those positions where the maxstress may exceed
                               the matrix strength, so that a new crack may
                               emerge at those positions
                    fun - the aforementioned equation
                    lam_min - the obtained load level
                    crack - the new crack position                 
        '''
        maxstress = self.matrix_stress_field(self.maxload)        
        possible = np.where(self.mstrength <= maxstress)[0]
        fun = lambda load: min(self.mstrength[possible] - \
                                self.matrix_stress_field(load)[possible])
#         fun = lambda load: self.mstrength[possible] - \
#                                self.matrix_stress_field(load)[possible]                        
#         def fun(load):
#             fun =[]
#             for i, j in enumerate(possible):
#                 fun.append(self.mstrength[j]-)
            
        lam_min = brentq(fun, load, self.maxload) + 1e-12
#         lam = newton_krylov(fun, np.zeros_like(possible))
#         lam_min = min(lam) + 1e-10
#         plt.plot(self.xgrid, self.mstrength)
#         plt.plot(self.xgrid, self.matrix_stress_field(lam_min))
#         plt.show()
        crack = self.xgrid[np.where(self.matrix_stress_field(lam_min) >= \
                                    self.mstrength)[0][0]]
        self.crack_list.append(crack)
#         maxstress = self.matrix_stress_field(self.maxload)
#         possible = np.where(self.mstrength <= maxstress)
#         lam_min = self.maxload
#         for x in self.xgrid[possible]:
#             fun = lambda load: self.matrix_stress(x, load) - \
#                     self.mstrength[np.where(self.xgrid == x)[0][0]]
#             lam = brentq(fun, load, self.maxload)
#             if lam < lam_min:
#                 lam_min = lam
#                 crack = x
#         self.crack_list.append(crack)
        return lam_min
    
    def next_load2(self, load):
        '''
        determine the next crack load level under which a new crack emerges
        and the new crack position, the load level is obtained by solving the
        equation matrix_stress(x, lambda) - matrix_strength(x) = 0 for each
        possible material point, and the next load level lam_min is the minimum
        of all the lambdas.
        parameters: load - the initial guess
                    maxstress - the stress field in the matrix corresponding
                                to the maximum load level and current crack
                                distribution
                    possible - those positions where the maxstress may exceed
                               the matrix strength, so that a new crack may
                               emerge at those positions
                    fun - the aforementioned equation
                    lam_min - the obtained load level
                    crack - the new crack position                 
        '''
        if self.crack_list:
            maxstress = self.matrix_stress_field(self.maxload)        
            possible = np.where(self.mstrength <= maxstress)[0]
            distance_arr = self.min_distance[possible]
            strength_arr = self.mstrength[possible]
            lam_min = self.maxload
            for i, x in enumerate(distance_arr):
                fun = lambda load: strength_arr[i] - \
                        self.interp.interpolate_single(x, load)
                lam = brentq(fun, load, self.maxload)
                if lam_min > lam:
                    lam_min = lam
                    xgrid = self.xgrid[possible]
                    crack = xgrid[i]
            self.crack_list.append(crack)
                    
#                 print fun(lambda_arr[i])
        else:
            idx = np.argsort(self.mstrength)[0]
            crack = self.xgrid[idx]
            lam_min = self.mstrength[idx]
            self.crack_list.append(crack)
#         print self.crack_list
        return lam_min
        
        
                                      
#     def evaluate(self):
#         load = 0
#         load_list = []
#         strain_list = []
#         while True:
#             m_stress = self.matrix_stress_field(load)
#             r_strain = self.reinf_strain_field(load, m_stress)
#             avg_strain = np.trapz(r_strain, self.xgrid)/self.xgrid[-1]
#             load_list.append(load)
#             strain_list.append(avg_strain)
#             try:
#                 load = self.next_load(load)
#             except:
#                 break
#         return load_list, strain_list, m_stress
        
    def stress_strain_diagram(self):
        '''
        Generate an array of load steps, and evaluate the strain of the 
        specimen corresponding to each load step.
        parameters: loadsteps - the generated array of load steps
                    load_record - the array record the actual load steps,
                                  some elements in the loadsteps array are
                                  replaced by the cracking loads
                    strain_record -  the array to record the average strain 
                                     of the specimen
        '''
        loadsteps = np.linspace(0., self.maxload, 101)
        load_record = copy.copy(loadsteps)
        strain_record = np.zeros_like(loadsteps)
        i = 0
#         frame = plt.figure(figsize=(9,4))
#         ax = frame.add_subplot(111)
        for load in loadsteps:
            m_stress = self.matrix_stress_field(load)
            if np.any(m_stress >= self.mstrength):
                crack_load = self.next_load(load/2)
#                 crack_load2 = self.next_load2(load/2)
#                 print crack_load
#                 print crack_load2
                m_stress = self.matrix_stress_field(crack_load)
                load = crack_load
                load_record[i] = crack_load

#             ax.cla()
#             ax.plot(self.xgrid, self.mstrength, \
#                     color='black', label='Strength Field')
#             ax.plot(self.xgrid, m_stress, \
#                     color='blue', label='Stress Field')
#             ax.set_ylim([0.0, 30])
#             filename = 'frame%s.png'%i
#             frame.savefig(filename)

            r_strain = self.reinf_strain_field(load, m_stress)
            avg_strain = np.trapz(r_strain, self.xgrid)/self.xgrid[-1]
            strain_record[i] = avg_strain
            i += 1
        return load_record, strain_record, m_stress, r_strain
    
    def crack_width(self, reinf_strain, m_stress):
        '''evaluate the crack with according the reinforcement strain'''
        m_strain = self.matrix_strain_field(m_stress)
        i = 0
#         j = 0
        width_arr = np.zeros_like(self.crack_list)
#         present = min(self.crack_list)
        distance = abs(self.xgrid[:, None] - \
                       np.array(self.crack_list)[None, :])
        crack_distr = np.array(self.crack_list)[np.argmin(distance, axis=1)]
        for crack in self.crack_list:
            width_arr[i] = sum(reinf_strain[np.where(crack_distr == crack)]- \
                m_strain[np.where(crack_distr == crack)])* \
                (self.xgrid[1] - self.xgrid[0])
            i += 1
#         width = 0
#         for x_coord in self.xgrid:
#             crack = min(self.crack_list, key=lambda x: abs(x-x_coord))
#             if crack != present:
#                 width_arr[j] = width
#                 width = 0
#                 j += 1
#             width += (reinf_strain[i] - m_strain[i])*self.rfield.interval
#             present = crack
#             i += 1
#         width_arr[-1] = width
#         print sum(reinf_strain-m_strain)
#         print sum(width_arr)
        return width_arr
    

if __name__ == '__main__':

#     def test():
        reinf = ContinuousFibers(r=0.0035,
                              tau=RV('weibull_min', loc=0.0, shape=1., scale=4.),
                              V_f=0.01,
                              E_f=180e3,
                              xi=fibers_MC(m=2.0, sV0=0.003),
                              label='carbon',
                              n_int=100)
        
#         model = CompositeCrackBridge(E_m=25e3,
#                                      reinforcement_lst=[reinf],
#                                      Ll=100.,
#                                      Lr=100.,
#                                      )

        model = CrackBridgeConstantBond(T = 80.,
                                        E_m = 25e3,
                                        E_r = 180e3,
                                        v_r = 0.05)
    
        CB = CrackBridge(ccb=model)
        
        interpolater = Interpolater(cb=CB)
        
        r_f = RandomField(lacor = 2, 
                         mean = 25, 
                         stdev = 1,
    #                      distribution = 'Weibull',
                         length = 100.,
                         n_p = 501)
        
    #     plt.plot(r_f.xgrid, r_f.random_field)
    #     plt.show()
    
        ctt = CompositeTensileTest(xgrid = r_f.xgrid,
                                   mstrength = r_f.random_field,
                                   interp=interpolater,
    #                                t=0.1,
                                   crack_list=[],
                                   maxload=35.)
        
        result = ctt.stress_strain_diagram()
    #     result = ctt.evaluate()
        strain = ctt.reinf_strain_field(ctt.maxload, result[2])
        crack_arr = ctt.crack_width(strain, result[2])
        
#         print [result[1]]  
                
        fig = plt.figure(figsize=(9,10))
        l_d = fig.add_subplot(311, xlabel='strain', ylabel='stress')
        l_d.plot(result[1], result[0], linewidth=2)
        m_stress = fig.add_subplot(312, xlabel='length', ylabel='stress')
        m_stress.plot(ctt.xgrid, ctt.mstrength, \
                 color='black', label='Strength Field')
        m_stress.plot(ctt.xgrid, result[2], \
                 color='blue', label='Stress Field')
        cra = fig.add_subplot(313, xlabel='Crack Width', ylabel='Number')
        cra.hist(crack_arr, bins=20)
        plt.subplots_adjust(left=0.1, right=0.9, bottom= 0.05, top=0.95, hspace=0.5)
        plt.show()
        
#     import cProfile
#        
#     cProfile.run('test()', sort=1)