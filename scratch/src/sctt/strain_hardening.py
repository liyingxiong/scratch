'''
'''
from enthought.traits.api import \
    HasStrictTraits, Instance, Property, cached_property, Float, Array
from composite_crack_bridge import \
    Reinforcement, CompositeCrackBridge
from random_field_1D import \
    RandomField
import numpy as np
from scipy.optimize import brentq
import copy
from matplotlib import pyplot as plt


class StrainHardening(HasStrictTraits):
    '''modelling the strain hardening behavior in tensile test'''
    
    reinf = Instance(Reinforcement)
    crack_arr = Array(value=np.array([]).reshape(0,3))
    '''
    a two-dimensional array, in which each row contains information of a crack
    row[0] = coordinate of the left boundary of the crack bridge
    row[1] = index of the node at crack plane
    row[2] = coordinate of the right boundary of the crack bridge
    '''
    matrix_strength = Instance(RandomField)
    '''the random matrix strength field'''
    max_load = Float(load_input=True)
    '''the maximum load'''   
        
    def evaluate_matrix(self, load):
        '''evaluates the stress and cracks in matrix, strain in reinforcement'''
        if len(self.crack_arr) == 0:
            matrix_stress = np.ones_like(self.matrix_strength.random_field)*load
            crack_width = []
            avg_strain = 0.
        else:
            matrix_stress = np.array([])
            crack_width = []
            reinf_strain = np.array([])
            interval = self.matrix_strength.interval
            for crack in self.crack_arr:
                left = self.matrix_strength.xgrid[crack[0]/interval+1: crack[1]]
                if crack[0] == 0. and crack[1] != 0.:
                    '''manually add the point(x=0)'''
                    left = np.append(0., left)
                free = np.array([self.matrix_strength.xgrid[crack[1]]])
                right = self.matrix_strength.xgrid \
                        [crack[1]+1: crack[2]/interval+1]
                crack_bridge = CompositeCrackBridge(reinforcement=self.reinf,
                                                    sigma_matrix=load,
                                                    coord_left=left,
                                                    coord_free=free,
                                                    coord_right=right)
                matrix_stress = np.append(matrix_stress, \
                                          crack_bridge.stress_m_arr)
                crack_width.append(crack_bridge.w)
                reinf_strain = np.append(reinf_strain, crack_bridge.strain_arr)
            avg_strain = np.trapz(reinf_strain, \
                    self.matrix_strength.xgrid) / self.matrix_strength.length
        return matrix_stress, crack_width, avg_strain
    
    def find_cracks(self, stress):
        '''compare the stress and strength of matrix to get the cracks'''
        redundance = self.matrix_strength.random_field - stress
#         print np.where(redundance < 0)[0]
        return np.where(redundance < 0)[0]
    
    def n_cracks(self,sigma_p):
        '''check the number of new cracks under certain load level'''
        cracks = self.find_cracks(self.evaluate_matrix(sigma_p)[0])
        return len(cracks) - 1.
    
    def next_load(self, sigma_p):
        '''evaluates the next load level under which only one crack emerges'''
        return brentq(self.n_cracks, sigma_p, self.max_load)
        ''' len(cracks) = 1, i.e., only one new crack emerges'''
   
    def update_crack_arr(self, current_arr, new_crack):
        '''add a new crack to the crack_arr'''
        new_crack = np.array([0, new_crack[0], 0])
        current_arr = np.vstack([current_arr, new_crack])
        current_arr = current_arr[np.argsort(current_arr[:,1])]
        '''sort the array according to the position of cracks'''
        xgrid = self.matrix_strength.xgrid
        posi = np.where(current_arr==new_crack[1])[0][0]
        if posi == 0:
            '''the new crack is the left-most crack'''
            current_arr[posi][0] = 0.
            if len(current_arr) == 1:
                current_arr[posi][2] = xgrid[-1]
            else:
                r_crack = current_arr[1]
                current_arr[posi][2] = \
                (xgrid[new_crack[1]] + xgrid[r_crack[1]])/2
                current_arr[1][0] = current_arr[posi][2]
        elif posi == len(current_arr) - 1:
            '''the new crack if the right-most and not the only crack'''
            l_crack = current_arr[posi - 1]
            current_arr[posi][0] = \
            (xgrid[new_crack[1]] + xgrid[l_crack[1]])/2
            current_arr[posi-1][2] = current_arr[posi][0]
            current_arr[posi][2] = xgrid[-1]
        else:
            r_crack = current_arr[posi + 1]
            l_crack = current_arr[posi - 1]
            current_arr[posi][0] = \
            (xgrid[new_crack[1]] + xgrid[l_crack[1]])/2
            current_arr[posi][2] = \
            (xgrid[new_crack[1]] + xgrid[r_crack[1]])/2
            current_arr[posi-1][2] = current_arr[posi][0]
            current_arr[posi+1][0] = current_arr[posi][2]
        return current_arr

    load_steps = Property(depends_on='+load_input, matrix_strength')
    '''defines load steps and evaluate crack distribution'''
    @cached_property
    def _get_load_steps(self):
        steps = np.linspace(np.amin(self.matrix_strength.random_field) - 0.1, \
                            self.max_load, 200)
        crack_dict = {}
        crack_load_list = []
        crack_load = 0.
        while np.any(self.evaluate_matrix(self.max_load)[0] > \
                     self.matrix_strength.random_field):
            crack_load = self.next_load(crack_load)
#             print crack_load
            crack_load_list.append(crack_load)
            new_crack = self.find_cracks( \
                        self.evaluate_matrix(crack_load)[0])
#             print self.crack_arr
            self.crack_arr = self.update_crack_arr(self.crack_arr, new_crack)
#             print 'updated:' 
#             print self.crack_arr
            crack_dict[crack_load] = copy.copy(self.crack_arr)
        loadsteps = np.sort(np.hstack([steps, np.array(crack_load_list)]))
        return loadsteps, crack_dict
    
    def evaluate(self):
        crack_dict = self.load_steps[1]
        self.crack_arr = np.array([]).reshape(0,3)
        strain_arr = np.array([])
#         frame = plt.figure(figsize=(9,4))
#         plt.ylim([0.2, 0.6])
#         ax = frame.add_subplot(111)
#         i = 1
        for load in self.load_steps[0]:
            if load in crack_dict:
                self.crack_arr = crack_dict[load]
            strain_arr = np.append(strain_arr,self.evaluate_matrix(load)[2])
            mat_stress = self.evaluate_matrix(load)[0]
#             ax.cla()
#             ax.plot(test.matrix_strength.xgrid, \
#                     test.matrix_strength.random_field, \
#                     color='black', label='Strength Field')
#             ax.plot(test.matrix_strength.xgrid, mat_stress, \
#                     color='blue', label='Stress Field')
#             ax.set_ylim([0.0, 0.7])
#             i += 1
#             filename = 'frame%s.png'%i
#             frame.savefig(filename)
        return strain_arr
            
                
if __name__ == '__main__':
    
    r_f = RandomField(lacor = 1, 
                     mean = 0.5, 
                     stdev = .05,
                     length = 100.,
                     n_p = 1001)
    
    reinforcement = Reinforcement(r=1.,
                                  tau=0.5,
                                  E_r=100.,
                                  xi=100.,
                                  v_f=0.1)
       
    test = StrainHardening(matrix_strength=r_f,
                           max_load = 0.7,
                           reinf=reinforcement)
    
    strain = test.evaluate()
    stress = test.load_steps[0]
    mat_sta = test.evaluate_matrix(test.max_load)
    print sum(mat_sta[1])
     
#     from matplotlib import pyplot as plt
     
    fig = plt.figure(figsize=(9,10))
    l_d = fig.add_subplot(311, xlabel='strain', ylabel='stress')
    l_d.plot(strain, stress, linewidth=2)
    l_d.set_ylim([0.3, 0.7])
    stress = fig.add_subplot(312, xlabel='length', ylabel='stress')
    stress.plot(test.matrix_strength.xgrid, test.matrix_strength.random_field, \
             color='black', label='Strength Field')
    stress.plot(test.matrix_strength.xgrid, mat_sta[0], \
             color='blue', label='Stress Field')
    cra = fig.add_subplot(313, xlabel='Crack Width', ylabel='Number')
    cra.hist(mat_sta[1], bins=20)
    plt.subplots_adjust(left=0.1, right=0.9, bottom= 0.05, top=0.95, hspace=0.5)
    plt.show()

#     new = np.array([]).reshape(0,3)
#     new = test.update_crack_arr(new, [240.])
#     new = test.update_crack_arr(new, [140.])
#     new = test.update_crack_arr(new, [740.])
#     print new
    
#     stress = test.evaluate_matrix_stress(0.42388470658)

#     print np.amin(test.matrix_strength.random_field)
#     
#     from matplotlib import pyplot as plt

#     cracking = test.load_steps[1]
#     print cracking
#     print test.load_steps[0]
#     print test.evaluate_matrix_stress(1.)[1]
#      
#     print test.n_cracks(0.)
#       
#     from matplotlib import pyplot as plt
#     plt.plot(test.matrix_strength.xgrid, stress)
#     plt.show()
#             stress = self.evaluate_matrix_stress(crack_load)
#             plt.plot(test.matrix_strength.xgrid, stress)
#             plt.plot(test.matrix_strength.xgrid, test.matrix_strength.random_field)
#             plt.show()

     
     
    
    