'''
'''
from enthought.traits.api import \
    HasStrictTraits, List, Instance, Property, cached_property, Float
from composite_crack_bridge import \
    Reinforcement, CompositeCrackBridge
from random_field_1D import \
    RandomField
import numpy as np
from scipy.optimize import brentq

class StrainHardening(HasStrictTraits):
    '''modelling the strain hardening behavior in tensile test'''
    
    reinf = Instance(Reinforcement)
    crack_list = List
    '''
    a nested list, in which each element contains information of a crack
    list_element[0] = coordinate of the left boundary of the crack bridge
    list_element[1] = index of the node at crack plane
    list_element[2] = coordinate of the right boundary of the crack bridge
    '''
    matrix_strength = Instance(RandomField)
    '''the random matrix strength field'''
    max_load = Float(load_input=True)
    '''the maximum load'''   
        
    def evaluate_matrix(self, load):
        '''evaluates the stress and cracks in matrix'''
        if len(self.crack_list) == 0:
            matrix_str = np.ones_like(self.matrix_strength.random_field)*load
            crack_width = []
        else:
            matrix_str = np.array([])
            crack_width = []
            interval = self.matrix_strength.interval
            for crack in self.crack_list:
                left = self.matrix_strength.xgrid[crack[0]/interval+1: crack[1]]
                if crack[0] == 0. and crack[1] != 0.:
                    '''manually add the point(x=0)'''
                    left = np.append(0., left)
                free = np.array([self.matrix_strength.xgrid[crack[1]]])
                right = self.matrix_strength.xgrid[crack[1]+1: crack[2]/interval+1]
                crack_bridge = CompositeCrackBridge(reinforcement=self.reinf,
                                                    sigma_matrix=load,
                                                    coord_left=left,
                                                    coord_free=free,
                                                    coord_right=right)
                matrix_str = np.append(matrix_str, crack_bridge.stress_m_arr)
                crack_width.append(crack_bridge.w)
        return matrix_str, crack_width
    
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
   
    def update_crack_list(self, new_crack):
        '''add a new crack to the crack_list'''
        new_crack = [0, new_crack[0], 0]
        self.crack_list.append(new_crack)
        self.crack_list.sort(key=lambda crack:crack[1])
        xgrid = self.matrix_strength.xgrid
        posi = self.crack_list.index(new_crack)
        if posi == 0:
            '''the new crack is the left-most crack'''
            self.crack_list[posi][0] = 0.
            if len(self.crack_list) == 1:
                self.crack_list[posi][2] = xgrid[-1]
            else:
                r_crack = self.crack_list[1]
                self.crack_list[posi][2] = \
                (xgrid[new_crack[1]] + xgrid[r_crack[1]])/2
                self.crack_list[1][0] = self.crack_list[posi][2]
        elif posi == len(self.crack_list) - 1:
            '''the new crack if the right-most and not the only crack'''
            l_crack = self.crack_list[posi - 1]
            self.crack_list[posi][0] = \
            (xgrid[new_crack[1]] + xgrid[l_crack[1]])/2
            self.crack_list[posi-1][2] = self.crack_list[posi][0]
            self.crack_list[posi][2] = xgrid[-1]
        else:
            r_crack = self.crack_list[posi + 1]
            l_crack = self.crack_list[posi - 1]
            self.crack_list[posi][0] = \
            (xgrid[new_crack[1]] + xgrid[l_crack[1]])/2
            self.crack_list[posi][2] = \
            (xgrid[new_crack[1]] + xgrid[r_crack[1]])/2
            self.crack_list[posi-1][2] = self.crack_list[posi][0]
            self.crack_list[posi+1][0] = self.crack_list[posi][2]
        return self.crack_list

    load_steps = Property(depends_on='+load_input, matrix_strength')
    '''defines load steps and evaluate crack distribution'''
    @cached_property
    def _get_load_steps(self):
        steps = np.linspace(np.amin(self.matrix_strength.random_field), \
                            self.max_load, 1000)
        crack_dict = {}
        crack_load_list = []
        crack_load = 0
        while np.any(self.evaluate_matrix(self.max_load)[0] > \
                     self.matrix_strength.random_field):
            crack_load = self.next_load(crack_load)
            crack_load_list.append(crack_load)
            new_crack = self.find_cracks( \
                        self.evaluate_matrix(crack_load)[0])
            self.crack_list = self.update_crack_list(new_crack)
            crack_dict[crack_load] = self.crack_list
        loadsteps = np.sort(np.hstack([steps, np.array(crack_load_list)]))
        return loadsteps, crack_dict
    
    def evaluate(self):
        crack_dict = self.load_steps[1]
        self.crack_list = []
        disp_list = []
        for load in self.load_steps[0]:
            if load in crack_dict:
                self.crack_list = crack_dict[load]
            disp = sum(self.evaluate_matrix(load)[1])
            disp_list.append(disp)
        return disp_list
            
                
if __name__ == '__main__':
    
    r_f = RandomField(lacor = 3, 
                     mean = 0.5, 
                     stdev = .05,
                     length = 100.,
                     n_p = 1001)
    
    reinforcement = Reinforcement(r=1.,
                                  tau=0.25,
                                  E_r=100.,
                                  xi=100.,
                                  v_f=0.1)
       
    test = StrainHardening(matrix_strength=r_f,
                           max_load = 0.8,
                           reinf=reinforcement)
    
    disp = test.evaluate()
    load = test.load_steps[0]
    mat_sta = test.evaluate_matrix(test.max_load)
    
    from matplotlib import pyplot as plt
    
    fig = plt.figure(figsize=(9,10))
    l_d = fig.add_subplot(311, xlabel='load', ylabel='displacement')
    l_d.plot(load, disp, linewidth=2)
    stress = fig.add_subplot(312, xlabel='length', ylabel='stress')
    stress.plot(test.matrix_strength.xgrid, test.matrix_strength.random_field, \
             color='black', label='Strength Field')
    stress.plot(test.matrix_strength.xgrid, mat_sta[0], \
             color='blue', label='Stress Field')
    cra = fig.add_subplot(313, xlabel='Crack Width', ylabel='Number')
    cra.hist(mat_sta[1], bins=20)
    plt.subplots_adjust(left=0.1, right=0.9, bottom= 0.05, top=0.95, hspace=0.5)
    plt.show()

    
#     new = test.update_crack_list(240.)
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

     
     
    
    