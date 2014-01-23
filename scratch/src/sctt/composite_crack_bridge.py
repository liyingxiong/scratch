'''
'''
from etsproxy.traits.api import \
    HasStrictTraits, Property, cached_property, Float, Int, Instance, Array
# from enthought.traits.ui.api import \
#     View, Item
import numpy as np

class Reinforcement(HasStrictTraits):
    '''defines non-stochastic reinforcement'''
    
    r = Float
    '''radius of the reinforcement'''
    tau = Float
    '''frictional stress at bonding interface'''
    E_r = Float
    '''the modulus of elasticity of reinforcement'''
    xi = Float
    '''breaking strain of the reinforcement'''
    v_f = Float
    '''volume fraction of the reinforcement'''
    

class CompositeCrackBridge(HasStrictTraits):    
    
    reinforcement = Instance(Reinforcement)
    
    sigma_matrix = Float
    '''the given load'''
    w = Float
    '''the crack width'''
#     E_m = Float
#     '''the modulus of elasticity of matrix'''
    position = Float
    '''the global coordinate of the crack plane'''
    coord_left = Array(l_input=True)
    coord_free = Array(l_input=True)
    '''the free length of the reinforcement'''
    coord_right = Array(l_input=True)
    

    t = Property(Float, depends_on='reinforcement')
    '''the bond intensity'''
    @cached_property
    def _get_t(self):
        return 2 * self.reinforcement.tau / self.reinforcement.r
    
#     ratio = Property(depens_on='reinforcement')
#     '''the ratio of reinforcement volume to matrix volume'''
#     @cached_property
#     def _get_ratio(self):
#         return self.reinforcement.v_f / (1 - self.reinforcement.v_f)
    
    sigma = Property(Float, depends_on='reinforcement, sigma_matrix')
    '''the stress in reinforcement at the crack plane'''
    @cached_property
    def _get_sigma(self):
        ratio = self.reinforcement.v_f / (1 - self.reinforcement.v_f)
        return self.sigma_matrix / ratio
    
#     coord = Property(depends_on='+l_input')
#     @cached_property
#     def _get_coord(self):
#         length = self.l_left + self.l_right
#         n_p = 100
#         interval = length/(n_p - 1)
#         coordinate = np.array(np.linspace(0, length, n_p))
#         l_left_db = self.l_left - self.l_free/2
#         '''the debonding length in the left'''
#         coord_left = coordinate[: l_left_db/interval + 1]
#         coord_free = coordinate[l_left_db/interval + 1: \
#                                 (l_left_db + self.l_free)/interval + 1]
#         coord_right = coordinate[(l_left_db + self.l_free)/interval + 1: ]
#         return coordinate, coord_left, coord_free, coord_right       
    
    stress_arr = Property(depends_on='sigma, reinforcement, +l_input')
    '''evaluates stress in the reinforcement on each point'''
    def _get_stress_arr(self):
        sigma_left = np.ones_like(self.coord_left)*self.sigma + \
             (self.coord_left - np.ones_like(self.coord_left)* \
              self.coord_free[0])*self.t
        sigma_free = np.ones_like(self.coord_free) * self.sigma
        sigma_right = np.ones_like(self.coord_right)*self.sigma - \
             (self.coord_right - np.ones_like(self.coord_right)* \
              self.coord_free[-1])*self.t
        crude = np.hstack([sigma_left, sigma_free, sigma_right])
        return crude.clip(min=0)  
    
    strain_arr = Property(depends_on='sigma, reinforcement')
    '''evaluates strain in the reinforcement on each point'''
    def _get_strain_arr(self):
        return self.stress_arr / self.reinforcement.E_r
    
    w = Property(depends_on='+l_input, sigma, reinforcement')
    '''integrates strain along the reinforcement'''
    def _get_w(self):
        w = np.trapz(self.strain_arr, x=self.coord[0])
#        print w
        return w
    
    stress_m_arr = Property(depends_on='+l_input, sigma, reinforcement')
    '''evaluates the stress profile of the matrix'''
    def _get_stress_m_arr(self):
        ratio = self.reinforcement.v_f / (1 - self.reinforcement.v_f)
        return np.ones_like(self.stress_arr)*self.sigma*ratio - \
               self.stress_arr*ratio
    
    
class CrackBridgeShow(HasStrictTraits):
    '''plots the load-crack width curve'''
    
#     crack = Instance(CompositeCrackBridge)
    sigma_max = Float(input=True)
    '''maximum loading level'''
    n_step = Int(input=True)
    '''number of loading steps'''
    
    loading_arr = Property(depends_on='+input')
    '''defines the loading steps'''
    @cached_property
    def _get_loading_arr(self):
        return np.linspace(0, self.sigma_max, self.n_step)
    
    w_arr = Property(depends_on='crack, +input')
    '''evaluate the array contains the crack widths'''
    def _get_w_arr(self):
        w_arr = []
        for loading in self.loading_arr:
            CCB = CompositeCrackBridge(reinforcement=reinf,
                             sigma=loading,
                             l_left=10.,
                             l_free=2.,
                             l_right=12.)
            w_arr.append(CCB.w)
        return np.array(w_arr)
                
    
if __name__ == '__main__':
    
    reinf = Reinforcement(r=1.,
                          tau=0.5,
                          E_r=100.,
                          xi=100.,
                          v_f=0.1)
    left = np.linspace(5, 15, 11)
    free = [16.,]
    right = np.linspace(17, 25, 9)
    CCBridge = CompositeCrackBridge(reinforcement=reinf,
                                   sigma_matrix=0.9,
                                   coord_left=left,
                                   coord_free=free,
                                   coord_right=right)
    x_coord = np.hstack([CCBridge.coord_left, CCBridge.coord_free, CCBridge.coord_right])
#     CCBridge.w
#     print CCBridge.w
#     CBShow = CrackBridgeShow(sigma_max=20.,
#                              n_step=100)
#     CBShow.w_arr
    from matplotlib import pyplot as plt
    plt.plot(x_coord, CCBridge.stress_m_arr)
#     plt.figure(figsize=(8,6))
#     plt.plot(CBShow.loading_arr, CBShow.w_arr, linewidth=2)
#     plt.xlabel('Stress')
#     plt.ylabel('Crack Width')
    plt.show()
    
