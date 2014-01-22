'''
'''
from etsproxy.traits.api import \
    HasStrictTraits, Property, cached_property, Float, Int, Instance
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
    

class CompositeCrackBridge(HasStrictTraits):    
    
    reinforcement = Instance(Reinforcement)
    
    sigma = Float(input=True)
    '''the given load'''
    w = Float
    '''the crack width'''
#     E_m = Float
#     '''the modulus of elasticity of matrix'''
    l_left = Float(l_input=True)
    l_free = Float(l_input=True)
    '''the free length of the reinforcement'''
    l_right = Float(l_input=True)

    t = Property(Float, depends_on='reinforcement')
    '''the bond intensity'''
    @cached_property
    def _get_t(self):
        return 2 * self.reinforcement.tau / self.reinforcement.r
    
    coord = Property(depends_on='+l_input')
    '''defines the coordinates of the points for evaluation and plotting'''
    @cached_property
    def _get_coord(self):
        length = self.l_left + self.l_right
        n_p = 100
        interval = length/(n_p - 1)
        coordinate = np.array(np.linspace(0, length, n_p))
        l_left_db = self.l_left - self.l_free/2
        '''the debonding length in the left'''
        coord_left = coordinate[: l_left_db/interval + 1]
        coord_free = coordinate[l_left_db/interval + 1: \
                                (l_left_db + self.l_free)/interval + 1]
        coord_right = coordinate[(l_left_db + self.l_free)/interval + 1: ]
        return coordinate, coord_left, coord_free, coord_right       
    
    stress_arr = Property(depends_on='sigma, reinforcement')
    '''evaluates stress in the reinforcement on each point'''
    def _get_stress_arr(self):
        sigma_left = np.ones_like(self.coord[1])*self.sigma + \
             (self.coord[1] - np.ones_like(self.coord[1])* \
             (self.l_left - self.l_free/2))*self.t
        sigma_free = np.ones_like(self.coord[2]) * self.sigma
        sigma_right = np.ones_like(self.coord[3])*self.sigma - \
             (self.coord[3] - np.ones_like(self.coord[3])* \
             (self.l_left +self.l_free/2))*self.t
        crude = np.hstack([sigma_left, sigma_free, sigma_right])
        return crude.clip(min=0)
    
    strain_arr = Property(depends_on='sigma, reinforcement')
    '''evaluates strain in the reinforcement on each point'''
    def _get_strain_arr(self):
        return self.stress_arr / self.reinforcement.E_r
    
    w = Property(depends_on='+l_input, sigma, reinforcement')
    def _get_w(self):
        '''integrate strain along the reinforcement'''
        w = np.trapz(self.strain_arr, x=self.coord[0])
#        print w
        return w
    
    
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
                          xi=100.)
    
#     CCBridge = CompositeCrackBridge(reinforcement=reinf,
#                                    sigma=10.,
#                                    l_left=10.,
#                                    l_free=2.,
#                                    l_right=12.)
#     CCBridge.w
#     print CCBridge.w
    CBShow = CrackBridgeShow(sigma_max=20.,
                             n_step=100)
    
    CBShow.w_arr
    from matplotlib import pyplot as plt
#     plt.plot(CCBridge.coord[0], CCBridge.stress_arr)
    plt.figure(figsize=(8,6))
    plt.plot(CBShow.loading_arr, CBShow.w_arr, linewidth=2)
    plt.xlabel('Stress')
    plt.ylabel('Crack Width')
    plt.show()
    
