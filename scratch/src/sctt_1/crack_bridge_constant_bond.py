'''
'''
from etsproxy.traits.api import \
    HasStrictTraits, Instance, Float, List, Property, \
    Array
import numpy as np

class CrackBridgeConstantBond(HasStrictTraits):
    
    T = Float()
    '''the bond intensity, T=2*tau/r'''
    E_m = Float()
    E_r = Float()
    v_r = Float()
    
    
    xgrid = np.linspace(0, 50, 100)
    
    def matrix_stress(self, load):
        '''Evaluate the matrix stress field'''
        stress = self.T*self.v_r/(1 - self.v_r)*self.xgrid
        return stress.clip(max = load)
    
    def reinf_strain(self, load):
        '''Evaluate the reinforcement strain field'''
        crude_stress = load*self.E_r/self.E_m + load*(1-self.v_r)/self.v_r \
                        - self.T*self.xgrid
        reinf_stress = crude_stress.clip(min=load*self.E_r/self.E_m)
        strain = reinf_stress / self.E_m
        return strain
    
if __name__ == '__main__':
    
    CB = CrackBridgeConstantBond(T = 30.,
                                 E_m = 25e3,
                                 E_r = 180e3,
                                 v_r = 0.05)
    
    loadsteps = np.linspace(0, 60, 100)
    
    matrix_stress = np.empty([len(loadsteps), len(CB.xgrid)])
    reinf_strain = np.empty([len(loadsteps), len(CB.xgrid)])
        
    for i,load in enumerate(loadsteps):
        matrix_stress[i,] = CB.matrix_stress(load)
        reinf_strain[i,] = CB.reinf_strain(load)
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    
    fig = plt.figure(figsize=(6,10))
    ax1 = fig.add_subplot(211, projection='3d')
    X, Y = np.meshgrid(CB.xgrid, loadsteps)
    ax1.plot_wireframe(X, Y, matrix_stress, rstride=10, cstride=10)
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot_wireframe(X, Y, reinf_strain, rstride=10, cstride=10)
    plt.show()



    
    
    
        
    
    
    