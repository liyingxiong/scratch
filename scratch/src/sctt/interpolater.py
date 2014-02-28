'''
'''
import numpy as np
from etsproxy.traits.api import \
    HasStrictTraits, Instance, Float, List, Property, \
    cached_property, Array
from scipy.interpolate import interp2d, RectBivariateSpline


class CrackBridge(HasStrictTraits):
    
    t = Float(0.1)
    E_r = Float(1000.)
    E_m = Float(100.)
    v_r = Float(0.1)
    load_arr = np.linspace(0., 1., 41)
    distance_arr = np.linspace(0, 15., 121)
    
    @staticmethod
    def heaviside(x):
        return x >= 0.0
    
    def matrix_stress(self):
#         for load in self.load_arr:
        stress = self.load_arr[:, None] - \
            (self.load_arr[:, None]/self.t - self.distance_arr[None, :])* \
            self.t*self.heaviside(self.load_arr[:, None]/self.t - \
                                  self.distance_arr[None, :])
        return stress
    

class Interpolater(HasStrictTraits):
    
    cb = Instance(CrackBridge)
    
    def interpolate(self, distance_arr, load):
        stress = self.cb.matrix_stress()
        X = self.cb.distance_arr
        Y = self.cb.load_arr
        f_stress = interp2d(X, Y, stress)
        order = np.argsort(distance_arr)
        mat_stress = f_stress(distance_arr[order], load)
        return mat_stress[np.argsort(order)]
        
if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    
    CB = CrackBridge()
    stress = CB.matrix_stress()
#     print CB.distance_arr.shape
#     print stress.shape
    
    interp = Interpolater(cb = CB)
    distance = np.array([1, 0, 1, 2, 1, 0, 1, 2])
    f = interp.interpolate(distance, 0.555)

    fig = plt.figure(figsize=(6,10))
    ax1 = fig.add_subplot(211, projection='3d')
    X, Y = np.meshgrid(CB.distance_arr, CB.load_arr)
    ax1.plot_wireframe(X, Y, stress, rstride=1, cstride=2)
    ax2 = fig.add_subplot(212)
    ax2.plot(range(8), f)
    plt.show()


            
        
    
    
    
    