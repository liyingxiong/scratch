
from etsproxy.traits.api import \
    HasStrictTraits, Instance
from random_field_1D import RandomField
import numpy as np

class StrengthField(HasStrictTraits):
    sf = Instance(RandomField)

#class CrackBridge(HasStrictTraits):

if __name__ == '__main__':
    
    sf = RandomField( lacor = 1.,
                      mean = 0.25, 
                      stdev = .1,
                      length = 100.,
                      n_p = 501)

    
    sf2 = RandomField( lacor = 1.,
                      mean = 0.25, 
                      stdev = .1,
                      length = 100.,
                      n_p = 501)

    
    
    x = sf.xgrid
    y = sf.random_field
    y2 = sf2.random_field
    sf.configure_traits
    from matplotlib import pyplot as plt
    plt.plot(x,y)
    plt.plot(x,y2)
    plt.show()





