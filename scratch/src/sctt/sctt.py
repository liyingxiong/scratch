
from etsproxy.traits.api import \
    HasStrictTraits, Instance
from stats.misc.random_field.random_field_1D import \
    RandomField
import numpy as np

class StrengthField(HasStrictTraits):
    sf = Instance(RandomField)

#class CrackBridge(HasStrictTraits):

if __name__ == '__main__':
    
    sf = RandomField( seed = True,
                      lacor = 1.,
                      xgrid = np.linspace( 0., 100, 400 ),
                      nsim = 1,
                      loc = .0,
                      shape = 15.,
                      scale = 2.5,
                      non_negative_check = True,
                      distribution = 'Weibull'
                    )
    x = sf.xgrid
    y = sf.random_field
    sf.configure_traits
    from matplotlib import pyplot as plt
    plt.plot(x,y)
    plt.show()





