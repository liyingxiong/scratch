'''
'''
from composite_tensile_test import CompositeTensileTest
from interpolater import CrackBridge, Interpolater
from matplotlib import pyplot as plt
from random_field_1D import RandomField
from interpolater import CrackBridge, Interpolater
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
    import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from spirrid.rv import RV

reinf = ContinuousFibers(r=0.0035,
                      tau=RV('weibull_min', loc=0.0, shape=1., scale=4.),
                      V_f=0.01,
                      E_f=180e3,
                      xi=fibers_MC(m=2.0, sV0=0.003),
                      label='carbon',
                      n_int=100)

model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=100.,
                             Lr=100.,
                                 )

CB = CrackBridge(ccb=model)

interpolater = Interpolater(cb=CB)

fig = plt.figure(figsize=(10,8))
l_d = fig.add_subplot(111, xlabel='strain', ylabel='stress')
# l_d2= fig.add_subplot(212, xlabel='strain', ylabel='stress')


field_list = [None]*5
for field in field_list:
    field = RandomField(lacor = 20, 
                     mean = 25, 
                     stdev = 2,
                     length = 100.,
                     n_p = 501)
    ctt = CompositeTensileTest(rfield=field,
                               interp=interpolater,
#                                t=0.1,
                               crack_list=[],
                               maxload=100)
    result = ctt.evaluate_loadsteps()
    l_d.plot(result[1], result[0])
#     l_d2.plot(result[1], result[0])
#     l_d2.set_xlim([0.003, 0.009])
#     l_d2.set_ylim([0.25, 0.6])

plt.subplots_adjust(left=0.1, right=0.9, bottom= 0.15, top=0.95, hspace=0.5)    
plt.show()


