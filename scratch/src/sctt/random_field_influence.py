'''
'''
from composite_tensile_test import CompositeTensileTest
from interpolater import CrackBridge, Interpolater
from matplotlib import pyplot as plt
from random_field_1D import RandomField



CB = CrackBridge()

interpolater = Interpolater(cb=CB)

fig = plt.figure(figsize=(10,8))
l_d = fig.add_subplot(211, xlabel='strain', ylabel='stress')
l_d2= fig.add_subplot(212, xlabel='strain', ylabel='stress')


field_list = [None]*5
for field in field_list:
    field = RandomField(lacor = 30, 
                     mean = 0.5, 
                     stdev = .05,
                     length = 100.,
                     n_p = 501)
    ctt = CompositeTensileTest(rfield=field,
                               interp=interpolater,
                               t=0.1,
                               crack_list=[],
                               maxload=0.7)
    result = ctt.evaluate_loadsteps()
    l_d.plot(result[1], result[0])
    l_d2.plot(result[1], result[0])
    l_d2.set_xlim([0.003, 0.009])
    l_d2.set_ylim([0.25, 0.6])

plt.subplots_adjust(left=0.1, right=0.9, bottom= 0.15, top=0.95, hspace=0.5)    
plt.show()


