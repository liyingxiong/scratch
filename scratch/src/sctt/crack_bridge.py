'''
'''
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx \
    import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement \
    import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import \
    WeibullFibers, fibers_MC
from spirrid.rv import RV
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

    
reinf = ContinuousFibers(r=0.0035,
                      tau=RV('weibull_min', loc=0.0, shape=3., scale=1.),
                      V_f=0.01,
                      E_f=180e3,
                      xi=fibers_MC(m=5.0, sV0=0.003),
                      label='carbon',
                      n_int=100)

model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=15.,
                             Lr=15.,
                             )
w_arr = np.linspace(1e-15, 0.5, 400)
sigma_m_list = []
epsm_list = []
for w in w_arr:
    model.w = w
    model.damage
    print model.damage.shape
    sigma_c = np.sum(model._epsf0_arr*model.sorted_stats_weights \
                  *model.sorted_V_f*model.sorted_nu_r \
                  *model.sorted_E_f*(1. - model.damage))
    sigma_m = sigma_c/(1 - model.V_f_tot + reinf.E_f/model.E_m*model.V_f_tot)
#     print model._x_arr.shape 
    if sigma_m_list:
        if sigma_m < sigma_m_list[-1]:
            break
    sigma_m_list.append(sigma_m)
#     print model._epsm_arr.shape
    epsm_list.append(model._epsm_arr[101::])

# plt.plot(model._x_arr, model._epsm_arr)
# print model.V_f_tot
# print model.E_m
# print reinf.E_f

epsm_arr = np.array(epsm_list)
sigm_arr = epsm_arr * model.E_m
sigma_m_arr = np.array(sigma_m_list)

# print sigm_arr.shape
# print model._x_arr.shape
# print sigma_m_arr.shape

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(model._x_arr[101::], sigma_m_arr)
ax1.plot_wireframe(X, Y, sigm_arr, rstride=1, cstride=1)
# ax2 = fig.add_subplot(212)
# ax2.plot(range(8), f)
plt.show()
    


# plt.plot(model._x_arr, model._epsm_arr*model.E_m)
# plt.show()





