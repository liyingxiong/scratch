'''
Created on 22.09.2016

@author: Yingxiong
'''
import numpy as np
from Haigh_Westergaard_Cartesian_Spherical import haigh_westergaard_to_cartesian, cartesian_to_spherical
from stress_strain_explorer_3d import get_envelope
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


xi, theta1 = np.mgrid[-0.50:1.:20j, 0:2 * np.pi:19j]
theta1 += 0.01
#     xi, theta = np.mgrid[-0.5:1:8j, -np.pi / 6.:np.pi / 6.:5j]
# xi, theta1 = np.mgrid[-0.5:1:8j, 0:np.pi / 3. - 0.01:8j]
rho = np.sqrt(1 - xi ** 2)
x, y, z = haigh_westergaard_to_cartesian(xi, rho, theta1)
r, theta, phi = cartesian_to_spherical(x, y, z)
theta = theta.flatten()
phi = phi.flatten()

sig1_max_arr, sig2_max_arr, sig3_max_arr, eps1_max_arr, eps2_max_arr, eps3_max_arr, frac_max_arr, sig1_record, sig2_record, sig3_record, frac_record = get_envelope(
    xi, theta, phi, Epp=5e-3, Efp=250e-6, h=0.01, G_f=0.0001117, f_t=3.0, max_strain=0.003)

np.savetxt('D:\\data\\envelope\\sig1_max_arr.txt', sig1_max_arr)
np.savetxt('D:\\data\\envelope\\sig2_max_arr.txt', sig2_max_arr)
np.savetxt('D:\\data\\envelope\\sig3_max_arr.txt', sig3_max_arr)
np.savetxt('D:\\data\\envelope\\frac_max_arr.txt', frac_max_arr)

# sig1_arr = np.array(sig1_record).flatten()
# sig2_arr = np.array(sig2_record).flatten()
# sig3_arr = np.array(sig3_record).flatten()
# frac_arr = np.array(frac_record).flatten()
points = np.vstack((sig1_record, sig2_record, sig3_record)).T
print sig1_record
print sig2_record
print sig3_record

print sig1_record
print sig2_record
print sig3_record

# print points.type
# print points

points = points.astype(float)


# sig1 = np.array(sig1_record)[:, -1].reshape(xi.shape)
# sig2 = np.array(sig2_record)[:, -1].reshape(xi.shape)
# sig3 = np.array(sig3_record)[:, -1].reshape(xi.shape)
# frac = np.array(frac_record)[:, -1].reshape(xi.shape)


xi, theta1 = np.mgrid[-0.50:-0.8:30j, 0:2 * np.pi:37j]
theta1 += 0.01
rho = np.sqrt(1 - xi ** 2)
x, y, z = haigh_westergaard_to_cartesian(xi, rho, theta1)
r, theta, phi = cartesian_to_spherical(x, y, z)
theta = theta.flatten()
phi = phi.flatten()

sig1_max_arr2, sig2_max_arr2, sig3_max_arr2, eps1_max_arr2, eps2_max_arr2, eps3_max_arr2, frac_max_arr2, sig1_record2, sig2_record2, sig3_record2, frac_record2 = \
    get_envelope(xi, theta, phi, Epp=5e-3, Efp=250e-6, h=0.01,
                 G_f=0.0001117, f_t=3.0, max_strain=0.0005)

# sig1_arr2 = np.array(sig1_record2).flatten()
# sig2_arr2 = np.array(sig2_record2).flatten()
# sig3_arr2 = np.array(sig3_record2).flatten()
# frac_arr2 = np.array(frac_record2).flatten()
points2 = np.vstack((sig1_record2, sig2_record2, sig3_record2)).T

points = np.vstack((points, points2))
points = points.astype(float)
frac_arr = np.hstack((frac_record, frac_record2))
np.savetxt('D:\\data\\envelope\\points.txt', points)
np.savetxt('D:\\data\\envelope\\frac_arr.txt', frac_arr)


#
# xi = sig1_arr * np.sqrt(3.) / 3. + sig2_arr * \
#     np.sqrt(3.) / 3. + sig3_arr * np.sqrt(3.) / 3.
# print np.amax(xi)
#
# from mayavi import mlab
# mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# e = mlab.mesh(sig1_arr, sig2_arr, sig3_arr, scalars=frac_arr)
# mlab.axes(e)
# mlab.show()

rho, theta = np.mgrid[0:10:100j, 0:2 * np.pi:100j]
xi = -2.
x, y, z = haigh_westergaard_to_cartesian(xi, rho, theta)

coord = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
val = griddata(points, frac_arr, coord)
val = val.reshape(rho.shape)
x[val == np.nan] = np.nan
y[val == np.nan] = np.nan
z[val == np.nan] = np.nan

from mayavi import mlab
mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
e = mlab.mesh(x, y, z, scalars=val)
e1 = mlab.mesh(sig1_max_arr, sig2_max_arr, sig3_max_arr, scalars=frac_max_arr)
mlab.axes(e1)
mlab.show()

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
# v = np.linspace(np.amin(val), np.amax(val), 5, endpoint=True)
# ax.contour(theta, rho, val, v, linewidths=0.5, colors='k')
p = ax.contourf(theta, rho, val)
plt.colorbar(p)
plt.show()
