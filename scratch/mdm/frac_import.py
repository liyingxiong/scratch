'''
Created on 22.09.2016

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from Haigh_Westergaard_Cartesian_Spherical import haigh_westergaard_to_cartesian
from scipy.interpolate import griddata

sig1_max_arr = np.loadtxt('D:\\data\\envelope\\sig1_max_arr.txt')
sig2_max_arr = np.loadtxt('D:\\data\\envelope\\sig2_max_arr.txt')
sig3_max_arr = np.loadtxt('D:\\data\\envelope\\sig3_max_arr.txt')
frac_max_arr = np.loadtxt('D:\\data\\envelope\\frac_max_arr.txt')
points = np.loadtxt('D:\\data\\envelope\\points.txt')
frac_arr = np.loadtxt('D:\\data\\envelope\\frac_arr.txt')

rho, theta = np.mgrid[0:10:1000j, 0:2 * np.pi:1000j]
xi = -2.
x, y, z = haigh_westergaard_to_cartesian(xi, rho, theta)

coord = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
val = griddata(points, frac_arr, coord)
val = val.reshape(rho.shape)

x[np.isnan(val)] = 0
y[np.isnan(val)] = 0
z[np.isnan(val)] = 0

from mayavi import mlab
mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
e = mlab.mesh(x, y, z, scalars=val)
e1 = mlab.mesh(
    sig1_max_arr, sig2_max_arr, sig3_max_arr, color=(0, 0, 1), opacity= 0.4)
xx = yy = zz = np.arange(-20, 10, 0.1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)
mlab.plot3d(yx, yy, yz, line_width=0.05, tube_radius=0.05)
mlab.plot3d(zx, zy, zz, line_width=0.05, tube_radius=0.05)
mlab.plot3d(xx, xy, xz, line_width=0.05, tube_radius=0.05)
# mlab.axes(e1)

from Willam_Warnke_surface import rho_5
x_lower_limit = -8.
x_upper_limit = 8.

xi, theta = np.mgrid[x_lower_limit:x_upper_limit:100j, 0:np.pi / 3:20j]

# the symmetry of the yielding surface (0<theta<pi/3)
theta = np.hstack(
    (theta, theta[:, ::-1], theta, theta[:, ::-1], theta, theta[:, ::-1]))
xi = np.hstack((xi, xi, xi, xi, xi, xi))
r = rho_5(xi, theta)
r[r < 0] = 0

# the actual coordinates in Haigh-Westergaard coordinates
xi, theta = np.mgrid[x_lower_limit:x_upper_limit:100j, 0:2 * np.pi:120j]

sig1 = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * r * np.cos(theta)
sig2 = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * \
    r * -np.sin(np.pi / 6 - theta)
sig3 = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * \
    r * -np.sin(np.pi / 6 + theta)

s = mlab.mesh(sig1, sig2, sig3,  color=(0, 1, 0), opacity= 0.4)

mlab.show()

# fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
# p = ax.contourf(theta, rho, val)
# plt.colorbar(p)
# plt.show()
