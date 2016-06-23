'''
Created on 20.06.2016

@author: Yingxiong
'''
import numpy as np
from scipy.interpolate import griddata, interp2d, RectSphereBivariateSpline
import matplotlib.pyplot as plt

phi_arr = np.linspace(-np.pi / 2, np.pi, 100, endpoint=True)
theta_arr = np.linspace(-np.pi / 2, np.pi, 100, endpoint=True)

phi, theta = np.meshgrid(phi_arr, theta_arr)

r = 5.

x = r * np.cos(phi) * np.sin(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(theta)

print x
print y
print z

#interp = interp2d(x.flatten(), z.flatten(), y.flatten())
x_range = np.linspace(0, 3, 100, endpoint=True)

interp = griddata(np.vstack([x.flatten(), z.flatten()]).T, y.flatten(), np.vstack(
    [x_range, 4. * np.ones_like(x_range)]).T)


from mayavi import mlab
s = mlab.mesh(x, y, z)
# mlab.pipeline.scalar_field(x.flatten(), y.flatten(), z.flatten())
mlab.axes(s)
mlab.show()

#x_range = np.linspace(0,5, 20)
#y = interp(x_range, 5)
# print y

plt.plot(x_range, interp)
plt.show()
