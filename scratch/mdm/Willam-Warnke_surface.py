'''
Created on 22.06.2016

@author: Yingxiong

the Willam-Warnke yield surface, 
https://en.wikipedia.org/wiki/Willam-Warnke_yield_criterion
'''
from __future__ import division
import numpy as np
from mayavi import mlab
import matplotlib.pylab as plt


def rho_3(xi, theta):
    '''three parameter model'''

    sig_c = 1.
    sig_t = 0.3
    sig_b = 1.7

    rc = np.sqrt(6 / 5) * sig_b * sig_t / \
        (3 * sig_b * sig_t + sig_c * (sig_b - sig_t))
    rt = np.sqrt(6 / 5) * sig_b * sig_t / (sig_c * (2 * sig_b + sig_t))

    u = 2 * rc * (rc ** 2 - rt ** 2) * np.cos(theta)

    a = 4 * (rc ** 2 - rt ** 2) * np.cos(theta) ** 2 + \
        5 * rt ** 2 - 4 * rt * rc
    v = rc * (2 * rt - rc) * np.sqrt(a)

    w = 4 * (rc ** 2 - rt ** 2) * np.cos(theta) ** 2 + (rc - 2 * rt) ** 2

    r = (u + v) / w
    z = sig_b * sig_t / sig_c / (sig_b - sig_t)

    lambda_bar = 1 / np.sqrt(5) / r
    B_bar = 1 / np.sqrt(3) / z

    return -(B_bar * xi - sig_c) / lambda_bar


def rho_5(xi, theta):
    '''five parameter model'''

    ft = 0.15  # uniaxial tensile strength
    fcu = 1.  # uniaxial compressive strength
    fcb = 1.8  # biaxial compressive strength

    a_z = ft / fcu
    a_u = fcb / fcu

    x = 3.67
    q1 = 1.59
    q2 = 1.94

    a2_numerator = np.sqrt(6 / 5) * x * (a_z - a_u) - \
        np.sqrt(6 / 5) * a_z * a_u + q1 * (2 * a_u + a_z)
    a2_denominator = (2 * a_u + a_z) * (x ** 2 - 2 / 3. *
                                        a_u * x + 1 / 3. * a_z * x - 2 / 9. * a_z * a_u)
    a2 = a2_numerator / a2_denominator
    a1 = 1 / 3. * (2 * a_u - a_z) * a2 + np.sqrt(6 / 5) * \
        (a_z - a_u) / (2 * a_u + a_z)
    a0 = 2 / 3. * a_u * a1 - 4 / 9. * a_u ** 2 * a2 + np.sqrt(2 / 15.) * a_u

    x0 = (-a1 - np.sqrt(a1 ** 2 - 4 * a0 * a2)) / (2 * a2)

    b2 = (q2 * (x0 + 1 / 3) - np.sqrt(2 / 15.) * (x0 + x)) / \
        ((x + x0) * (x - 1 / 3.) * (x0 + 1 / 3.))
    b1 = (x + 1 / 3) * b2 + (np.sqrt(6 / 5) - 3 * q2) / (3 * x - 1)
    b0 = -x0 * b1 - x0 ** 2 * b2

    r1 = a0 + a1 * (xi / fcu) + a2 * (xi / fcu) ** 2
    r2 = b0 + b1 * (xi / fcu) + b2 * (xi / fcu) ** 2

    r_numerator = 2 * r2 * (r2 ** 2 - r1 ** 2) * np.cos(theta) + r2 * (2 * r1 - r2) * \
        np.sqrt(4 * (r2 ** 2 - r1 ** 2) * np.cos(theta)
                ** 2 + 5 * r1 ** 2 - 4 * r1 * r2)
    r_denominator = 4 * (r2 ** 2 - r1 ** 2) * \
        np.cos(theta) ** 2 + (r2 - 2 * r1) ** 2
    r = r_numerator / r_denominator
    return r * fcu


xi, theta = np.mgrid[-5:1:100j, 0:np.pi / 3:20j]

# the symmetry of the yielding surface (0<theta<pi/3)
theta = np.hstack(
    (theta, theta[:, ::-1], theta, theta[:, ::-1], theta, theta[:, ::-1]))
xi = np.hstack((xi, xi, xi, xi, xi, xi))
r = rho_3(xi, theta)
r[r < 0] = 0

# the actual coordinates in Haigh-Westergaard coordinates
xi, theta = np.mgrid[-5:1:100j, 0:2 * np.pi:120j]

sig1 = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * r * np.cos(theta)
sig2 = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * r * -np.sin(np.pi / 6 - theta)
sig3 = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * r * -np.sin(np.pi / 6 + theta)

from mayavi import mlab
mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
s = mlab.mesh(sig1, sig2, sig3)
mlab.axes(s)
mlab.show()
