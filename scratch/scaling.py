import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import brentq


def get_test_data_dir():
    return os.path.join('D:\\',
                        '2017-06-22-TTb-sig-eps-dresden-girder')


def get_test_files(test_file_names):
    return [os.path.join(get_test_data_dir(), file_name)
            for file_name in test_file_names
            ]

file_names_800 = [
    'tt-dk1-800tex.txt',
    'tt-dk2-800tex.txt',
    'tt-dk3-800tex.txt',
    'tt-dk4-800tex.txt'
]


file_names_3300 = [
    'tt-dk1-3300tex.txt',
    'tt-dk2-3300tex.txt',
    'tt-dk3-3300tex.txt',
    'tt-dk4-3300tex.txt'
]


Em = 11456 #matrix modulus 
Ef = 190e3 #fiber modulus
# vf = 0.0171 #reinforcement ratio
# T = 12. #bond intensity
sig_cu = 20.0 #[MPa]
x = np.linspace(0, 1000, 1000) #specimen discretization
sig_mu_x = np.linspace(1.98, 3.25, 1000) #matrix strength field
# slack = 0.0013 # slack strain

def eps_sig(vf = 0.0171, T= 12., slack = 0.0013):
    def cb(z, sig_c): #Eq.(3) and Eq. (9)
        sig_m = np.minimum(z * T * vf / (1 - vf), Em*sig_c/(vf*Ef + (1-vf)*Em)) #matrix stress
        esp_f = slack + (sig_c-sig_m) / vf / Ef #reinforcement strain
        return  sig_m, esp_f
    
    def get_z_x(x, XK): #Eq.(5)
        z_grid = np.abs(x[:, np.newaxis] - np.array(XK)[np.newaxis, :])
        return np.amin(z_grid, axis=1)
    
    def get_lambda_z(sig_mu, z):
        fun = lambda sig_c: sig_mu - cb(z, sig_c)[0]
        try: # search for the local crack load level 
            return brentq(fun, 0, sig_cu)
        except: # solution not found (shielded zone) return the ultimate composite stress
            return sig_cu
    
    def get_sig_c_K(z_x):
        get_lambda_x = np.vectorize(get_lambda_z)
        lambda_x = get_lambda_x(sig_mu_x, z_x) #Eq. (6)
        y_idx = np.argmin(lambda_x) #Eq. (7) and Eq.(8)
        return lambda_x[y_idx], x[y_idx]
    
    def get_cracking_history():
        XK = [0.] #position of the first crack
        sig_c_K = [0., np.amin(sig_mu_x)]
        eps_c_K = [0., np.amin(sig_mu_x)/(vf*Ef + (1-vf)*Em)]
        while True:
            z_x = get_z_x(x, XK)
            sig_c_k, y_i = get_sig_c_K(z_x)
            if sig_c_k == sig_cu: break
            XK.append(y_i)
            sig_c_K.append(sig_c_k)
            eps_c_K.append(np.trapz(cb(get_z_x(x, XK), sig_c_k)[1], x)/1000.) #Eq. (10)
        sig_c_K.append(sig_cu)
        eps_c_K.append(np.trapz(cb(get_z_x(x, XK), sig_cu)[1], x)/1000.)
        return sig_c_K, eps_c_K
    
    sig_c_K, eps_c_K = get_cracking_history()
    
#     plt.plot([0.0, sig_cu/(Ef*vf)], [0.0, sig_cu])
    
    return eps_c_K, sig_c_K

if __name__ == '__main__':
    
#     for file in get_test_files(file_names_800):
#         data = np.loadtxt(file)
#         xdata, ydata = data.T
#         xdata *= 0.001
#         xdata = xdata[:np.argmax(ydata)]
#         ydata = ydata[:np.argmax(ydata)]
#         del_idx = np.arange(60)
#         xdata = np.delete(xdata, del_idx)
#         ydata = np.delete(ydata, del_idx)
#         xdata = np.append(0.0, xdata)
#         ydata = np.append(0.0, ydata)
#          
#         plt.plot(xdata, ydata)
         
    for test_file in get_test_files(file_names_3300):
        data = np.loadtxt(test_file)
        xdata, ydata = data.T
        xdata *= 0.001
        xdata = xdata[:np.argmax(ydata)]
        ydata = ydata[:np.argmax(ydata)]
        del_idx = np.arange(60)
        xdata = np.delete(xdata, del_idx)
        ydata = np.delete(ydata, del_idx)
        xdata = np.append(0.0, xdata)
        ydata = np.append(0.0, ydata)
         
        plt.plot(xdata, ydata, alpha=0.5, color='k', lw=1)
 
    eps_c_K, sig_c_K = eps_sig(vf =0.0086)
    plt.plot(eps_c_K, sig_c_K, '--', lw=2, label = 'vf = 0.0086')
    
    eps_c_K, sig_c_K = eps_sig()
    plt.plot(eps_c_K, sig_c_K, 'k--', lw=2, label = 'vf = 0.00171 (test)')
    
    eps_c_K, sig_c_K = eps_sig(vf = 0.0342)
    plt.plot(eps_c_K, sig_c_K, '--', lw=2, label = 'vf = 0.00342')


    plt.xlim(0., 0.012)
    
    plt.legend()
         
    plt.show()

    