#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:04:04 2024

@author: gabriel
"""

import numpy as np
from superconductor import Superconductor
import multiprocessing
from pathlib import Path

L_x = 100 #100  
L_y = 100 #100
w_0 = 10
Delta = 0.2
mu = -39
theta = np.pi/2
B = 1*Delta
B_x = B * np.cos(theta) #initial values
B_y = B * np.sin(theta) #initial values
Lambda_R = 0.56 #5*Delta/k_F
Lambda_D = 0.1
g_xx = 1
g_yy = 1
g_xy = 0
g_yx = 0
Omega = 0 #0.02
superconductor_params = {"w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B_x":B_x, "B_y":B_y, "Lambda_R":Lambda_R,
          "Lambda_D": Lambda_D
          }

Gamma_0 = 0.01
Gamma_1 = 0.3
alpha = 0
beta = 0
Beta = 1000
k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
k_y_values = 2*np.pi*np.arange(0, L_x)/L_y
# k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
# k_y_values = np.pi*np.arange(-L_y, L_x)/L_y
n_cores = 8
points = 2*n_cores
# epsrel=1e-01

# omega_values = np.linspace(-45, 0, 100)

# part = "paramagnetic"
# part = "diamagnetic"
part = "total"
# fermi_function = lambda omega: 1/(1 + np.exp(Beta*omega))
# fermi_function = lambda omega: 1 - np.heaviside(omega, 1)
params = {
    "Gamma_0":Gamma_0, "alpha":alpha,
    "beta":beta, "Omega":Omega, "part":part,
    "theta":theta, "L_x":L_x, "L_y":L_y, "Gamma_1": Gamma_1
    }

def fermi_function(omega):
    return np.heaviside(-omega, 1)

S = Superconductor(**superconductor_params)

# E_k = S.plot_spectrum(k_x_values, k_y_values)
# S.plot_spectral_density(omega_values,
#                         k_x=-np.pi/2, k_y=-np.pi/2, Gamma=Gamma)

def integrate(B):
    # S.B_x = B * np.cos(theta)
    # S.B_y = B * np.sin(theta)
    S.B_x = B * ( g_xx * np.cos(theta) + g_xy * np.sin(theta) ) 
    S.B_y = B * ( g_yx * np.cos(theta) + g_yy * np.sin(theta) )
    Gamma = Gamma_0 + Gamma_1 * B**2
    return [S.get_response_function_quad(0, 0, L_x, L_y, Gamma, fermi_function, Omega, Delta, part),
            S.get_response_function_quad(1, 1, L_x, L_y, Gamma, fermi_function, Omega, Delta, part)]

if __name__ == "__main__":
    B_values = np.linspace(0, 1.5*Delta, points)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    K = np.array(results_pooled)
    
    data_folder = Path("Data/")
    name = f"Response_kernel_vs_B_with_field_dissorder_mu={mu}_L={L_x}_Gamma_0={Gamma_0}_Gamma_1={Gamma_1}_Omega={Omega}_Lambda_R={Lambda_R}_Lambda_D={Lambda_D}_B_in_(0-{np.round(np.max(B_values),2)})_Delta={Delta}_points={points}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , K=K, B_values=B_values,
             **params, **superconductor_params)
