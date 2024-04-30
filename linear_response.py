#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:31:57 2024

@author: gabriel
"""
import numpy as np
from superconductor import Superconductor
import matplotlib.pyplot as plt

L_x = 20
L_y = 20
w_0 = 10
Delta = 0
mu = -32
theta = np.pi/2
B = 2*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.56 #5*Delta/k_F
Omega = 0
t = 0
A_x = 0
A_y = 0
params = {"w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda,
          }

k_x_values = np.pi/L_x*np.arange(-L_x, L_x)
k_y_values = np.pi/L_y*np.arange(-L_y, L_y)
Gamma = 0.01
alpha = 0
beta = 0
Beta = 1000
omega_values = np.linspace(-30, 60, 100)

# S = Superconductor(**params)
# S.plot_spectrum(k_x_values, k_y_values)
# S.plot_spectral_density(np.linspace(-100, 100, 100),
#                          k_x=np.pi, k_y=np.pi, Gamma=Gamma)

S = Superconductor(**params)

#%% DC-conductivity
sigma = S.get_conductivity(alpha, beta, L_x, L_y, omega_values, Gamma, Beta, Omega)

#%% Convergence in size

L_values = np.linspace(10, 100, 10)
sigma = np.zeros((len(L_values), 2))
for i, L in enumerate(L_values):
    L_x = L
    L_y = L
    sigma[i, :] = S.get_conductivity(alpha, beta, L_x, L_y, omega_values, Gamma, Beta, Omega)
    print(i)
    
fig, ax = plt.subplots()
ax.plot(L_values, sigma[:, 0], "o")