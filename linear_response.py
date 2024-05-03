#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:31:57 2024

@author: gabriel
"""
import numpy as np
from superconductor import Superconductor
import matplotlib.pyplot as plt

L_x = 100
L_y = 100
w_0 = 10
Delta = 0
mu = -32
theta = np.pi/2
B = 0 #2*Delta
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

k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
Gamma = 0.1
alpha = 0
beta = 0
Beta = 1000
omega_values = np.linspace(-40, 5, 2)
# omega_values = [-10]

S = Superconductor(**params)
# S.plot_spectrum(k_x_values, k_y_values)
# S.plot_spectral_density(np.linspace(-100, 100, 100),
#                          k_x=np.pi, k_y=np.pi, Gamma=Gamma)

#%% DC-conductivity
sigma = S.get_conductivity(alpha, beta, L_x, L_y, omega_values, Gamma, Beta, Omega)
print(sigma)

#%% Convergence in size

L_values = np.linspace(10, 100, 10)
sigma = np.zeros((len(L_values), 2), dtype=complex)
for i, L in enumerate(L_values):
    L_x = L
    L_y = L
    sigma[i, :] = S.get_conductivity(alpha, beta, L_x, L_y, omega_values, Gamma, Beta, Omega)
    print(i)
    
fig, ax = plt.subplots()
ax.plot(L_values, np.real(sigma[:, 0]), "o", label="Real part")
ax.plot(L_values, np.imag(sigma[:, 0]), "o", label="Imaginary part")
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$\sigma(\omega=-10, \Omega=0)$")
plt.legend()

#%% Conductivity vs omega

omega_values = np.linspace(-40, 5, 100)
sigma = np.zeros((len(omega_values), 2), dtype=complex)
for i, omega in enumerate(omega_values):
    L_x = 100
    L_y = 100
    sigma[i, :] = S.get_conductivity(alpha, beta, L_x, L_y, [omega], Gamma, Beta, Omega)
    print(i)
    
fig, ax = plt.subplots()
ax.plot(omega_values, np.real(sigma[:, 0]), "o", label="Real part")
ax.plot(omega_values, np.imag(sigma[:, 0]), "o", label="Imaginary part")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\sigma(\omega, \Omega=0)$")
plt.legend()