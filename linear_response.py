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
Delta = 0.2
mu = -40
theta = np.pi/2
B = 0
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

k_x_values = np.pi/L_x*np.arange(-L_x, L_x)#2*np.pi/L_x*np.arange(0, L_x)
k_y_values = np.pi/L_x*np.arange(-L_y, L_y)#2*np.pi/L_y*np.arange(0, L_y)
Gamma = 0.1
alpha = 0
beta = 0
Beta = 1000

omega_values = np.linspace(-45, 0, 100)

S = Superconductor(**params)

# S.plot_spectrum(k_x_values, k_y_values)
# S.plot_spectral_density(omega_values,
#                         k_x=-np.pi, k_y=-np.pi, Gamma=Gamma)
                          

#%% DC-conductivity
sigma = S.get_conductivity_zero_Temperature(alpha, beta, L_x, L_y, omega_values, Gamma, Omega)
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

#%% Conductivity vs B
B_values = np.linspace(0, Delta, 10)

sigma_xx = np.zeros((len(B_values), 2), dtype=complex)
sigma_yy = np.zeros((len(B_values), 2), dtype=complex)
n = np.zeros(len(B_values))
for i, B in enumerate(B_values):
    S.B_x = B * np.cos(theta)
    S.B_y = B * np.sin(theta)
    sigma_xx[i, 0], sigma_xx[i, 1] = S.get_conductivity_zero_Temperature(0, 0, L_x, L_y, omega_values, Gamma, Omega)
    sigma_yy[i, 0], sigma_yy[i, 1] = S.get_conductivity_zero_Temperature(1, 1, L_x, L_y, omega_values, Gamma, Omega)
    print(i)
    
fig, ax = plt.subplots()
ax.plot(B_values/Delta, sigma_xx[:,0], "-o",  label=r"$\sigma^{(L)}_{xx}$")
ax.plot(B_values/Delta, sigma_xx[:,1], "-o",  label=r"$\sigma^{(R)}_{xx}$")
ax.plot(B_values/Delta, sigma_yy[:,0], "-o",  label=r"$\sigma^{(L)}_{yy}$")
ax.plot(B_values/Delta, sigma_yy[:,1], "-o",  label=r"$\sigma^{(R)}_{yy}$")

ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={B:.2}" + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$\sigma(B_y)$")
ax.legend()
plt.tight_layout()