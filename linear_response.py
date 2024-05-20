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

# omega_values = np.linspace(-45, 0, 100)

# part = "paramagnetic"
# part = "diamagnetic"
part="total"
# fermi_function = lambda omega: 1/(1 + np.exp(Beta*omega))
# fermi_function = lambda omega: 1 - np.heaviside(omega, 1)
def fermi_function(omega):
    return 1 - np.heaviside(omega, 1)

S = Superconductor(**params)

# E_k = S.plot_spectrum(k_x_values, k_y_values)
# S.plot_spectral_density(omega_values,
#                         k_x=-np.pi/2, k_y=-np.pi/2, Gamma=Gamma)
                          

#%% DC-conductivity

# K = S.get_response_function(alpha, beta, L_x, L_y, omega_values, Gamma, fermi_function, Omega, part)
K = S.get_response_function_quad(alpha, beta, L_x, L_y, Gamma, fermi_function, Omega, part)
print(K)

#%% Convergence in size

L_values = np.linspace(10, 100, 10)
K = np.zeros((len(L_values), 2), dtype=complex)
for i, L in enumerate(L_values):
    L_x = L
    L_y = L
    K[i, :] = S.get_response_function_quad(alpha, beta, L_x, L_y, Gamma, fermi_function, Omega, part)
    print(i)
    
fig, ax = plt.subplots()
ax.plot(L_values, np.real(K[:, 0]), "o", label="Real part")
ax.plot(L_values, np.imag(K[:, 0]), "o", label="Imaginary part")
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$K_{xx}(\Omega=0)$")
ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={np.round(B, 2)}" + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}")
ax.annotate(f"{part}", (0.5, 0.75), xycoords="figure fraction")
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
B_values = np.linspace(0, 3*Delta, 10)

K_xx = np.zeros((len(B_values), 2), dtype=complex)
K_yy = np.zeros((len(B_values), 2), dtype=complex)
n = np.zeros(len(B_values))
for i, B in enumerate(B_values):
    S.B_x = B * np.cos(theta)
    S.B_y = B * np.sin(theta)
    K_xx[i, 0], K_xx[i, 1] = S.get_response_function_quad(0, 0, L_x, L_y, Gamma, fermi_function, Omega, "total")
    K_yy[i, 0], K_yy[i, 1] = S.get_response_function_quad(1, 1, L_x, L_y, Gamma, fermi_function, Omega, "total")
    print(i)
    
fig, ax = plt.subplots()
ax.plot(B_values/Delta, K_xx[:,0], "-o",  label=r"$K^{(L)}_{xx}$")
ax.plot(B_values/Delta, K_xx[:,1], "-o",  label=r"$K^{(R)}_{xx}$")
ax.plot(B_values/Delta, K_yy[:,0], "-o",  label=r"$K^{(L)}_{yy}$")
ax.plot(B_values/Delta, K_yy[:,1], "-o",  label=r"$K^{(R)}_{yy}$")

ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={B:.2}" + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$K(B_y)$")
ax.legend()
plt.tight_layout()

#%%
from pathlib import Path

data_folder = Path("Data/")

file_to_open = data_folder / "K_alpha_alpha_mu_-40_L=200.npz"
np.savez(file_to_open , K_xx=K_xx,
         K_yy=K_yy, **params,
         B_values=B_values)