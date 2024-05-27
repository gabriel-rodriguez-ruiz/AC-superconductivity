#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:31:57 2024

@author: gabriel
"""
import numpy as np
from superconductor import Superconductor
import matplotlib.pyplot as plt
from pathlib import Path

L_x = 50
L_y = 50
w_0 = 10
Delta = 0.2
mu = -40
theta = np.pi/2
B = 1*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.56 #5*Delta/k_F
Omega = 0
superconductor_params = {"w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda,
          }

Gamma = 0.1
alpha = 0
beta = 0
Beta = 1000
k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
k_y_values = 2*np.pi*np.arange(0, L_x)/L_y
# k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
# k_y_values = np.pi*np.arange(-L_y, L_x)/L_y

# epsrel=1e-01

# omega_values = np.linspace(-45, 0, 100)

# part = "paramagnetic"
# part = "diamagnetic"
part = "total"
# fermi_function = lambda omega: 1/(1 + np.exp(Beta*omega))
# fermi_function = lambda omega: 1 - np.heaviside(omega, 1)
params = {
    "Gamma":Gamma, "alpha":alpha,
    "beta":beta, "Omega":Omega, "part":part
    }

def fermi_function(omega):
    return np.heaviside(-omega, 1)

S = Superconductor(**superconductor_params)

# E_k = S.plot_spectrum(k_x_values, k_y_values)
# S.plot_spectral_density(omega_values,
#                         k_x=-np.pi/2, k_y=-np.pi/2, Gamma=Gamma)
                          

#%% DC-conductivity

# K = S.get_response_function(alpha, beta, L_x, L_y, omega_values, Gamma, fermi_function, Omega, part)
K = S.get_response_function_quad(alpha, beta, L_x, L_y, Gamma, fermi_function, Omega, part)
print(K)

#%% Convergence in size

L_values = np.linspace(1, 10, 10)
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

K_xx = np.zeros((len(B_values),  2)  , dtype=complex)
K_yy = np.zeros((len(B_values), 2), dtype=complex)
n = np.zeros(len(B_values))
for i, B in enumerate(B_values):
    S.B_x = B * np.cos(theta)
    S.B_y = B * np.sin(theta)
    K_xx[i, 0], K_xx[i, 1] = S.get_response_function_quad(0, 0, L_x, L_y, Gamma, fermi_function, Omega, part)
    K_yy[i, 0], K_yy[i, 1] = S.get_response_function_quad(1, 1, L_x, L_y, Gamma, fermi_function, Omega, part)
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
             +r"; $w_0$"+f"={w_0}"
             +r";$\Omega=$"+f"{Omega}"
             )
ax.text(1, 0.5, f"{part} part")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$K(B_y)$")
ax.legend()
plt.tight_layout()

#%% Normal density
n = S.get_normal_density(L_x, L_y, Gamma, fermi_function)

#%% Normal density vs. B
B_values = np.linspace(0, 3*Delta, 10)
n = np.zeros(len(B_values))

for i, B in enumerate(B_values):
    S.B_x = B * np.cos(theta)
    S.B_y = B * np.sin(theta)
    n[i] = S.get_normal_density(L_x, L_y, Gamma, fermi_function)
    print(i)
fig, ax = plt.subplots()
ax.plot(B_values/Delta, n, "-o",  label=r"$n$")
ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={B:.2}" + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r";$ \Omega=$"+f"{Omega}"
             )
ax.text(1, 0.5, f"{part} part")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$n(B_y)$")
ax.legend()
plt.tight_layout()
#%%
from pathlib import Path

data_folder = Path("Data/")

# file_to_open = data_folder / "K_alpha_alpha_quad_mu_-40_L=10-100.npz"
# np.savez(file_to_open , K_xx=K_xx,
#          K_yy=K_yy, **params,
#          B_values=B_values)

file_to_open = data_folder / "normal_density_mu_-40_L=50_total_B=0-3Delta.npz"
# np.savez(file_to_open , K_xx=K_xx, K_yy=K_yy, **params, B_values=B_values,
#          part=part, **superconductor_params)
np.savez(file_to_open , n=n, **superconductor_params)
#%% Normalized conductivity vs B

data_folder = Path("Data/")
file_to_open = data_folder / "K_alpha_alpha_quad_mu_-40_L=50_total_B=0-3Delta.npz"
Data = np.load(file_to_open)
file_to_open_n = data_folder / "normal_density_mu_-40_L=50_total_B=0-3Delta.npz"
data_n = np.load(file_to_open_n)

B_values = Data["B_values"]
K_xx = Data["K_xx"]
K_yy = Data["K_yy"]
Delta = Data["Delta"]
n = data_n["n"] * 2*np.pi

fig, ax = plt.subplots()
ax.plot(B_values/Delta, K_xx[:,0]/n, "-o",  label=r"$K^{(L)}_{xx}$")
ax.plot(B_values/Delta, K_xx[:,1]/n, "-o",  label=r"$K^{(R)}_{xx}$")
ax.plot(B_values/Delta, K_yy[:,0]/n, "-o",  label=r"$K^{(L)}_{yy}$")
ax.plot(B_values/Delta, K_yy[:,1]/n, "-o",  label=r"$K^{(R)}_{yy}$")

ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={np.round(B,2)}" + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $\Omega=$"+f"{Omega}"
             )
ax.text(1, 0.5, f"{part} part")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$K(B_y)$")
ax.legend()
plt.tight_layout()