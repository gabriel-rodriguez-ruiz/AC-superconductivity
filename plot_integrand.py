#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:10:29 2024

@author: gabriel
"""
import matplotlib.pyplot as plt
import numpy as np
from superconductor import Superconductor
import scipy

L_x = 20
L_y = 20
w_0 = 10
Delta = 0.2
mu = -40
theta = np.pi/2
B = 0 * Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.56 #5*Delta/k_F
Omega = 0#0.02
superconductor_params = {"w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda,
          }

Gamma = 0.01
alpha = 0
beta = 0
Beta = 1000
k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
k_y_values = 2*np.pi*np.arange(0, L_x)/L_y
# k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
# k_y_values = np.pi*np.arange(-L_y, L_x)/L_y

# epsrel=1e-01


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

#%%

k_x = 0
k_y = 0
omega_values = np.linspace(-45, 0, 4000)


integrand_omega_k_inductive = [S.get_integrand_omega_k_inductive(
    omega, k_x, k_y, alpha, beta, Gamma, fermi_function, Omega, part="total")
                               for omega in omega_values]
poles = S.get_Energy(k_x, k_y)

G_anomalous = [S.get_Green_function(omega, k_x, k_y, Gamma)[0,2] for omega in omega_values]
rho_02 = [S.get_spectral_density(omega, k_x, k_y, Gamma)[0,2] for omega in omega_values]

fig, ax = plt.subplots()

ax.plot(omega_values, integrand_omega_k_inductive)
ax.plot(poles, np.zeros_like(poles), "o", label=r"$E(k_x,k_y)$")
ax.plot(omega_values, np.imag(G_anomalous), label=r"$Im((\hat{G})_{0,2})$")
ax.plot(omega_values, np.real(G_anomalous), label=r"$Re((\hat{G})_{0,2})$")
ax.plot(omega_values, rho_02, label=r"$(\hat{\rho})_{0,2}$")

ax.set_xlabel(r"$\omega$")
ax.set_ylabel("Integrand")
ax.set_title(r"$k_x=$" + f"{np.round(k_x, 2)}"
             + "; $k_y=$" + f"{np.round(k_y, 2)}")
ax.legend()

E_k = S.plot_spectrum(k_x_values, k_y_values)
S.plot_spectral_density(omega_values,
                        k_x, k_y, Gamma=Gamma)
a = -45
b = 0
params = (k_x, k_y, alpha, beta, Gamma, fermi_function, Omega, part)
Integral, abserror, infodict = scipy.integrate.quad(S.get_integrand_omega_k_inductive,
                                a, b, args=params, points=poles,
                                full_output=True)

#%%
omega_values = np.linspace(-45, 0, 1000)
Integrand_omega = [S.get_integrand_omega_inductive(omega, alpha, beta, L_x, L_y, Gamma, fermi_function, Omega)
                   for omega in omega_values]
poles = np.array([S.get_Energy(k_x, k_y) for k_x in k_x_values for k_y in k_y_values]).flatten()


fig, ax = plt.subplots()
ax.plot(omega_values, Integrand_omega, label="Integrand")
ax.plot(poles, np.zeros_like(poles), "o", label=r"$E(k_x,k_y)$")

#%%

a = -45
b = 0
params = (alpha, beta, L_x, L_y, Gamma, fermi_function, Omega)
Integral, abserror, infodict = scipy.integrate.quad(S.get_integrand_omega_inductive,
                                a, b, args=params, #points=poles,
                                full_output=True)
