#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:25:03 2024

@author: gabriel
"""

from ZKMBsuperconductor import ZKMBSuperconductorKX
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

L_y = 200
k_x_values = np.linspace(-np.pi/6, np.pi/6, 100)
t = 10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = np.pi/2
B = 2*Delta_0#2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -40   #in the middle ot the topological phase

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }

E_k_x = np.zeros((len(k_x_values), 4*L_y))
for i, k_x in enumerate(k_x_values):
    S = ZKMBSuperconductorKX(k_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
                               B_x, B_y, B_z)
    E_k_x[i, :] = np.linalg.eigvalsh(S.matrix)

fig, axs = plt.subplots(1, 2)
axs[0].plot(k_x_values, E_k_x/Delta_0, color="black")
axs[0].plot(k_x_values[33:67], E_k_x[33:67,2*L_y-1:2*L_y+1]/Delta_0, color="red")

axs[0].set_xlabel(r"$k_x$")
axs[0].set_ylabel(r"$\frac{E(k_x)}{\Delta_0}$")
axs[0].set_title(r"$\mu=$"+f"{np.round(S.mu, 2)}")
fig.suptitle(r"$L_y=$"+f"{L_y}"
             +r"; $\lambda=$" + f"{S.Lambda:.2}"
             +r"; $\Delta_0=$" + f"{S.Delta_0}"
             +r"; $\Delta_1=$" + f"{S.Delta_1}"
             +r"; $w_0=$"+f"{S.t}" + "\n"
             +r"$B_x=$"+f"{np.round(B_x, 2)}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}"
             +r"; $B_z=$"+f"{np.round(B_z, 2)}")
axs[0].set_ylim((-2, 2))

k_x = 0
mu_values = np.linspace(-4*t - 2* np.sqrt(B**2 - Delta_0**2), -4*t + 2 * np.sqrt(B**2 - Delta_0**2), 100)
E_mu = np.zeros((len(mu_values), 4*L_y))
for i, mu in enumerate(mu_values):
    S = ZKMBSuperconductorKX(k_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
                               B_x, B_y, B_z)
    E_mu[i, :] = np.linalg.eigvalsh(S.matrix)

axs[1].plot(mu_values, E_mu/Delta_0, color="black")
axs[1].plot(mu_values[27:73], E_mu[27:73,2*L_y-1:2*L_y+1]/Delta_0, color="red")

axs[1].set_xlabel(r"$\mu$")
axs[1].set_ylabel(r"$\frac{E(\mu)}{\Delta_0}$")
axs[1].set_title(r"$k_x=$"+f"{k_x}")
axs[1].axvline(-4*t + np.sqrt(B**2 - Delta_0**2))
axs[1].axvline(-4*t - np.sqrt(B**2 - Delta_0**2))

axs[1].set_ylim((-2, 2))
plt.tight_layout()
