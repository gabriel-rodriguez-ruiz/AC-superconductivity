#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:10:18 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from superconductor import Superconductor
import scipy

L_x = 1
L_y = 1
w_0 = 10
Delta = 0.2
mu = -39
theta = np.pi/2
B = 0
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.56 #5*Delta/k_F
superconductor_params = {"w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda,
          }

# k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
# k_y_values = 2*np.pi*np.arange(0, L_y)/L_y

S = Superconductor(**superconductor_params)
# normal_params = superconductor_params
# normal_params.pop("Delta")
# N = Superconductor(Delta=0, **normal_params)

omega_values = np.linspace(-6*Delta, 6*Delta, 100)
Gamma = 0.01
density_of_states = np.array([S.get_density_of_states(omega, L_x, L_y, Gamma)
                       for omega in omega_values])
# normal_density_of_states = np.array([N.get_density_of_states(omega, L_x, L_y, Gamma)
#                        for omega in omega_values])

fig, ax = plt.subplots()
ax.plot(omega_values/Delta, density_of_states)

ax.set_xlabel(r"$\frac{\omega}{\Delta}$")
ax.set_ylabel(r"$\frac{\rho(\omega)}{\nu_0(\omega)}$")

fig.suptitle(r"$\lambda=$" + f"{S.Lambda:.2}"
             +r"; $\Delta=$" + f"{S.Delta}"
             + r"; $\mu=$"+f"{S.mu}"
             +r"; $w_0=$"+f"{S.w_0}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}"
             +r"; $\Gamma=$"+f"{Gamma}")
plt.tight_layout()

# a = -50
# b = 50
# Integral = scipy.integrate.quad(S.get_density_of_states,
#                                 a, b, args=(L_x, L_y, Gamma))


