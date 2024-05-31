#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:44:53 2024

@author: gabriel
"""

import numpy as np
from superconductor import Superconductor
import matplotlib.pyplot as plt
from pathlib import Path

L_x = 200
L_y = 200
w_0 = 10
Delta = 0.2
mu = -40
theta = np.pi/2
B = 11/10 * Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.56 #5*Delta/k_F
superconductor_params = {"w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda,
          }

# k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
# k_y_values = 2*np.pi*np.arange(0, L_y)/L_y
k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
k_y_values = np.pi*np.arange(-L_y, L_y)/L_y

S = Superconductor(**superconductor_params)

fig, ax = S.plot_spectrum(k_x_values, k_y_values, index_k_y=L_y)
fig.suptitle(r"$\lambda=$" + f"{S.Lambda:.2}"
             +r"; $\Delta=$" + f"{S.Delta}"
             + r"; $\mu=$"+f"{S.mu}"
             +r"; $w_0=$"+f"{S.w_0}"
             +r"; $B_y=$"+f"{B_y}")
plt.tight_layout()
