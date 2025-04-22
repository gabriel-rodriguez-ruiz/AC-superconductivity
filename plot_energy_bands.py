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

L_x = 100
L_y = 100
w_0 = 10
Delta = 0.2
mu = -39
theta = 0
B = 0.4 
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda_R = 0.56 #5*Delta/k_F
Lambda_D = 0
superconductor_params = {"w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B_x":B_x, "B_y":B_y, "Lambda_R":Lambda_R,
          "Lambda_D": Lambda_D
          }

# k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
# k_y_values = 2*np.pi*np.arange(0, L_y)/L_y
k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
k_y_values = np.pi*np.arange(-L_y, L_y)/L_y

S = Superconductor(**superconductor_params)

fig, ax = S.plot_spectrum(k_x_values, k_y_values, index_k_y=L_y)
fig.suptitle(r"$\lambda_R=$" + f"{np.round(S.Lambda_R,2)}"
             +r"; $\Delta=$" + f"{S.Delta}"
             + r"; $\mu=$"+f"{S.mu}"
             +r"; $w_0=$"+f"{S.w_0}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}")
plt.tight_layout()
