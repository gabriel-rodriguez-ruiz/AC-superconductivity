# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:50:11 2024

@author: Gabriel
"""

from ZKMBsuperconductor import ZKMBSuperconductorKXKY
import numpy as np
from Zak import Zak
import matplotlib.pyplot as plt

k_y = 0.1
L_x = 2000
# L_y = 200
# k_x_values = np.pi/L_x*np.arange(-L_x, L_x)
# k_y_values = np.pi/L_y*np.arange(-L_y, L_y)

t = 10   #10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = 0
B_values = np.linspace(0, 2*Delta_0, 30)       #2*Delta_0

B = None
B_x = None
B_y = None
B_z = None
mu = -40  #in the middle ot the topological phase

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }

Berry_B = np.zeros((len(B_values), 4))

for i, B in enumerate(B_values):
    B_x = B * np.sin(theta) * np.cos(phi)
    B_y = B * np.sin(theta) * np.sin(phi)
    B_z = B * np.cos(theta)
    superconductor_params["B_x"] = B_x
    superconductor_params["B_y"] = B_y
    superconductor_params["B_z"] = B_z
    Z = Zak(ZKMBSuperconductorKXKY, superconductor_params)
    Berry_B[i, :] = Z.get_Zak_Berry_phase(k_y, L_x)
    # Berry_B[i, :] = Z.get_Zak_log_phase(k_y, L_x)


fig, ax = plt.subplots()
ax.plot(B_values/Delta_0, Berry_B[:, 0], label=r"$E_1$")
ax.plot(B_values/Delta_0, Berry_B[:, 1], label=r"$E_2$")
ax.plot(B_values/Delta_0, Berry_B[:, 2], label=r"$E_3$")
ax.plot(B_values/Delta_0, Berry_B[:, 3], label=r"$E_4$")

ax.set_xlabel(r"$\frac{B}{\Delta_0}$")
ax.set_ylabel(r"$\cal{Z}$")
ax.legend()
ax.set_title(r"$k_y=$" + f"{k_y}" + r"$; \mu=$" + f"{mu}" +
             r"; $\lambda=$" + f"{Lambda}" + r"; $\Delta_0=$" + f"{Delta_0}"+
             r"; $L_x=$" + f"{L_x}")

plt.show()