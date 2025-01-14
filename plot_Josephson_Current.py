#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 07:30:46 2025

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np

data = np.load("Data/Josephson current.npz", allow_pickle=True)
J_0 = data["J_0"]
Josephson_current = data["Josephson_current"]
Josephson_current_k = data["Josephson_current_k"]

phi_values = np.linspace(0, 2*np.pi, 240)    #240
k_y_values = np.linspace(0, np.pi, 200)     #75

fig, ax = plt.subplots()
ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current/J_0)
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J(\phi)/J_0$")
ax.set_title("Josephson current")

fig, ax = plt.subplots()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")
# ax.set_title("Josephson current for given k\n"+
#              r"$\theta=$" + f"{np.round(theta, 2)}"+
#              r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
#              r"; $B=$" + f"{B}")

for i, k in enumerate(k_y_values):
    ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current_k[i,:],
            label=r"$k_y=$" + f"{np.round(k_y_values[i], 2)}",
            marker="o")

#ax.legend(fontsize= "xx-small")
plt.show()