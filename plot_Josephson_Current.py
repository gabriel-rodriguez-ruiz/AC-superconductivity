#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 07:30:46 2025

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "text.usetex": False})

fig, ax = plt.subplots()


data_folder = Path("Data/")
phi_angles = [r"$\pi/2$", r"$\pi/4$", r"$\pi/6$", r"$\pi/8$", r"$\pi/16$", "0"]
file_names = [
          "Josephson_current_theta_1.571_phi_angle_1.571_phi_values_240_k_y_values_200.npz",
          "Josephson_current_theta_1.571_phi_angle_0.785_phi_values_240_k_y_values_200.npz",
          "Josephson_current_theta_1.571_phi_angle_0.524_phi_values_240_k_y_values_200.npz",
          "Josephson_current_theta_1.571_phi_angle_0.393_phi_values_240_k_y_values_200.npz",
          "Josephson_current_theta_1.571_phi_angle_0.196_phi_values_240_k_y_values_200.npz",
          "Josephson current.npz"
          ]
for i, s in enumerate(file_names):
    file_to_open = data_folder / s
    
    
    data = np.load(file_to_open, allow_pickle=True)
    J_0 = data["J_0"]
    Josephson_current = data["Josephson_current"]
    Josephson_current_k = data["Josephson_current_k"]
    params = data["params"].item()
    phi_values = params["phi_values"]    #240
    k_y_values = params["k_y_values"]     #75
    
    ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current/J_0,
            label=r"$(\theta, \varphi)=$" + r"($\pi/2$, " + f"{phi_angles[i]})")


ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J(\phi)/J_0$")
# ax.set_title("Josephson current")
ax.legend()

plt.show()

#%% Plot Josephson_current_k

s = "Josephson current.npz"
theta = np.pi/2
phi_angle = 0
B = 0.4
file_to_open = data_folder / s


data = np.load(file_to_open, allow_pickle=True)
J_0 = data["J_0"]
Josephson_current = data["Josephson_current"]
Josephson_current_k = data["Josephson_current_k"]
params = data["params"].item()
phi_values = params["phi_values"]    #240
k_y_values = params["k_y_values"]     #75
dphi = np.diff(phi_values)


fig, ax = plt.subplots()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")
ax.set_title("Josephson current for given k\n"+
             r"$\theta=$" + f"{np.round(theta, 2)}"+
             r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
             r"; $B=$" + f"{B}")

for i, k in enumerate(k_y_values):
    ax.scatter(phi_values[:-1]/(2*np.pi), Josephson_current_k[i,:] / dphi,
            label=r"$k_y=$" + f"{np.round(k_y_values[i], 2)}",
            marker=".")

# ax.legend(fontsize= "xx-small")
plt.tight_layout()
plt.show()

#%% Plot effective mass approximation

def get_Josephson_current(k, phi, t_J, v):
    return (1/2 * t_J**2 * np.cos(phi/2) * np.sin(phi/2)
            / np.sqrt((v*k)**2 + t_J**2 * np.cos(phi/2)**2))

t_J = 5
v = 50
k_values = k_y_values #np.linspace(0, np.pi/2, 10)
fig, ax  = plt.subplots()
for k in k_values:
    ax.plot(phi_values[:-1]/(2*np.pi), [get_Josephson_current(k, phi, t_J, v) for
                                    phi in phi_values[:-1]])
    
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")

plt.show()