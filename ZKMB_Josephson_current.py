# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 08:52:18 2024

@author: Gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductorKY
from functions import phi_spectrum       
from junction import Junction, PeriodicJunction
import scipy
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False})

L_x = 200#500
t = 10
Delta_0 = 0.2#t/5     
Delta_1 = 0#t/20
Lambda = 0.56
phi_angle = np.pi/8
theta = np.pi/2   #np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -4*t#-2*t
t_J = t/2     #t/2#t/5
phi_values = np.linspace(0, 2*np.pi, 240)    #240
k_y_values = np.array([-0.01, 0, 0.01]) #np.linspace(0, 2*np.pi, 200)  #200
antiparallel = False

params = {"L_x":L_x, "t":t, "t_J":t_J,
          "Delta_0":Delta_0,
          "Delta_1":Delta_1,
          "mu":mu, "phi_values":phi_values,
          "k_y_values": k_y_values,
          "B": B, "phi_angle": phi_angle,
          "theta": theta, "antiparallel": antiparallel
          }

eigenvalues = []
for k_y in k_y_values:      ##############changed sign of Lambda
    eigenvalues_k = []
    print(k_y)
    for phi in phi_values:
        phi = np.array([phi])   #array of length 1
        S_ZKMB = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x, B_y, B_z)
        S_ZKMB2 = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x=(1-2*antiparallel)*B_x, B_y=(1-2*antiparallel)*B_y, B_z=(1-2*antiparallel)*B_z)
        J = Junction(S_ZKMB, S_ZKMB2, t_J, phi)
        energies = np.linalg.eigvalsh(J.matrix.toarray())
        energies = list(energies)
        eigenvalues_k.append(energies)
    eigenvalues.append(eigenvalues_k)
eigenvalues = np.array(eigenvalues)
E_phi = eigenvalues

#%% Plotting for a given k

fig, ax = plt.subplots()
j = 2   #index of k-value
for i in range(np.shape(E_phi)[2]):
    ax.plot(phi_values, E_phi[j, :, i], "ok", markersize=0.5)
    # ax.plot(phi_values, E_phi[j+1, :, i], "or", markersize=0.5)

ax.set_title(f"k={k_y_values[j]}")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$E_k$")
ax.set_ylim((-0.02, 0.02))
plt.show()

#%% Total energy

total_energy_k = np.zeros((len(k_y_values)//2 + 1, len(phi_values)))

for i in range(len(k_y_values)//2 + 1):
    E_phi_k_minus_k = (E_phi[i, :, :] + E_phi[len(k_y_values)-(i+1), :, :]) / 2
    E_positive = np.where(E_phi_k_minus_k > 0, E_phi_k_minus_k, np.zeros_like(E_phi_k_minus_k))
    total_energy_k[i] = np.sum(E_positive, axis=1)

fig, ax = plt.subplots()
# ax.plot(phi_values, total_energy_k[0, :], "ok", markersize=0.5)
ax.plot(phi_values, total_energy_k[1, :], "or", label=f"{k_y_values[1]}", markersize=0.5)
ax.legend()
#%% Josephson current

dphi = np.diff(phi_values)
Josephson_current_k = np.diff(-total_energy_k) / dphi
Josephson_current = np.sum(Josephson_current_k, axis=0)

J_0 = np.max(Josephson_current) 
fig, ax = plt.subplots()
ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current/J_0)
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J(\phi)/J_0$")
ax.set_title("Josephson current")

fig, ax = plt.subplots()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")
ax.set_title("Josephson current for given k\n"+
             r"$\theta=$" + f"{np.round(theta, 2)}"+
             r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
             r"; $B=$" + f"{B}")

for i in range(len(k_y_values)//2 + 1):
    ax.scatter(phi_values[:-1]/(2*np.pi), Josephson_current_k[i,:],
               marker=".",
               label=r"$|k_y|=$" + f"{np.abs(np.round(k_y_values[i], 3))}")

ax.legend(fontsize= "xx-small")
plt.show()

#%% Save 

np.savez(f"Data/Josephson_current_theta_{np.round(theta,3)}_phi_angle_{np.round(phi_angle, 3)}_phi_values_{len(phi_values)}_k_y_values_{len(k_y_values)}_tJ_{np.round(t_J, 3)}_antiparallel_{antiparallel}", Josephson_current=Josephson_current, J_0=J_0,
         Josephson_current_k=Josephson_current_k,
        params=params, k_y_values=k_y_values, phi_values=phi_values)