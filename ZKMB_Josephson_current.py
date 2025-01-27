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

L_x = 300
t = 10
Delta_0 = 0.2#t/5     
Delta_1 = 0#t/20
Lambda = 0.56
phi_angle = 0
theta = np.pi/2   #np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -4*t#-2*t
t_J = t/2       #t/2#t/5
phi_values = np.linspace(0.475*2*np.pi, 0.525*2*np.pi, 240)    #240
k_y_values = np.linspace(0, np.pi/20, 5)     #200

params = {"L_x":L_x, "t":t, "t_J":t_J,
          "Delta_0":Delta_0,
          "Delta_1":Delta_1,
          "mu":mu, "phi_values":phi_values,
          "k_y_values": k_y_values,
          "B": B, "phi_angle": phi_angle,
          "theta": theta
          }

eigenvalues = []
for k_y in k_y_values:
    eigenvalues_k = []
    print(k_y)
    for phi in phi_values:
        phi = np.array([phi])   #array of length 1
        S_ZKMB = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x, B_y, B_z)
        S_ZKMB2 = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x=B_x, B_y=B_y, B_z=B_z)
        J = Junction(S_ZKMB, S_ZKMB2, t_J, phi)
        energies = np.linalg.eigvalsh(J.matrix.toarray())
        energies = list(energies)
        eigenvalues_k.append(energies)
    eigenvalues.append(eigenvalues_k)
eigenvalues = np.array(eigenvalues)
E_phi = eigenvalues

#%% Plotting for a given k

fig, ax = plt.subplots()
j = 0   #index of k-value
for i in range(np.shape(E_phi)[2]):
    ax.plot(phi_values, E_phi[j, :, i], ".k", markersize=1)

ax.set_title(f"k={k_y_values[j]}")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$E_k$")
plt.show()
#%% Total energy

E_positive = E_phi[:, :, np.shape(E_phi)[2]//2:]
total_energy_k = np.sum(E_positive, axis=2)
total_energy = np.sum(total_energy_k, axis=0) 
phi_eq = phi_values[np.where(min(-total_energy)==-total_energy)]

#%% Josephson current

dphi = np.diff(phi_values)
Josephson_current = np.diff(-total_energy)
Josephson_current_k = np.diff(-total_energy_k)

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

for i, k in enumerate(k_y_values):
    ax.scatter(phi_values[:-1]/(2*np.pi), Josephson_current_k[i,:],
               marker=".",
               label=r"$k_y=$" + f"{np.round(k_y_values[i], 2)}")

ax.legend(fontsize= "xx-small")
plt.show()

#%% Save 

# np.savez(f"Data/Josephson_current_theta_{np.round(theta,3)}_phi_angle_{np.round(phi_angle, 3)}_phi_values_{len(phi_values)}_k_y_values_{len(k_y_values)}_near_phi_pi_over_2", Josephson_current=Josephson_current, J_0=J_0,
#          Josephson_current_k=Josephson_current_k,
#         params=params, k_y_values=k_y_values, phi_values=phi_values)