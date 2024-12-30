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

L_x = 300
t = 10
Delta_0 = 0.2#t/5     
Delta_1 = 0#t/20
Lambda = 0.56
phi_angle = 0
theta = np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -4*t#-2*t
t_J = t/2       #t/2#t/5
phi_values = np.linspace(0, 2*np.pi, 50)
k_y_values = np.linspace(0, np.pi/10, 5)

params = {"L_x":L_x, "t":t, "t_J":t_J,
          "Delta_0":Delta_0,
          "Delta_1":Delta_1,
          "mu":mu, "phi_values":phi_values,
          "k_y_values": k_y_values,
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
                                      Lambda, B_x=-B_x, B_y=-B_y, B_z=-B_z)
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
    ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current_k[i,:],
            label=r"$k_y=$" + f"{np.round(k_y_values[i], 2)}")

ax.legend(fontsize= "xx-small")
plt.show()

#%% Plotting of total energy

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="large")  # reduced tick label size
plt.rc("ytick", labelsize="large")
plt.rc('font', size=18) #controls default text size
plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=18) #fontsize of the x and y labels
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False
plt.rc('legend', fontsize=18) #fontsize of the legend

def energy(phi_values, E_0, E_J):
    # return E_0*(4*np.cos(phi_eq[0])*(1-np.cos(phi_values))-2*np.sin(phi_values)**2) 
    return E_J*(1-np.cos(phi_values)) - 2*E_0*np.sin(phi_values)**2
popt, pcov = scipy.optimize.curve_fit(energy, xdata = phi_values, ydata = -total_energy+total_energy[0])
E_0 = popt[0]
E_J = popt[1]
fig, ax = plt.subplots()
ax.plot(phi_values/(2*np.pi), -total_energy+total_energy[0], label="Numerical")
ax.plot(phi_values/(2*np.pi), energy(phi_values, E_0, E_J))
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$E(\phi)$")
ax.set_title(r"$\phi_{0}=$"+f"{(2*np.pi-phi_eq[0])/(2*np.pi):.2}"+r"$\times 2\pi$")

# plt.legend()
plt.tight_layout()
#np.savez("phi_eq=0.12", E=-total_energy+total_energy[0], E_fit=energy(phi_values, E_0, E_J), phi=phi_values/(2*np.pi),
#         params=params)

plt.show()
