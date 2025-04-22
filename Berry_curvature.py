# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 20:47:14 2025

@author: Gabriel
"""
import numpy as np

def berry_curvature_2d(hamiltonian, kx, ky, dk):
    """
    Calcula la curvatura de Berry para un Hamiltoniano bidimensional.

    Parameters:
        hamiltonian: función que toma (kx, ky) y devuelve la matriz Hamiltoniana
        kx, ky: arrays de puntos en las direcciones kx y ky
        dk: paso diferencial en el espacio k

    Returns:
        curvature: matriz de curvatura de Berry en el espacio (kx, ky)
    """
    nx, ny = len(kx), len(ky)
    curvature = np.zeros((nx, ny, 4))

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Obtén los puntos de k
            kx0, ky0 = kx[i], ky[j]
            kx_plus, kx_minus = kx[i+1], kx[i-1]
            ky_plus, ky_minus = ky[j+1], ky[j-1]

            # Calcula las matrices Hamiltonianas en los puntos adyacentes
            H0 = hamiltonian(kx0, ky0)
            Hkx_plus = hamiltonian(kx_plus, ky0)
            Hkx_minus = hamiltonian(kx_minus, ky0)
            Hky_plus = hamiltonian(kx0, ky_plus)
            Hky_minus = hamiltonian(kx0, ky_minus)

            # Obtén los autoestados
            _, psi0 = np.linalg.eigh(H0)
            _, psi_kx_plus = np.linalg.eigh(Hkx_plus)
            _, psi_kx_minus = np.linalg.eigh(Hkx_minus)
            _, psi_ky_plus = np.linalg.eigh(Hky_plus)
            _, psi_ky_minus = np.linalg.eigh(Hky_minus)

            # Producto escalar para calcular las fases de Berry
            # Calcula la curvatura de Berry
            F_kx_ky = np.zeros(4)
            for k in range(4):
                F_kx_ky[k] = np.angle(np.vdot(psi0[:, k], psi_kx_plus[:, k]) * np.vdot(psi_kx_plus[:, k], psi_ky_plus[:, k]) *
                                   np.vdot(psi_ky_plus[:, k], psi0[:, k]))
                curvature[i, j, k] = F_kx_ky[k] / (dk**2)
    return curvature

from ZKMBsuperconductor import ZKMBSuperconductorKXKY
import numpy as np
from Zak import Zak
import matplotlib.pyplot as plt
import matplotlib.cm as cm

k_y = 0
L_x = 1000

t = 10   #10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = 0
B = 2*Delta_0  #2*Delta_0       #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -40  #in the middle ot the topological phase

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }


# Ejemplo: Hamiltoniano bidimensional
def hamiltonian(kx, ky):
    return ZKMBSuperconductorKXKY(kx, ky, **superconductor_params).matrix

# Espacio k bidimensional
kx = np.linspace(-np.pi, np.pi, 50)
ky = np.linspace(-np.pi, np.pi, 50)
# kx = np.linspace(-0.1, 0.1, 50)
# ky = np.linspace(-0.1, 0.1, 50)
dk = kx[1] - kx[0]

# S = ZKMBSuperconductorKXKY(1, 1, **superconductor_params)


# Calcula la curvatura de Berry
curvatura = berry_curvature_2d(hamiltonian, kx, ky, dk)
print("Curvatura de Berry calculada.")

#%%

Z = np.sum(curvatura[:,:, 0:2], axis=2)
# Z = curvatura[:,:, 3]

fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
               origin='lower', extent=[kx.min(), kx.max(), ky.min(), ky.max()],
               vmax=abs(Z).max(), vmin=-abs(Z).max())

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")

plt.show()
