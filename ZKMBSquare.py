# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:00:13 2024

@author: Gabriel
"""

from ZKMBsuperconductor import ZKMBSparseSuperconductor, ZKMBSuperconductor,\
                                ZKMBSparseSuperconductorPeriodicInY
import numpy as np
import matplotlib.pyplot as plt
import scipy
from functions import get_components
from pauli_matrices import tau_y, sigma_y

L_x = 200
L_y = 1000#1000#700
t = 10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = np.pi/4
B = 2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -40   #in the middle ot the topological phase
k = 16

H = ZKMBSparseSuperconductor(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
                             B_x, B_y, B_z)
# H = ZKMBSuperconductor(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
#                              B_x, B_y, B_z)
# H = ZKMBSparseSuperconductorPeriodicInY(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
#                              B_x, B_y, B_z)

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H.matrix, k=k, sigma=0) 
# eigenvalues_sparse, eigenvectors_sparse = np.linalg.eigh(H.matrix) 

# C = np.kron(tau_y, sigma_y)     #charge conjugation operator
# M = np.kron(np.eye(L_x*L_y), C)
# U = eigenvectors_sparse
# D = np.linalg.inv(U) @ H.matrix @ U
# C_prime = U @ M @ np.linalg.inv(U)

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }

#%% Probability density
# eigenvectors_sparse = 1/2 * (eigenvectors_sparse - eigenvectors_sparse.conj())

index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2))
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components
    
# for i in index:
#     destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors[:,i], L_x, L_y)
#     probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
#     zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components


#%% Plot of probability density

index = 0

fig, ax = plt.subplots()
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower",
                  aspect=0.3) #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Probability density" + "\n" +
             f"E={np.round(eigenvalues_sparse[index], 6)}"
             + r"; $\mu=$" + f"{mu}"
            +r"; $\lambda=$" + f"{Lambda:.2}"
            +r"; $\Delta_0=$" + f"{Delta_0}"
            +r"; $\Delta_1=$" + f"{Delta_1}"
            +r"; $w_0=$"+f"{t}" + "\n"
            +r"$B_x=$"+f"{np.round(B_x, 2)}"
            +r"; $B_y=$"+f"{np.round(B_y, 2)}"
            +r"; $B_z=$"+f"{np.round(B_z, 2)}")

# ax.set_title(f"{k for k in superconductor_params.keys()}")
plt.tight_layout()
plt.show()

print("\007")

#%% 

fig, ax = plt.subplots()
image = ax.plot(probability_density[index][L_y//2, :]) 
ax.set_xlabel("x")
ax.set_ylabel(r"$|\Psi(x)|^2$")
ax.set_title("Probability density" + "\n" +
             f"E={np.round(eigenvalues_sparse[index], 6)}"
             + r"; $\mu=$" + f"{mu}"
            +r"; $\lambda=$" + f"{Lambda:.2}"
            +r"; $\Delta_0=$" + f"{Delta_0}"
            +r"; $\Delta_1=$" + f"{Delta_1}"
            +r"; $w_0=$"+f"{t}" + "\n"
            +r"$B_x=$"+f"{np.round(B_x, 2)}"
            +r"; $B_y=$"+f"{np.round(B_y, 2)}"
            +r"; $B_z=$"+f"{np.round(B_z, 2)}")

# ax.set_title(f"{k for k in superconductor_params.keys()}")
plt.tight_layout()
plt.show()

#%% Save
# from pathlib import Path

# data_folder = Path("Data/")


# name = f"Probability_density_Periodic_in_Y_mu_{mu}_L_x={L_x}_L_y={L_y}_B_x={np.round(B_x,3)}_B_y={np.round(B_y,3)}_B_z={np.round(B_z,3)}.npz"
# file_to_open = data_folder / name
# np.savez(file_to_open, probability_density=probability_density, k=k, index=index, L_x=L_x, L_y=L_y,
#          **superconductor_params)
# print("\007")