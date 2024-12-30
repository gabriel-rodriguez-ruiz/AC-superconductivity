# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:57:46 2024

@author: Gabriel
"""
from ZKMBsuperconductor import ZKMBSuperconductorKXKY
import numpy as np
from Zak import Zak
import matplotlib.pyplot as plt

k_y = 0
L_x = 1000

t = 10   #10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = 0
B = 2*Delta_0       #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -40  #in the middle ot the topological phase

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }

Z = Zak(ZKMBSuperconductorKXKY, superconductor_params)
Berry = Z.get_Zak_Berry_phase(k_y, L_x)
# Berry = Z.get_Zak_log_phase(k_y, L_x)

print(Berry)