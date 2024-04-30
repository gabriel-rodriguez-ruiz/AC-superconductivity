#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:31:57 2024

@author: gabriel
"""
import numpy as np
from superconductor import Superconductor

if __name__ == "__main__":
   L_x = 50
   L_y = 50
   w_0 = 10
   Delta = 0.2
   mu = -32
   theta = np.pi/4
   B = 3*Delta
   B_x = B * np.cos(theta)
   B_y = B * np.sin(theta)
   Lambda = 0.56 #5*Delta/k_F
   phi_x_values = [0]
   phi_y_values = [0]    #np.linspace(0, np.pi, 1)
   k_x_values = np.pi/L_x*np.arange(-L_x, L_x)
   k_y_values = np.pi/L_y*np.arange(-L_y, L_y)

   params = {"w_0":w_0, "Delta":Delta,
             "mu":mu,
             "phi_x":phi_x_values,
             "phi_y":phi_y_values,
             "B_x":B_x, "B_y":B_y, "Lambda":Lambda
             }
   S = Superconductor(**params)
   S.plot_spectrum(k_x_values, k_y_values)