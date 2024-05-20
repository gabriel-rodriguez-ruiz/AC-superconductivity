#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:58:46 2024

@author: gabriel
"""

import numpy as np
from superconductor import Superconductor
import matplotlib.pyplot as plt
from multiprocessing import Pool

def fermi_function(omega):
    return 1 - np.heaviside(omega, 1)

def integrate(L):
    L_x = L
    L_y = L
    return S.get_response_function_quad(alpha, beta, L_x, L_y, Gamma, fermi_function, Omega, part="total")

if __name__ == "__main__":
    L_x = 10
    L_y = 10
    w_0 = 10
    Delta = 0
    mu = -40
    theta = np.pi/2
    B = 0
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    Lambda = 0.56 #5*Delta/k_F
    Omega = 0
    t = 0
    A_x = 0
    A_y = 0
    params = {"w_0":w_0, "Delta":Delta,
              "mu":mu,
              "B_x":B_x, "B_y":B_y, "Lambda":Lambda,
              }
    
    k_x_values = np.pi/L_x*np.arange(-L_x, L_x)#2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = np.pi/L_x*np.arange(-L_y, L_y)#2*np.pi/L_y*np.arange(0, L_y)
    Gamma = 0.1
    alpha = 0
    beta = 0
    Beta = 1000
    
    omega_values = np.linspace(-45, 0, 200)
    
    part = "total"#"diamagnetic"#"paramagnetic"
    # fermi_function = lambda omega: 1/(1 + np.exp(Beta*omega))

    
    
    # E_k = S.plot_spectrum(k_x_values, k_y_values)
    # S.plot_spectral_density(omega_values,
    #                         k_x=0, k_y=0, Gamma=Gamma)
                              
    # K = S.get_response_function(alpha, beta, L_x, L_y, omega_values, Gamma, fermi_function, Omega, part)
    
    S = Superconductor(**params)    
    # K = S.get_response_function_quad(alpha, beta, L_x, L_y, Gamma, fermi_function, Omega, part="total")
    # print(K)
    
    L_values = np.linspace(10, 100, 10)
    K = np.zeros((len(L_values), 2), dtype=complex)
    P = Pool(4)
    results_pooled = P.map(integrate, L_values)
    # fig, ax = plt.subplots()
    # ax.plot(L_values, np.real(K[:, 0]), "o", label="Real part")
    # ax.plot(L_values, np.imag(K[:, 0]), "o", label="Imaginary part")
    # ax.set_xlabel(r"$L$")
    # ax.set_ylabel(r"$K_{xx}(\Omega=0)$")
    # plt.legend()
