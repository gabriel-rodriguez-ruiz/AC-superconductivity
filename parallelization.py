#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:04:04 2024

@author: gabriel
"""

import numpy as np
from superconductor import Superconductor
import matplotlib.pyplot as plt
import multiprocessing


if __name__ == "__main__":
    L_x = 3
    L_y = 3
    w_0 = 10
    Delta = 0
    mu = -40
    theta = np.pi/2
    B = 0
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    Lambda = 0.56 #5*Delta/k_F
    Omega = 0
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
    epsrel=1e-01
    
    # omega_values = np.linspace(-45, 0, 100)
    
    # part = "paramagnetic"
    # part = "diamagnetic"
    part = "total"
    # fermi_function = lambda omega: 1/(1 + np.exp(Beta*omega))
    # fermi_function = lambda omega: 1 - np.heaviside(omega, 1)
    def fermi_function(omega):
        return np.heaviside(-omega, 1)
    
    S = Superconductor(**params)
    
    # E_k = S.plot_spectrum(k_x_values, k_y_values)
    # S.plot_spectral_density(omega_values,
    #                         k_x=-np.pi/2, k_y=-np.pi/2, Gamma=Gamma)
    def integrate(L):
        L_x = L
        L_y = L
        return S.get_response_function_quad(alpha, beta, L_x, L_y, Gamma, fermi_function, Omega, part, epsrel)
    
    L_values = np.linspace(10, 100, 10)
    p = multiprocessing.Pool(10)
    results_pooled = p.map(integrate, L_values)
    K = np.array(results_pooled)
    
    fig, ax = plt.subplots()
    ax.plot(L_values, np.real(K[:, 0]), "o", label="Real part")
    ax.plot(L_values, np.imag(K[:, 0]), "o", label="Imaginary part")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$K_{xx}(\Omega=0)$")
    ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
                 +r"; $\Delta=$" + f"{Delta}"
                 +r"; $\theta=$" + f"{theta:.3}"
                 +f"; B={np.round(B, 2)}" + r"; $\mu$"+f"={mu}"
                 +r"; $w_0$"+f"={w_0}")
    ax.annotate(f"{part}", (0.5, 0.75), xycoords="figure fraction")
    plt.legend()