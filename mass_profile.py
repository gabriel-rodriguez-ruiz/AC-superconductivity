#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:35:11 2024

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.physics.quantum import TensorProduct

def get_phi(y):
    return 4 * np.arctan(np.exp(y))

def get_mass(y, m_0):
    phi = get_phi(y)
    return m_0*np.cos(phi/2)

def get_m_prime(y, m_0_prime):
    phi = get_phi(y)
    return m_0_prime*np.sin(phi)

def get_psi_1(y, m_0, m_prime):
    spinor = 1/2*np.array([1, -1, -1, 1])
    return 1/2*np.array([1, -1])

y = np.linspace(-10, 10, 1000)
m_0 = 1
m_0_prime = 1/10
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(y, get_phi(y),
        label=r"$\phi(y)$")
ax2.plot(y, get_m_prime(y, m_0_prime), label=r"$m'(y)$")
ax2.plot(y, get_mass(y, m_0), label=r"$m(y)$")
ax2.plot(y, get_m_prime(y, m_0_prime) + get_mass(y, m_0), label=r"$m(y)+m'(y)$")

ax1.set_xlabel(r"$y$")
ax2.set_xlabel(r"$y$")
ax1.set_ylabel(r"$\phi(y)$")
ax2.set_ylabel(r"$m(y)+m'(y)$")
plt.tight_layout()
plt.legend()

m, m_prime, Lambda = sp.symbols(("m", "m'", "lambda"))

tau_x = sp.Matrix([[0, 1],
                   [1, 0]])
tau_0 = sp.Matrix([[1, 0],
                   [0, 1]])
sigma_0 = sp.Matrix([[1, 0],
                     [0, 1]])
sigma_x = sp.Matrix([[0, 1],
                     [1, 0]])

M = m*TensorProduct(tau_x, sigma_0) + m_prime*TensorProduct(tau_0, sigma_x)

determinant = M.det()
eigenvalues = sp.solve(determinant, Lambda)
M.eigenvects()