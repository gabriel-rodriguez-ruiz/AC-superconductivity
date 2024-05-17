# -*- coding: utf-8 -*-
"""
Created on Sat May 11 09:26:06 2024

@author: gabri
"""
import sympy as sp
from pauli_matrices import tau_0, sigma_0, tau_z, tau_x, sigma_x
from sympy.physics.quantum import TensorProduct
import numpy as np

tau_0 = sp.Matrix(np.array(tau_0, dtype=int))
tau_z = sp.Matrix(np.array(tau_z, dtype=int)) 
tau_x = sp.Matrix(np.array(tau_x, dtype=int)) 
sigma_0 = sp.Matrix(np.array(sigma_0, dtype=int))
sigma_x = sp.Matrix(np.array(sigma_x, dtype=int))
sigma_y = sp.Matrix([[0, -sp.I],
                     [sp.I, 0]])

omega, eta, chi_k, Delta, Lambda= sp.symbols("omega eta chi_k Delta lambda", real=True)
H_k = (
       chi_k * TensorProduct(tau_z, sigma_0)
        +Delta * TensorProduct(tau_x, sigma_0)
        # +2*Lambda * TensorProduct(tau_z, sigma_y)
        # -2*Lambda * TensorProduct(tau_z, sigma_x)
       )
A = sp.Matrix(
              (omega + sp.I*eta) * TensorProduct(tau_0, sigma_0)
              - H_k
              )
B = sp.Matrix(
              (omega - sp.I*eta) * TensorProduct(tau_0, sigma_0)
              - H_k
              )

G_retarded = A.inv("ADJ")
G_advanced = B.inv("ADJ")
rho = sp.I * (G_retarded - G_advanced)
# Real = 1/2 * (G_retarded + G_advanced)
# rho_2 = -2*sp.im(G_retarded)
rho_3 = G_retarded * 2*eta*TensorProduct(tau_0, sigma_0) * G_advanced

#%%

G = sp.Matrix(2, 2, lambda i,j: sp.MatrixSymbol("G", 2, 2)[min(i,j), max(i,j)])
rho = sp.Matrix(2, 2, lambda i,j: sp.MatrixSymbol("rho", 2, 2)[min(i,j), max(i,j)])
# G = sp.Matrix(4, 4, lambda i,j: sp.MatrixSymbol("G", 4, 4)[min(i,j), max(i,j)])
# rho = sp.Matrix(4, 4, lambda i,j: sp.MatrixSymbol("rho", 4, 4)[min(i,j), max(i,j)])


# rho = sp.eye(4,4)
# rho = TensorProduct(tau_z, sigma_0)
# v = sp.Matrix(2, 2, lambda i,j: sp.MatrixSymbol("v", 2, 2)[min(i,j), max(i,j)])
# v = TensorProduct(tau_z, sigma_0)+TensorProduct(tau_0, sigma_x)
# v = sp.MatrixSymbol("v", 2,2)
# v = sp.Matrix([[v[0,0], 0],
#                 [0, v[1,1]]])
v = tau_0
P = sp.Trace(v*tau_z*(G*v*tau_z + G.transpose()*v*tau_z)*rho)
D = -sp.Trace(v*(G*v + G.transpose()*v)*rho)

# T = sp.Trace((G*v*TensorProduct(tau_z, sigma_0)*rho + rho*v*TensorProduct(tau_z, sigma_0)*G.transpose())*v*TensorProduct(tau_z, sigma_0))
# Q = sp.Trace((G*v*rho + rho*v*G.transpose())*v)

sp.simplify(D+P)