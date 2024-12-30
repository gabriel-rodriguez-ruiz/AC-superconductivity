# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:42:53 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class Zak():
    """A class for calculating the Zak topological invariant for a given periodic
    superconductor S with parameters other than k_x and k_y."""
    def __init__(self, S, superconductor_params):
        self.S = S
        self.superconductor_params = superconductor_params
    def get_eigenstates(self, k_x, k_y):
        """Returns the eigenstates of the Hamiltonian in colunms."""
        H = self.S(k_x, k_y, **self.superconductor_params).matrix
        eigenvalues, eigenvectors = np.linalg.eigh(H)   #eigh gives always a real first element
        for i in range(4):
            if eigenvectors[0, i] < 0:      #Assure the first element is positive real
                eigenvectors[:, i] *= -1
        return eigenvectors
    def get_eigenvalues(self, k_x, k_y):
        """Returns the eigenstates of the Hamiltonian in colunms."""
        H = self.S(k_x, k_y, **self.superconductor_params).matrix
        eigenvalues = np.linalg.eigvalsh(H)   
        return eigenvalues
    def _get_matrix_of_eigenstates(self, k_y, L_x):
        """Returns an array U with eigenvectors in columns for each k_x and a
        given k_y."""
        k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
        U = np.zeros((L_x, 4, 4), dtype=complex)
        for i, k_x in enumerate(k_x_values):
            U[i, :, :] = self.get_eigenstates(k_x, k_y)
        return U
    def get_Zak_Berry_phase(self, k_y, L_x):
        """Returns an array with the Berry phase for the four bands for a given
        k_y and discretization L_x."""
        U = self._get_matrix_of_eigenstates(k_y, L_x)
        derivative = np.diff(U, axis=0)
        sumand = np.zeros((L_x-1, 4), dtype=complex)
        for i in range(L_x-1):
            sumand[i, :] = np.diag(U[i, :, :].conj().T @ derivative[i, :, :])
        gamma = -np.imag(np.sum(sumand, axis=0))
        return gamma
    def get_Zak_log_phase(self, k_y, L_x):
        r"""
            Returns the Berry phase for a given k_y in a system of length L_x.
        .. math ::
            \gamma = - Im(ln(P)) \\
            P = <u_1 | u_2> <u_2|u_3> ... <u_{M-1} |u_1>
            """
        U = self._get_matrix_of_eigenstates(k_y, L_x)
        P = np.zeros((L_x, 4), dtype=complex)
        for i in range(L_x-1):
            P[i, :] = np.diag(U[i, :, :].conj().T @ U[i+1, :, :])
        P[L_x-1, :] = np.diag(U[L_x-1, :, :].conj().T @ U[0, :, :])
        gamma = -np.imag(np.log(np.prod(P, axis=0)))
        return gamma