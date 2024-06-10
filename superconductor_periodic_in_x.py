#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:03:28 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x
import matplotlib.pyplot as plt
import scipy
from hamiltonian import Hamiltonian

class Superconductor(Hamiltonian):
    r"""
    A class for a superconductor with spin-orbit coupling and magnetic
    field in the linear response regime of the driving amplitude.
        
    Parameters
    ----------    
    w_0 : float
        Hopping amplitude.
    mu : float
        Chemical potential.
    Delta : float
        Local s-wave superconducting gap.
    Lambda : float
        Rashba spin-orbit coupling.
    B_x : float
        Magnetic field in x direction.
    B_y : float
        Magnetic field in y direction.
    L_x : int
       Length in x direction.
    L_y : int
       Length in y direction. 
    """
    def __init__(self, L_x, L_y, w_0, mu, Delta,
                 B_x, B_y, Lambda):
        self.L_x = L_x
        self.L_y = L_y
        self.w_0 = w_0
        self.mu = mu
        self.Delta = Delta
        self.Lambda = Lambda
        self.B_x = B_x
        self.B_y = B_y
    def get_Hamiltonian(self, k_x):
        r""" Periodic Hamiltonian in x with magnetic field.
        
        .. math::
    
            H = \frac{1}{2}\sum_k H_{k}
            
            H_{k} = \sum_n^L \vec{c}^\dagger_n\left[ 
                \xi_k\tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0
                +2\lambda sin(k) \tau_z\sigma_y
                -\tau_0(B_x\sigma_x+B_y\sigma_y)
                \right]\vec{c}_n +
                \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 
                +i\lambda \tau_z\sigma_x
                )\vec{c}_{n+1}
                + H.c. \right)
            
            \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                       -c^\dagger_{-k,\uparrow})^T
        
            \xi_k = -2w_0cos(k) - \mu
            """
        H = ZKMBSuperconductorKX(k=k_x, L_y=self.L_y,
                                 t=self.w_0, mu=self.mu,
                                 Delta_0=self.Delta,
                                 Delta_1=0, Lambda=self.Lambda,
                                 B_x=self.B_x, B_y=self.B_y,
                                 B_z=0)
        return H.matrix
    def get_Green_function(self, omega, k_x, site_y, Gamma):
        r"""
        .. math::
            G_{\mathbf{k}}(\omega) = [\omega\tau_0\sigma_0 - H_{\mathbf{k}} + i\Gamma\tau_0\sigma_0]^{-1}
        Parameters
        ----------
        omega : float
            Frequency.
        k_x : float
            Momentum in x direction.
        site_y : int
            Site in y direction.
        Gamma : float
            Damping.

        Returns
        -------
        ndarray
            4L_y x 4L_y Green function.

        """
        H_k_x = self.get_Hamiltonian(k_x)
        return np.linalg.inv(omega * np.eye(4*self.L_y)
                             - H_k_x
                             + 1j * Gamma * np.eye(4*self.L_y)
                             )
    def get_spectral_density(self, omega_values, k_x, k_y, Gamma):
        """ Returns the spectral density.

        Parameters
        ----------
        omega_values : float or ndarray
            Frequency values.
        k_x : float
            Momentum in x direction.
        k_y : float
            Momentum in y direction.
        Gamma : float
            Damping.

        Returns
        -------
        ndarray
            Spectral density.
        """
        if np.size(omega_values)==1:
            G_k = self.get_Green_function(omega_values, k_x, k_y, Gamma)
            # return G_k @ (2*Gamma*np.kron(tau_0, sigma_0)) @ G_k.conj().T
            return 1j * (G_k - G_k.conj().T)
        else:
            rho = np.zeros((len(omega_values), 4 , 4), dtype=complex)
            for i, omega in enumerate(omega_values):
                rho[i, :, :] = self.get_spectral_density(omega, k_x, k_y, Gamma)
            return rho
    