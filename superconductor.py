#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:12:01 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x

class Superconductor():
    r"""
    A class for a superconductor with spin-orbit coupling and magnetic
    field in the linear response regime of the driving amplitude.
        
    Parameters
    ----------    
    phi_x : float
        Flux in x.
    phi_y : float
        Flux in y.
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
    Omega : float
        Driving frequency.
    A_x : float
        Amplitude of the driving in x direction.
    A_y : float
        Amplitude of the driving in y direction.
    t : float
        Time.
    """
    def __init__(self, phi_x=0, phi_y=0,
                 w_0=0, mu=0, Delta=0, B_x=0, B_y=0, Lambda=0,
                 Omega=0, A_x=0, A_y=0, t=0):
        self.phi_x = phi_x
        self.phi_y = phi_y
        self.w_0 = w_0
        self.mu = mu
        self.Delta = Delta
        self.Lambda = Lambda
        self.B_x = B_x
        self.B_y = B_y
        self.Omega = Omega
        self.A_x = A_x
        self.A_y = A_y
        self.t = t
    def get_Hamiltonian(self, k_x, k_y):
        r""" Periodic Hamiltonian in x and y with flux.
        
        .. math::

            H = \frac{1}{2}\sum_{\mathbf{k}} \psi_{\mathbf{k}}^\dagger H_{\mathbf{k}} \psi_{\mathbf{k}}
            
            H_{\mathbf{k}} =  
                \xi_k(t)\tau_z\sigma_0 + \Delta \tau_x\sigma_0
                + \lambda_{k_x}(t)\tau_z\sigma_y
                + \lambda_{k_y}(t)\tau_z\sigma_x                
                -B_x\tau_0\sigma_x - B_y\tau_0\sigma_y 
            
            \vec{c}_k = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                       -c^\dagger_{-k,\uparrow})^T
        
            \xi_k(t) = -2w_0(cos(k_x+\phi_x)+cos(k_y+\phi_y)) - \mu
                        +2 w_0 (A_{1,x}sin(k_x+\phi_x) + A_{1,y}sin(k_y+\phi_y)) cos(\Omega t)
            
            \lambda_{k_x}(t) = 2\lambda \left[sin(k_x+\phi_x)
            + cos(k_x+\phi_x)A_{1,x} cos(\Omega t)\right]
            
            \lambda_{k_y}(t) =  - 2\lambda \left[sin(k_y+\phi_y)
             + cos(k_y+\phi_y)A_{1,y} cos(\Omega t)\right]
        """
        chi_k = (
                 -2*self.w_0*((np.cos(k_x + self.phi_x)
                                + np.cos(k_y + self.phi_y)))
                 - self.mu
                 + 2*self.w_0*(self.A_x * np.sin(k_x + self.phi_x)
                               + self.A_y * np.sin(k_y + self.phi_y))
                 * np.cos(self.Omega*self.t)
                 )
        Lambda_k_x = (
                      2*self.Lambda*(np.sin(k_x + self.phi_x) 
                         + self.A_x * np.cos(k_x + self.phi_x)
                         * np.cos(self.Omega * self.t) )
                      )
        Lambda_k_y = ( 
                      -2*self.Lambda*(np.sin(k_y + self.phi_y) 
                         + self.A_y * np.cos(k_y + self.phi_y)
                         * np.cos(self.Omega * self.t) )
                      )
        H = (
             chi_k * np.kron(tau_z, sigma_0)
             + Lambda_k_x * np.kron(tau_z, sigma_y)
             + Lambda_k_y * np.kron(tau_z, sigma_x)
             - self.B_x * np.kron(tau_0, sigma_x)
             - self.B_y * np.kron(tau_0, sigma_y)
             + self.Delta * np.kron(tau_x, sigma_0)
             ) * 1/2
        return H
    def get_Green_function(self, omega, k_x, k_y, Gamma):
        r"""
        .. math::
            G_{\mathbf{k}}(\omega) = [\omega\tau_0\sigma_0 - H_{\mathbf{k}} + i\Gamma\tau_0\sigma_0]^{-1}
        Parameters
        ----------
        omega : float
            Frequency.
        k_x : float
            Momentum in x direction.
        k_y : float
            Momentum in y direction.
        Gamma : float
            Damping.

        Returns
        -------
        ndarray
            4x4 Green function.

        """
        H_k = self.get_Hamiltonian(k_x, k_y)
        return np.linalg.inv(omega*np.kron(tau_0, sigma_0)
                             - H_k
                             + 1j * Gamma * np.kron(tau_0, sigma_0)
                             )
    def get_spectral_density(self, omega, k_x, k_y, Gamma):
        """ Returns the spectral density.

        Parameters
        ----------
        omega : float
            Frequency.
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
        G_k = self.get_Green_function(omega, k_x, k_y, Gamma)
        return -2 * np.imag(G_k)
    def get_Fermi_function(self, omega, beta):
        """ Fermi function"""
        return 1/(1 + np.exp(-beta*omega))
