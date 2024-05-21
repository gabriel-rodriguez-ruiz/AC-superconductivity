#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:12:01 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x
import matplotlib.pyplot as plt
import scipy

class Superconductor():
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
    Omega : float
        Driving frequency.
    A_x : float
        Amplitude of the driving in x direction.
    A_y : float
        Amplitude of the driving in y direction.
    t : float
        Time.
    """
    def __init__(self, w_0=0, mu=0, Delta=0,
                 B_x=0, B_y=0, Lambda=0):
        self.w_0 = w_0
        self.mu = mu
        self.Delta = Delta
        self.Lambda = Lambda
        self.B_x = B_x
        self.B_y = B_y
    def get_velocity_0(self, k_x, k_y):
        v_0_k_x = (
                   2*self.w_0*np.sin(k_x) * np.kron(tau_0, sigma_0)
                   + 2*self.Lambda*np.cos(k_x) * np.kron(tau_0, sigma_y)
                   )
        v_0_k_y = (
                   2*self.w_0*np.sin(k_y) * np.kron(tau_0, sigma_0)
                   - 2*self.Lambda*np.cos(k_y) * np.kron(tau_0, sigma_x)
                   )
        return [v_0_k_x, v_0_k_y]
    def get_velocity_1(self, k_x, k_y):
        v_1_k_x = (
                   2*self.w_0*np.cos(k_x) * np.kron(tau_z, sigma_0)
                   - 2*self.Lambda*np.sin(k_x) * np.kron(tau_z, sigma_y)
                   )
        v_1_k_y = (
                   2*self.w_0*np.cos(k_y) * np.kron(tau_z, sigma_0)
                   + 2*self.Lambda*np.sin(k_y) * np.kron(tau_z, sigma_x)
                   )
        return [v_1_k_x, v_1_k_y]
    def get_Hamiltonian(self, k_x, k_y):
        r""" Periodic Hamiltonian in x and y with flux.
        
        .. math::

            H = \frac{1}{2}\sum_{\mathbf{k}} \psi_{\mathbf{k}}^\dagger H_{\mathbf{k}} \psi_{\mathbf{k}}
            
            H_{\mathbf{k}} =  
                \xi_k\tau_z\sigma_0 + \Delta \tau_x\sigma_0
                + \lambda_{k_x}\tau_z\sigma_y
                + \lambda_{k_y}\tau_z\sigma_x                
                -B_x\tau_0\sigma_x - B_y\tau_0\sigma_y 
            
            \vec{c}_k = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                       -c^\dagger_{-k,\uparrow})^T
        
            \xi_k = -2w_0(cos(k_x)+cos(k_y)) - \mu
            
            \lambda_{k_x} = 2\lambda sin(k_x)

            \lambda_{k_y}(t) =  - 2\lambda sin(k_y)
        """
        chi_k = -2*self.w_0*(np.cos(k_x) + np.cos(k_y)) - self.mu
        Lambda_k_x = 2*self.Lambda*np.sin(k_x)
        Lambda_k_y = -2*self.Lambda*np.sin(k_y) 
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
    def get_Energy(self, k_x_values, k_y_values):
        if np.size([k_x_values, k_y_values])==2:
            H = self.get_Hamiltonian(k_x_values, k_y_values)
            return np.linalg.eigvalsh(H)
        else:
            energies = np.zeros((len(k_x_values), len(k_y_values),
                                 4))
            for i, k_x in enumerate(k_x_values):
                for j, k_y in enumerate(k_y_values):
                    for k in range(4):
                        H = self.get_Hamiltonian(k_x, k_y)
                        energies[i, j, k] = np.linalg.eigvalsh(H)[k]
            return energies
    def plot_spectrum(self, k_x_values, k_y_values):
        E = self.get_Energy(k_x_values, k_y_values)
        L_y = len(k_y_values)//2
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(k_x_values, E[:,L_y,0])
        ax1.plot(k_x_values, E[:,L_y,1])
        ax1.plot(k_x_values, E[:,L_y,2])
        ax1.plot(k_x_values, E[:,L_y,3])
        ax1.set_xlabel(r"$k_x$")
        ax1.set_ylabel(r"$E(k_x,k_y=$"+f"{np.round(k_y_values[L_y],2)})")
        X, Y = np.meshgrid(k_x_values, k_y_values)
        C1 = ax2.contour(Y, X, E[:,:,1]>0, 0, colors="C1") #notice the inversion of X and Y
        C2 = ax2.contour(Y, X, E[:,:,2]<0, 0, colors="C2")
        C3 = ax2.contour(Y, X, E[:,:,0], 10, colors="C0")
        ax2.clabel(C1, inline=True, fontsize=10)
        ax2.clabel(C2, inline=True, fontsize=10)
        ax2.clabel(C3, inline=True, fontsize=10)
        ax2.set_xlabel(r"$k_x$")
        ax2.set_ylabel(r"$k_y$")
        plt.tight_layout()
    def plot_spectral_density(self, omega_values, k_x, k_y, Gamma):
        rho = self.get_spectral_density(omega_values, k_x, k_y, Gamma)
        fig, axs = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                axs[i,j].plot(omega_values, rho[:, i,j], label=r"$(kx, ky)=$"+f"({np.round(k_x,2)}, {np.round(k_y,2)})")
        fig.supxlabel(r"$\omega$")
        fig.supylabel(r"$\hat{\rho}_{\mathbf{k}}(\omega)$")
        fig.suptitle(r"$(kx, ky)=$"+f"({np.round(k_x,2)}, {np.round(k_y,2)})")
        plt.tight_layout()
    def integrate_spectral_density(self, k_x, k_y, a, b, Gamma):
        f = self.get_spectral_density
        return scipy.integrate.quad_vec(f, a, b, args=(k_x, k_y, Gamma))[0]
    def get_response_function(self, alpha, beta, L_x, L_y, omega_values, Gamma, Fermi_function, Omega, part="total"):
        """Returns the response function element (alpha, beta)
        Fermi function should be a function f(omega).
        If part=0, it calculates the paramegnetic part.
        If part=1, it calculates the diamagnetic and paramagentic
        """
        d, p = self.__select_part(part)
        dw = np.diff(omega_values)[0]
        k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
        k_y_values = 2*np.pi/L_y*np.arange(0, L_y)        
        integrand_inductive = np.zeros((len(k_x_values), len(k_y_values),
                                        len(omega_values)), dtype=complex)
        integrand_ressistive = np.zeros((len(k_x_values), len(k_y_values),
                                         len(omega_values)), dtype=complex)
        for i, k_x in enumerate(k_x_values):
            for j, k_y in enumerate(k_y_values):
                v_0 = self.get_velocity_0(k_x, k_y)
                v_1 = self.get_velocity_1(k_x, k_y)
                for k, omega in enumerate(omega_values):
                    rho = self.get_spectral_density(omega, k_x, k_y, Gamma)
                    G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma)
                    G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma)
                    G_plus_Omega_dagger = G_plus_Omega.conj().T
                    G_minus_Omega_dagger = G_minus_Omega.conj().T
                    fermi_function = Fermi_function(omega)
                    if alpha==beta:
                        integrand_inductive[i, j, k] = (
                            1/(2*np.pi) * fermi_function
                                * np.trace(
                                           d * rho @ v_1[alpha]
                                           + p * 1/2 * rho
                                           @ (
                                              v_0[alpha]
                                              @ (G_plus_Omega
                                                 + G_minus_Omega)
                                              @ v_0[beta] 
                                              + v_0[beta]
                                              @ (G_plus_Omega_dagger
                                                 + G_minus_Omega_dagger)
                                              @ v_0[alpha]
                                              )
                                           )
                            )
                    else:
                        integrand_inductive[i, j, k] = (
                            1/(2*np.pi) * fermi_function
                                * np.trace(
                                           d * rho @ v_1[alpha]
                                           + p * 1/2 * rho
                                           @ (
                                              v_0[alpha]
                                              @ (G_plus_Omega
                                                 + G_minus_Omega)
                                              @ v_0[beta] 
                                              + v_0[beta]
                                              @ (G_plus_Omega_dagger
                                                 + G_minus_Omega_dagger)
                                              @ v_0[alpha]
                                              )
                                           )
                            )
                    integrand_ressistive[i, j, k] = (
                        1/(2*np.pi) * fermi_function
                           * np.trace(
                                      1j/2 * rho
                                      @ (
                                         v_0[alpha]
                                         @ (G_plus_Omega
                                            - G_minus_Omega)
                                         @ v_0[beta]              
                                         - v_0[beta]
                                         @ (G_plus_Omega_dagger
                                            - G_minus_Omega_dagger)
                                         @ v_0[alpha]
                                         )
                                      )
                        )
        integral_inductive = np.sum(integrand_inductive, axis=2) * dw
        K_inductive = 1/(L_x*L_y) * np.sum(integral_inductive)
        integral_ressistive = np.sum(integrand_ressistive, axis=2) * dw
        K_ressistive = 1/(L_x*L_y) * np.sum(integral_ressistive)
        return [K_inductive, K_ressistive]
    def __select_part(self, part):
        if part=="paramagnetic":
            p = 1
            d = 0
        elif part=="diamagnetic":
            p = 0
            d = 1
        else:
            p = 1
            d = 1
        return [d, p]
    def get_integrand_omega_inductive(self, omega, alpha, beta, L_x, L_y, Gamma, Fermi_function, Omega, part="total"):
        r"""Returns the integrand of the response function element (alpha, beta)
        Fermi function should be a function f(omega).
        If part=0, it calculates the paramegnetic part.
        If part=1, it calculates the diamagnetic and paramagnetic
        
        .. math::
            \frac{1}{2}\sum_{\mathbf{k}} \int \frac{d\omega}{2\pi} f(\omega) Tr\left( \hat{\rho}^{0}_{\mathbf{k}}(\omega) \hat{v}^{(1)}_{k_\alpha}(t)  \right)
            
        """
        d, p = self.__select_part(part)
        k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
        k_y_values = 2*np.pi/L_y*np.arange(0, L_y)        
        integrand_inductive = np.zeros((len(k_x_values), len(k_y_values)),
                                        dtype=complex)
        for i, k_x in enumerate(k_x_values):
            for j, k_y in enumerate(k_y_values):
                v_0 = self.get_velocity_0(k_x, k_y)
                v_1 = self.get_velocity_1(k_x, k_y)
                rho = self.get_spectral_density(omega, k_x, k_y, Gamma)
                G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma)
                G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma)
                G_plus_Omega_dagger = G_plus_Omega.conj().T
                G_minus_Omega_dagger = G_minus_Omega.conj().T
                fermi_function = Fermi_function(omega)
                if alpha==beta:
                    integrand_inductive[i, j] = (
                        1/(2*np.pi) * fermi_function
                            * np.trace(
                                       d * rho @ v_1[alpha]
                                       + p * 1/2 * rho
                                       @ (
                                          v_0[alpha]
                                          @ (G_plus_Omega
                                             + G_minus_Omega)
                                          @ v_0[beta] 
                                          + v_0[beta]
                                          @ (G_plus_Omega_dagger
                                             + G_minus_Omega_dagger)
                                          @ v_0[alpha]
                                          )
                                       )
                        )
                else:
                    integrand_inductive[i, j] = (
                        1/(2*np.pi) * fermi_function
                            * np.trace(
                                       d * rho @ v_1[alpha]
                                       + p * 1/2 * rho
                                       @ (
                                          v_0[alpha]
                                          @ (G_plus_Omega
                                             + G_minus_Omega)
                                          @ v_0[beta] 
                                          + v_0[beta]
                                          @ (G_plus_Omega_dagger
                                             + G_minus_Omega_dagger)
                                          @ v_0[alpha]
                                          )
                                       )
                        )
        return integrand_inductive
    def get_integrand_omega_ressistive(self, omega, alpha, beta, L_x, L_y, Gamma, Fermi_function, Omega, part="total"):
        r"""Returns the integrand of the response function element (alpha, beta)
        Fermi function should be a function f(omega).
        If part=0, it calculates the paramegnetic part.
        If part=1, it calculates the diamagnetic and paramagnetic
        
        .. math::
            \frac{1}{2}\sum_{\mathbf{k}} \int \frac{d\omega}{2\pi} f(\omega) Tr\left( \hat{\rho}^{0}_{\mathbf{k}}(\omega) \hat{v}^{(1)}_{k_\alpha}(t)  \right)
            
        """
        d, p = self.__select_part(part)
        k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
        k_y_values = 2*np.pi/L_y*np.arange(0, L_y)        
        integrand_ressistive = np.zeros((len(k_x_values), len(k_y_values)),
                                        dtype=complex)
        for i, k_x in enumerate(k_x_values):
            for j, k_y in enumerate(k_y_values):
                v_0 = self.get_velocity_0(k_x, k_y)
                rho = self.get_spectral_density(omega, k_x, k_y, Gamma)
                G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma)
                G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma)
                G_plus_Omega_dagger = G_plus_Omega.conj().T
                G_minus_Omega_dagger = G_minus_Omega.conj().T
                fermi_function = Fermi_function(omega)
                integrand_ressistive[i, j] = (
                    1/(2*np.pi) * fermi_function
                       * np.trace(
                                  1j/2 * rho
                                  @ (
                                     v_0[alpha]
                                     @ (G_plus_Omega
                                        - G_minus_Omega)
                                     @ v_0[beta]              
                                     - v_0[beta]
                                     @ (G_plus_Omega_dagger
                                        - G_minus_Omega_dagger)
                                     @ v_0[alpha]
                                     )
                                  )
                    )
        return integrand_ressistive
    def get_integrand_omega_k_inductive(self, omega, k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, part="total"):
        r"""Returns the integrand of the response function element (alpha, beta)
        at omega and (k_x, k_y)
        """
        d, p = self.__select_part(part)    
        v_0 = self.get_velocity_0(k_x, k_y)
        v_1 = self.get_velocity_1(k_x, k_y)
        rho = self.get_spectral_density(omega, k_x, k_y, Gamma)
        G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma)
        G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma)
        G_plus_Omega_dagger = G_plus_Omega.conj().T
        G_minus_Omega_dagger = G_minus_Omega.conj().T
        fermi_function = Fermi_function(omega)
        if alpha==beta:
            integrand_inductive = (
                1/(2*np.pi) * fermi_function
                    * np.trace(
                               d * rho @ v_1[alpha]
                               + p * 1/2 * rho
                               @ (
                                  v_0[alpha]
                                  @ (G_plus_Omega
                                     + G_minus_Omega)
                                  @ v_0[beta] 
                                  + v_0[beta]
                                  @ (G_plus_Omega_dagger
                                     + G_minus_Omega_dagger)
                                  @ v_0[alpha]
                                  )
                               )
                )
        else:
            integrand_inductive = (
                1/(2*np.pi) * fermi_function
                    * np.trace(
                               d * rho @ v_1[alpha]
                               + p * 1/2 * rho
                               @ (
                                  v_0[alpha]
                                  @ (G_plus_Omega
                                     + G_minus_Omega)
                                  @ v_0[beta] 
                                  + v_0[beta]
                                  @ (G_plus_Omega_dagger
                                     + G_minus_Omega_dagger)
                                  @ v_0[alpha]
                                  )
                               )
                )
        return integrand_inductive
    def get_integrand_omega_k_ressistive(self, omega, k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, part="total"):
        r"""Returns the integrand of the response function resistive element (alpha, beta)
            for a given omega and (k_x, k_y)
        """
        v_0 = self.get_velocity_0(k_x, k_y)
        rho = self.get_spectral_density(omega, k_x, k_y, Gamma)
        G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma)
        G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma)
        G_plus_Omega_dagger = G_plus_Omega.conj().T
        G_minus_Omega_dagger = G_minus_Omega.conj().T
        fermi_function = Fermi_function(omega)
        integrand_ressistive = (
            1/(2*np.pi) * fermi_function
               * np.trace(
                          1j/2 * rho
                          @ (
                             v_0[alpha]
                             @ (G_plus_Omega
                                - G_minus_Omega)
                             @ v_0[beta]              
                             - v_0[beta]
                             @ (G_plus_Omega_dagger
                                - G_minus_Omega_dagger)
                             @ v_0[alpha]
                             )
                          )
            )
        return integrand_ressistive
    def get_response_function_quad(self, alpha, beta, L_x, L_y, Gamma, Fermi_function, Omega, part="total"):
        inductive_integrand = self.get_integrand_omega_k_inductive
        ressistive_integrand = self.get_integrand_omega_k_ressistive
        a = -45
        b = 0
        k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
        k_y_values = 2*np.pi/L_x*np.arange(0, L_y)
        K_inductive_k = np.zeros((len(k_x_values), len(k_y_values)),
                                dtype=complex)
        K_ressistive_k = np.zeros((len(k_x_values), len(k_y_values)),
                                dtype=complex)
        for i, k_x in enumerate(k_x_values):
            for j, k_y in enumerate(k_y_values):
                E_k = self.get_Energy(k_x, k_y)
                poles = list(E_k[np.where(E_k<=0)])
                params = (k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, part)
                K_inductive_k[i, j] = scipy.integrate.quad_vec(inductive_integrand, a, b, args=params, points=poles, workers=-1)[0]
                K_ressistive_k[i, j] = scipy.integrate.quad_vec(ressistive_integrand, a, b, args=params, points=poles, workers=-1)[0]
        K_inductive = 1/(L_x*L_y) * np.sum(K_inductive_k)
        K_ressistive = 1/(L_x*L_y) * np.sum(K_ressistive_k)
        return [K_inductive, K_ressistive]
