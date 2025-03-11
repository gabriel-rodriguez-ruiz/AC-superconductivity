#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:22:22 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plt.rcParams.update({
    "text.usetex": True})

data_folder = Path("Data/")
file_to_open = data_folder / "Response_kernel_vs_B_with_field_dissorder_mu=-40_L=100_Gamma_0=0.01_Gamma_1=0.3_Omega=0_Lambda=0.56_B_in_(0-0.3)_Delta=0.2.npz"
Data = np.load(file_to_open)

K = Data["K"]
B_values = Data["B_values"]
Lambda = Data["Lambda"]
Delta = Data["Delta"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
part = Data["part"]
Omega = Data["Omega"]
# Gamma = Data["Gamma"]
Gamma_0 = Data["Gamma_0"]

if "U" in list(Data.keys()):
    U = Data["U"]
else:
    U = None
# L_x = Data["L_x"]
# L_y = Data["L_y"]

fig, ax = plt.subplots()
ax.plot(B_values/Delta, K[:, 0, 0], "-o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
ax.plot(B_values/Delta, K[:, 1, 0], "-o",  label=r"$K^{(L)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
# ax.plot(B_values/Delta, 1.25 - 0.25*(B_values/Delta)**2, "--")
# ax.plot(B_values/Delta, 1.25 - 0.55*(B_values/Delta)**2, "--")
# ax.plot(B_values/Delta, 0.154 - 0.025*(B_values/Delta)**2, "--")
# ax.plot(B_values/Delta, 0.154 - 0.04*(B_values/Delta)**2, "--")

# ax.plot(B_values/Delta, np.sqrt(1/(0+1/K[:, 0, 0])), "-o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
# ax.plot(B_values/Delta, np.sqrt(1/(0+1/K[:, 1, 0])), "-o",  label=r"$K^{(L)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
# ax.plot(B_values/Delta, 0.246 - 0.01*(B_values/Delta)**2, "--")
# ax.plot(B_values/Delta, 0.246 - 0.015*(B_values/Delta)**2, "--")

# ax.set_xlim((0, 1))

# f_kin_perp = K[:, 0, 0]/K[0, 0, 0]
# f_kin_par = K[:, 1, 0]/K[0, 0, 0]
# f_geom = 1
# Delta_f_perp = f_geom*f_kin_perp/np.sqrt(f_kin_perp**2 + f_geom**2)
# Delta_f_par = f_geom*f_kin_par/np.sqrt(f_kin_par**2 + f_geom**2)

# ax.plot(B_values/Delta, Delta_f_perp/Delta_f_perp[0], "-o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
# ax.plot(B_values/Delta, Delta_f_par/Delta_f_perp[0], "-o",  label=r"$K^{(L)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")

# f_geom = 1
# Delta_f_perp = f_kin_perp/np.sqrt(f_kin_perp**2 + f_geom**2)
# Delta_f_par = f_kin_par/np.sqrt(f_kin_par**2 + f_geom**2)

# ax.plot(B_values/Delta, Delta_f_perp/Delta_f_perp[0], "-o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
# ax.plot(B_values/Delta, Delta_f_par/Delta_f_perp[0], "-o",  label=r"$K^{(L)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")


# ax.plot(B_values/Delta, K[:, 0, 1]/K[0, 0, 0], "-o",  label=r"$K^{(R)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)})")
# ax.plot(B_values/Delta, K[:, 1, 1]/K[0, 0, 0], "-o",  label=r"$K^{(R)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)})")


ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             + r"; $\mu=$"+f"{mu}"
             +r"; $w_0=$"+f"{w_0}"
             +r"; $\Gamma_0=$"+f"{Gamma_0}")
# ax.annotate(f"L={L_x}", (0.5, 0.75), xycoords="figure fraction")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$K(B_y,\Omega=$"+f"{Omega})")
ax.legend()
plt.tight_layout()
plt.show()

#%%
file_to_open = data_folder / "Response_kernel_vs_B_with_field_dissorder_mu=-40_L=100_Gamma_0=0.01_Gamma_1=0.3_Omega=0_Lambda=0.56_B_in_(0-0.3)_Delta=0.2.npz"
Data = np.load(file_to_open)

K = Data["K"]
B_values = Data["B_values"]
Lambda = Data["Lambda"]
Delta = Data["Delta"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
part = Data["part"]
Omega = Data["Omega"]
Gamma = Data["Gamma"]
if "U" in list(Data.keys()):
    U = Data["U"]
# L_x = Data["L_x"]
# L_y = Data["L_y"]

ax.plot(B_values**2/Delta, np.sqrt(K[:, 0, 0]/K[0, 0, 0]), "--o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda}"+r", $U=$"+f"{U})")
ax.plot(B_values**2/Delta, np.sqrt(K[:, 1, 0]/K[0, 1, 0]), "--o",  label=r"$K^{(L)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda}"+r", $U=$"+f"{U})")
# ax.plot(B_values/Delta, K[:, 0, 1]/K[0, 0, 0], "--o",  label=r"$K^{(R)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)})")
# ax.plot(B_values/Delta, K[:, 1, 1]/K[0, 0, 0], "--o",  label=r"$K^{(R)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)})")

ax.legend()
plt.tight_layout()