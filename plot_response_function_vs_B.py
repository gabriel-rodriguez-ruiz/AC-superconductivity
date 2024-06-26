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
file_to_open = data_folder / "Response_kernel_vs_B_mu=-40_L=100_Gamma=0.1_Omega=0.npz"
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
# L_x = Data["L_x"]
# L_y = Data["L_y"]

fig, ax = plt.subplots()
ax.plot(B_values/Delta, K[:, 0, 0]/K[0, 0, 0], "-o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
ax.plot(B_values/Delta, K[:, 1, 0]/K[0, 0, 0], "-o",  label=r"$K^{(L)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")

# ax.plot(B_values/Delta, K[:, 0, 1]/K[0, 0, 0], "-o",  label=r"$K^{(R)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)})")
# ax.plot(B_values/Delta, K[:, 1, 1]/K[0, 0, 0], "-o",  label=r"$K^{(R)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)})")


ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             + r"; $\mu=$"+f"{mu}"
             +r"; $w_0=$"+f"{w_0}"
             +r"; $\Gamma=$"+f"{Gamma}")
# ax.annotate(f"L={L_x}", (0.5, 0.75), xycoords="figure fraction")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$K(B_y,\Omega=$"+f"{Omega})")
ax.legend()
plt.tight_layout()

#%%
file_to_open = data_folder / "Response_kernel_vs_B_mu=-40_L=100_Gamma=0.1_Omega=0.05_Lambda=0.npz"
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
# L_x = Data["L_x"]
# L_y = Data["L_y"]

ax.plot(B_values/Delta, K[:, 0, 0]/K[0, 0, 0], "--o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
ax.plot(B_values/Delta, K[:, 1, 0]/K[0, 0, 0], "--o",  label=r"$K^{(L)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
# ax.plot(B_values/Delta, K[:, 0, 1]/K[0, 0, 0], "--o",  label=r"$K^{(R)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)})")
# ax.plot(B_values/Delta, K[:, 1, 1]/K[0, 0, 0], "--o",  label=r"$K^{(R)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)})")

ax.legend()
plt.tight_layout()