#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:22:22 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

data_folder = Path("Data/")
file_to_open = data_folder / "Response_kernel_vs_B_mu=-40_L=100.npz"
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

fig, ax = plt.subplots()
ax.plot(B_values/Delta, K[:, 0], "-o",  label=r"$K^{(L)}_{xx}$")
ax.plot(B_values/Delta, K[:, 1], "-o",  label=r"$K^{(L)}_{yy}$")
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$K_{xx}(\Omega=0)$")
ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             + r"; $\mu=$"+f"{mu}"
             +r"; $w_0=$"+f"{w_0}"
             +r"; $\Omega=$"+f"{Omega}")
ax.annotate(f"{part}", (0.5, 0.75), xycoords="figure fraction")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$K(B_y)$")
ax.legend()
plt.tight_layout()
plt.legend()