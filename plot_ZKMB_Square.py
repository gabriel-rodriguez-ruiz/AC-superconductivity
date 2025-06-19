#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:51:56 2025

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as colors
import matplotlib as mpl

plt.rcParams['text.usetex'] = True

data_folder = data_folder = Path("Data/")

file_name = "Probability_density_Periodic_in_Y_mu_-40_L_x=200_L_y=300_B_x=0.4_B_y=0.0_B_z=0.0.npz"

file_to_open = data_folder / file_name


data = np.load(file_to_open, allow_pickle=True)
probability_density = data["probability_density"]
L_x = data["L_x"]
L_y = data["L_y"]
index = data["index"]

file_name2 = "Probability_density_Square_mu_-40_L_x=200_L_y=1000_B_x=0.4_B_y=0.0_B_z=0.0.npz"

file_to_open2 = data_folder / file_name2


data2 = np.load(file_to_open2, allow_pickle=True)
probability_density2 = data2["probability_density"]
L_x2 = data2["L_x"]
L_y2 = data2["L_y"]
index2 = data2["index"]


index = 0

# cmap = mpl.cm.binary
# norm = mpl.colors.Normalize(vmin=np.min(probability_density[0]),
#                             vmax=np.max(probability_density[0]))


fig, axs = plt.subplots(1, 2, dpi=600)
image = axs[0].imshow(probability_density[index], cmap="binary", origin="lower",
                      norm=colors.PowerNorm(gamma=1),
                      aspect=1) #I have made the transpose and changed the origin to have xy axes as usually
# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#              cax=ax, orientation='horizontal', label='Some Units')

image2 = axs[1].imshow(probability_density2[index2], cmap="binary", origin="lower",
                      # norm=colors.LogNorm(vmin=probability_density[index].min(),
                      #                     vmax=probability_density[index].max()),
                      norm=colors.PowerNorm(gamma=1),
                  aspect=0.3) #I have made the transpose and changed the origin to have xy axes as usually
cbar = fig.colorbar(image, orientation="horizontal", ticks=[np.min(probability_density[index]),
                                                     np.max(probability_density[index])],
                    pad=-0.75,
                    shrink=0.8) #pad=0.11
# cbar.ax.set_xticklabels([np.format_float_scientific(cbar.get_ticks()[0],1),
#                          np.format_float_scientific(cbar.get_ticks()[1],1)],
#                         fontsize=8)
cbar.ax.set_xticklabels([r"$7.4\times 10^{-8}$",
                         r"$1.3\times 10^{-4}$"],
                         fontsize=6)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.tick_params(pad=-1)
cbar2 = fig.colorbar(image2,
                     orientation="horizontal",
                     ticks=[np.min(probability_density2[index]),
                            np.max(probability_density2[index])],
                     pad=-0.75,
                     shrink=0.8)
# cbar2.ax.set_xticklabels([np.format_float_scientific(cbar2.get_ticks()[0],1),
#                          np.format_float_scientific(cbar2.get_ticks()[1],1)],
#                          fontsize=8)
cbar2.ax.set_xticklabels([r"$3.5\times 10^{-9}$",
                         r"$6.1\times 10^{-5}$"],
                         fontsize=6)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.tick_params(pad=-1)

axs[0].set_xlabel(r"$l_x$", fontsize=8, labelpad=0.5)
axs[0].set_ylabel(r"$l_y$", fontsize=8, labelpad=2)
axs[0].set_xticks([0, 100, 200], [r"$0$", r"$100$", r"$200$"])
axs[0].set_yticks([0, 75, 150, 225, 300], [r"$0$", r"$75$", r"$150$", r"$225$", r"$300$"])

axs[1].set_xlabel(r"$l_x$", fontsize=8, labelpad=0.5)
axs[1].set_ylabel(r"$l_y$", fontsize=8, labelpad=0)
axs[1].set_xticks([0, 100, 200], [r"$0$", r"$100$", r"$200$"])
axs[1].set_yticks([0, 250, 500, 750, 1000], [r"$0$", r"$250$", r"$500$", r"$750$", r"$1000$"])

axs[0].tick_params(labelsize=8, pad=1)
axs[1].tick_params(labelsize=8, pad=1)

fig.set_figheight(3)
fig.set_figwidth(3+3/8)

axs[0].text(-55, 272, "(a)")
axs[1].text(-55, 905, "(b)")

axs[0].set_title(r"PBC", pad=10)
axs[1].set_title(r"OBC", pad=10)

# fig.set_size_inches(3 + 3/8, 3 + 3/8)


# plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

plt.savefig("/home/gabriel/OneDrive/Doctorado-DESKTOP-JBOMLCA/Papers propios/Antichiral/Figures/Figure_edge_states.svg",
            dpi=600,
            format="svg",
            bbox_inches="tight")
plt.show()
