import numpy as np
import matplotlib.pyplot as plt

def plot_topography(ax, topo_data, background=True, contours=True):
    if background:
        ax.pcolormesh(
            topo_data.easting, topo_data.northing, topo_data.values.T, rasterized=True,
            cmap='Greys', zorder=-20, alpha=0.8)

    if contours:
        ax.pc = ax.contour(
            topo_data.easting, topo_data.northing, topo_data.values.T, 
            colors='k', zorder=-10,
            levels=np.arange(0, 3000, 100),
            linewidths=0.5, alpha=0.5)

        ax.cl = ax.clabel(
            ax.pc,
            levels=np.arange(0, 3000, 100),
            inline=True, fontsize=6, fmt='%1.0f', colors='k', use_clabeltext=True)

