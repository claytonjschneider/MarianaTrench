import numpy as np
import os, shutil
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm

class DepthMap:
    def __init__(
        self,
        depth_path='mariana_depth.csv',
        lon_path='mariana_longitude.csv',
        lat_path='mariana_latitude.csv'):

        self.depth = np.genfromtxt(depth_path, dtype='float', delimiter=',')
        self.lon = np.genfromtxt(lon_path, dtype='float', delimiter=',')
        self.lat = np.genfromtxt(lat_path, dtype='float', delimiter=',')

    def contour_plot(self):
        X, Y = np.meshgrid(self.lon, self.lat)
        Z = self.depth.T
        fig, ax = plt.subplots()
        CS = ax.contour(
            X, Y, Z,
            cmap='Greys_r',
            # alpha=0.5
            )

        ax.imshow(
            Z,
            alpha=0.8,
            extent=[
                min(self.lon),
                max(self.lon),
                max(self.lat),
                min(self.lat)
            ],
            cmap='YlGnBu_r')

        # plt.xlim([min(self.lon), max(self.lon)])
        # plt.ylim([min(self.lat), max(self.lat)])

        # ax.clabel(CS, inline=False, fontsize=10)
        ax.set_title('Mariana Trench Contour Map')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # plt.legend()
        cbar = plt.colorbar(CS)
        # cbar.set_label('Meters Relative to Sea Level')
        plt.show()

    def depth_plot(self):
        X, Y = np.meshgrid(self.lon, self.lat)
        Z = self.depth.T
        fig, ax = plt.subplots()

        plt.imshow(Z)

        ax.set_title('Mariana Trench Depth Plot')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.show()
