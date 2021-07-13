import numpy as np
import os, shutil
import matplotlib.pyplot as plt
# import argparse
import math

# from tqdm import tqdm

def mag(x):
    return math.sqrt(sum(i**2 for i in x))

def scale_mag_1(x):
    return np.array([np.true_divide(ui, mag(x)) for ui in x])

class DepthMap:
    def __init__(
        self,
        depth_path='mariana_depth.csv',
        lon_path='mariana_longitude.csv',
        lat_path='mariana_latitude.csv'):

        self.depth = np.genfromtxt(depth_path, dtype='float', delimiter=',')
        # self.depth = np.flip(self.depth, axis=1)
        self.lon = np.genfromtxt(lon_path, dtype='float', delimiter=',')
        self.lat = np.genfromtxt(lat_path, dtype='float', delimiter=',')

        # get A
        # and A_T * A
        self.A = self.depth
        self.ATA = np.dot(self.A.T,self.A)

    def contour_plot(self, eigenvectors=False):
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

        if eigenvectors is True:
            origin = [0, 0]

            plt.quiver(*origin, self.u, color=['r'])

        ax.set_title('Mariana Trench Contour Map')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        cbar = plt.colorbar(CS)

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

    def deepest_point(self):
        self.deep = np.min(self.depth)

        return self.deep

    def trench_average(self, threshold=-6000):
        flat_depth = self.depth.flatten()
        trench_depth = [loc for loc in flat_depth if loc < threshold]
        self.trench_depth = np.mean(trench_depth)

        return(self.trench_depth)

    def first_eigenvector(self):
        # randomly generated vector of size 1440
        self.u = np.random.rand(self.ATA.shape[0])
        # scale entire vector by its magnitude, to make magnitude = 1
        self.u = scale_mag_1(self.u)

        small_diff = False

        #
        i = 0
        while not small_diff:
            u_ = scale_mag_1(np.dot(self.ATA, self.u))

            diff = mag(self.u - u_)
            print("Diff:", diff)

            self.u = u_

            if diff < 1e-3:
                small_diff = True

            # if i == 10:
            #     break
            i += 1

        self.u = u_
        print("It took", i, "iterations to find the first eigen vector.")
        return self.u
