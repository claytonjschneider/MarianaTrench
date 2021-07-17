import numpy as np
import os, shutil
import matplotlib.pyplot as plt

from tqdm import tqdm

def mag(x):
    """
    return magnitude of a vector,
    (square root of sum of squares)
    """
    return np.sqrt(sum(i**2 for i in x))

def scale_mag_1(x):
    """
    scales a vector by its magnitude,
    returning a new vector with magnitude of ~1
    """
    return np.array([np.true_divide(ui, mag(x)) for ui in x])

def get_eigen_value(A, v):
    """
    Given matrix A and eigenvector v, returns lambda s.t.
    Av = lambda * v
    """
    Av = np.dot(A, v)
    print("Mag v, should be 1:", mag(v))
    lmb = mag(Av) / mag(v)
    return lmb

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

        # get depth as matrix A
        self.A = self.depth
        # and ATA = Atranspose * A
        self.ATA = np.dot(self.A.T, self.A)

    def plot(self, eigenvectors=False, deepest_point=False):
        X, Y = np.meshgrid(self.lon, np.flip(self.lat))
        Z = np.flip(self.depth, axis=1).T
        fig, ax = plt.subplots()

        # plot contour lines around arbitrary depth thresholds
        CS = ax.contour(
            X, Y, Z,
            cmap='Greys_r',
            # alpha=0.5
            )

        # plot as pixel-wise depth map
        IM = ax.imshow(
            Z,
            alpha=0.8,
            extent=[
                min(self.lon),
                max(self.lon),
                min(self.lat),
                max(self.lat)
            ],
            cmap='YlGnBu_r')

        if deepest_point is True:
            DP = ax.scatter(*self.deepest_point, color='r')

        ax.set_title('Mariana Trench Contour Map')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # cbar = plt.colorbar(CS)
        cbar = plt.colorbar(IM)

        plt.show()

    def deepest(self):
        deepest_point = np.unravel_index(
            np.argmin(self.depth),
            self.depth.shape)
        self.deepest_point = (
            self.lon[deepest_point[0]],
            self.lat[deepest_point[1]]
        )
        self.deepest_depth = np.min(self.depth)

        return (self.deepest_point, self.deepest_depth)

    def trench_depth(self, floor_depth=6000):
        flat_depth = self.depth.flatten()
        trench_depth = [loc for loc in flat_depth if loc < (floor_depth*-1)]
        self.trench_depth = np.mean(trench_depth)

        return(self.trench_depth)

    def get_first_eigen(self, plot=True):
        # randomly generated vector of size m
        # (length of latitudes, in this case?)
        u1 = np.random.rand(self.ATA.shape[0])
        # scale entire vector by its magnitude, to make magnitude = 1
        u1 = scale_mag_1(u1)

        diff = 1 # initialize high difference
        i = 0 # count iterations before success
        while diff > 1e-3:
            un = scale_mag_1(np.dot(self.ATA, u1))

            # diff is the magnitude of the difference between the two
            diff = mag(u1 - un)
            print("Diff:", diff)

            u1 = un
            i += 1

        self.eigen_vectors = []
        self.eigen_vectors.append(u1)

        self.eigen_values = []
        self.eigen_values.append(
            get_eigen_value(self.ATA, u1))

        print("It took", i, "iterations to resolve the first eigen vector.")

        if plot is True:
            fig, ax = plt.subplots()

            ax.plot(
                range(len(self.eigen_vectors[0])),
                self.eigen_vectors[0])

            ax.set_title("First Eigen Vector, with Eigen Value = {value:.2f}".format(value=self.eigen_values[0]))
            ax.set_xlabel("1 -> N = {N:.0f}".format(N=len(self.eigen_vectors[0])))
            ax.set_ylabel("Value of x_i")

            plt.show()

        return self.eigen_vectors[0]

    def gs(self, k=50):
        """
        Compute the k largest eigenvalues and associated eigenvectors of the
        system by using the fact that every eigenvector of a symmetric matrix
        (such as ATA!) must be orthogonal to all the previous ones, using the
        Gram-Schmidt Orthogonalization process.
        """
        # a. initialize V1 to Vk as a matrix of zeros
        Vs = np.zeros((k, self.ATA.shape[0]), dtype=float)

        # initialize u_n as first eigen vector?
        # un = self.eigen_vectors[0]

        # looking for k largest eigenvalues and associated eigenvectors
        # of ATA
        # b. for i = 1 to k
        for i in tqdm(range(len(Vs))):
            print("Doing i")

            # i. randomly generated vector of size m
            # (length of latitudes, in this case?)
            # scale entire vector by its magnitude, to make magnitude = 1
            u1 = scale_mag_1(np.random.rand(self.ATA.shape[0]))
            un = u1 # at first, u_n is u_1 and random

            diff = 1 # set initial diff too high to trip while loop
            while diff > 1e-3:

                print("Doing ii")
                # ii. u_(n+1) = A^T*A*u_n
                u1more = np.dot(self.ATA, un)

                print("Doing iii")
                # iii. u_(n+1) = u_(n+1) - Sigma_j^(i-1)(u_(n+1)^T * V_j) * V_j
                u1more = u1more - np.sum([
                    np.dot(np.dot(u1more.T, Vs[j]), Vs[j]) for j in range(i)
                ])

                print("Doing iv")
                # iv. u_(n+1) = u_(n+1) / || u_(n+1) ||
                # just norm mag
                u1more = scale_mag_1(u1more)

                diff = mag(u1more - un)
                print("Diff:", diff)

                un = u1more

            Vs[i] = un
