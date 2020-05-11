"""@author Akash Pallath
Partially adapted from: https://gist.github.com/rmcgibbo/6094172
Using trained autoencoder and combined datafile:
- Generates collective variable data, reconstructs trajectory
- Plots trajectory and reconstructed trajectory together
- Plots collective variable histogram
- Identifies boundaries for next iteration
"""

import numpy as np
import torch

from autoencoder import AE, AE_model

import pickle #for loading saved model
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""Muller-Brown potential class"""
class MullerBrown():
    def __init__(self, imgprefix=""):
        self.a = [-1, -1, -6.5, 0.7]
        self.b = [0, 0, 11, 0.6]
        self.c = [-10, -10, -6.5, 0.7]
        self.A = [-200, -100, -170, 15]
        self.x_bar = [1, 0, -0.5, -1]
        self.y_bar = [0, 0.5, 1.5, 1]
        self.imgprefix = imgprefix

    def potential(self, x, y):
        """Compute the potential at a given point x,y"""
        value = 0
        for i in range(4):
            value += self.A[i] * np.exp(self.a[i] * (x - self.x_bar[i])**2 + \
                self.b[i] * (x - self.x_bar[i]) * (y - self.y_bar[i]) + self.c[i] * (y - self.y_bar[i])**2)
        return value

    def plot(self, ax=None, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, trj=[], reconst_trj=[] , biases=np.array([])):
        """Plot the Muller potential"""
        grid_width = max(maxx - minx, maxy - miny) / 200.0
        xx, yy = np.mgrid[minx:maxx:grid_width, miny:maxy:grid_width]
        V = self.potential(xx, yy)
        plt.clf()
        fig, ax = plt.subplots()
        lines = []
        line1, = ax.plot(trj[:, 0], trj[:, 1], color='k')
        lines.append(line1)
        line2, = ax.plot(reconst_trj[:, 0], reconst_trj[:, 1], color='r')
        lines.append(line2)

        def update(num, x, y, rx, ry, lines):
            lines[0].set_data(x[:num], y[:num])
            lines[1].set_data(rx[:num], ry[:num])
            return lines

        ani = animation.FuncAnimation(fig, update, len(trj), fargs=[trj[:, 0], trj[:, 1], \
            reconst_trj[:, 0], reconst_trj[:, 1], lines], interval=10, blit=True)

        plt.contourf(xx, yy, V.clip(max=200), 40, cmap='jet')
        plt.plot(biases[:,0], biases[:,1], 'bo')
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        ani.save(self.imgprefix+'_mb_traj.mp4', writer='ffmpeg', bitrate=10000, dpi=300)

parser = argparse.ArgumentParser()
parser.add_argument("datafile", help="File containing trajectory/merged trajectories")
parser.add_argument("model", help="File containing trained autoencoder model")
parser.add_argument("--show", action="store_true", help="Show interactive matplotlib plots")
parser.add_argument("--dataprefix", help="Output prefix for boundary data")
parser.add_argument("--imgprefix", help="Output prefix for images")

args = parser.parse_args()

m = open(args.model, 'rb')
ae = pickle.load(m)
m.close()

if args.imgprefix is None:
    imgprefix = "img{}".format(ae.last_epoch)
else:
    imgprefix = args.imgprefix

if args.dataprefix is None:
    dataprefix = "data{}".format(ae.last_epoch)
else:
    dataprefix = args.dataprefix

"""Read and normalize trajectories"""

trajs = []
f = open(args.datafile, 'r')
for l in f:
    if l[0]!='#':
        x = [float(i) for i in l.strip().split()]
        x = np.array(x)
        trajs.append(x)
trajs = np.array(trajs)
mu = np.mean(trajs, axis=0)
std = np.std(trajs, axis=0)
trajs = (trajs - mu)/std

trajs = torch.from_numpy(trajs).type(torch.FloatTensor)

"""Get CV and reconstructed trajectories"""

# Get CV
cv = ae.net.encoder(trajs)
# Get reconstructed trajectories
reconst_trajs = ae.net.decoder(cv)

trajs = trajs.data.numpy() * std + mu
reconst_trajs = reconst_trajs.data.numpy() * std + mu

"""CV histogram"""
cv = cv.data.numpy()

plt.figure()
plt.hist(cv, bins=20)
plt.savefig(imgprefix+'_hist.png')

n, x_edges = np.histogram(cv, bins=20)
x = np.zeros(len(x_edges) - 1)
for i in range(len(x_edges) - 1):
    x[i] = (x_edges[i] + x_edges[i+1])/2
p = -np.exp(-n)
plt.figure()
plt.plot(x, p, 'x-')
plt.savefig(imgprefix+'_p.png')

v = np.zeros(len(p))
for i in range(len(p)):
    if i == 0:
        v[i] = p[i] - p[i+1]
    elif i == len(p) - 1:
        v[i] = p[i] - p[i-1]
    else:
        v[i] = p[i] - 0.5*(p[i+1] + p[i-1])
plt.figure()
plt.plot(x, v, 'x-')
plt.savefig(imgprefix+'_v.png')

#Save v[i]
xv = np.transpose(np.vstack((x, v)))
np.savetxt(dataprefix+'_v.dat', xv, header='cv      v')

"""Select CV bias points based on formula in Ferguson and Chen"""
bias_pts = [0.15, -0.15]

"""Save animations of true and reconstructed trajectories with bias points"""
biases = np.array(bias_pts).reshape(-1, 1)
biases = torch.from_numpy(biases).type(torch.FloatTensor)
bias_orig = ae.net.decoder(biases).data.numpy()

mb = MullerBrown(imgprefix)
mb.plot(trj=trajs, reconst_trj = reconst_trajs, biases = bias_orig * std + mu)
