"""
@author Kevin Juan
2D simulation of particle in Muller-Brown Potential
Partially adapted from: https://gist.github.com/rmcgibbo/6094172
Dependencies:
- OpenMM
- multiprocessing
- sys
- os
"""

from simtk.openmm.app import*
from simtk.openmm import*
from simtk.unit import*
from sys import stdout
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_threads = str(mp.cpu_count())


class MullerBrown(CustomExternalForce):
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    A = [-200, -100, -170, 15]
    x_bar = [1, 0, -0.5, -1]
    y_bar = [0, 0.5, 1.5, 1]

    def __init__(self):
        force = '1000 * z^2'
        for i in range(4):
            fmt = dict(a = self.a[i], b = self.b[i], c = self.c[i], A = self.A[i], x_bar = self.x_bar[i], y_bar = self.y_bar[i])
            force += '''+ {A} * exp({a} * (x - {x_bar})^2 + {b} * (x - {x_bar}) * (y - {y_bar}) + {c} * (y - {y_bar})^2)'''.format(**fmt)
        super(MullerBrown, self).__init__(force)

    @classmethod
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        value = 0
        for i in range(4):
            value += cls.A[i] * np.exp(cls.a[i] * (x - cls.x_bar[i])**2 + \
                cls.b[i] * (x - cls.x_bar[i]) * (y - cls.y_bar[i]) + cls.c[i] * (y - cls.y_bar[i])**2)
        return value

    @classmethod
    def plot(cls, ax=None, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, trj=[]):
        "Plot the Muller potential"
        grid_width = max(maxx - minx, maxy - miny) / 200.0
        xx, yy = np.mgrid[minx:maxx:grid_width, miny:maxy:grid_width]
        V = cls.potential(xx, yy)
        plt.clf()
        fig, ax = plt.subplots()
        line, = ax.plot(trj[:, 0], trj[:, 1], color='k')

        def update(num, x, y, line):
            line.set_data(x[:num], y[:num])
            return line,
        ani = animation.FuncAnimation(fig, update, len(trj), fargs=[trj[:, 0], trj[:, 1], line], interval=10, blit=True)
        plt.contourf(xx, yy, V.clip(max=200), 40, cmap='jet')
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        ani.save('mb_traj.mp4', writer='ffmpeg', bitrate=10000, dpi=300)


# Simulation Parameters
n = 1
mass = 1 * dalton
temperature = 300 * kelvin
friction = 100 / picosecond
timestep = 10.0 * femtosecond

# Initial Position
# init_coord = np.array([[1.0, 1.75, 0]])
init_coord = (np.random.rand(n, 3) * np.array([2.7, 1.8, 1])) + np.array([-1.5, -0.2, 0])

# Main Simulation
system = System()
mullerbrown = MullerBrown()
for i in range(n):
    system.addParticle(mass)
    mullerbrown.addParticle(i, [])
system.addForce(mullerbrown)
integrator = LangevinIntegrator(temperature, friction, timestep)
platform = Platform.getPlatformByName('CPU')
properties = {'Threads': num_threads}
context = Context(system, integrator, platform, properties)
context.setPositions(init_coord)
context.setVelocitiesToTemperature(temperature)
xy = init_coord

for i in range(1000):
    pos = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
    if i != 0:
        xy = np.vstack((xy, pos))
    PE = context.getState(getEnergy=True).getPotentialEnergy()
    KE = context.getState(getEnergy=True).getKineticEnergy()
    TE = KE + PE
    print("Step: {} | Energy: {}".format(i, TE))
    integrator.step(1)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
MullerBrown.plot(trj=xy)
plt.show()
