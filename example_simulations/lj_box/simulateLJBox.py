"""
@author Akash Pallath
Example OpenMM simulation of Lennard-Jones fluid in a box
Dependencies:
- OpenMM
- multiprocessing
- pyDOE
- numpy
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import numpy as np
from pyDOE import *

import multiprocessing as mp

num_threads = str(mp.cpu_count())

"""Basic parameters"""
n = 1000 #number of particles
rho_reduced  = 0.05 #reduced density
mass = 39.9 * amu #mass of argon
sigma = 3.4 * angstroms
epsilon = 0.238 * kilocalories_per_mole #argon

system = System()

#set box size
rho_actual = rho_reduced/(sigma**3)
volume = n/rho_actual
box_edge = volume ** (1/3)
a = Quantity((box_edge,     0 * angstrom, 0 * angstrom))
b = Quantity((0 * angstrom, box_edge,     0 * angstrom))
c = Quantity((0 * angstrom, 0 * angstrom, box_edge))
system.setDefaultPeriodicBoxVectors(a, b, c)

""" Specify nonbonded interactions
Reference: http://docs.openmm.org/7.4.0/userguide/theory.html#nonbondedforce"""
"""LJ cutoff and correction parameters"""
rcutoff = 3.0 * sigma #LJ-cutoff
rswitch = rcutoff - sigma;

#specify non-bonded interactions (LJ for short-range, Coulomb for long-range)
#No charges => only LJ
f = NonbondedForce()
f.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
f.setCutoffDistance(rcutoff)

#as potential is truncated at cutoff, use a switching function (applied after
#rswitch) to make the energy go smoothly to 0 at cutoff
f.setUseSwitchingFunction(True)
f.setSwitchingDistance(rswitch)

#Add dispersion correction
f.setUseDispersionCorrection(True)

#Add particles
for i in range(n):
    system.addParticle(mass)
    f.addParticle(0 * elementary_charge, sigma, epsilon)

#Generate random particle positions using Latin Hypercube Sampling
pos = Quantity(np.zeros([n, 3], np.float32), nanometers)
box_vectors = system.getDefaultPeriodicBoxVectors()
#generate samples on a latin hypercube grid (1x1x1)
x = lhs(3, samples=n)
#scale and set positions
for dim in range(3):
    l = box_vectors[dim][dim] #diagonal element = length of box
    pos[:, dim] = Quantity(x[:, dim] * l / l.unit, l.unit)

"""Add nonbonded interactions to system and prepare topology"""
system.addForce(f)
top = Topology()
element = Element.getBySymbol('Ar')
chain = top.addChain()
for i in range(system.getNumParticles()):
    residue = top.addResidue('Ar', chain)
    top.addAtom('Ar', element, residue)

#define simulation variables
temp = 298.15*kelvin
friction_coeff = 1/picosecond
timestep = 2*femtoseconds

#integrator
integrator = LangevinIntegrator(temp, friction_coeff, timestep)

#create a simulation object (similar to GROMACS executable)
platform = Platform.getPlatformByName('CPU')
properties = {'Threads': num_threads}
simulation = Simulation(top, system, integrator, platform, properties)

#initial atom positions
simulation.context.setPositions(pos)

#energy minimization
print("Begin Energy Minimization...")
simulation.minimizeEnergy()
print("Energy Minimization Done!")

# Save energy minimized positions to PDB file (for VMD init state)
print("Saving Positions")
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('eminimized.pdb', 'w'))
print('Done')

#append data every 100 steps
simulation.reporters.append(StateDataReporter("state.txt", 100, step=True, potentialEnergy=True, temperature=True))
#append structures to PDB every 500 steps
simulation.reporters.append(PDBReporter('traj.pdb', 500))
#create (rewrite) checkpoint every 5000 steps
simulation.reporters.append(CheckpointReporter('checkpnt.chk', 5000))

#simulation steps
print("Begin NVT...")
simulation.step(100000)
print("NVT Done!")
