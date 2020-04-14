"""
@author Akash Pallath
Example OpenMM simulation of water box
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import numpy as np
import multiprocessing as mp

num_threads = str(mp.cpu_count())

#specify one or more xml files from which to load force field definitions
forcefield = ForceField('tip4pew.xml')

#create empty topology
emptytop = Topology()
#create empty coordinates
emptypos = Quantity((), angstrom)

#create Modeller
modeller = Modeller(emptytop, emptypos)

#Add solvent
modeller.addSolvent(forcefield, boxSize=Vec3(2.5, 2.5, 2.5)*nanometers, model='tip4pew')

#Create system
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

#define simulation variables
temp = 298.15*kelvin
friction_coeff = 1/picosecond
timestep = 2*femtoseconds

#integrator
integrator = LangevinIntegrator(temp, friction_coeff, timestep)

#create a simulation object (similar to GROMACS executable)
platform = Platform.getPlatformByName('CPU')
properties = {'Threads': num_threads}
simulation = Simulation(modeller.topology, system, integrator, platform, properties)

#initial atom positions
simulation.context.setPositions(modeller.positions)

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
simulation.step(10000)
print("NVT Done!")

#add pressure coupling (isotropic)
press = 1*bar
system.addForce(MonteCarloBarostat(press, temp))

#simulation steps
print("Begin NPT...")
simulation.step(10000)
print("NPT Done!")
