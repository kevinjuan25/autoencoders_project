"""OpenMM example simulation
Edits: Akash Pallath"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import multiprocessing as mp

num_threads = str(mp.cpu_count())

#load PDB from disk
pdb = PDBFile('input.pdb')

#specify one or more xml files from which to load force field definitions
#amber14-all: AMBER14 force field
#amber14/tip3pfb: TIP3P-FB water model
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

#create system
#read topology, use particle mesh Ewald for long range electrostatics with a 1nm cutoff,
#   constrain bonds involving hydrogen (allows longer timesteps)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

#define simulation variables
temp = 300*kelvin
friction_coeff = 1/picosecond
timestep = 4*femtoseconds

#integrator
integrator = LangevinIntegrator(temp, friction_coeff, timestep)

#create a simulation object (similar to GROMACS executable)
platform = Platform.getPlatformByName('CPU')
properties = {'Threads': num_threads}
simulation = Simulation(pdb.topology, system, integrator, platform, properties)

#initial atom positions
simulation.context.setPositions(pdb.positions)

#energy minimization
print("Begin Energy Minimization...")
simulation.minimizeEnergy()
print("Energy Minimization Done!")

#append data every 100 steps
simulation.reporters.append(StateDataReporter("state.txt", 100, step=True, potentialEnergy=True, temperature=True))
#append structures to PDB every 1000 steps
simulation.reporters.append(PDBReporter('traj.pdb', 1000))
#create (rewrite) checkpoint every 5000 steps
simulation.reporters.append(CheckpointReporter('checkpnt.chk', 5000))

#simulation steps
print("Begin NVT...")
simulation.step(10000)
print("NVT Done!")
