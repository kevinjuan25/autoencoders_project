"""
@author Akash Pallath
Methane (OPLS-AA, modified) in water (TIP3p or SPC/E) unbiased simulation,
reading from GROMACS input files
**Do not use TIP4P water model when using OpenMM to read GROMACS files
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import multiprocessing as mp

num_threads = str(mp.cpu_count())

includeDir = '../../GMX_TOP'
gro = GromacsGroFile('conf.gro')
top = GromacsTopFile('topol.top', periodicBoxVectors=gro.getPeriodicBoxVectors(),includeDir=includeDir)

system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,constraints=HBonds)
#(constrained H-bonds allows using a timestep higher than 1 fs)

#integrator/thermostat parameters
temp = 300*kelvin
friction = 1/picosecond
timestep = 2*femtoseconds

integrator = LangevinIntegrator(temp, friction, timestep)
simulation = Simulation(top.topology, system, integrator)
simulation.context.setPositions(gro.positions)

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
