"""
@author Kevin Juan
Alanine dipeptide in water unbiased simulation
Dependencies:
- OpenMM
- multiprocessing
- sys
- os
"""
import os
from simtk.openmm.app import*
from simtk.openmm import*
from simtk.unit import*
from sys import stdout
import multiprocessing as mp

num_threads = str(mp.cpu_count())

os.chdir('..')
pdb_file = os.path.abspath('./pdb_files/alanine_dipeptide.pdb')

pdb = PDBFile(pdb_file)
forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, model='tip3p', boxSize=Vec3(3, 3, 3) * nanometers, ionicStrength=0 * molar)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometers, constraints=AllBonds, ewaldErrorTolerance=0.0005)
integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
platform = Platform.getPlatformByName('CPU')
properties = {'Threads': num_threads}
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)

# Energy Minimization
print("Begin Energy Minimization...")
simulation.minimizeEnergy()
print("Energy Minimization Done!")

# Equilibration
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
print("Begin Equilibration...")
simulation.step(25000)
print("Equilibration Done!")

# Production Run
simulation.reporters.append(PDBReporter('./unbiased_simulations/ala_output.pdb', 1000))
print("Begin Production...")
simulation.step(400000)
print("Production Done!")
