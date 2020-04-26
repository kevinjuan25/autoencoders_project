"""
@author Kevin Juan
Alanine dipeptide in water unbiased simulation
Dependencies:
- OpenMM
- Openmm-Torch
- multiprocessing
- sys
- os
"""
import os
from simtk.openmm.app import*
from simtk.openmm import*
from simtk.unit import*
from openmmtorch import*
from sys import stdout
import multiprocessing as mp

num_threads = str(mp.cpu_count())

os.chdir('..')
pdb_file = os.path.abspath('./pdb_files/alanine_dipeptide.pdb')

pdb = PDBFile(pdb_file)
os.chdir('./openmm_torch_simulations/')

# Define force field
forcefield = ForceField('amber99sb.xml', 'tip3p.xml')

# Modify model to add solvent
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, model='tip3p', boxSize=Vec3(3, 3, 3) * nanometers, ionicStrength=0 * molar)

# Define simulation
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometers, constraints=AllBonds, ewaldErrorTolerance=0.0005)
harmonic = TorchForce('harmonic_pbc.pt')
harmonic.setUsesPeriodicBoundaryConditions(True)
system.addForce(harmonic)

integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)
platform = Platform.getPlatformByName('CPU')
properties = {'Threads': num_threads}
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)

# Output to console
simulation.reporters.append(StateDataReporter(stdout, 1, step=True,
        potentialEnergy=True, temperature=True))

# Energy Minimization
print("Begin Energy Minimization...")
simulation.minimizeEnergy(maxIterations=1000)
print("Energy Minimization Done!")

# Save energy minimized positions to PDB file (for VMD init state)
print("Saving Positions")
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('./ala_minimized.pdb', 'w'))
print("Done")

# Add reporters in the beginning, later, discard equilibration region
simulation.reporters.append(StateDataReporter('./state.txt', 1, step=True, potentialEnergy=True, temperature=True, volume=True))
simulation.reporters.append(PDBReporter('./ala_output.pdb', 1))

# Equilibration
print("Begin Equilibration...")
simulation.step(25000)
print("Equilibration Done!")
# Save checkpoint
simulation.saveCheckpoint('ala_equil.chk')

# Create (rewrite) checkpoint every 10000 steps during production
simulation.reporters.append(CheckpointReporter('checkpnt.chk', 5000))

# Production
print("Begin Production...")
simulation.step(40000)
print("Production Done!")
