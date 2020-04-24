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
from openmmtorch import*
import multiprocessing as mp

num_threads = str(mp.cpu_count())

os.chdir('..')
pdb_file = os.path.abspath('./pdb_files/alanine_dipeptide.pdb')

pdb = PDBFile(pdb_file)
forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, model='tip3p', boxSize=Vec3(3, 3, 3) * nanometers, ionicStrength=0 * molar)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometers, constraints=AllBonds, ewaldErrorTolerance=0.0005)
harmonic = TorchForce('.openmm_torch_simulations/harmonic_pbc.pt')
system.addForce(harmonic)
integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)
platform = Platform.getPlatformByName('CPU')
properties = {'Threads': num_threads}
simulation = Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)

# Energy Minimization
print("Begin Energy Minimization...")
simulation.minimizeEnergy()
print("Energy Minimization Done!")

# Save energy minimized positions to PDB file (for VMD init state)
print("Saving Positions")
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('ala_minimized.pdb', 'w'))
print('Done')

# Add reporters in the beginning, later, discard equilibration region
simulation.reporters.append(StateDataReporter("state.txt", 1000, step=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(PDBReporter('./unbiased_simulations/ala_output.pdb', 1000))

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
