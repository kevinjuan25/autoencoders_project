# autoencoders_project

## Example simulations

`example_simulations/`

This directory contains Python scripts for OpenMM simulations of test 2-D and 3-D systems,
demonstrating various capabilities of OpenMM.

Systems modelled:
- Protein in water
- Methane in water, configuration and topology generated using `gmx pdb2gmx` and `gmx solvate` utilities
- Box of water
- Box of gaseous Lennard-Jones particles
- 2-D Lennard-Jones particles

## Unbiased simulations:

`unbiased_simulations`

Contains Python scripts for OpenMM simulations of:
- 2-D particles on a Muller-Brown potential energy surface
- Alanine dipeptide in water

## Test simulations with openmm-torch:

`openmm_torch_test_simulations`

Contains Python scripts for test simulations using OpenMM-torch to add biasing forces on
- 2-D particles on a Muller-Brown potential energy surface
- Alanine dipeptide in water
- Methane in water

## Reference autoencoder model:

`autoencoder`

Contains code for PyTorch model to train an autoencoder on an MB particle trajectory

## Modified GROMACS topology files (for OpenMM to read)

`GMX_TOP`

Can modify topology files in this directory and include them in OpenMM simulation scripts.
