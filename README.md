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
- Hard sphere alkane (TraPPE potential) in vacuum (or non-interacting SPC/E water) (References: 
[Martin and Siepmann](https://pubs.acs.org/doi/pdf/10.1021/jp972543%2B)
[Ferguson et al](https://pubs.acs.org/doi/pdf/10.1021/jp811229q`))
- Alanine dipeptide in water

## Modified GROMACS topology files (for OpenMM to read)

`GMX_TOP`

Can modify topology files in this directory and include them in OpenMM simulation scripts.
