#!/bin/bash
export GMXLIB=~/Documents/autoencoders_project/GMX_TOP

#Get GROMACS input files from PDB
#Select OPLS-AA force field from GMXLIB directory and TIP3P water when prompted
gmx pdb2gmx -f methane.pdb

#solvate (SPC, SPC/E, TIP3P models are in spc216.gro)
gmx solvate -cp conf.gro -o conf.gro -cs spc216 -p topol.top -box 2.3 2.3 2.3
