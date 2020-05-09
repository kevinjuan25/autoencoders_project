"""@author Akash Pallath
Using trained autoencoder and combined datafile:
- Generates collective variable data
- Plots collective variable histogram
- Identifies boundaries for next iteration
"""

import numpy as np
import torch

from autoencoder import AE, AE_model

import pickle #for loading model
import argparse #for passing command line arguments
