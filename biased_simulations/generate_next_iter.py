"""@author Akash Pallath
Using trained autoencoder, generate umbrella files for next round of biased CV discovery
"""

import numpy as np
import torch

from autoencoder import AE, AE_model
import pickle
import os

"""Begin user-defined - for CV round 0 -> 1"""
"""
previous_dir = 'iter_0_unbiased'
previous_model = 'iter_0_convnet_5000.pkl'
next_dir = 'iter_1_biased' #this directory need not exist, this script will create it
norm_mu = np.array([-0.03690885, 0.4591628, -0.00053193])
norm_std = np.array([0.08342291, 0.04366507, 0.0370001 ])
biases = [7.051497459411621094e+00, 8.423364639282226562e+00, 1.116709995269775391e+01, 1.253896713256835938e+01]
bias_strength = 1
"""
"""End"""

"""Begin user-defined - for CV round 1 -> 2"""
previous_dir = 'iter_1_biased'
previous_model = 'iter_1_convnet_2000.pkl'
next_dir = 'iter_2_biased' #this directory need not exist, this script will create it
norm_mu = np.array([0.09876496, 0.3183415,  0.08421195])
norm_std = np.array([0.1627713,  0.11958472, 0.15355152])
biases = [9.813648223876953125e+01, 1.171137390136718750e+02, 1.455796203613281250e+02, 1.645568847656250000e+02]
bias_strength = 10
"""End"""

m = open(previous_dir + "/" + previous_model, 'rb')
ae = pickle.load(m)
m.close()

for idx, bias in enumerate(biases):
    if not os.path.exists(next_dir+"/"+str(idx)):
        os.makedirs(next_dir+"/"+str(idx))
    #bias_strength * (cv - bias_around)**2
    ae.save_torchscript_model(bias_around=np.array(bias), bias_strength=np.array(bias_strength), norm_mu=norm_mu, norm_std=norm_std, filename=next_dir+"/"+str(idx)+"/umbrella.pt")
