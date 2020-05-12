"""@author Akash Pallath
Using trained autoencoder, generate umbrella files for next round of biased CV discovery
"""

import numpy as np
import torch

from autoencoder import AE, AE_model
import pickle
import os

"""Begin user-defined"""
previous_dir = 'iter_1_biased'
previous_model = 'iter_1_convnet_500.pkl'
next_dir = 'iter_2_biased' #this directory need not exist, this script will create it
norm_mu = np.array([-0.03690885, 0.4591628, -0.00053193])
norm_std = np.array([0.08342291, 0.04366507, 0.0370001 ])
biases = [-4.135866165161132812e-01, -2.580527365207672119e-01, -1.802857816219329834e-01]
"""End"""

m = open(previous_dir + "/" + previous_model, 'rb')
ae = pickle.load(m)
m.close()

for idx, bias in enumerate(biases):
    if not os.path.exists(next_dir+"/"+str(idx)):
        os.makedirs(next_dir+"/"+str(idx))
    #bias_strength * (cv - bias_around)**2
    ae.save_torchscript_model(bias_around=np.array(bias), bias_strength=np.array(5e3), norm_mu=norm_mu, norm_std=norm_std, filename=next_dir+"/"+str(idx)+"/umbrella.pt")
