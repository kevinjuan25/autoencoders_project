"""@author Akash Pallath
Using trained autoencoder, generate umbrella files for next round of biased CV discovery
"""

import numpy as np
import torch

from autoencoder import AE, AE_model
import pickle
import os

"""Begin user-defined"""
previous_dir = 'iter_0_unbiased'
previous_model = 'iter_0_convnet_5000.pkl'
next_dir = 'iter_1_biased' #this directory need not exist, this script will create it
biases = [0.15, -0.15]
"""End"""

m = open(previous_dir + "/" + previous_model, 'rb')
ae = pickle.load(m)
m.close()

for idx, bias in enumerate(biases):
    if not os.path.exists(next_dir+"/"+str(idx)):
        os.makedirs(next_dir+"/"+str(idx))
    ae.save_torchscript_model(bias_around=bias, bias_strength=50, filename=next_dir+"/"+str(idx)+"/umbrella.pt")
