"""@author Akash Pallath
PyTorch autoencoder plots code
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from autoencoder import AE, AE_model

import pickle #for loading saved model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="File containing trained autoencoder model (default: trained_convnet.pkl)")
parser.add_argument("--show", action="store_true", help="Show interactive matplotlib plots")
parser.add_argument("--lossoutfile", help="File to save output image of loss history to (default: loss_history_<num_epochs>.png")

args = parser.parse_args()
if args.file is None:
    filename = "trained_convnet.pkl"
else:
    filename = args.file

f = open(filename, 'rb')
ae = pickle.load(f)
f.close()

if args.lossoutfile is None:
    imgfile = "loss_history_{}.png".format(ae.last_epoch)
else:
    imgfile = args.lossoutfile


loss_history = np.array(ae.loss_history)
test_loss_history = np.array(ae.test_loss_history)

plt.figure(dpi=150)
ax = plt.gca()
ax.plot(loss_history[:,0], loss_history[:,1], 'x-', label="Training loss")
ax.plot(test_loss_history[:,0], test_loss_history[:,1], '+-', label="Test loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("MSE loss")
ax.legend()
ax.set_title("Losses after {} epochs [{:.1f}s]".format(ae.last_epoch, ae.train_time))
plt.savefig(imgfile)
if args.show:
    plt.show()
