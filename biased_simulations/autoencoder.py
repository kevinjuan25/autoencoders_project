"""@author Akash Pallath
PyTorch autoencoder class (AE_module) with ability to:
- set umbrella potential bias
- save trained module as TorchScript
- restart training from saved state
- store train and test loss history
Dataset: Muller-Brown particle trajectories
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import MB_traj_dataset
import torch.nn as nn
import torch.optim as optim
import timeit

class AE(nn.Module):
    #Autoencoder with umbrella sampling force
    def __init__(self):
        super(AE, self).__init__()

        # Encoder layers
        self.enc1L = nn.Linear(3, 4)
        self.enc1A = nn.ReLU()
        self.enc2L = nn.Linear(4, 2)
        self.enc2A = nn.ReLU()
        self.enc3L = nn.Linear(2, 1)
        self.enc3A = nn.ReLU()

        # Decoder layers
        self.dec1L = nn.Linear(1, 2)
        self.dec1A = nn.ReLU()
        self.dec2L = nn.Linear(2, 4)
        self.dec2A = nn.ReLU()
        self.dec3L = nn.Linear(4, 3)
        self.dec3A = nn.ReLU()

        #Bias variables
        self.bias_around = 0 #bias around this value of latent variable
        self.bias_strength = 1 #strength of biasing umbrella

    def encoder(self, x):
        z = self.enc1L(x)
        z = self.enc1A(z)
        z = self.enc2L(z)
        z = self.enc2A(z)
        z = self.enc3L(z)
        z = self.enc3A(z)
        return z

    def decoder(self, z):
        x = self.dec1L(z)
        x = self.dec1A(x)
        x = self.dec2L(x)
        x = self.dec2A(x)
        x = self.dec3L(x)
        x = self.dec3A(x)
        return x

    def forward(self, x):
        #forward is used for umbrella sampling
        c = self.encoder(x)
        return self.bias_strength * (c - self.bias_value)**2

class AE_model:
    def __init__(self, datafile='mb_traj.dat'):
        # Hyperparameters
        self.LRATE = 3e-4

        # Dataset parameters
        self.datafile = datafile
        self.dataset = MB_traj_dataset(self.datafile)
        self.test_ratio = 0.2

        # Autoencoder architecture
        self.net = AE()

        # Loss function
        self.loss_fn = nn.MSELoss(reduction="mean")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.LRATE)

        # Epoch stopped at
        self.last_epoch = 0

        # Loss history
        self.loss_history = []
        self.test_loss_history = []

        # Training time
        self.train_time = 0

    def train(self, num_epochs = 10, batch_size = 128):
        # Train-test split
        dataset_size = len(self.dataset)
        split = int(np.floor(self.test_ratio * dataset_size))

        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices = indices[split:], indices[:split]
        self.train_size = len(train_indices)
        self.test_size = len(test_indices)

        # Data samplers and train-test dataloaders
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler)

        # Record time elapsed
        start_time = timeit.default_timer()

        loss_history = []
        test_loss_history = []

        for epoch in range(num_epochs):
            epochlosses = []
            testepochlosses = []

            for it, (features, labels) in enumerate(train_loader):
                # Reset gradients
                self.optimizer.zero_grad()
                # Forward pass
                latent = self.net.encoder(features) #encoder
                outputs = self.net.decoder(latent) #decoder
                # Compute loss
                loss = self.loss_fn(outputs, features)
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimizer.step()

                elapsed = timeit.default_timer() - start_time
                start_time = timeit.default_timer()
                self.train_time += elapsed

                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Time: %2fs'
                       %(epoch + self.last_epoch + 1, self.last_epoch + num_epochs, \
                       it+1, int(np.ceil(self.train_size/batch_size)), loss.data, elapsed))

                epochlosses.append(loss.data.numpy())

            # Average loss over the epoch
            epochloss = np.mean(np.array(epochlosses))
            loss_history.append([epoch + self.last_epoch + 1, epochloss])

            print("Epoch [{}/{}], Training loss: {:.4f}".format(\
                epoch + self.last_epoch + 1, self.last_epoch + num_epochs, epochloss))

            # Calculate epoch test loss every X epochs
            if epoch % 1 == 0:
                for it, (features, labels) in enumerate(test_loader):
                    # Forward pass
                    latent = self.net.encoder(features)
                    outputs = self.net.decoder(latent)
                    # Compute loss
                    loss = self.loss_fn(outputs, features)
                    # Update test losses
                    testepochlosses.append(loss.data.numpy())

                testepochloss = np.mean(np.array(testepochlosses))
                test_loss_history.append([epoch + self.last_epoch + 1, testepochloss])

                print("Epoch [{}/{}], Testing loss = {:.4f}".format(\
                    epoch + self.last_epoch + 1, self.last_epoch + num_epochs, testepochloss))

        self.last_epoch += num_epochs

        # Store loss history as list for plots
        self.loss_history.extend(loss_history)
        self.test_loss_history.extend(test_loss_history)

    def save_torchscript_model(self, bias_around, bias_strength, filename="umbrella.pt"):
        self.net.bias_around = bias_around
        self.net.bias_strength = bias_strength
        module = torch.jit.script(self.net)
        module.save(filename)
