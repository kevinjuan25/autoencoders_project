"""@author Akash Pallath
PyTorch MB trajectory data loader
"""

import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MB_traj_dataset(Dataset):
    """Muller-Brown PES particle trajectory dataset"""
    def __init__(self, file_location):
        self.location = file_location
        f = open(self.location)
        self.traj = []
        for l in f:
            if l[0]!='#':
                self.traj.append([float(i) for i in l.strip().split()])
        self.traj = np.array(self.traj)
        #don't normalize data - gives z-dimension data equal importance which it
        #does not deserve. Model doesn't need to learn variation in z-direction

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame = self.traj[idx]
        label = np.array(0)

        return torch.from_numpy(frame).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)

# Test
if __name__ == '__main__':
    test_dataset = MB_traj_dataset('mb_traj.dat')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    for idx, (frames, _) in enumerate(test_loader):
        print(frames)
        print(frames.shape)
        break
