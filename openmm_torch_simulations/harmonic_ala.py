"""
@author Kevin Juan
PyTorch implementation of harmonic force
Dependencies:
- PyTorch
"""
import torch


class ForceModule(torch.nn.Module):
    def forward(self, positions, boxvectors):
        boxsize = boxvectors.diag()
        periodicPositions = positions - torch.floor(positions/boxsize)*boxsize
        return torch.sum(periodicPositions**2)

harmonic = ForceModule()
module = torch.jit.script(ForceModule())
module.save('harmonic_ala.pt')
