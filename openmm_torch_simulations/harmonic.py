"""
@author Kevin Juan
PyTorch implementation of harmonic force
Dependencies:
- PyTorch
"""
import torch


class ForceModule(torch.nn.Module):
    def forward(self, positions):
        return torch.sum(positions ** 2)

harmonic = ForceModule()
module = torch.jit.script(ForceModule())
module.save('harmonic.pt')
