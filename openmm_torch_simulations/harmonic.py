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

class ForceModulePBC(torch.nn.Module):
    def forwarf(self, positions, boxvectors):
        boxsize = boxvectors.diag()
        pbc_pos = positions - torch.floor(positions / boxsize) * boxsize
        return torch.sum(pbc_pos ** 2)

harmonic = ForceModule()
positions = torch.rand(3)
module_no_pbc = torch.jit.trace(harmonic, positions)
module_no_pbc.save('harmonic_no_pbc.pt')

module_pbc = torch.jit.script(ForceModulePBC())
module_pbc.save('harmonic_pbc.pt')
