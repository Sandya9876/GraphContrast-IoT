import torch
import torch.nn.functional as F

def info_nce(z1, z2, temperature=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    positives = torch.cat([torch.arange(B,2*B), torch.arange(0,B)]).to(z.device)
    return F.cross_entropy(sim, positives)
