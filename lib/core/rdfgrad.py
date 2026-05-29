import sys
sys.path.append('')

import torch

from utils.transforms import quaternion_hamilton_product

def exp_map(q, rgrad, device='cuda'):
    alpha = torch.norm(rgrad[:, :, 1:], dim=-1)
    tmp = torch.sin(alpha) / alpha
    beta = torch.ones_like(alpha).to(device)
    mask = (alpha > 0.)
    beta[mask] = tmp[mask]

    exp_v = torch.ones_like(rgrad).to(device)
    exp_v[:, :, 0] = torch.cos(alpha)
    exp_v[:, :, 1:] = torch.einsum('...d, ...->...d', rgrad[:, :, 1:], beta)

    prod = quaternion_hamilton_product(q, exp_v)

    prod = prod / torch.norm(prod, dim=-1).unsqueeze(-1)

    mask = (prod[:, :, 0] < 0)
    prod[mask] *= -1
    return prod

def egrad2rgrad(egrad, q, device='cuda', norm=True):
    bs, n_joints, _ = q.shape

    Id = torch.eye(4).to(device)
    Id = Id.expand(bs, n_joints, 4, 4)
    P = Id - torch.einsum('...ij, ...jk -> ...ik', q.unsqueeze(-1), q.unsqueeze(2)) 


    v = torch.einsum('...ij, ...jk -> ...ik', P, egrad.unsqueeze(-1)) 
    if norm:
        v = torch.nn.functional.normalize(v, dim=2)


    rmat = torch.eye(4)
    rmat = rmat.expand(bs, n_joints, 4, 4).to(device)

    rmat[:, :, 0, 0] = 0.
    rmat[:, :, 1, 0] = -q[:, :, 1] / (1 + q[:, :, 0])
    rmat[:, :, 2, 0] = -q[:, :, 2] / (1 + q[:, :, 0])
    rmat[:, :, 3, 0] = -q[:, :, 3] / (1 + q[:, :, 0])

    rgrad = torch.einsum('...ij, ...jk -> ...ik', rmat, v)
    rgrad = rgrad.squeeze(-1)

    return rgrad


def rdfgrad(egrad, q, dist, step_size, device='cuda', norm=True):

    rgrad = egrad2rgrad(egrad, q, device=device, norm=norm)
    rgrad = -step_size * dist * rgrad

    res = exp_map(q, rgrad, device=device)

    return res
