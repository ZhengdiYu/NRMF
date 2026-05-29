import torch
from acceleration_prior import dev
from acceleration_sampling import anneal_combine


def bare_om(net, om_std, tv_std, om, tv, clean_q, cmask, legi, W, acc, accsc):
    om_n = (om[:W].unsqueeze(0).to(dev) / om_std)
    tv_n = (tv[:W].unsqueeze(0).to(dev) / tv_std)
    torch.manual_seed(0)
    xo = om_n.clone()
    xo[:, :, legi] = torch.randn_like(xo[:, :, legi])
    rec, _ = anneal_combine(net, acc, accsc, xo.clone(), tv_n.clone(), cmask, 0.3)
    return rec
