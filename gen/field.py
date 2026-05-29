''















import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gen.dataset import accel_from_vel


def sigma_embedding(sigma, dim):
    ''
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=sigma.device, dtype=torch.float32) / max(half - 1, 1))
    args = torch.log(sigma).clamp(min=-20.0).unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class WindowDistanceField(nn.Module):
    def __init__(self, n_joints=22, d_model=256, n_layers=4, n_heads=8, ff=1024, max_len=64, dropout=0.0,
                 cond_mask=False, use_accel=False, acc_scale=1.0, per_joint_sigma=False, root_head=False,
                 hybrid_head=False, accel_manifold=False, acc_lambda=1.0):
        super().__init__()
        self.n_joints = n_joints
        self.cond_mask = cond_mask
        self.use_accel = use_accel
        hybrid_head = hybrid_head or accel_manifold
        self.hybrid_head = hybrid_head
        self.accel_manifold = accel_manifold
        self.acc_lambda = acc_lambda
        pjs = per_joint_sigma or hybrid_head
        self.per_joint_sigma = pjs
        self.root_head = root_head and per_joint_sigma and not hybrid_head
        self.register_buffer('acc_scale', torch.tensor(float(acc_scale)))                 
        self.register_buffer('leg_idx', torch.tensor([1, 2, 4, 5, 7, 8, 10, 11]), persistent=False)         
        self.register_buffer('nonleg_idx', torch.tensor([0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]), persistent=False)  


        self.in_dim = n_joints * 3 + 3 + (use_accel * n_joints * 3) + (cond_mask * (n_joints + 1)) \
            + (pjs * (n_joints + 1))
        self.d_model = d_model
        self.inp = nn.Linear(self.in_dim, d_model)
        self.sigma_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout=dropout,
                                           batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        if accel_manifold:
            self.head = nn.Linear(d_model, 2 * (8 + 1))
        elif hybrid_head:
            self.head = nn.Linear(d_model, 8 + 1)
        else:
            self.head = nn.Linear(d_model, n_joints + 1 if pjs else 1)
        if self.root_head:
            self.root_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))

    def distance(self, omega, transv, sigma, mask=None):
        ''


        B, N = omega.shape[:2]
        feats = [omega.reshape(B, N, -1), transv]
        if self.use_accel:
            feats.append((accel_from_vel(omega) / self.acc_scale).reshape(B, N, -1))
        if self.cond_mask:
            if mask is None:
                mask = torch.zeros(B, N, self.n_joints + 1, device=omega.device, dtype=omega.dtype)
            feats.append(mask)
        if self.per_joint_sigma:
            sj = sigma if sigma.dim() == 2 else sigma.view(B, 1).expand(B, self.n_joints + 1)
            feats.append(torch.log(sj).clamp(min=-20.0).detach().unsqueeze(1).expand(B, N, self.n_joints + 1))
            sig_emb = torch.exp(torch.log(sj).mean(-1))
        else:
            sig_emb = sigma
        x = torch.cat(feats, dim=-1)
        h = self.inp(x) + self.pos[:, :N]
        h = h + self.sigma_mlp(sigma_embedding(sig_emb, self.d_model)).unsqueeze(1)
        h = self.encoder(h)
        d = F.softplus(self.head(h))
        if self.hybrid_head:
            return d
        if self.root_head:
            d = torch.cat([F.softplus(self.root_mlp(h)), d[..., 1:]], dim=-1)
        return d if self.per_joint_sigma else d.squeeze(-1)

    def _hybrid_energy(self, d9, sj):
        ''
        d_leg = d9[..., :8]; d_coh = d9[..., 8]
        s_leg = sj[:, self.leg_idx]
        nl = torch.cat([self.nonleg_idx, torch.tensor([self.n_joints], device=sj.device)])
        s_coh = torch.exp(torch.log(sj[:, nl]).mean(-1))
        E_leg = (d_leg ** 2 / (2.0 * s_leg.unsqueeze(1) ** 2)).sum(dim=(1, 2))
        E_coh = (d_coh ** 2 / (2.0 * s_coh.unsqueeze(1) ** 2)).sum(dim=1)
        return E_leg + E_coh

    def energy(self, omega, transv, sigma, mask=None):
        ''
        d = self.distance(omega, transv, sigma, mask)
        if self.accel_manifold:
            sj = sigma if sigma.dim() == 2 else sigma.view(-1, 1).expand(d.shape[0], self.n_joints + 1)
            return self._hybrid_energy(d[..., :9], sj) + self.acc_lambda * self._hybrid_energy(d[..., 9:], sj)
        if self.hybrid_head:
            sj = sigma if sigma.dim() == 2 else sigma.view(-1, 1).expand(d.shape[0], self.n_joints + 1)
            return self._hybrid_energy(d, sj)
        if self.per_joint_sigma:
            sj = sigma if sigma.dim() == 2 else sigma.view(-1, 1).expand(d.shape[0], self.n_joints + 1)
            return (d ** 2 / (2.0 * sj.unsqueeze(1) ** 2)).sum(dim=(1, 2))
        return (d ** 2 / (2.0 * sigma.view(-1, 1) ** 2)).sum(dim=-1)

    def energy_vel(self, omega, transv, sigma, mask=None):
        ''
        if not self.accel_manifold:
            return self.energy(omega, transv, sigma, mask)
        d = self.distance(omega, transv, sigma, mask)
        sj = sigma if sigma.dim() == 2 else sigma.view(-1, 1).expand(d.shape[0], self.n_joints + 1)
        return self._hybrid_energy(d[..., :9], sj)

    def score(self, omega, transv, sigma, mask=None):
        ''

        omega = omega.detach().requires_grad_(True)
        transv = transv.detach().requires_grad_(True)
        E = self.energy(omega, transv, sigma, mask).sum()
        g_om, g_tv = torch.autograd.grad(E, [omega, transv], create_graph=self.training)
        return -g_om, -g_tv

    def score_vel(self, omega, transv, sigma, mask=None):
        ''

        omega = omega.detach().requires_grad_(True)
        transv = transv.detach().requires_grad_(True)
        E = self.energy_vel(omega, transv, sigma, mask).sum()
        g_om, g_tv = torch.autograd.grad(E, [omega, transv], create_graph=self.training)
        return -g_om, -g_tv


class PoseField(nn.Module):
    ''


    def __init__(self, n_joints=21, d_model=512, n_layers=5, dropout=0.0):
        super().__init__()
        self.n_joints = n_joints; self.in_dim = n_joints * 3; self.d_model = d_model
        self.inp = nn.Linear(self.in_dim, d_model)
        self.sigma_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                                   nn.Dropout(dropout), nn.Linear(d_model, d_model)) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 1)

    def distance(self, pose, sigma):
        ''
        B = pose.shape[0]
        h = self.inp(pose.reshape(B, -1)) + self.sigma_mlp(sigma_embedding(sigma, self.d_model))
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))
        return F.softplus(self.head(h)).squeeze(-1)

    def energy(self, pose, sigma):
        d = self.distance(pose, sigma)
        return (d ** 2 / (2.0 * sigma ** 2)).sum()

    def score(self, pose, sigma):
        pose = pose.detach().requires_grad_(True)
        E = self.energy(pose, sigma)
        return -torch.autograd.grad(E, pose, create_graph=self.training)[0]
