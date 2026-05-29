''








import sys, math; sys.path.insert(0, ''); sys.path.insert(0, 'motionlib')
import numpy as np, torch
from gen.dataset import process_npz, accel_from_vel
from gen.sample import load
from utils.transforms import axis_angle_to_quaternion as aa2q
from kinematics import fk_joints, FEET, LEG
from rendering import integrate
from humor.body_model.body_model import BodyModel
dev = 'cuda'
ROOT = 'data'
BMP = 'body_models/smplh/neutral/model.npz'
ACC = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/accel.pt'
HYB = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/field_hybrid.pt'   
SEQS = [('run_walk', ROOT + '/Transitions_mocap/mazen_c3d/run_walk_poses_253_frames_30_fps.npz'),
        ('turn180',  ROOT + '/Transitions_mocap/mazen_c3d/JOOF_turntwist180_poses_151_frames_30_fps.npz'),
        ('jumpjack', ROOT + '/SSM_synced/20160930_50032/jumping_jacks_sync_poses_94_frames_30_fps.npz')]
LAMBDAS = [0.0, 0.1, 0.3, 1.0]
FR = [0, 30, 60, 89]


def anneal_combine(hyb, acc, accsc, xo, xt, cmask, lam, n_levels=40, n_corr=4, step=0.15, s0=1.0, smin=0.1):
    ''
    B, nj = xo.shape[0], hyb.n_joints
    sigmas = torch.exp(torch.linspace(math.log(s0), math.log(smin), n_levels))
    keep = cmask[:, :, :nj].unsqueeze(-1).bool(); clean = xo.clone()
    def score(xo, xt, sig):
        sv = torch.full((B, nj + 1), sig, device=dev)
        s_om, s_tv = hyb.score(xo, xt, sv, cmask)
        if lam > 0:
            xr = xo.detach().requires_grad_(True)
            al = accel_from_vel(xr) / accsc
            Ea = acc.energy(al, xt.detach(), sv).sum()
            s_om = s_om + lam * (-torch.autograd.grad(Ea, xr)[0])
        return s_om, s_tv
    for sig in sigmas.tolist():
        xo = torch.where(keep, clean, xo)
        s_om, s_tv = score(xo, xt, sig); xo = xo + sig**2 * s_om; xt = xt + sig**2 * s_tv
        for _ in range(n_corr):
            xo = torch.where(keep, clean, xo)
            s_om, s_tv = score(xo, xt, sig); eps = step * sig**2
            xo = xo + 0.5*eps*s_om + math.sqrt(eps)*torch.randn_like(xo); xt = xt + 0.5*eps*s_tv + math.sqrt(eps)*torch.randn_like(xt)
    return torch.where(keep, clean, xo), xt
