''














import os; os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')                 
import sys, argparse; sys.path.insert(0, ''); sys.path.insert(0, 'motionlib'); sys.path.insert(0, '.')
import numpy as np, torch
from gen.dataset import process_npz, FPS, accel_from_vel
from gen.sample import load, anneal
from gen.optimize import velocity_field_query
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)
from kinematics import fk_joints, FEET, LEG
from noise import rohm_noise_aa, geo_idx
from contact_refinement import foot_metrics, knee_drift, integ_full
from acceleration_sampling import anneal_combine
from humor.body_model.body_model import BodyModel
dev = 'cuda'
ROOT = 'data'
BMP = 'body_models/smplh/neutral/model.npz'
B1 = 'checkpoints/field_perjoint.pt'; HYB = 'checkpoints/field_hybrid.pt'
ACC = 'checkpoints/accel.pt'
SEQS = [('run_walk',  ROOT + '/Transitions_mocap/mazen_c3d/run_walk_poses_253_frames_30_fps.npz'),
        ('fwdwalk',   ROOT + '/SFU/0007/0007_Walking001_poses_786_frames_30_fps.npz'),
        ('turn180',   ROOT + '/Transitions_mocap/mazen_c3d/JOOF_turntwist180_poses_151_frames_30_fps.npz'),
        ('jumpjacks', ROOT + '/SSM_synced/20160930_50032/jumping_jacks_sync_poses_94_frames_30_fps.npz')]
ROOTJ = [0]; BODYJ = list(range(1, 22)); K = 5; LAM = 0.3; FR = [0, 30, 60, 89]


def accel_query(theta, qa, om_std, acc, accsc, sg, tv_n):
    ''

    q = aa2q(theta)
    cq = torch.cat([qmul(qa, q[:, 0, :]).unsqueeze(1), q[:, 1:, :]], 1)
    om = FPS * qlog(qmul(cq[1:], qconj(cq[:-1])))
    om_n = (om / om_std).unsqueeze(0)
    al = accel_from_vel(om_n) / accsc
    sgf = torch.full((1, acc.n_joints + 1), float(sg.reshape(-1)[0]), device=dev)
    return acc.energy(al, tv_n, sgf).mean()


def denoise_pose(noisy_q, qa, field, om_std, tv_n, sg, acc=None, accsc=None, lam=LAM,
                 iters=500, lr=0.005, w_data=2.0, w_smooth=4.0):
    ''
    theta = qlog(noisy_q).detach().clone().requires_grad_(True); opt = torch.optim.Adam([theta], lr=lr)
    for it in range(iters):
        q = aa2q(theta); L_data = w_data * ((q - noisy_q) ** 2).sum(-1).mean()
        om0 = qlog(qmul(q[1:], qconj(q[:-1])))
        L_smooth = w_smooth * ((om0[1:] - om0[:-1]) ** 2).sum(-1).mean()
        g_ds = torch.autograd.grad(L_data + L_smooth, theta, retain_graph=True)[0]; gdn = g_ds.norm()
        dv = velocity_field_query(theta, qa, om_std, field, sg, tv_n, None)
        g_v = torch.autograd.grad(dv, theta, retain_graph=True)[0]
        g = g_ds + (gdn / (g_v.norm() + 1e-9)) * g_v
        if acc is not None:
            da = accel_query(theta, qa, om_std, acc, accsc, sg, tv_n)
            g_a = torch.autograd.grad(da, theta, retain_graph=True)[0]
            g = g + lam * (gdn / (g_a.norm() + 1e-9)) * g_a
        theta.grad = g; opt.step()
    return aa2q(theta.detach())


def recover(field, om_std, tv_std, om, tv, q_clean, cmask, legi, W, acc=None, accsc=None, lam=LAM):
    om_n = (om[:W].unsqueeze(0).to(dev) / om_std); tv_n = (tv[:W].unsqueeze(0).to(dev) / tv_std)
    torch.manual_seed(0); xo = om_n.clone(); xo[:, :, legi] = torch.randn_like(xo[:, :, legi])
    if acc is None:
        rec, _ = anneal(field, xo.clone(), tv_n.clone(), 1.0, n_levels=40, n_corrector=4, sigma_min=0.1, cmask=cmask, langevin=True)
    else:
        rec, _ = anneal_combine(field, acc, accsc, xo.clone(), tv_n.clone(), cmask, lam)
    return integ_full(rec, q_clean, legi, om_std, W)
