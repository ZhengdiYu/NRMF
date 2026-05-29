''










import sys, math, argparse; sys.path.insert(0, ''); sys.path.insert(0, 'motionlib')
import numpy as np, torch
from gen.dataset import process_npz
from gen.sample import load, anneal
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)
from kinematics import fk_joints, FEET, LEG, FPS
from gen.optimize import make_contact, contact_term
from humor.body_model.body_model import BodyModel
dev = 'cuda'
ROOT = 'data'
BMP = 'body_models/smplh/neutral/model.npz'
CK = 'checkpoints/field_perjoint.pt'             
SEQS = [('run_walk',  ROOT + '/Transitions_mocap/mazen_c3d/run_walk_poses_253_frames_30_fps.npz'),
        ('turn180',   ROOT + '/Transitions_mocap/mazen_c3d/JOOF_turntwist180_poses_151_frames_30_fps.npz'),
        ('jumpjacks', ROOT + '/SSM_synced/20160930_50032/jumping_jacks_sync_poses_94_frames_30_fps.npz'),
        ('fwdwalk',   ROOT + '/SFU/0007/0007_Walking001_poses_786_frames_30_fps.npz')]
FR = [0, 30, 60, 89]


def integ_full(rec_om, q_clean, legi, om_std, W):
    ''
    tau = aa2q((rec_om[0] * om_std)[:, legi] / FPS); poses = [q_clean[0, legi]]
    for t in range(W):
        poses.append(qmul(tau[t], poses[-1]))
    q = q_clean[:W + 1].clone(); q[:, legi] = torch.stack(poses); return q


def recover(net, om_n, tv_n, cmask, legi):
    ''
    torch.manual_seed(0)
    xo = om_n.clone(); xo[:, :, legi] = torch.randn_like(xo[:, :, legi])
    rec, _ = anneal(net, xo.clone(), tv_n.clone(), 1.0, n_levels=40, n_corrector=4, sigma_min=0.1, cmask=cmask, langevin=True)
    return rec


def contact_refine(q_rec, legi, obsj, betas, bm, transT, *, gt_low0=None, iters=200, lr=5e-3,
                   w_contact=1.0, w_data_obs=2.0, w_data_leg=0.2, w_smooth=4.0, w_floor=0.0, floor=None, verbose=False):
    ''





    fk = lambda q: fk_joints(q, betas, bm, transT)
    contact = make_contact(q_rec, q_rec, fk, gt_low0=gt_low0)
    base = q_rec.detach()
    theta_leg = qlog(q_rec[:, legi]).detach().clone().requires_grad_(True); opt = torch.optim.Adam([theta_leg], lr=lr)
    for it in range(iters):
        q = base.index_copy(1, legi, aa2q(theta_leg))
        L_data = w_data_leg * ((q[:, legi] - q_rec[:, legi]) ** 2).sum(-1).mean()
        om0 = qlog(qmul(q[1:], qconj(q[:-1])))
        L_smooth = w_smooth * ((om0[1:] - om0[:-1]) ** 2).sum(-1).mean()
        L_reg = L_data + L_smooth
        if w_floor > 0 and floor is not None:
            L_reg = L_reg + w_floor * fk_joints(q, betas, bm, transT)[:, FEET, 2].sub(floor).clamp(max=0).pow(2).mean()
        g_ds = torch.autograd.grad(L_reg, theta_leg, retain_graph=True)[0]; gdn = g_ds.norm()
        lc, ci = contact_term(q, contact); g_c = torch.autograd.grad(lc, theta_leg)[0]
        if verbose and it == 0:
            print(f"      [refine] legs-only | ||g_ds||={gdn:.3f} ||g_c||={g_c.norm():.3f} contact0 skate={ci['skate']:.4f} pen={ci['pen']:.4f}", flush=True)
        theta_leg.grad = g_ds + w_contact * (gdn / (g_c.norm() + 1e-9)) * g_c
        opt.step()
    q_out = base.index_copy(1, legi, aa2q(theta_leg.detach()))
    return q_out, contact


def foot_metrics(J, floor):
    fz = J[:, FEET, 2] - floor
    horiz = (J[1:, FEET, :2] - J[:-1, FEET, :2]).norm(dim=-1) * FPS
    low = (fz[:-1] < 0.05); skate = horiz[low]; pen = (-fz).clamp(min=0)
    q90 = torch.quantile(skate, .9).item() if skate.numel() else 0.0
    return dict(fh_med=fz.median().item(), fh_max=fz.max().item(), pen=pen.max().item(),
                sk_mean=(skate.mean().item() if skate.numel() else 0.0), sk_p90=q90)


def knee_drift(q, q_clean, legi, W):
    ''
    d = qlog(qmul(q[:W + 1, legi], qconj(q_clean[:W + 1, legi]))).norm(dim=-1).mean(0)
    return (d[2] + d[3]).item() / 2
