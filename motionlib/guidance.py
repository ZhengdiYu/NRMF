''






















import os; os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys, math, argparse; sys.path.insert(0, ''); sys.path.insert(0, 'motionlib')
sys.path.insert(0, '.'); sys.path.insert(0, '.')
import numpy as np, torch
from gen.dataset import process_npz, FPS, accel_from_vel
from gen.sample import load
from gen.optimize import pose_field_query, make_contact, contact_term
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)
from kinematics import fk_joints, FEET, LEG
from contact_refinement import foot_metrics, integ_full
from acceleration_prior import recover as arecover, dev, BMP, ACC
from contacts import load_net, predicted_contact
from pose_refinement import pose_refine, load_pose
from humor.body_model.body_model import BodyModel


def stance(J):
    return (J[:, 7, :2] - J[:, 8, :2]).norm(dim=-1).mean().item()


def kneed(q, q_clean):
    lg = torch.tensor([4, 5], device=dev)
    return qlog(qmul(q[:, lg], qconj(q_clean[:q.shape[0], lg]))).norm(dim=-1).mean().item()


def legspd(q, legi):
    ql = q[:, legi]
    return qlog(qmul(ql[1:], qconj(ql[:-1]))).norm(dim=-1).mean().item()


B1 = 'checkpoints/field_perjoint.pt'
R = 'data'
SEQS = [('dribblekick', R + '/SSM_synced/20160930_50032/dribble_kick_sync_poses_141_frames_30_fps.npz'),
        ('airkick_run', R + '/Transitions_mocap/mazen_c3d/airkick_running_poses_217_frames_30_fps.npz'),
        ('crouchwalk',  R + '/Transitions_mocap/mazen_c3d/crouchwalk_crawl_poses_261_frames_30_fps.npz'),
        ('skipping',    R + '/SFU/0008/0008_Skipping001_poses_581_frames_30_fps.npz'),
        ('fwdwalk_EASY', R + '/SFU/0007/0007_Walking001_poses_786_frames_30_fps.npz')]  


def _integ_leg_diff(xo, q_clean, legi, om_std, W):
    ''

    tau = aa2q((xo[0] * om_std)[:, legi] / FPS)
    poses = [q_clean[0, legi]]
    for t in range(W):
        poses.append(qmul(tau[t], poses[-1]))
    return q_clean[:W + 1].clone().index_copy(1, legi, torch.stack(poses))


def anneal_guided(field, acc, accsc, xo, xt, cmask, legi, *, q_clean, om_std, betas, bm, transT, bodyi,
                  pf, pose_std, gt_low0, guide_pose, guide_contact, tail_sigma=0.3, tail_lo=0.0, sig_floor=0.15,
                  guide_iters=2, pose_sg=0.15, acc_lam=0.3, n_levels=40, n_corr=4, step=0.15,
                  s0=1.0, smin=0.1, verbose=False, return_om=False):
    ''






    B, nj = xo.shape[0], field.n_joints
    W = xo.shape[1]
    sigmas = torch.exp(torch.linspace(math.log(s0), math.log(smin), n_levels))
    keep = cmask[:, :, :nj].unsqueeze(-1).bool(); clean = xo.clone()
    fk = lambda q: fk_joints(q, betas, bm, transT)
    ctx = [None]
    sgt = torch.tensor([pose_sg], device=dev)
    dbg = [False]

    def score(xo, xt, sig):
        sv = torch.full((B, nj + 1), sig, device=dev)
        s_om, s_tv = field.score(xo, xt, sv, cmask)
        if acc is not None and acc_lam > 0:
            xr = xo.detach().requires_grad_(True)
            Ea = acc.energy(accel_from_vel(xr) / accsc, xt.detach(), sv).sum()
            s_om = s_om + acc_lam * (-torch.autograd.grad(Ea, xr)[0])
        return s_om, s_tv

    def guide_step(xo, sig):
        ''
        if ctx[0] is None and guide_contact > 0:
            with torch.no_grad():
                qn = _integ_leg_diff(xo.detach(), q_clean, legi, om_std, W)
            ctx[0] = make_contact(qn, qn, fk, gt_low0=gt_low0)
        mag = max(sig, sig_floor); delta = torch.zeros_like(xo); info = []
        if guide_pose > 0:
            xr = xo.detach().requires_grad_(True)
            Ep = pose_field_query(qlog(_integ_leg_diff(xr, q_clean, legi, om_std, W)), bodyi, pose_std, pf, sgt)
            gp = torch.autograd.grad(Ep, xr)[0]
            delta = delta + (guide_pose * mag / (gp.norm() + 1e-9)) * gp; info.append(('pose', gp.norm().item()))
        if guide_contact > 0:
            xr = xo.detach().requires_grad_(True)
            Ec, _ = contact_term(_integ_leg_diff(xr, q_clean, legi, om_std, W), ctx[0])
            gc = torch.autograd.grad(Ec, xr)[0]
            delta = delta + (guide_contact * mag / (gc.norm() + 1e-9)) * gc; info.append(('contact', gc.norm().item()))
        if verbose and not dbg[0]:
            dbg[0] = True
            print(f"      [guide@sig={sig:.3f}] mag={mag:.3f} " +
                  " ".join(f"||g_{n}||={g:.2f}" for n, g in info) + f" ||step||={delta.norm():.4f}", flush=True)
        return delta

    guided = (guide_pose > 0 or guide_contact > 0)
    for sig in sigmas.tolist():
        xo = torch.where(keep, clean, xo)
        s_om, s_tv = score(xo, xt, sig)
        xo = xo + sig ** 2 * s_om; xt = xt + sig ** 2 * s_tv
        for _ in range(n_corr):
            xo = torch.where(keep, clean, xo)
            s_om, s_tv = score(xo, xt, sig); eps = step * sig ** 2
            xo = xo + 0.5 * eps * s_om + math.sqrt(eps) * torch.randn_like(xo)
            xt = xt + 0.5 * eps * s_tv + math.sqrt(eps) * torch.randn_like(xt)
        if guided and tail_lo < sig <= tail_sigma:
            for _ in range(guide_iters):
                xo = torch.where(keep, clean, xo)
                xo = xo - guide_step(xo, sig)
    xo = torch.where(keep, clean, xo)
    if return_om:
        return xo
    return integ_full(xo, q_clean, legi, om_std, W)
