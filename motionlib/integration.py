''













import os; os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys, argparse; sys.path.insert(0, ''); sys.path.insert(0, 'motionlib')
sys.path.insert(0, '.'); sys.path.insert(0, '.')
import numpy as np, torch
from gen.dataset import process_npz, FPS
from gen.sample import load
from gen.optimize import pose_field_query
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)
from kinematics import fk_joints, FEET, LEG
from acceleration_prior import recover as arecover, dev, BMP, ACC
from contact_refinement import foot_metrics, integ_full
from guidance import legspd, stance, kneed, SEQS
from pose_refinement import load_pose
from lib.core.rdfgrad import egrad2rgrad, exp_map
from humor.body_model.body_model import BodyModel

B1 = 'checkpoints/field_perjoint.pt'


def _pi_r_leg(q_leg, q_obs_full, legi, bodyi, pf, pose_std, alpha, sg, iters, selfscale=False):
    ''


    q_leg = q_leg.detach()
    for _ in range(iters):
        ql = q_leg.detach().requires_grad_(True)
        q_full = q_obs_full.clone().index_copy(0, legi, ql).unsqueeze(0)
        E = pose_field_query(qlog(q_full), bodyi, pose_std, pf, sg)
        g = torch.autograd.grad(E, ql)[0]
        rg = egrad2rgrad(g.unsqueeze(0), ql.detach().unsqueeze(0), device=dev, norm=True)
        mag = alpha * (E.detach() if selfscale else 1.0)
        q_leg = exp_map(ql.detach().unsqueeze(0), -mag * rg, device=dev)[0]
    return q_leg


def integ_projected(rec_om, q_clean, legi, om_std, W, *, pf, pose_std, bodyi, alpha, proj_iters=1, sg=0.15, selfscale=False):
    ''

    sgt = torch.tensor([sg], device=dev)
    tau = aa2q((rec_om[0] * om_std)[:, legi] / FPS)
    poses = [q_clean[0, legi]]
    for t in range(W):
        q_next = qmul(tau[t], poses[-1])
        if alpha > 0:
            q_next = _pi_r_leg(q_next, q_clean[t + 1], legi, bodyi, pf, pose_std, alpha, sgt, proj_iters, selfscale)
        poses.append(q_next)
    q = q_clean[:W + 1].clone(); q[:, legi] = torch.stack(poses); return q


def cum_drift(q, q_clean, legi, W):
    ''

    d = qlog(qmul(q[:W + 1, legi], qconj(q_clean[:W + 1, legi]))).norm(dim=-1)
    per_frame = d.mean(-1) * 180 / np.pi
    early = per_frame[:30].mean().item(); late = per_frame[-30:].mean().item()
    return per_frame.mean().item(), late - early




def _recover_om(field, om_std, tv_std, om, tv, q_clean, cmask, legi, W, acc, accsc):
    ''
    from acceleration_prior import anneal_combine
    from gen.sample import anneal
    om_n = (om[:W].unsqueeze(0).to(dev) / om_std); tv_n = (tv[:W].unsqueeze(0).to(dev) / tv_std)
    torch.manual_seed(0); xo = om_n.clone(); xo[:, :, legi] = torch.randn_like(xo[:, :, legi])
    if acc is None:
        rec, _ = anneal(field, xo.clone(), tv_n.clone(), 1.0, n_levels=40, n_corrector=4, sigma_min=0.1, cmask=cmask, langevin=True)
    else:
        rec, _ = anneal_combine(field, acc, accsc, xo.clone(), tv_n.clone(), cmask, 0.3)
    return rec
