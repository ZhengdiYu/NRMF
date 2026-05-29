''







import os; os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys, argparse; sys.path.insert(0, ''); sys.path.insert(0, 'motionlib')
sys.path.insert(0, '.'); sys.path.insert(0, '.')
import numpy as np, torch
from gen.dataset import process_npz
from gen.sample import load
from gen.optimize import velocity_field_query, pose_field_query, make_contact, contact_term
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)
from kinematics import fk_joints, FEET, LEG
from acceleration_prior import accel_query, dev, BMP, ACC, ROOTJ, BODYJ, K
from noise import rohm_noise_aa, geo_idx
from contacts import load_net, predicted_contact
from contact_refinement import foot_metrics, contact_refine
from guidance import SEQS
from humor.body_model.body_model import BodyModel
SG = 0.15


def denoise_full(noisy_q, qa, field, om_std, tv_n, sgv, acc, accsc, pf, pose_std, bodyi,
                 contact=None, w_pose=1.0, w_contact=1.0, w_vel=1.0, vel_selfscale=False, iters=400, lr=0.005, w_data=2.0, w_smooth=4.0,
                 lam=0.3, contact_start=0.7, sig_schedule='fixed', sig_start=2.0, sig_min=0.5, w_floor=0.0, floor=None):
    ''


    import math
    theta = qlog(noisy_q).detach().clone().requires_grad_(True); opt = torch.optim.Adam([theta], lr=lr)
    anneal = (sig_schedule == 'anneal'); nj = field.n_joints
    if anneal:
        sigmas = torch.exp(torch.linspace(math.log(sig_start), math.log(sig_min), iters, device=dev))
    sgt = torch.tensor([SG], device=dev); c0 = int(contact_start * iters)
    for it in range(iters):
        if anneal:
            s = float(sigmas[it]); sgv = torch.full((1, nj + 1), s, device=dev); sgt = torch.tensor([s], device=dev)
        q = aa2q(theta)
        L_data = w_data * ((q - noisy_q) ** 2).sum(-1).mean()
        om0 = qlog(qmul(q[1:], qconj(q[:-1]))); L_smooth = w_smooth * ((om0[1:] - om0[:-1]) ** 2).sum(-1).mean()
        L_reg = L_data + L_smooth
        if w_floor > 0 and floor is not None and contact is not None:
            L_reg = L_reg + w_floor * contact['fk'](q)[:, FEET, 2].sub(floor).clamp(max=0).pow(2).mean()
        g = torch.autograd.grad(L_reg, theta, retain_graph=True)[0]; gdn = g.norm()
        dv = velocity_field_query(theta, qa, om_std, field, sgv, tv_n, None)
        g_v = torch.autograd.grad(dv, theta, retain_graph=True)[0]; gvn = g_v.norm() + 1e-9
        if vel_selfscale:
            g = g + w_vel * dv.detach() * (g_v / gvn)
        else:
            g = g + w_vel * (gdn / gvn) * g_v
        if acc is not None:
            da = accel_query(theta, qa, om_std, acc, accsc, sgv, tv_n)
            g_a = torch.autograd.grad(da, theta, retain_graph=True)[0]; g = g + lam * (gdn / (g_a.norm() + 1e-9)) * g_a
        if w_pose > 0:
            dp = pose_field_query(theta, bodyi, pose_std, pf, sgt)
            g_p = torch.autograd.grad(dp, theta, retain_graph=True)[0]; g = g + w_pose * (gdn / (g_p.norm() + 1e-9)) * g_p
        if contact is not None and it >= c0:
            lc, _ = contact_term(q, contact)
            g_c = torch.autograd.grad(lc, theta, retain_graph=True)[0]; g = g + w_contact * (gdn / (g_c.norm() + 1e-9)) * g_c
        theta.grad = g; opt.step()
    return aa2q(theta.detach())
