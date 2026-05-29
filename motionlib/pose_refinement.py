''





import os; os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys; sys.path.insert(0, ''); sys.path.insert(0, 'motionlib')
sys.path.insert(0, '.'); sys.path.insert(0, '.')
import numpy as np, torch
from gen.dataset import process_npz
from gen.sample import load
from gen.field import PoseField
from gen.optimize import pose_field_query, make_contact, contact_term
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)
from kinematics import fk_joints, FEET, LEG
from contact_refinement import recover, integ_full, foot_metrics, knee_drift, SEQS, dev, BMP, FR
from contacts import load_net, predicted_contact
from humor.body_model.body_model import BodyModel

W_POSE = 1.0; W_CONTACT = 1.0; SG = 0.15
_NRDF = [None]


def get_nrdf():
    if _NRDF[0] is None:
        import yaml
        c = './checkpoints/NRDF/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1'
        nopt = yaml.safe_load(open(c + '/config.yaml')); nopt['train']['device'] = dev
        from lib.model.nrdf import NRDF
        nd = NRDF(nopt).to(dev)
        nd.load_state_dict(torch.load(c + '/checkpoints/checkpoint_epoch_best.tar', map_location=dev)['model_state_dict'])
        nd.eval(); _NRDF[0] = nd
    return _NRDF[0]


def load_pose(p='checkpoints/pose.pt'):
    ck = torch.load(p, map_location=dev); a = ck['args']; a = a if isinstance(a, dict) else vars(a)
    pf = PoseField(d_model=a['d_model'], n_layers=a['layers']).to(dev); pf.load_state_dict(ck['model']); pf.eval()
    return pf, ck['pose_std'].to(dev)


def pose_refine(q_rec, legi, betas, bm, transT, *, pose_src=None, pf=None, pose_std=None, bodyi=None, nrdf=None,
                w_pose=0.0, gt_low0=None, w_contact=0.0, iters=200, lr=5e-3, w_data_leg=0.2, w_smooth=4.0,
                w_floor=0.0, floor=None, sg=SG):
    ''
    fk = lambda q: fk_joints(q, betas, bm, transT)
    contact = make_contact(q_rec, q_rec, fk, gt_low0=gt_low0) if w_contact > 0 else None
    base = q_rec.detach(); sgt = torch.tensor([sg], device=dev)
    theta_leg = qlog(q_rec[:, legi]).detach().clone().requires_grad_(True); opt = torch.optim.Adam([theta_leg], lr=lr)
    for it in range(iters):
        q = base.index_copy(1, legi, aa2q(theta_leg))
        L = w_data_leg * ((q[:, legi] - q_rec[:, legi]) ** 2).sum(-1).mean()
        om0 = qlog(qmul(q[1:], qconj(q[:-1]))); L = L + w_smooth * ((om0[1:] - om0[:-1]) ** 2).sum(-1).mean()
        if w_floor > 0 and floor is not None:
            L = L + w_floor * fk(q)[:, FEET, 2].sub(floor).clamp(max=0).pow(2).mean()
        g = torch.autograd.grad(L, theta_leg, retain_graph=True)[0]; gdn = g.norm(); total = g
        if w_pose > 0:
            dp = nrdf(q[:, 1:], train=False)['dist_pred'].abs().mean() if pose_src == 'nrdf' \
                else pose_field_query(qlog(q), bodyi, pose_std, pf, sgt)
            gp = torch.autograd.grad(dp, theta_leg, retain_graph=True)[0]
            total = total + w_pose * (gdn / (gp.norm() + 1e-9)) * gp
        if w_contact > 0:
            lc, _ = contact_term(q, contact); gc = torch.autograd.grad(lc, theta_leg, retain_graph=True)[0]
            total = total + w_contact * (gdn / (gc.norm() + 1e-9)) * gc
        theta_leg.grad = total; opt.step()
    return base.index_copy(1, legi, aa2q(theta_leg.detach()))
