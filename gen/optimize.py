''



















import math
import torch
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)

FPS = 30


LEG = [1, 2, 4, 5, 7, 8, 10, 11]
FEET = [7, 8, 10, 11]



PJ_QUERY_OVERRIDE = None





def sigma_schedule(kind, iters, sig_start, sig_min, dev='cuda'):
    ''


    if kind == 'fixed':
        return torch.full((iters,), float(sig_start), device=dev)
    return torch.exp(torch.linspace(math.log(sig_start), math.log(sig_min), iters, device=dev))


def build_data_mask(task, W1, n_joints=22, occ=LEG, dev='cuda'):
    ''

    if task == 'generate':
        return torch.zeros(W1, n_joints, dtype=torch.bool, device=dev)
    m = torch.ones(W1, n_joints, dtype=torch.bool, device=dev)
    if task == 'recover':
        m[:, occ] = False
    return m





def velocity_field_query(theta, qalign, om_std, field, sg, tv_n, field_mask=None):
    ''


    q = aa2q(theta)
    cq = torch.cat([qmul(qalign, q[:, 0, :]).unsqueeze(1), q[:, 1:, :]], 1)
    om = FPS * qlog(qmul(cq[1:], qconj(cq[:-1])))
    om_n = (om / om_std).unsqueeze(0)
    if getattr(field, 'per_joint_sigma', False):                                   
        sg = torch.full((1, field.n_joints + 1), float(sg.reshape(-1)[0]), device=om_n.device)
        if PJ_QUERY_OVERRIDE is not None:
            for j, s in PJ_QUERY_OVERRIDE: sg[:, j] = s
    return field.distance(om_n, tv_n, sg, field_mask).mean()


def pose_field_query(theta, bodyi, pose_std, field, sg):
    ''
    return field.distance((theta[:, bodyi] / pose_std),
                          torch.full((theta.shape[0],), sg.item(), device=theta.device)).mean()





def make_contact(q_clean, q_init, fk, *, h_thresh=0.08, gt_low0=None,
                 w_skate=1.5, w_floor=0.3, w_pen=5.0, start_frac=0.7):
    ''


    with torch.no_grad():
        Jc = fk(q_clean); Ji = fk(q_init)
        floor = float(torch.cat([Jc[:, FEET, 2], Ji[:, FEET, 2]]).min())
        if gt_low0 is not None:
            low0 = gt_low0.float()
        else:
            h0 = Ji[:, FEET, 2] - floor
            low0 = (h0 < h_thresh).float()
        planted0 = low0[:-1] * low0[1:]
    return {'fk': fk, 'floor': floor, 'low0': low0, 'planted0': planted0,
            'w_skate': w_skate, 'w_floor': w_floor, 'w_pen': w_pen, 'start_frac': start_frac}


def contact_term(q_full, c):
    ''

    J = c['fk'](q_full)                                                             
    h = J[:, FEET, 2] - c['floor']                                                  
    horiz = (J[1:, FEET, :2] - J[:-1, FEET, :2]).norm(dim=-1) * FPS
    L_skate = (c['planted0'] * horiz ** 2).mean()
    L_floor = (c['low0'] * h ** 2).mean()
    L_pen = (h.clamp(max=0) ** 2).mean()
    loss = c['w_skate'] * L_skate + c['w_floor'] * L_floor + c['w_pen'] * L_pen
    return loss, {'skate': L_skate.item(), 'floor': L_floor.item(), 'pen': L_pen.item()}





def _integ_from_anchor(om_un, q0, W):
    ''

    tau = aa2q(om_un / FPS)
    ps = [q0]
    for t in range(W):
        ps.append(qmul(tau[t], ps[-1]))
    return torch.stack(ps)


def recover_init(q_clean, om_std, sig_start, W, occ=LEG, seed=0):
    ''


    dev = q_clean.device
    theta = qlog(q_clean[:W + 1].clone())
    occ = torch.as_tensor(occ, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    leg_om = torch.randn(W, len(occ), 3, generator=g, device=dev) * sig_start * om_std[occ]
    poses = _integ_from_anchor(leg_om, q_clean[0, occ], W)
    theta[:, occ] = qlog(poses)
    return theta.detach()





def unified_optimize(anchor_q, qalign, tv_n, *, task, sigma_kind, fields, om_std, pose_std, bodyi,
                     data_mask, use, w_data=2.0, w_smooth=4.0, ratio=1.0, iters=500, lr=5e-3,
                     sig_start=2.0, sig_min=0.5, contact=None,
                     field_mask=None, init_theta=None, grad_cap=None,
                     root_field=None, root_om_std=None):
    ''










    dev = anchor_q.device
    theta = (init_theta if init_theta is not None else qlog(anchor_q)).detach().clone().requires_grad_(True)
    sigs = sigma_schedule(sigma_kind, iters, sig_start, sig_min, dev)
    opt = torch.optim.Adam([theta], lr=lr)
    mask = data_mask[..., None].float()
    logs = {}
    for it in range(iters):
        sg = sigs[it:it + 1]
        q = aa2q(theta)
        om0 = qlog(qmul(q[1:], qconj(q[:-1])))
        L_data = w_data * ((mask * (q - anchor_q)) ** 2).sum(-1).mean()
        L_smooth = w_smooth * ((om0[1:] - om0[:-1]) ** 2).sum(-1).mean()
        L_ds = L_data + L_smooth
        opt.zero_grad()
        extra = list(use) + (['contact'] if (contact is not None and task != 'generate'
                                             and it >= int(contact['start_frac'] * iters)) else [])
        if extra:
            g_ds = torch.autograd.grad(L_ds, theta, retain_graph=True)[0]
            total = g_ds.clone()
            gdn = g_ds.norm()
            for nm in use:
                if nm == 'a3':
                    d = velocity_field_query(theta, qalign, om_std, fields['a3'], sg, tv_n, field_mask)
                    g_pr = torch.autograd.grad(d, theta, retain_graph=True)[0]
                    if root_field is not None:
                        d_r = velocity_field_query(theta, qalign, root_om_std, root_field, sg, tv_n, None)
                        g_r = torch.autograd.grad(d_r, theta, retain_graph=True)[0]
                        g_pr = g_pr.clone(); g_pr[:, 0] = g_r[:, 0]
                else:
                    d = pose_field_query(theta, bodyi, pose_std, fields['pose'], sg)
                    g_pr = torch.autograd.grad(d, theta, retain_graph=True)[0]
                scale = (ratio / len(use)) * gdn / (g_pr.norm() + 1e-9)
                if grad_cap is not None: scale = scale.clamp(max=grad_cap)
                total = total + scale * g_pr
            if 'contact' in extra:
                lc, cinfo = contact_term(q, contact)
                g_c = torch.autograd.grad(lc, theta, retain_graph=True)[0]
                scale = gdn / (g_c.norm() + 1e-9)
                if grad_cap is not None: scale = scale.clamp(max=grad_cap)
                total = total + scale * g_c
                logs.update({f'contact_{k}': v for k, v in cinfo.items()})
            theta.grad = total
        else:
            L_ds.backward()
        opt.step()
    logs.update({'L_data': L_data.item(), 'L_smooth': L_smooth.item(), 'sigma_end': float(sigs[-1])})
    return {'q': aa2q(theta.detach()), 'logs': logs}
