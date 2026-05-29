''








import os; os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys, argparse, math; sys.path.insert(0, ''); sys.path.insert(0, '.')
sys.path.insert(0, '.')
import numpy as np, torch, trimesh, pyrender
from PIL import Image, ImageDraw
import torch.backends.cuda as bc; bc.enable_flash_sdp(False); bc.enable_mem_efficient_sdp(False)
from gen.dataset import process_npz
from gen.sample import load, anneal
from utils.transforms import axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul, quaternion_log_map as qlog
from utils.transforms import quaternion_log_map as _q2aa
def verts(quat, betas, faces, bm):
    T = len(quat); aa = _q2aa(quat); root = aa[:, 0, :]; body = aa[:, 1:, :].reshape(T, 63)
    bt = torch.tensor(betas[:16], dtype=torch.float32, device=dev)[None].repeat(T, 1)
    out = bm(root_orient=root, pose_body=body, betas=bt, trans=torch.zeros(T, 3, device=dev))
    return out.v.detach().cpu().numpy()
from humor.body_model.body_model import BodyModel
dev = 'cuda'; FPS = 30
LEG = [1, 2, 4, 5, 7, 8, 10, 11]

_NRDF = [None]
def get_nrdf():
    if _NRDF[0] is None:
        import yaml; ND = '.'
        c = ND + '/checkpoints/NRDF/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1'
        nopt = yaml.safe_load(open(c + '/config.yaml')); nopt['train']['device'] = dev
        from lib.model.nrdf import NRDF
        nd = NRDF(nopt).to(dev)
        nd.load_state_dict(torch.load(c + '/checkpoints/checkpoint_epoch_best.tar', map_location=dev)['model_state_dict'])
        nd.eval(); _NRDF[0] = nd
    return _NRDF[0]


def _legposes(legw_un, q_anchor, W, scheme='onesided', q_anchor1=None):
    ''



    if scheme == 'logdiff':
        full = aa2q(legw_un * (2.0 / FPS))
        poses = [q_anchor, q_anchor1]
        for k in range(1, W): poses.append(qmul(full[k], poses[k - 1]))
        return torch.stack(poses)
    leg_tau = aa2q(legw_un / FPS)
    poses = [q_anchor]
    for t in range(W): poses.append(qmul(leg_tau[t], poses[-1]))
    return torch.stack(poses)


def recover_guided(net, om_n, tv_n, mask_om, legi, om_std, q_anchor, q_clean_full, W,
                   sig0=1.0, sig_min=0.1, n_levels=40, n_corr=4, step=0.15, nrdf_w=0.6, nrdf_iters=2,
                   cmask=None, scheme='onesided', q_anchor1=None):
    ''

    nd = get_nrdf(); om_std_leg = om_std[legi]
    def noise(x, s): return x + s * torch.randn_like(x)
    def obs_fill(s): return om_n if cmask is not None else noise(om_n, s)
    xo = om_n.clone(); xo[:, :, legi] = torch.randn_like(xo[:, :, legi]) * sig0
    xt = tv_n.clone()
    for sig in torch.exp(torch.linspace(math.log(sig0), math.log(sig_min), n_levels)).tolist():
        sv = torch.full((1,), sig, device=dev)
        xo = torch.where(mask_om, obs_fill(sig), xo)
        s_om, _ = net.score(xo, xt, sv, cmask); xo = xo + sig * sig * s_om
        for _ in range(n_corr):
            xo = torch.where(mask_om, obs_fill(sig), xo)
            s_om, _ = net.score(xo, xt, sv, cmask); eps = step * sig * sig
            xo = xo + 0.5 * eps * s_om + math.sqrt(eps) * torch.randn_like(xo)
        for _ in range(nrdf_iters):
            legw = xo[:, :, legi].detach().requires_grad_(True)
            qleg = _legposes(legw[0] * om_std_leg, q_anchor, W, scheme, q_anchor1)
            qfull = q_clean_full[:W + 1].clone().index_copy(1, legi, qleg)
            E = nd(qfull[:, 1:], train=False)['dist_pred'].abs().mean()
            g = torch.autograd.grad(E, legw)[0]
            gstep = nrdf_w * max(sig, 0.25)
            with torch.no_grad(): xo[:, :, legi] = xo[:, :, legi] - gstep * g
    return torch.where(mask_om, om_n, xo)


def make_ground(cx, cy, z, size=14.0, tile=0.4):
    n = int(size / tile); V, F, C = [], [], []
    for i in range(n):
        for j in range(n):
            x0 = cx - size / 2 + i * tile; y0 = cy - size / 2 + j * tile; b = len(V)
            V += [[x0, y0, z], [x0 + tile, y0, z], [x0 + tile, y0 + tile, z], [x0, y0 + tile, z]]
            F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
            c = [175, 182, 200] if (i + j) % 2 == 0 else [95, 105, 130]
            C += [c, c]
    m = trimesh.Trimesh(np.array(V, np.float32), np.array(F), process=False)
    m.visual.face_colors = np.array(C, np.uint8); return m


_R = [None]
def render_scene(body_v, faces, col, ground, ctr, dist, W=380, H=470):
    sc = pyrender.Scene(bg_color=[0.93, 0.96, 1.0, 1.0], ambient_light=[0.32, 0.32, 0.38])
    bmesh = trimesh.Trimesh(body_v, faces, process=False)
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=col + [1.0], metallicFactor=0.0, roughnessFactor=0.6)
    sc.add(pyrender.Mesh.from_trimesh(bmesh, material=mat, smooth=True))
    sc.add(pyrender.Mesh.from_trimesh(ground, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0); cp = np.eye(4)
    eye = np.array([ctr[0] + 0.28 * dist, ctr[1] - dist, ctr[2] + 0.48 * dist])
    look = np.array([ctr[0], ctr[1], ctr[2] - 0.15])
    fwd = look - eye; fwd /= np.linalg.norm(fwd); up = np.array([0, 0, 1.])
    right = np.cross(fwd, up); right /= np.linalg.norm(right); up2 = np.cross(right, fwd)
    cp[:3, 0] = right; cp[:3, 1] = up2; cp[:3, 2] = -fwd; cp[:3, 3] = eye
    sc.add(cam, pose=cp)
    for off in ([3, -3, 4], [-3, -1, 4], [0, -4, 3]):
        lp = np.eye(4); lp[:3, 3] = [ctr[0] + off[0], ctr[1] + off[1], ctr[2] + off[2]]
        sc.add(pyrender.DirectionalLight(intensity=3.0), pose=lp)
    if _R[0] is None: _R[0] = pyrender.OffscreenRenderer(W, H)
    c, _ = _R[0].render(sc); return c


def integrate(rec_om, q_clean, legi, om_std, W, scheme='onesided'):
    qleg = _legposes((rec_om[0] * om_std)[:, legi, :], q_clean[0, legi], W, scheme, q_clean[1, legi])
    q = q_clean[:W + 1].clone(); q[:, legi] = qleg; return q
