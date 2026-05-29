''







import sys, argparse; sys.path.insert(0, ''); sys.path.insert(0, '.')
import numpy as np, torch
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
from gen.dataset import process_npz
from gen.sample import load, anneal
from utils.transforms import axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul, quaternion_log_map as q2aa
from humor.body_model.body_model import BodyModel
dev = 'cuda'; FPS = 30
LEG = [1, 2, 4, 5, 7, 8, 10, 11]
ANK = [7, 8]; TOE = [10, 11]; FEET = ANK + TOE


def fk_joints(q_full, betas, bm, trans):
    ''
    T = q_full.shape[0]
    root = q2aa(q_full[:, 0]); body = q2aa(q_full[:, 1:]).reshape(T, -1)
    bt = torch.as_tensor(np.asarray(betas), dtype=torch.float32, device=dev)[:16].unsqueeze(0).expand(T, -1)
    tr = torch.as_tensor(trans, dtype=torch.float32, device=dev)
    out = bm(root_orient=root, pose_body=body, betas=bt, trans=tr)
    return out.Jtr


def integ_legs(rec_om, q_clean, legi, om_std, W):
    tau = aa2q((rec_om[0] * om_std)[:, legi] / FPS); poses = [q_clean[0, legi]]
    for t in range(W): poses.append(qmul(tau[t], poses[-1]))
    q = q_clean[:W + 1].clone(); q[:, legi] = torch.stack(poses); return q


def foot_stats(J, floor, tag):
    ''
    fz = J[:, FEET, 2] - floor
    horiz = (J[1:, FEET, :2] - J[:-1, FEET, :2]).norm(dim=-1) * FPS
    low = (fz[:-1] < 0.05)
    skate = horiz[low]
    pen = (-fz).clamp(min=0)
    print(f"  {tag:24s}: foot-height med {fz.median():.3f} max {fz.max():.3f} | penetration max {pen.max():.3f} "
          f"| SKATE(low-foot speed) mean {skate.mean() if skate.numel() else torch.tensor(0.):.3f} "
          f"p90 {torch.quantile(skate, .9) if skate.numel() else torch.tensor(0.):.3f} m/s", flush=True)
