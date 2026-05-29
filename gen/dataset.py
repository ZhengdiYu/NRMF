''












import os, re, glob
import numpy as np, torch
from torch.utils.data import Dataset
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)

FPS = 30


def _mat_to_quat(R):
    R = R.reshape(-1, 3, 3)
    w = torch.sqrt(torch.clamp(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], min=1e-8)) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    q = torch.stack([w, x, y, z], -1)
    return q / q.norm(dim=-1, keepdim=True)


def process_npz(path, fps=FPS, vel_scheme='onesided', return_pose=False):
    ''




    d = np.load(path, allow_pickle=True)
    aa = np.concatenate([d['root_orient'].reshape(-1, 1, 3), d['pose_body'].reshape(-1, 21, 3)], 1)  
    aa = torch.tensor(aa, dtype=torch.float32)
    q = aa2q(aa)
    R = torch.tensor(np.asarray(d['world2aligned_rot']), dtype=torch.float32)  
    qalign = _mat_to_quat(R)
    q_canon = torch.cat([qmul(qalign, q[:, 0, :]).unsqueeze(1), q[:, 1:, :]], 1)
    if vel_scheme == 'logdiff' and q_canon.shape[0] >= 3:                   
        mid = (fps / 2) * qlog(qmul(q_canon[2:], qconj(q_canon[:-2])))
        e0 = (fps / 2) * qlog(qmul(q_canon[2:3], qconj(q_canon[0:1])))
        eL = (fps / 2) * qlog(qmul(q_canon[-1:], qconj(q_canon[-3:-2])))
        omega = torch.cat([e0, mid, eL], 0)[:-1]
    else:
        tau = qmul(q_canon[1:], qconj(q_canon[:-1]))
        omega = fps * qlog(tau)
    trans = torch.tensor(np.asarray(d['trans']), dtype=torch.float32)       
    vw = fps * (trans[1:] - trans[:-1])
    transv = torch.einsum('tij,tj->ti', R[1:], vw)                          
    if return_pose:
        return omega, transv, q_canon, qalign
    return omega, transv


def accel_from_vel(om):
    ''



    a = torch.zeros_like(om)
    a[..., 1:-1, :, :] = (om[..., 2:, :, :] - om[..., :-2, :, :]) * 0.5
    a[..., 0, :, :] = om[..., 1, :, :] - om[..., 0, :, :]
    a[..., -1, :, :] = om[..., -1, :, :] - om[..., -2, :, :]
    return a


def rohm_noise_canon(q_canon, qalign, k_deg):
    ''




    from scipy.spatial.transform import Rotation as _Rot
    B, M, J, _ = q_canon.shape; dev = q_canon.device
    raw_root = qmul(qconj(qalign), q_canon[:, :, 0])
    raw_q = torch.cat([raw_root.unsqueeze(2), q_canon[:, :, 1:]], 2)
    rotvec = qlog(raw_q).reshape(-1, 3).cpu().numpy()
    eul = _Rot.from_rotvec(rotvec).as_euler('zxy', degrees=True)           
    noise = (torch.randn(B, M, J, 3) * k_deg.detach().view(B, 1, 1, 1).cpu()).reshape(-1, 3).numpy()
    rv_n = _Rot.from_euler('zxy', eul + noise, degrees=True).as_rotvec()   
    q_raw_n = aa2q(torch.tensor(rv_n, dtype=torch.float32, device=dev).reshape(B, M, J, 3))
    return torch.cat([qmul(qalign, q_raw_n[:, :, 0]).unsqueeze(2), q_raw_n[:, :, 1:]], 2)


def process_npz_pose(path):
    ''

    d = np.load(path, allow_pickle=True)
    return torch.tensor(d['pose_body'].reshape(-1, 21, 3), dtype=torch.float32)   


class PoseDataset(Dataset):
    ''
    def __init__(self, files, stride=2, cache=True):
        self.files = list(files); self.cache = {} if cache else None
        self.index = []; skipped = 0
        for fi, f in enumerate(self.files):
            nf = parse_nframes(f)
            if nf is None:
                try: nf = int(np.load(f, allow_pickle=True)['pose_body'].shape[0])
                except Exception: skipped += 1; continue
            for t in range(0, nf, stride): self.index.append((fi, t))
        self.skipped = skipped

    def __len__(self): return len(self.index)

    def _get(self, fi):
        if self.cache is not None and fi in self.cache: return self.cache[fi]
        p = process_npz_pose(self.files[fi])
        if self.cache is not None: self.cache[fi] = p
        return p

    def __getitem__(self, i):
        fi, t = self.index[i]; p = self._get(fi)
        return {'pose': p[min(t, p.shape[0] - 1)].clone()}   


def parse_nframes(path):
    m = re.search(r'_(\d+)_frames_', os.path.basename(path))
    return int(m.group(1)) if m else None


def list_amass_files(root, split='train'):
    from utils.data_utils import amass_splits
    ds = set(amass_splits[split])
    return sorted(p for p in glob.glob(root + '/**/*.npz', recursive=True)
                  if '_frames_' in p and p.split('/')[-3] in ds)


class WindowMotionDataset(Dataset):
    ''


    def __init__(self, files, window=31, stride=10, fps=FPS, cache=True, vel_scheme='onesided', pose_noise=False):
        self.files = list(files); self.window = window; self.stride = stride; self.fps = fps
        self.vel_scheme = vel_scheme
        self.pose_noise = pose_noise
        self.cache = {} if cache else None
        self.index = []
        skipped = 0
        for fi, f in enumerate(self.files):
            nf = parse_nframes(f)
            if nf is None:
                try: nf = int(np.load(f, allow_pickle=True)['trans'].shape[0])
                except Exception: skipped += 1; continue
            ntr = nf - 1
            if ntr < window: skipped += 1; continue
            for s in range(0, ntr - window + 1, stride):
                self.index.append((fi, s))
        self.skipped = skipped

    def __len__(self):
        return len(self.index)

    def _get_file(self, fi):
        if self.cache is not None and fi in self.cache:
            return self.cache[fi]
        out = process_npz(self.files[fi], self.fps, self.vel_scheme, return_pose=self.pose_noise)
        if self.cache is not None:
            self.cache[fi] = out
        return out

    def __getitem__(self, i):
        fi, s = self.index[i]
        N = self.window
        if self.pose_noise:
            om, tv, qc, qa = self._get_file(fi)
            return {'omega': om[s:s + N].clone(), 'transv': tv[s:s + N].clone(),
                    'q_canon': qc[s:s + N + 1].clone(), 'qalign': qa[s:s + N + 1].clone()}
        om, tv = self._get_file(fi)
        return {'omega': om[s:s + N].clone(), 'transv': tv[s:s + N].clone()}
