import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def geo_idx(a, b, idx):
    return torch.rad2deg(2 * torch.arccos((a[:, idx] * b[:, idx]).sum(-1).abs().clamp(max=1 - 1e-7))).mean().item()


def rohm_noise_aa(body_aa_np, k, rng):
    flat = body_aa_np.reshape(-1, 3)
    eul = Rot.from_rotvec(flat).as_euler('zxy', degrees=True) + rng.normal(0.0, k, (flat.shape[0], 3))
    return Rot.from_euler('zxy', eul, degrees=True).as_rotvec().reshape(body_aa_np.shape).astype(np.float32)
