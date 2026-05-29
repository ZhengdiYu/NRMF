import torch
import numpy as np


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    ''











    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )


    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_hamilton_product(a, b):

    ndim = len(a.shape)

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    res = torch.concat((
        (aw * bw - ax * bx - ay * by - az * bz).unsqueeze(-1),
        (aw * bx + ax * bw + ay * bz - az * by).unsqueeze(-1),
        (aw * by - ax * bz + ay * bw + az * bx).unsqueeze(-1),
        (aw * bz + ax * by - ay * bx + az * bw).unsqueeze(-1)
    ), dim=ndim-1)

    return res

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''










    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)


    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def batch_rodrigues_np(aa):
    ''
    if torch.is_tensor(aa):
        aa = aa.numpy()

    batch_size = aa.shape[0]
    rotation_matrices = []

    for i in range(batch_size):
        angle = np.linalg.norm(aa[i])
        if angle < 1e-8:

            rot_mat = np.eye(3)
        else:
            axis = aa[i] / angle
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)


            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])

            rot_mat = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

        rotation_matrices.append(rot_mat)

    return np.array(rotation_matrices)

def rotation_matrix_to_quaternion(
    rotation_matrix: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    ''









    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    rotation_matrix_vec: torch.Tensor = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt((trace + 1.0).clamp_min(eps)) * 2.0
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1():
        sq = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(eps)) * 2.0
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2():
        sq = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(eps)) * 2.0
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3():
        sq = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(eps)) * 2.0
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion






def normalize_quaternion(q):
    ''



    norm = torch.norm(q, dim=-1, keepdim=True)

    norm = torch.clamp(norm, min=1e-8)
    return q / norm

def normalize_quaternion_np(q):
    ''



    norm = np.linalg.norm(q, axis=-1, keepdims=True)

    norm = np.clamp(norm, a_min=1e-8, a_max=None)
    return q / norm

def is_normalized(q, tolerance=1e-6):
    ''









    if isinstance(q, torch.Tensor):
        norm = torch.norm(q, dim=-1)
        return torch.all(torch.abs(norm - 1.0) <= tolerance)
    else:
        norm = np.linalg.norm(q, axis=-1)
        return np.all(np.abs(norm - 1.0) <= tolerance)

def quaternion_conjugate(q):
    ''



    q_conj = q.clone()
    q_conj[..., 1:] *= -1
    return q_conj

def apply_canonical_transform(poses_mat, cur_align_rot, invert=False):
    ''









    B, J, _, _ = poses_mat.shape
    assert cur_align_rot.shape == (3, 3)

    root_orient_mat = poses_mat[:, 0, :, :]
    body_orient_mat = poses_mat[:, 1:, :, :]

    if invert:
        raise ValueError('Not implemented')
        cur_align_rot = np.linalg.inv(cur_align_rot)

    alinged_root_orient_matrix = np.matmul(cur_align_rot, root_orient_mat.copy())
    aligned_poses_mat = np.concatenate([alinged_root_orient_matrix[:, np.newaxis, :, :], body_orient_mat], axis=1)

    return aligned_poses_mat

def compute_quaternion_transition(pose_prev, pose_curr, normalize=False):
    ''











    if normalize:
        pose_prev = normalize_quaternion(pose_prev)
        pose_curr = normalize_quaternion(pose_curr)


    pose_prev_conj = quaternion_conjugate(pose_prev)
    transition = quaternion_hamilton_product(pose_curr, pose_prev_conj)

    return transition

def quaternion_log_map(q, eps=1e-8):
    ''












    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    w = q[..., :1]
    v = q[..., 1:]

    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))
    w = w * sign
    v = v * sign
    v_norm = v.norm(dim=-1, keepdim=True)
    theta = 2.0 * torch.atan2(v_norm, w.clamp(-1.0, 1.0))

    small = v_norm < eps
    factor = torch.where(small, 2.0 / w.clamp_min(eps), theta / v_norm.clamp_min(eps))
    return v * factor


def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    denominator = denominator.clone()
    denominator[denominator.abs() < eps] += eps
    return numerator / denominator
