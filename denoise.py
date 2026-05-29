"""NRMF motion denoising.

Recovers clean full-body motion from a noisy input pose sequence (all joints observed
and perturbed). Writes results/denoise.gif with three columns per sequence:

    input (noisy)  |  ground truth  |  NRMF denoised

Run:  python denoise.py
"""
import _bootstrap  # noqa: F401  (must be first)
import numpy as np
import torch
from gen.dataset import process_npz
from gen.sample import load
from gen.optimize import make_contact
from utils.transforms import axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul
from kinematics import fk_joints, FEET
from acceleration_prior import dev, BMP, ACC
from contacts import load_net, predicted_contact
from noise import rohm_noise_aa
from pose_refinement import load_pose
from denoising import denoise_full, SG
from humor.body_model.body_model import BodyModel
from _render import make_panels, save_gif

FIELD = 'checkpoints/field_hybrid.pt'
POSE = 'checkpoints/pose.pt'
SEQS = [
    ('dribblekick', 'data/SSM_synced/20160930_50032/dribble_kick_sync_poses_141_frames_30_fps.npz'),
    ('airkick',     'data/Transitions_mocap/mazen_c3d/airkick_running_poses_217_frames_30_fps.npz'),
    ('crouchwalk',  'data/Transitions_mocap/mazen_c3d/crouchwalk_crawl_poses_261_frames_30_fps.npz'),
    ('walk',        'data/SSM_synced/20160330_03333/walking_01_poses_143_frames_30_fps.npz'),
]
LABELS = ['input (noisy)', 'ground truth', 'NRMF denoised']
COLORS = [[.75, .5, .5], [.45, .8, .45], [.30, .55, .95]]
NOISE_LEVEL = 5  # per-frame pose perturbation


def main():
    field, om_std, tv_std, W = load(FIELD)
    acc, _, _, _ = load(ACC)
    accsc = float(acc.acc_scale)
    W1 = W + 1
    bodyi = torch.arange(1, 22, device=dev)
    pf, ps = load_pose(POSE)
    cnet, cnetom, _ = load_net()
    sgv = torch.full((1, field.n_joints + 1), SG, device=dev)
    faces = np.load(BMP)['f'].astype(np.int64)
    common = dict(w_pose=0.5, w_vel=0.5, sig_schedule='anneal', sig_start=2.0, sig_min=0.5)

    rows = []
    for name, path in SEQS:
        d = np.load(path, allow_pickle=True)
        betas = d['betas']
        aa = np.concatenate([d['root_orient'].reshape(-1, 1, 3), d['pose_body'].reshape(-1, 21, 3)], 1)
        clean_q = aa2q(torch.tensor(aa, dtype=torch.float32, device=dev))[:W1]
        om, tv, _, qalign = process_npz(path, return_pose=True)
        qa = qalign[:W1].to(dev)
        bm = BodyModel(bm_path=BMP, num_betas=16, batch_size=W1).to(dev)
        tr = np.asarray(d['trans'][:W1], dtype=np.float32)
        tr = tr - tr[0]
        transT = torch.tensor(tr, device=dev)
        fk = lambda q: fk_joints(q, betas, bm, transT)
        floor = float(fk(clean_q)[:, FEET, 2].min().detach())
        tvn = (tv[:W].unsqueeze(0).to(dev) / tv_std)
        noisy_q = aa2q(torch.tensor(rohm_noise_aa(aa[:W1], NOISE_LEVEL, np.random.default_rng(0)),
                                    device=dev, dtype=torch.float32))

        # pass 1: pose-only denoise to predict foot contacts; pass 2: full stack + contact + floor
        q0 = denoise_full(noisy_q, qa, field, om_std, tvn, sgv, acc, accsc, pf, ps, bodyi, contact=None, **common)
        qc = q0.clone()
        qc[:, 0] = qmul(qa, q0[:, 0])
        pred = predicted_contact(cnet, qc, cnetom)
        ctx = make_contact(q0, q0, fk, gt_low0=pred)
        denoised = denoise_full(noisy_q, qa, field, om_std, tvn, sgv, acc, accsc, pf, ps, bodyi,
                                contact=ctx, w_contact=1.0, w_floor=5.0, floor=floor, **common)

        rows.append(make_panels(name, [noisy_q, clean_q, denoised], LABELS, COLORS, betas, faces, bm, tr, W1))
        print(f"  [{name}] done", flush=True)

    out = save_gif(rows, faces, 'results/denoise.gif', W1)
    print(f"saved {out}", flush=True)


if __name__ == '__main__':
    main()
