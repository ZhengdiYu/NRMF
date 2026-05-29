"""NRMF motion recovery (leg in-filling).

The leg joints are treated as unobserved; NRMF infers them from the observed upper body
and root, using the motion prior. Writes results/recover_<model>.gif with three columns:

    input (legs occluded)  |  ground truth  |  NRMF recovered

Three recovery models are provided (see README):
    combo          hybrid field + in-loop pose prior + post-hoc contact grounding   (default)
    proj_hybrid    hybrid field + in-loop guidance + RMF-integrator pose projection
    combo_perjoint  per-joint field + in-loop pose guidance + post-hoc contact grounding

Run:  python recover.py --model combo
"""
import _bootstrap  # noqa: F401  (must be first)
import argparse
import numpy as np
import torch
from gen.dataset import process_npz
from gen.sample import load
from utils.transforms import axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul
from kinematics import fk_joints, FEET, LEG
from acceleration_prior import dev, BMP, ACC
from contacts import load_net, predicted_contact
from contact_refinement import integ_full, contact_refine
from pose_refinement import load_pose, pose_refine
from guidance import anneal_guided
from integration import integ_projected
from sampling import bare_om
from humor.body_model.body_model import BodyModel
from _render import make_panels, save_gif

POSE = 'checkpoints/pose.pt'
POSE_EIK = 'checkpoints/pose_eik.pt'
MODELS = {
    'combo':         'checkpoints/field_hybrid.pt',
    'proj_hybrid':   'checkpoints/field_hybrid.pt',
    'combo_perjoint': 'checkpoints/field_perjoint.pt',
}
SEQS = [
    ('dribblekick', 'data/SSM_synced/20160930_50032/dribble_kick_sync_poses_141_frames_30_fps.npz'),
    ('airkick',     'data/Transitions_mocap/mazen_c3d/airkick_running_poses_217_frames_30_fps.npz'),
    ('crouchwalk',  'data/Transitions_mocap/mazen_c3d/crouchwalk_crawl_poses_261_frames_30_fps.npz'),
    ('walk',        'data/SSM_synced/20160330_03333/walking_01_poses_143_frames_30_fps.npz'),
]
LABELS = ['input (legs occluded)', 'ground truth', 'NRMF recovered']
COLORS = [[.7, .7, .72], [.45, .8, .45], [.30, .55, .95]]
ALPHA = 0.005     # RMF-integrator projection step
W_FLOOR = 15.0    # floor-penetration weight in the leg refinement


def recover_seq(model, field, om_std, tv_std, W, om, tv, clean_q, qa, betas, bm, transT, floor,
                acc, accsc, pf, ps, pfe, pse, bodyi, cnet, cnetom):
    legi = torch.tensor(LEG, device=dev)
    obsj = torch.tensor([j for j in range(22) if j not in LEG], device=dev)
    cmask = torch.ones(1, W, field.n_joints + 1, device=dev)
    cmask[:, :, legi] = 0.0
    om_n = (om[:W].unsqueeze(0).to(dev) / om_std)
    tv_n = (tv[:W].unsqueeze(0).to(dev) / tv_std)

    def seed():
        torch.manual_seed(0)
        x = om_n.clone()
        x[:, :, legi] = torch.randn_like(x[:, :, legi])
        return x

    gk = dict(q_clean=clean_q, om_std=om_std, betas=betas, bm=bm, transT=transT, bodyi=bodyi,
              pf=pf, pose_std=ps, tail_sigma=0.5, guide_iters=4)

    # bare prior recovery -> predicted foot contacts (shared)
    omb = bare_om(field, om_std, tv_std, om, tv, clean_q, cmask, legi, W, acc, accsc)
    qb = integ_full(omb, clean_q, legi, om_std, W)
    qcb = qb.clone(); qcb[:, 0] = qmul(qa, qb[:, 0])
    pred = predicted_contact(cnet, qcb, cnetom)

    if model in ('combo', 'combo_perjoint'):
        q_inp = anneal_guided(field, acc, accsc, seed(), tv_n.clone(), cmask, legi,
                              gt_low0=None, guide_pose=1.0, guide_contact=0.0, **gk)
        qcp = q_inp.clone(); qcp[:, 0] = qmul(qa, q_inp[:, 0])
        predp = predicted_contact(cnet, qcp, cnetom)
        return pose_refine(q_inp, legi, betas, bm, transT, pose_src='field', pf=pf, pose_std=ps,
                           bodyi=bodyi, w_pose=0.0, gt_low0=predp, w_contact=1.0, w_floor=W_FLOOR, floor=floor)
    if model == 'proj_hybrid':
        # COMBO base (in-loop pose, no in-loop contact) -> RMF-integrator with the eikonal
        # pose field + self-scaling (distance-magnitude) step -> post-hoc contact + floor
        om_com = anneal_guided(field, acc, accsc, seed(), tv_n.clone(), cmask, legi,
                               gt_low0=None, guide_pose=1.0, guide_contact=0.0, return_om=True, **gk)
        q_proj = integ_projected(om_com, clean_q, legi, om_std, W, pf=pfe, pose_std=pse, bodyi=bodyi,
                                 alpha=ALPHA, selfscale=True)
        qcp = q_proj.clone(); qcp[:, 0] = qmul(qa, q_proj[:, 0])
        predp = predicted_contact(cnet, qcp, cnetom)
        return contact_refine(q_proj, legi, obsj, betas, bm, transT, gt_low0=predp,
                              w_floor=W_FLOOR, floor=floor)[0]
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='combo', choices=list(MODELS))
    a = ap.parse_args()
    field, om_std, tv_std, W = load(MODELS[a.model])
    acc, _, _, _ = load(ACC)
    accsc = float(acc.acc_scale)
    W1 = W + 1
    legi = torch.tensor(LEG, device=dev)
    bodyi = torch.arange(1, 22, device=dev)
    pf, ps = load_pose(POSE)
    pfe, pse = load_pose(POSE_EIK)
    cnet, cnetom, _ = load_net()
    faces = np.load(BMP)['f'].astype(np.int64)

    rows = []
    for name, path in SEQS:
        d = np.load(path, allow_pickle=True)
        betas = d['betas']
        aa = np.concatenate([d['root_orient'].reshape(-1, 1, 3), d['pose_body'].reshape(-1, 21, 3)], 1)
        clean_q = aa2q(torch.tensor(aa, dtype=torch.float32, device=dev))
        om, tv, _, qalign = process_npz(path, return_pose=True)
        qa = qalign[:W1].to(dev)
        bm = BodyModel(bm_path=BMP, num_betas=16, batch_size=W1).to(dev)
        tr = np.asarray(d['trans'][:W1], dtype=np.float32)
        tr = tr - tr[0]
        transT = torch.tensor(tr, device=dev)
        floor_frames = 61 if a.model == 'combo_perjoint' else W1
        floor = float(fk_joints(clean_q[:W1], betas, bm, transT)[:floor_frames, FEET, 2].min().detach())

        recovered = recover_seq(a.model, field, om_std, tv_std, W, om, tv, clean_q, qa, betas, bm,
                                transT, floor, acc, accsc, pf, ps, pfe, pse, bodyi, cnet, cnetom)

        # visualize the input as the observed body with legs frozen at frame 0 (occluded)
        q_occ = clean_q[:W1].clone()
        q_occ[:, legi] = clean_q[0:1, legi]
        rows.append(make_panels(name, [q_occ, clean_q[:W1], recovered], LABELS, COLORS, betas, faces, bm, tr, W1))
        print(f"  [{name}] done", flush=True)

    out = save_gif(rows, faces, f'results/recover_{a.model}.gif', W1)
    print(f"saved {out}", flush=True)


if __name__ == '__main__':
    main()
