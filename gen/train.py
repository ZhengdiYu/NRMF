import os, sys, glob, argparse, time, math; sys.path.insert(0, '')
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
from torch.utils.data import DataLoader
from gen.dataset import WindowMotionDataset, list_amass_files, accel_from_vel, rohm_noise_canon, FPS
from gen.field import WindowDistanceField
from utils.transforms import (quaternion_hamilton_product as qmul, quaternion_conjugate as qconj,
                              quaternion_log_map as qlog)
dev = 'cuda'
ROOT = 'data/amass_processed'
LEG = [1, 2, 4, 5, 7, 8, 10, 11]


SIGMAX_J_DEFAULT = [24, 8, 8, 24, 8, 8, 24, 8, 8, 24, 8, 8, 14, 12, 12, 14, 8, 8, 6, 6, 7, 7, 8]


ACCEL_SIGMAX_J  = [35, 15, 15, 21, 11, 13, 28, 13, 12, 37, 15, 15, 19, 16, 15, 18, 11, 11, 9, 8, 10, 8, 8]


def sample_masks(B, N, nj, mask_prob):


    obs = torch.zeros(B, N, nj); obs_tv = torch.zeros(B, N)
    legi = torch.tensor(LEG)
    r = torch.rand(B); sub = torch.rand(B)
    for b in range(B):
        if r[b] >= mask_prob:
            continue
        s = sub[b].item()
        if s < 0.35:
            obs[b] = 1.0; obs[b, :, legi] = 0.0; obs_tv[b] = 1.0
        elif s < 0.70:
            obs[b] = (torch.rand(nj) < 0.5).float().unsqueeze(0).expand(N, nj)
            obs_tv[b] = float(torch.rand(1).item() < 0.7)
        elif s < 0.85:
            gl = int(torch.randint(N // 6, N // 2 + 1, (1,)).item()); g0 = int(torch.randint(0, max(1, N - gl), (1,)).item())
            obs[b] = 1.0; obs_tv[b] = 1.0; obs[b, g0:g0 + gl] = 0.0; obs_tv[b, g0:g0 + gl] = 0.0
        else:
            obs[b] = (torch.rand(N, nj) < 0.5).float(); obs_tv[b] = float(torch.rand(1).item() < 0.5)
    return torch.cat([obs, obs_tv.unsqueeze(-1)], dim=-1)


def compute_norm(ds, n_windows=500):

    oms, tvs = [], []
    idx = torch.randperm(len(ds))[:min(n_windows, len(ds))]
    for i in idx.tolist():
        b = ds[i]; oms.append(b['omega']); tvs.append(b['transv'])
    om = torch.stack(oms); tv = torch.stack(tvs)
    om_std = om.reshape(-1, 22, 3).std(0).clamp_min(1e-4)
    tv_std = tv.reshape(-1, 3).std(0).clamp_min(1e-4)
    acc_scale = accel_from_vel(om / om_std).std().clamp_min(1e-4)
    return om_std, tv_std, acc_scale


def cycle(dl):
    while True:
        for b in dl:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT); ap.add_argument('--split', default='train')
    ap.add_argument('--max_files', type=int, default=0)
    ap.add_argument('--window', type=int, default=31); ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--d_model', type=int, default=256); ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=8); ap.add_argument('--ff', type=int, default=1024)
    ap.add_argument('--save_every', type=int, default=0)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--sw', default='none', choices=['none', 'inv'])
    ap.add_argument('--eval_every', type=int, default=1000); ap.add_argument('--val_n', type=int, default=256)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--decay', action='store_true')
    ap.add_argument('--sigma_min', type=float, default=0.01); ap.add_argument('--sigma_max', type=float, default=2.0)
    ap.add_argument('--val_ref', type=float, default=0.0)
    ap.add_argument('--objective', default='dist', choices=['dist', 'dsm', 'both'])
    ap.add_argument('--eikonal', type=float, default=0.0)
    ap.add_argument('--init_from', default='')
    ap.add_argument('--mask', action='store_true')
    ap.add_argument('--mask_prob', type=float, default=0.6)
    ap.add_argument('--sel_denoise', action='store_true')
    ap.add_argument('--accel', default='none', choices=['none', 'joint', 'only'])
    ap.add_argument('--vel_scheme', default='onesided', choices=['onesided', 'logdiff'])
    ap.add_argument('--per_joint_sigma', action='store_true')
    ap.add_argument('--pj_scalar', action='store_true')
    ap.add_argument('--pose_noise', action='store_true')
    ap.add_argument('--k_max', type=float, default=8.0)
    ap.add_argument('--root_decorr', action='store_true')
    ap.add_argument('--root_boost', type=float, default=1.0)
    ap.add_argument('--root_head', action='store_true')
    ap.add_argument('--hybrid_head', action='store_true')
    ap.add_argument('--accel_manifold', action='store_true')
    ap.add_argument('--acc_lambda', type=float, default=1.0)
    ap.add_argument('--legacy_eval_seed', action='store_true')
    ap.add_argument('--out', default='checkpoints/gen/p1_sanity'); ap.add_argument('--log_every', type=int, default=50)
    a = ap.parse_args()
    assert sum([a.pose_noise, a.per_joint_sigma, a.pj_scalar, a.hybrid_head]) <= 1, "per_joint_sigma/pose_noise/pj_scalar/hybrid_head are mutually exclusive; pick one"
    assert not (a.accel_manifold and (a.pose_noise or a.per_joint_sigma or a.pj_scalar)), "accel_manifold builds on the hybrid head; don't combine with per_joint_sigma/pose_noise/pj_scalar"
    if a.accel_manifold: assert a.accel == 'joint', "accel_manifold needs --accel joint (alpha in the input)"


    sigmax_j = torch.tensor(SIGMAX_J_DEFAULT, dtype=torch.float32, device=dev)
    if a.accel == 'only':
        sigmax_j = torch.tensor(ACCEL_SIGMAX_J, dtype=torch.float32, device=dev)
        print("using ACCEL_SIGMAX_J (acceleration-calibrated per-joint sigma) for --accel only", flush=True)

    files = list_amass_files(a.root, a.split)
    if a.max_files: files = files[:a.max_files]
    ds = WindowMotionDataset(files, window=a.window, stride=a.stride, vel_scheme=a.vel_scheme, pose_noise=a.pose_noise)
    print(f"files={len(files)} windows={len(ds)} (skipped {ds.skipped})", flush=True)
    nrm = f'gen/norm_w{a.window}.pt' if a.vel_scheme == 'onesided' else f'gen/norm_w{a.window}_{a.vel_scheme}.pt'
    if os.path.exists(nrm) and 'acc' in torch.load(nrm):
        st = torch.load(nrm); om_std, tv_std, acc_scale = st['om'], st['tv'], st['acc']; print(f"loaded cached norm {nrm}", flush=True)
    else:
        om_std, tv_std, acc_scale = compute_norm(ds); torch.save({'om': om_std, 'tv': tv_std, 'acc': acc_scale}, nrm); print(f"computed+saved norm {nrm}", flush=True)
    om_std = om_std.to(dev); tv_std = tv_std.to(dev); acc_scale = float(acc_scale)
    print(f"norm: om_std mean {om_std.mean():.3f} | tv_std {tv_std.tolist()}", flush=True)


    vds = WindowMotionDataset(list_amass_files(a.root, 'vald'), window=a.window, stride=a.stride, vel_scheme=a.vel_scheme)
    vi = torch.randperm(len(vds))[:a.val_n].tolist()
    val_om = torch.stack([vds[i]['omega'] for i in vi]).to(dev) / om_std
    val_tv = torch.stack([vds[i]['transv'] for i in vi]).to(dev) / tv_std
    if a.accel == 'only':
        val_om = accel_from_vel(val_om) / acc_scale
    print(f"val: {len(vi)} windows from vald split (held out)", flush=True)
    best = -1e9

    net = WindowDistanceField(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, ff=a.ff,
                              max_len=a.window + 1, cond_mask=a.mask,
                              use_accel=(a.accel == 'joint'), acc_scale=acc_scale,
                              per_joint_sigma=a.per_joint_sigma, root_head=a.root_head,
                              hybrid_head=a.hybrid_head, accel_manifold=a.accel_manifold,
                              acc_lambda=a.acc_lambda).to(dev); net.train()
    if a.init_from:
        ick = torch.load(a.init_from, map_location=dev); miss, unexp = net.load_state_dict(ick['model'], strict=False)
        assert not unexp and all('acc_scale' in m for m in miss), f"init_from load mismatch: {miss} {unexp}"
        print(f"init_from {a.init_from} (step {ick.get('step','?')}, val {ick.get('val_score','?')})", flush=True)
    print(f"model params: {sum(p.numel() for p in net.parameters())/1e6:.1f}M | d_model={a.d_model} layers={a.layers} window={a.window} batch={a.batch} mask={a.mask} accel={a.accel} eikonal={a.eikonal}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    legi = torch.tensor(LEG, device=dev)


    eval_sigmas = [s for s in (0.1, 0.5, 2.0, 8.0, 16.0) if s <= a.sigma_max + 1e-6] or [a.sigma_max]

    def evaluate():

        net.eval(); out = {}; _rng = torch.get_rng_state(); _crng = torch.cuda.get_rng_state()
        for sg in eval_sigmas:
            torch.manual_seed(0); eo = torch.randn_like(val_om); et = torch.randn_like(val_tv)
            xo = val_om + sg * eo; xt = val_tv + sg * et
            nvv = val_om.shape[0]
            sv = torch.full((nvv, net.n_joints + 1), sg, device=dev) if net.per_joint_sigma else torch.full((nvv,), sg, device=dev)
            so, st = net.score(xo, xt, sv)
            c = torch.nn.functional.cosine_similarity(so.reshape(len(vi), -1), (-eo).reshape(len(vi), -1), dim=-1).mean().item()
            eb = (xo - val_om).norm(dim=-1).mean(); ea = (xo + sg * sg * so - val_om).norm(dim=-1).mean()
            out[sg] = (c, 100 * (1 - (ea / eb).item()))
        if not a.legacy_eval_seed: torch.set_rng_state(_rng); torch.cuda.set_rng_state(_crng)
        net.train(); return out

    def evaluate_masked():


        net.eval(); nv = val_om.shape[0]; out = {}; _rng = torch.get_rng_state(); _crng = torch.cuda.get_rng_state()
        mask = torch.ones(nv, a.window, net.n_joints + 1, device=dev); mask[:, :, legi] = 0.0
        mom = mask[:, :, :net.n_joints].unsqueeze(-1)
        for sg in (0.3, 0.6):
            torch.manual_seed(0); eo = torch.randn_like(val_om) * (1 - mom)
            xo = val_om + sg * eo; xt = val_tv
            sv = torch.full((nv, net.n_joints + 1), sg, device=dev) if net.per_joint_sigma else torch.full((nv,), sg, device=dev)
            so, st = net.score(xo, xt, sv, mask)
            sl = so[:, :, legi]; el = eo[:, :, legi]
            c = torch.nn.functional.cosine_similarity(sl.reshape(nv, -1), (-el).reshape(nv, -1), dim=-1).mean().item()
            eb = (xo - val_om)[:, :, legi].norm(); ea = (xo + sg * sg * so - val_om)[:, :, legi].norm()
            out[sg] = (c, 100 * (1 - (ea / eb).item()))
        if not a.legacy_eval_seed: torch.set_rng_state(_rng); torch.cuda.set_rng_state(_crng)
        net.train(); return out

    gen = cycle(DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True))
    lsm = lmin = float('nan'); t0 = time.time()
    for step in range(1, a.steps + 1):
        if a.warmup or a.decay:
            if step < a.warmup:
                lr_t = a.lr * step / max(1, a.warmup)
            elif a.decay:
                prog = (step - a.warmup) / max(1, a.steps - a.warmup)
                lr_t = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
            else:
                lr_t = a.lr
            for g in opt.param_groups: g['lr'] = lr_t
        b = next(gen)
        om = (b['omega'].to(dev) / om_std)
        tv = (b['transv'].to(dev) / tv_std)
        if a.accel == 'only':
            om = accel_from_vel(om) / acc_scale
        B = om.shape[0]
        mask = None
        if a.mask:
            mask = sample_masks(B, om.shape[1], net.n_joints, a.mask_prob).to(dev)
            mom = mask[:, :, :net.n_joints].unsqueeze(-1); mtv = mask[:, :, net.n_joints:]
        if a.pose_noise:
            qc = b['q_canon'].to(dev); qa = b['qalign'].to(dev)
            k = torch.rand(B, device=dev) * a.k_max
            q_noisy = rohm_noise_canon(qc, qa, k)
            om_noisy = FPS * qlog(qmul(q_noisy[:, 1:], qconj(q_noisy[:, :-1])))
            Delta = om_noisy / om_std - om
            cap = sigmax_j[:net.n_joints].view(1, 1, net.n_joints, 1)
            Delta = torch.maximum(torch.minimum(Delta, cap), -cap)
            e_tv = torch.randn_like(tv)
            if a.mask: Delta = Delta * (1 - mom); e_tv = e_tv * (1 - mtv)
            sigma = (Delta.pow(2).mean((1, 2, 3)) + 1e-12).sqrt().clamp_min(a.sigma_min)
            sg4 = sigma.view(-1, 1, 1, 1); sg3 = sigma.view(-1, 1, 1)
            e_om = Delta / sg4
            x_om = om + Delta; x_tv = tv + sg3 * e_tv; wsig = sigma
        elif a.per_joint_sigma:
            u = torch.rand(B, 1, device=dev).expand(B, net.n_joints + 1).clone()
            if a.root_decorr: u[:, 0] = torch.rand(B, device=dev)
            sigma = torch.exp(u * (sigmax_j.log() - math.log(a.sigma_min)) + math.log(a.sigma_min))
            sg4 = sigma[:, :net.n_joints].view(B, 1, net.n_joints, 1); sg3 = sigma[:, net.n_joints:].view(B, 1, 1)
            e_om = torch.randn_like(om); e_tv = torch.randn_like(tv)
            if a.mask: e_om = e_om * (1 - mom); e_tv = e_tv * (1 - mtv)
            x_om = om + sg4 * e_om; x_tv = tv + sg3 * e_tv
            wsig = torch.exp(torch.log(sigma).mean(-1))
        elif a.pj_scalar:
            u = torch.rand(B, 1, device=dev).expand(B, net.n_joints + 1).clone()
            sigma_pj = torch.exp(u * (sigmax_j.log() - math.log(a.sigma_min)) + math.log(a.sigma_min))
            sg4 = sigma_pj[:, :net.n_joints].view(B, 1, net.n_joints, 1); sg3 = sigma_pj[:, net.n_joints:].view(B, 1, 1)
            e_om = torch.randn_like(om); e_tv = torch.randn_like(tv)
            if a.mask: e_om = e_om * (1 - mom); e_tv = e_tv * (1 - mtv)
            x_om = om + sg4 * e_om; x_tv = tv + sg3 * e_tv
            wsig = torch.exp(torch.log(sigma_pj).mean(-1)); sigma = wsig
        elif a.hybrid_head or a.accel_manifold:
            u = torch.rand(B, 1, device=dev).expand(B, net.n_joints + 1).clone()
            sigma = torch.exp(u * (sigmax_j.log() - math.log(a.sigma_min)) + math.log(a.sigma_min))
            sg4 = sigma[:, :net.n_joints].view(B, 1, net.n_joints, 1); sg3 = sigma[:, net.n_joints:].view(B, 1, 1)
            e_om = torch.randn_like(om); e_tv = torch.randn_like(tv)
            if a.mask: e_om = e_om * (1 - mom); e_tv = e_tv * (1 - mtv)
            x_om = om + sg4 * e_om; x_tv = tv + sg3 * e_tv
            wsig = torch.exp(torch.log(sigma).mean(-1))
        else:
            logs = torch.rand(B, device=dev) * (torch.log(torch.tensor(a.sigma_max)) - torch.log(torch.tensor(a.sigma_min))) + torch.log(torch.tensor(a.sigma_min))
            sigma = logs.exp(); e_om = torch.randn_like(om); e_tv = torch.randn_like(tv)
            if a.mask: e_om = e_om * (1 - mom); e_tv = e_tv * (1 - mtv)
            sg4 = sigma.view(-1, 1, 1, 1); sg3 = sigma.view(-1, 1, 1)
            x_om = om + sg4 * e_om; x_tv = tv + sg3 * e_tv; wsig = sigma

        jw = torch.ones(net.n_joints + 1, device=dev)
        if a.per_joint_sigma: jw[0] = a.root_boost
        if a.hybrid_head or a.accel_manifold:
            tgt_leg = (sg4 * e_om)[:, :, net.leg_idx, :].norm(dim=-1)
            nlp = (sg4 * e_om)[:, :, net.nonleg_idx, :]
            tgt_coh = torch.sqrt(nlp.pow(2).sum((2, 3)) + (sg3 * e_tv).pow(2).sum(2) + 1e-8)
            tgt = torch.cat([tgt_leg, tgt_coh.unsqueeze(-1)], dim=-1)
            if a.accel_manifold:
                an = accel_from_vel(sg4 * e_om) / acc_scale
                atgt_leg = an[:, :, net.leg_idx, :].norm(dim=-1)
                atgt_coh = an[:, :, net.nonleg_idx, :].pow(2).sum((2, 3)).add(1e-8).sqrt()
                tgt = torch.cat([tgt, atgt_leg, atgt_coh.unsqueeze(-1)], dim=-1)
            d_pred = net.distance(x_om, x_tv, sigma, mask)
            ld_err = ((d_pred - tgt) ** 2).mean((1, 2))
        elif a.per_joint_sigma:
            tgt = torch.cat([(sg4 * e_om).norm(dim=-1), (sg3 * e_tv).norm(dim=-1, keepdim=True)], dim=-1)
            d_pred = net.distance(x_om, x_tv, sigma, mask)
            ld_err = (((d_pred - tgt) ** 2) * jw).mean((1, 2))
        else:
            tgt = torch.sqrt((sg4 * e_om).pow(2).sum((2, 3)) + (sg3 * e_tv).pow(2).sum(2) + 1e-8)
            d_pred = net.distance(x_om, x_tv, sigma, mask)
            ld_err = ((d_pred - tgt) ** 2).mean(1)
        w = (1.0 / wsig).clamp(max=20.0) if a.sw == 'inv' else torch.ones_like(wsig)
        w = w / w.mean()
        wv = (a.val_ref / wsig).clamp(max=1.0).pow(2) if a.val_ref > 0 else torch.ones_like(wsig)
        ld = wv * ld_err
        if a.objective == 'dist':
            loss = (w * ld).mean()
        else:
            s_om, s_tv = (net.score_vel if a.accel_manifold else net.score)(x_om, x_tv, sigma, mask)
            jwo = jw[:net.n_joints].view(1, 1, net.n_joints, 1)
            ls = (((sg4 * s_om + e_om) ** 2) * jwo).mean((1, 2, 3)) + ((sg3 * s_tv + e_tv) ** 2).mean((1, 2))
            loss = (w * (ld + ls if a.objective == 'both' else ls)).mean()
        l_eik = None
        if a.eikonal > 0:
            xg_om = x_om.detach().requires_grad_(True); xg_tv = x_tv.detach().requires_grad_(True)
            dg = net.distance(xg_om, xg_tv, sigma, mask)
            g_om, g_tv = torch.autograd.grad(dg.sum(), [xg_om, xg_tv], create_graph=True)
            gn = torch.sqrt(g_om.pow(2).sum((2, 3)) + g_tv.pow(2).sum(2) + 1e-8)
            l_eik = ((gn - 1.0) ** 2).mean()
            loss = loss + a.eikonal * l_eik
        opt.zero_grad(); loss.backward()
        if a.clip > 0: torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step()
        lsm = loss.item() if step == 1 else 0.98 * lsm + 0.02 * loss.item()
        lmin = lsm if step == 1 else min(lmin, lsm)
        if step % a.log_every == 0 or step == 1:
            eik_s = f" | eik {l_eik.item():.3f} ||gradd|| {gn.mean().item():.2f}" if l_eik is not None else ""
            print(f"step {step:6d} | loss {loss.item():.4f} | ema {lsm:.4f} | d {d_pred.mean().item():.3f} vs tgt {tgt.mean().item():.3f}{eik_s} | {step/(time.time()-t0):.1f} it/s", flush=True)
        if a.save_every and step % a.save_every == 0:
            os.makedirs(a.out, exist_ok=True)
            torch.save({'model': net.state_dict(), 'om_std': om_std.cpu(), 'tv_std': tv_std.cpu(), 'acc_scale': acc_scale, 'args': vars(a)}, f'{a.out}/ckpt.pt')
        if a.eval_every and step % a.eval_every == 0:
            m = evaluate()
            msg = "denoise " + " ".join(f"s{sg}:cos{c:+.2f}/dn{d:+.0f}%" for sg, (c, d) in m.items())
            dn_sel = 100 * sum(c for c, _ in m.values())
            if a.mask:
                mm = evaluate_masked(); rec_sel = sum(d for _, d in mm.values())
                msg += " | recover " + " ".join(f"s{sg}:cos{c:+.2f}/leg-dn{d:+.0f}%" for sg, (c, d) in mm.items())
                sel = dn_sel if a.sel_denoise else rec_sel
            else:
                sel = dn_sel
            star = ""
            if sel > best:
                best = sel; star = " *BEST*"; os.makedirs(a.out, exist_ok=True)
                torch.save({'model': net.state_dict(), 'om_std': om_std.cpu(), 'tv_std': tv_std.cpu(), 'acc_scale': acc_scale,
                            'args': vars(a), 'step': step, 'val_score': sel}, f'{a.out}/best.pt')
            print(f"  [VAL {step}] {msg} | sel {sel:+.0f}{star}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({'model': net.state_dict(), 'om_std': om_std.cpu(), 'tv_std': tv_std.cpu(), 'acc_scale': acc_scale,
                'args': vars(a)}, f'{a.out}/ckpt.pt')
    print(f"\nDONE | ema loss {lsm:.4f} (min {lmin:.4f}) | saved {a.out}/ckpt.pt", flush=True)


if __name__ == '__main__':
    main()
