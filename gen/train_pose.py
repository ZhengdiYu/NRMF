import os, sys, argparse, time, math; sys.path.insert(0, '')
import torch
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
from torch.utils.data import DataLoader
from gen.dataset import PoseDataset, list_amass_files
from gen.field import PoseField
dev = 'cuda'
ROOT = 'data/amass_processed'


def cycle(dl):
    while True:
        for b in dl: yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT); ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--batch', type=int, default=512); ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--d_model', type=int, default=512); ap.add_argument('--layers', type=int, default=5)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--clip', type=float, default=0.5); ap.add_argument('--decay', action='store_true', default=True)
    ap.add_argument('--sigma_min', type=float, default=0.01); ap.add_argument('--sigma_max', type=float, default=2.0)
    ap.add_argument('--objective', default='both', choices=['dist', 'dsm', 'both'])
    ap.add_argument('--eikonal', type=float, default=0.0)
    ap.add_argument('--init_from', default='')
    ap.add_argument('--eval_every', type=int, default=1000); ap.add_argument('--val_n', type=int, default=2048)
    ap.add_argument('--log_every', type=int, default=1000); ap.add_argument('--out', default='checkpoints/gen/pose1')
    a = ap.parse_args()

    ds = PoseDataset(list_amass_files(a.root, 'train'), stride=a.stride)
    print(f"train poses: {len(ds)} (skipped {ds.skipped})", flush=True)
    nrm = 'gen/norm_pose.pt'
    if os.path.exists(nrm):
        pose_std = torch.load(nrm)['pose']; print(f"loaded cached norm {nrm}", flush=True)
    else:
        idx = torch.randperm(len(ds))[:20000].tolist()
        ps = torch.stack([ds[i]['pose'] for i in idx])
        pose_std = ps.reshape(-1, 21, 3).std(0).clamp_min(1e-4)
        torch.save({'pose': pose_std}, nrm); print(f"computed+saved norm {nrm}", flush=True)
    pose_std = pose_std.to(dev)
    print(f"norm: pose_std mean {pose_std.mean():.3f}", flush=True)

    vds = PoseDataset(list_amass_files(a.root, 'vald'), stride=a.stride)
    vi = torch.randperm(len(vds))[:a.val_n].tolist()
    val = torch.stack([vds[i]['pose'] for i in vi]).to(dev) / pose_std
    print(f"val: {len(vi)} poses (held out)", flush=True); best = -1e9

    net = PoseField(d_model=a.d_model, n_layers=a.layers).to(dev); net.train()
    if a.init_from:
        ick = torch.load(a.init_from, map_location=dev); net.load_state_dict(ick['model'])
        print(f"init_from {a.init_from} (step {ick.get('step','?')}, val {ick.get('val','?')})", flush=True)
    print(f"PoseField params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M eikonal={a.eikonal}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def evaluate():
        net.eval(); out = {}; _r = torch.get_rng_state(); _c = torch.cuda.get_rng_state()
        for sg in (0.1, 0.5):
            torch.manual_seed(0); e = torch.randn_like(val); x = val + sg * e
            s = net.score(x, torch.full((val.shape[0],), sg, device=dev))
            c = torch.nn.functional.cosine_similarity(s.reshape(len(vi), -1), (-e).reshape(len(vi), -1), -1).mean().item()
            eb = (x - val).norm(dim=-1).mean(); ea = (x + sg * sg * s - val).norm(dim=-1).mean()
            out[sg] = (c, 100 * (1 - (ea / eb).item()))
        torch.set_rng_state(_r); torch.cuda.set_rng_state(_c); net.train(); return out

    gen = cycle(DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=6, drop_last=True))
    lsm = float('nan'); t0 = time.time()
    for step in range(1, a.steps + 1):
        if step < a.warmup: lr_t = a.lr * step / max(1, a.warmup)
        else: lr_t = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup)))) if a.decay else a.lr
        for g in opt.param_groups: g['lr'] = lr_t
        pose = (next(gen)['pose'].to(dev) / pose_std)
        B = pose.shape[0]
        logs = torch.rand(B, device=dev) * (math.log(a.sigma_max) - math.log(a.sigma_min)) + math.log(a.sigma_min)
        sigma = logs.exp(); sg3 = sigma.view(-1, 1, 1)
        e = torch.randn_like(pose); x = pose + sg3 * e
        tgt = torch.sqrt((sg3 * e).pow(2).sum((1, 2)) + 1e-8)
        d_pred = net.distance(x, sigma)
        ld = (d_pred - tgt) ** 2
        if a.objective == 'dist': loss = ld.mean()
        else:
            s = net.score(x, sigma); ls = ((sg3 * s + e) ** 2).mean((1, 2))
            loss = (ld + ls if a.objective == 'both' else ls).mean()
        gn_log = 0.0
        if a.eikonal > 0:
            xg = x.detach().requires_grad_(True); dg = net.distance(xg, sigma)
            gx = torch.autograd.grad(dg.sum(), xg, create_graph=True)[0]
            gn = torch.sqrt(gx.pow(2).sum((1, 2)) + 1e-8); l_eik = ((gn - 1.0) ** 2).mean()
            loss = loss + a.eikonal * l_eik; gn_log = gn.mean().item()
        opt.zero_grad(); loss.backward()
        if a.clip > 0: torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step()
        lsm = loss.item() if step == 1 else 0.98 * lsm + 0.02 * loss.item()
        if step % a.log_every == 0 or step == 1:
            es = f" | ||gradd|| {gn_log:.2f}" if a.eikonal > 0 else ""
            print(f"step {step:6d} | loss {loss.item():.4f} | ema {lsm:.4f} | d {d_pred.mean():.3f} vs tgt {tgt.mean():.3f}{es} | {step/(time.time()-t0):.1f} it/s", flush=True)
        if a.eval_every and step % a.eval_every == 0:
            m = evaluate(); sel = sum(d for _, d in m.values())
            star = ""
            if sel > best:
                best = sel; star = " *BEST*"; os.makedirs(a.out, exist_ok=True)
                torch.save({'model': net.state_dict(), 'pose_std': pose_std.cpu(), 'args': vars(a), 'step': step, 'val': sel}, f'{a.out}/best.pt')
            print(f"  [VAL {step}] " + " ".join(f"s{sg}:cos{c:+.2f}/dn{d:+.0f}%" for sg, (c, d) in m.items()) + f" | sel {sel:+.0f}{star}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({'model': net.state_dict(), 'pose_std': pose_std.cpu(), 'args': vars(a)}, f'{a.out}/ckpt.pt')
    print(f"\nDONE | ema {lsm:.4f} | saved {a.out}/ckpt.pt", flush=True)


if __name__ == '__main__':
    main()
