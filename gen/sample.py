''







import sys, argparse, math; sys.path.insert(0, '')
import torch
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
from gen.dataset import WindowMotionDataset, list_amass_files
from gen.field import WindowDistanceField
dev = 'cuda'
ROOT = 'data'


def load(ckpt):
    ck = torch.load(ckpt, map_location=dev); ar = ck['args']; window = ar.get('window', 31)
    net = WindowDistanceField(d_model=ar['d_model'], n_layers=ar['layers'], n_heads=ar.get('heads', 8),
                              ff=ar.get('ff', 1024), max_len=window + 1, cond_mask=ar.get('mask', False),
                              use_accel=(ar.get('accel', 'none') == 'joint'),
                              acc_scale=ck.get('acc_scale', 1.0),
                              per_joint_sigma=ar.get('per_joint_sigma', False),
                              root_head=ar.get('root_head', False),
                              hybrid_head=ar.get('hybrid_head', False),
                              accel_manifold=ar.get('accel_manifold', False),
                              acc_lambda=ar.get('acc_lambda', 1.0)).to(dev)
    missing, unexpected = net.load_state_dict(ck['model'], strict=False)  
    assert not unexpected and all('acc_scale' in m for m in missing), f"unexpected load mismatch: {missing} {unexpected}"
    net.eval()
    return net, ck['om_std'].to(dev), ck['tv_std'].to(dev), window      


@torch.no_grad()
def _noise(x, s):
    return x + s * torch.randn_like(x)


def anneal(net, xo, xt, sigma_start, n_levels=20, n_corrector=2, step=0.15, sigma_min=0.02,
           obs=None, langevin=True, cmask=None):
    ''



    sigmas = torch.exp(torch.linspace(math.log(sigma_start), math.log(sigma_min), n_levels))
    B = xo.shape[0]
    if cmask is not None:
        nj = net.n_joints
        keep_om = cmask[:, :, :nj].unsqueeze(-1).bool(); keep_tv = cmask[:, :, nj:].bool()
        clean_om = xo.clone(); clean_tv = xt.clone()
    pj = getattr(net, 'per_joint_sigma', False)                  
    def _sv(s): return torch.full((B, net.n_joints + 1), s, device=dev) if pj else torch.full((B,), s, device=dev)
    for sig in sigmas.tolist():
        sv = _sv(sig)
        if obs is not None:
            mo, vo, mt, vt = obs
            xo = torch.where(mo, _noise(vo, sig), xo); xt = torch.where(mt, _noise(vt, sig), xt)
        if cmask is not None:
            xo = torch.where(keep_om, clean_om, xo); xt = torch.where(keep_tv, clean_tv, xt)
        s_om, s_tv = net.score(xo, xt, sv, cmask)
        xo = xo + sig ** 2 * s_om; xt = xt + sig ** 2 * s_tv
        for _ in range(n_corrector if langevin else 0):
            sv2 = _sv(sig)
            if cmask is not None:
                xo = torch.where(keep_om, clean_om, xo); xt = torch.where(keep_tv, clean_tv, xt)
            s_om, s_tv = net.score(xo, xt, sv2, cmask)
            eps = step * sig ** 2
            xo = xo + 0.5 * eps * s_om + math.sqrt(eps) * torch.randn_like(xo)
            xt = xt + 0.5 * eps * s_tv + math.sqrt(eps) * torch.randn_like(xt)
    if obs is not None:
        mo, vo, mt, vt = obs
        xo = torch.where(mo, vo, xo); xt = torch.where(mt, vt, xt)
    if cmask is not None:
        xo = torch.where(keep_om, clean_om, xo); xt = torch.where(keep_tv, clean_tv, xt)
    return xo, xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/gen/p1_both/ckpt.pt')
    ap.add_argument('--split', default='test'); ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--sigma', type=float, default=0.5); ap.add_argument('--levels', type=int, default=20)
    ap.add_argument('--corrector', type=int, default=0)   
    a = ap.parse_args()
    net, om_std, tv_std, window = load(a.ckpt)
    files = list_amass_files(ROOT, a.split)[:300]
    ds = WindowMotionDataset(files, window=window, stride=20)
    idx = torch.randperm(len(ds))[:a.n].tolist()
    om = torch.stack([ds[i]['omega'] for i in idx]).to(dev) / om_std
    tv = torch.stack([ds[i]['transv'] for i in idx]).to(dev) / tv_std
    torch.manual_seed(0)
    xo = om + a.sigma * torch.randn_like(om); xt = tv + a.sigma * torch.randn_like(tv)
    eb = (xo - om).norm(dim=-1).mean().item()
    do, dt = anneal(net, xo.clone(), xt.clone(), a.sigma, n_levels=a.levels,
                    n_corrector=a.corrector, langevin=a.corrector > 0)
    ea = (do - om).norm(dim=-1).mean().item()
    mode = f'Langevin(c={a.corrector})' if a.corrector > 0 else 'deterministic'
    print(f"ckpt={a.ckpt} | {a.split} split | {a.n} windows | denoise sigma={a.sigma}, {a.levels} levels, {mode}", flush=True)
    print(f"  annealed denoise err: {eb:.3f} -> {ea:.3f}  ({100*(1-ea/eb):+.0f}%)", flush=True)


if __name__ == '__main__':
    main()
