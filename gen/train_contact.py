import os, sys, argparse, time, math, glob, re; sys.path.insert(0, '')
import numpy as np, torch
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
from torch.utils.data import Dataset, DataLoader
from gen.dataset import process_npz, list_amass_files, parse_nframes, FPS
from gen.contact_net import ContactNet
from utils.transforms import (axis_angle_to_quaternion as aa2q, quaternion_hamilton_product as qmul,
                              quaternion_conjugate as qconj, quaternion_log_map as qlog)
dev = 'cuda'
ROOT = 'data/amass_processed'
FEET = [7, 8, 10, 11]


class ContactDataset(Dataset):

    def __init__(self, files, window=90, stride=10, fps=FPS, max_files=None):
        self.files = list(files)[:max_files] if max_files else list(files)
        self.window = window; self.stride = stride; self.fps = fps; self.cache = {}; self.index = []; skipped = 0
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

    def __len__(self): return len(self.index)

    def _get(self, fi):
        if fi in self.cache: return self.cache[fi]
        _, _, qc, _ = process_npz(self.files[fi], self.fps, 'onesided', return_pose=True)
        c = np.asarray(np.load(self.files[fi], allow_pickle=True)['contacts'])[:, FEET]
        c = torch.tensor(c, dtype=torch.float32)
        out = (qc, c); self.cache[fi] = out; return out

    def __getitem__(self, i):
        fi, s = self.index[i]; N = self.window; qc, c = self._get(fi)
        return {'q': qc[s:s + N + 1].clone(), 'y': c[s:s + N + 1].clone()}


def add_noise(q, k):

    eps = k * torch.randn(q.shape[0], q.shape[1], 22, 3, device=q.device)
    qn = qmul(aa2q(eps), q)
    return qn / qn.norm(dim=-1, keepdim=True)


def derive_om(q, om_std):

    om = FPS * qlog(qmul(q[:, 1:], qconj(q[:, :-1])))
    omf = torch.cat([torch.zeros_like(om[:, :1]), om], dim=1)
    return omf / om_std


def cycle(dl):
    while True:
        for b in dl: yield b


def f1(pred, y):
    tp = (pred * y).sum(); fp = (pred * (1 - y)).sum(); fn = ((1 - pred) * y).sum()
    return (2 * tp / (2 * tp + fp + fn + 1e-9)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT); ap.add_argument('--window', type=int, default=90); ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--d_model', type=int, default=256); ap.add_argument('--layers', type=int, default=4); ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--warmup', type=int, default=2000); ap.add_argument('--decay', action='store_true', default=True)
    ap.add_argument('--noise_max', type=float, default=0.15)
    ap.add_argument('--eval_every', type=int, default=1000); ap.add_argument('--val_n', type=int, default=2048)
    ap.add_argument('--log_every', type=int, default=500); ap.add_argument('--out', default='checkpoints/gen/contact_net')
    ap.add_argument('--max_files', type=int, default=None)
    a = ap.parse_args()

    om_std = torch.load(f'gen/norm_w{a.window}.pt')['om'].to(dev) if os.path.exists(f'gen/norm_w{a.window}.pt') else torch.ones(22, 3, device=dev)
    print(f"om_std loaded (mean {om_std.mean():.3f})", flush=True)
    ds = ContactDataset(list_amass_files(a.root, 'train'), window=a.window, stride=a.stride, max_files=a.max_files)
    print(f"train windows: {len(ds)} (skipped {ds.skipped} files)", flush=True)
    vds = ContactDataset(list_amass_files(a.root, 'vald'), window=a.window, stride=a.stride, max_files=a.max_files)
    vi = torch.randperm(len(vds))[:a.val_n].tolist()
    vq = torch.stack([vds[i]['q'] for i in vi]).to(dev); vy = torch.stack([vds[i]['y'] for i in vi]).to(dev)
    p = vy.mean().item(); pos_weight = torch.tensor((1 - p) / max(p, 1e-3), device=dev)
    print(f"val: {len(vi)} windows | positive contact fraction {p:.2f} -> pos_weight {pos_weight.item():.2f}", flush=True)

    net = ContactNet(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, max_len=a.window + 8).to(dev); net.train()
    print(f"ContactNet params: {sum(pp.numel() for pp in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr); bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def evaluate():
        net.eval(); out = {}; _r = torch.get_rng_state(); _c = torch.cuda.get_rng_state()
        for k in (0.0, a.noise_max):
            torch.manual_seed(0); qk = add_noise(vq, k) if k > 0 else vq
            pred = net.predict(qk, derive_om(qk, om_std))
            out[k] = f1(pred, vy)
        torch.set_rng_state(_r); torch.cuda.set_rng_state(_c); net.train(); return out

    gen = cycle(DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True))
    lsm = float('nan'); t0 = time.time(); best = -1.0
    for step in range(1, a.steps + 1):
        if step < a.warmup: lr_t = a.lr * step / max(1, a.warmup)
        else: lr_t = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup)))) if a.decay else a.lr
        for g in opt.param_groups: g['lr'] = lr_t
        b = next(gen); q0 = b['q'].to(dev); y = b['y'].to(dev); B = q0.shape[0]
        k = torch.rand(B, 1, 1, 1, device=dev) * a.noise_max
        q = add_noise(q0, k); logits = net(q, derive_om(q, om_std))
        loss = bce(logits, y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        lsm = loss.item() if step == 1 else 0.98 * lsm + 0.02 * loss.item()
        if step % a.log_every == 0 or step == 1:
            print(f"step {step:6d} | loss {loss.item():.4f} | ema {lsm:.4f} | {step/(time.time()-t0):.1f} it/s", flush=True)
        if a.eval_every and step % a.eval_every == 0:
            m = evaluate(); sel = m[0.0] + m[a.noise_max]; star = ""
            if sel > best:
                best = sel; star = " *BEST*"; os.makedirs(a.out, exist_ok=True)
                torch.save({'model': net.state_dict(), 'om_std': om_std.cpu(), 'args': vars(a), 'step': step, 'val_f1': sel}, f'{a.out}/best.pt')
            print(f"  [VAL {step}] F1 clean {m[0.0]:.3f} | noised(k{a.noise_max}) {m[a.noise_max]:.3f} | sel {sel:.3f}{star}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({'model': net.state_dict(), 'om_std': om_std.cpu(), 'args': vars(a)}, f'{a.out}/ckpt.pt')
    print(f"\nDONE | ema {lsm:.4f} | saved {a.out}/ckpt.pt", flush=True)


if __name__ == '__main__':
    main()
