"""Shared 3-column renderer: input | ground truth | prediction, one row per sequence."""
import numpy as np
from rendering import render_scene, make_ground, verts
from PIL import Image, ImageDraw


def make_panels(name, qs, labels, colors, betas, faces, bm, tr, W1):
    """qs: list of [W1,22,4] quats. Returns (name, [(label, verts, color)...], ground, cam-track)."""
    V = [(labels[i], verts(q, betas, faces, bm) + tr[:, None, :], colors[i]) for i, q in enumerate(qs)]
    gt = V[1][1]  # ground truth column defines the floor + camera track
    floor = float(np.percentile(gt[..., 2], 1))
    ground = make_ground(float(gt[:, :, 0].mean()), float(gt[:, :, 1].mean()), floor)
    mz = float(gt[..., 2].mean())
    track = np.stack([gt[:, :, 0].mean(1), gt[:, :, 1].mean(1), np.full(W1, mz)], 1)
    return (name, V, ground, track)


def save_gif(rows, faces, out_path, W1, cam=2.7, duration=90):
    step = max(1, W1 // 40)
    frames = []
    for t in range(0, W1, step):
        row_imgs = []
        for name, V, ground, track in rows:
            imgs = []
            for label, vv, col in V:
                im = render_scene(vv[t], faces, col, ground, track[t], cam)
                pim = Image.fromarray(im)
                ImageDraw.Draw(pim).text((6, 4), f"{name} | {label}", fill=(0, 0, 0))
                imgs.append(np.asarray(pim))
            row_imgs.append(np.concatenate(imgs, 1))
        frames.append(Image.fromarray(np.concatenate(row_imgs, 0)))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    return out_path
