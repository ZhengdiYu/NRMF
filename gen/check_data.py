"""Validate a motion dataset before training. Checks the three things gen/train*.py rely on:
  (1) the split registration in utils/data_utils.py:amass_splits,
  (2) the directory + filename layout that list_amass_files expects,
  (3) the per-npz field schema that gen/dataset.py:process_npz reads.
Run this first, on your own data, to catch problems before a training run.

  python gen/check_data.py --root <DATA>              # velocity / pose fields
  python gen/check_data.py --root <DATA> --contacts   # also require the contacts field (for train_contact.py)
"""
import os, sys, argparse, glob; sys.path.insert(0, '')
import numpy as np
from gen.dataset import parse_nframes, list_amass_files

# field -> expected number of values per frame (shape after the frame axis)
REQUIRED = {'root_orient': 3, 'pose_body': 63, 'trans': 3, 'world2aligned_rot': 9}
FEET = [7, 8, 10, 11]


def check_file(path, need_contacts):
    """Return a list of problem strings for one npz (empty list = OK)."""
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return [f"cannot load: {e}"]
    keys = set(getattr(d, 'files', []) or list(d.keys()))
    probs, Ns = [], []
    for k, want in REQUIRED.items():
        if k not in keys:
            probs.append(f"missing field '{k}'"); continue
        a = np.asarray(d[k]); Ns.append(a.shape[0]); nper = int(np.prod(a.shape[1:]))
        if nper != want:
            probs.append(f"'{k}': {nper} values/frame, expected {want}")
    if need_contacts:
        if 'contacts' not in keys:
            probs.append("missing 'contacts' (needed only for train_contact.py)")
        else:
            c = np.asarray(d['contacts']); Ns.append(c.shape[0])
            if c.ndim < 2 or c.shape[1] <= max(FEET):
                probs.append(f"'contacts' must have > {max(FEET)} columns (foot joints {FEET})")
    if Ns and len(set(Ns)) > 1:
        probs.append(f"inconsistent frame counts across fields: {sorted(set(Ns))}")
    if parse_nframes(path) is None:
        probs.append("filename lacks a '_<N>_frames_' token (works via a slower shape fallback, but rename recommended)")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dataset root: <root>/<DATASET>/<subject>/<name>_<N>_frames_..._fps.npz')
    ap.add_argument('--contacts', action='store_true', help='also require the contacts field (train_contact.py)')
    ap.add_argument('--sample', type=int, default=20, help='files to schema-check per split')
    a = ap.parse_args()

    total = len(glob.glob(a.root + '/**/*.npz', recursive=True))
    print(f"root: {a.root}   ({total} .npz in tree)\n")
    ok = True
    for split in ('train', 'vald'):
        try:
            files = list_amass_files(a.root, split)
        except KeyError:
            print(f"[{split}] '{split}' not defined in utils/data_utils.py:amass_splits -- add your dataset names\n"); ok = False; continue
        print(f"[{split}] {len(files)} files matched (dataset in amass_splits AND '_frames_' in name AND dataset at path depth -3)")
        if not files:
            print("    -> 0 files. Check that:")
            print(f"       (1) amass_splits['{split}'] lists your dataset folder names, and")
            print("       (2) layout is <root>/<DATASET>/<subject>/<name>_<N>_frames_..._fps.npz\n")
            ok = False; continue
        bad = 0
        for f in files[:a.sample]:
            probs = check_file(f, a.contacts)
            if probs:
                bad += 1
                if bad <= 5:
                    print(f"    PROBLEM  {os.path.relpath(f, a.root)}")
                    for p in probs:
                        print(f"        - {p}")
        print(f"    schema OK on first {min(a.sample, len(files))} files" if bad == 0
              else f"    {bad}/{min(a.sample, len(files))} sampled files have schema problems")
        ok = ok and bad == 0
        print()
    print("RESULT:", "ready to train" if ok else "fix the problems above before training")


if __name__ == '__main__':
    main()
