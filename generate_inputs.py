import argparse
import zlib
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def crossing_number(skel):
    """0->1 transitions around the 8-ring: >=3 means a real branch point
    (unlike a raw neighbour count, staircase corners are not flagged)."""
    h, w = skel.shape
    p = np.pad(skel, 1)
    ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING]
    t = np.zeros(skel.shape, np.int32)
    for i in range(8):
        t += (ring[i] == 0) & (ring[(i + 1) % 8] == 1)
    return t


def order_path(pix):
    """Order a set of (y,x) skeleton pixels into a walk, starting at an end."""
    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in NEIGH if (p[0] + dy, p[1] + dx) in pix]
    ends = [p for p in pix if len(nbrs(p)) == 1]
    cur = ends[0] if ends else next(iter(pix))
    path, seen = [cur], {cur}
    while True:
        nxt = [q for q in nbrs(cur) if q not in seen]
        if not nxt:
            break
        nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))  # prefer 4-connected
        cur = nxt[0]
        seen.add(cur)
        path.append(cur)
    return path


def build_branches(skel, dist, junction_margin):
    """Split skeleton into junction-free branches. Returns (owner map, {id: ordered path})."""
    junc = (skel == 1) & (crossing_number(skel) >= 3)
    zone = np.zeros_like(skel)
    for y, x in np.argwhere(junc):
        cv2.circle(zone, (int(x), int(y)), int(np.ceil(dist[y, x])) + junction_margin, 1, -1)
    branch = ((skel == 1) & (zone == 0)).astype(np.uint8)
    _, lab = cv2.connectedComponents(branch, connectivity=8)
    owner = lab.astype(np.int32)
    owner[(skel == 1) & (zone == 1)] = -1          # junction zone: off limits
    ys, xs = np.nonzero(lab)
    order = np.argsort(lab[ys, xs], kind="stable")
    ys, xs = ys[order], xs[order]
    ids = lab[ys, xs]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    branches = {}
    for yy, xx, i in zip(np.split(ys, cuts), np.split(xs, cuts), np.split(ids, cuts)):
        pix = set(zip(yy.tolist(), xx.tolist()))
        path = order_path(pix)
        if len(path) >= 0.8 * len(pix):             # skip messy components
            branches[int(i[0])] = path
    return owner, branches


def add_gaps(gray, rng, gaps_per_1000=12.0, min_gap_mult=1.0, max_gap_mult=4.0,
             min_gap=5, max_gap=28, min_piece=10, junction_margin=4, tries=30):
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    lines = (binary < 127).astype(np.uint8)
    empty = np.zeros_like(binary)
    if not lines.any():
        return binary, binary, empty, 0
    skel = skeletonize(lines > 0).astype(np.uint8)
    dist = cv2.distanceTransform(lines, cv2.DIST_L2, 5)
    owner, branches = build_branches(skel, dist, junction_margin)

    usable = {k: p for k, p in branches.items() if len(p) >= 2 * min_piece + min_gap}
    if not usable:
        return binary, binary, empty, 0
    keys = list(usable)
    lens = np.array([len(usable[k]) for k in keys], float)
    n_target = max(1, int(round(gaps_per_1000 * lens.sum() / 1000.0)))
    probs = lens / lens.sum()

    h, w = binary.shape
    used = {k: [] for k in keys}               # index extents already cut per branch
    erase = np.zeros_like(binary)
    placed = 0
    for _ in range(n_target * tries):
        if placed >= n_target:
            break
        k = keys[rng.choice(len(keys), p=probs)]
        path = usable[k]
        n = len(path)
        c = int(rng.integers(min_piece, n - min_piece))
        cy, cx = path[c]
        t = max(1.0, 2 * dist[cy, cx] - 1)                 # local stroke thickness
        r = int(np.ceil(dist[cy, cx])) + 1                 # eraser radius, fully severs
        L = float(np.clip(rng.uniform(min_gap_mult, max_gap_mult) * t, min_gap, max_gap))
        L = max(L, 2 * r + 1)
        s = int(round(L - 2 * r))                          # path length between end caps
        a = c - s // 2
        b = a + s
        piece = int(max(min_piece, 2.5 * t))
        lo, hi = a - r - piece, b + r + piece
        if lo < 0 or hi > n - 1:
            continue
        if any(lo <= e1 and hi >= e0 for e0, e1 in used[k]):
            continue

        seg = path[a:b + 1]
        pad = 2 * r + 8
        y0 = max(0, min(p[0] for p in seg) - pad); y1 = min(h, max(p[0] for p in seg) + pad + 1)
        x0 = max(0, min(p[1] for p in seg) - pad); x1 = min(w, max(p[1] for p in seg) + pad + 1)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for (py, px) in seg:
            ri = int(np.ceil(dist[py, px])) + 1
            cv2.circle(m, (px - x0, py - y0), ri, 1, -1)

        dm = cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3)
        ow = owner[y0:y1, x0:x1]
        if np.any((ow != 0) & (ow != k) & (dm <= dist[y0:y1, x0:x1] + 0.5)):
            continue

        erase[y0:y1, x0:x1] |= m
        used[k].append((a - r, b + r))
        placed += 1

    out = binary.copy()
    out[erase > 0] = 255
    return binary, out, (erase * 255).astype(np.uint8), placed


def run(input_dir, output_dir, gt_dir=None, mask_dir=None, seed=0, **kw):
    src, dst = Path(input_dir), Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for d in (gt_dir, mask_dir):
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
    files = sorted(list(src.glob("*.png")) + list(src.glob("*.jpg")))
    ok = skipped = err = total = 0
    for f in files:
        try:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            rng = np.random.default_rng([seed, zlib.crc32(f.name.encode())])
            clean, broken, mask, n = add_gaps(img, rng, **kw)
            if n == 0:
                skipped += 1
                continue
            name = f.with_suffix(".png").name
            cv2.imwrite(str(dst / name), broken)
            if gt_dir:
                cv2.imwrite(str(Path(gt_dir) / name), clean)
            if mask_dir:
                cv2.imwrite(str(Path(mask_dir) / name), mask)
            ok += 1
            total += n
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            err += 1
    print("-" * 40)
    print(f"Saved: {ok} | Skipped (no valid gap site): {skipped} | Errors: {err}")
    print(f"Gaps: {total} total, {total / max(ok, 1):.1f} per image")
    print(f"Output Directory: {dst.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="gap generator")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--gt-dir")
    ap.add_argument("--mask-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gaps-per-1000", type=float, default=12.0)
    ap.add_argument("--min-gap-mult", type=float, default=1.0)
    ap.add_argument("--max-gap-mult", type=float, default=4.0)
    ap.add_argument("--min-gap", type=int, default=5)
    ap.add_argument("--max-gap", type=int, default=28)
    ap.add_argument("--min-piece", type=int, default=10)
    ap.add_argument("--junction-margin", type=int, default=4)
    a = ap.parse_args()
    run(a.input_dir, a.output_dir, a.gt_dir, a.mask_dir, a.seed,
        gaps_per_1000=a.gaps_per_1000, min_gap_mult=a.min_gap_mult, max_gap_mult=a.max_gap_mult,
        min_gap=a.min_gap, max_gap=a.max_gap, min_piece=a.min_piece, junction_margin=a.junction_margin)

"""
Usage:
  python generate_inputs_v2.py ./Outputs ./Inputs --gt-dir ./GT --mask-dir ./Masks
  python generate_inputs_v2.py ./Outputs_skeletonized ./Inputs_skeletonized \
         --gt-dir ./GT_skel --mask-dir ./Masks_skel
"""
