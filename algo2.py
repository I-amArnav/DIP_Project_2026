import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
_KERNEL = np.ones((3, 3), np.uint8)
_KERNEL[1, 1] = 0


def find_endpoints(skel):
    count = cv2.filter2D(skel, -1, _KERNEL, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((skel == 1) & (count == 1))
    return list(zip(ys.tolist(), xs.tolist()))


def walk(skel, start, max_len):
    """Follow the skeleton from `start`. Returns (path[(y,x)...], hit_junction)."""
    h, w = skel.shape
    path = [start]
    visited = {start}
    cur = start
    junction = False
    while len(path) < max_len:
        y, x = cur
        cands = []
        for dy, dx in NEIGH:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) not in visited:
                cands.append((ny, nx))
        if not cands:
            break
        if len(cands) > 1:
            adjacent = all(
                max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
                for i, a in enumerate(cands) for b in cands[i + 1:]
            )
            if not adjacent:
                junction = True
                break
            for c in cands[1:]:
                visited.add(c)
        cur = cands[0]
        visited.add(cur)
        path.append(cur)
    return path, junction


def prune_spurs_dt(skel, dist, k=1.5, slack=1.5, iters=3):
    """Remove branches shorter than k * (stroke radius at the junction) + slack."""
    rmax = float(dist[skel > 0].max())
    maxlen = int(k * rmax + slack) + 3
    for _ in range(iters):
        removed = False
        for p in find_endpoints(skel):
            path, junction = walk(skel, p, maxlen)
            if junction and (len(path) - 1) <= k * dist[path[-1]] + slack:
                for q in path[:-1]:
                    skel[q] = 0
                removed = True
        if not removed:
            break
    return skel


# endpoint description: robust tip position, tangent and stroke width
def describe_endpoint(p, skel, lines, dist, g_width):
    h, w = lines.shape
    path, _ = walk(skel, p, int(np.clip(4 * g_width, 8, 40)))
    pts = np.array(path, dtype=np.float64)                    # (y, x)
    d_path = dist[np.array(path)[:, 0], np.array(path)[:, 1]]
    width = max(1.0, 2 * float(np.median(d_path[min(2, len(d_path) - 1):])) - 1)

    p_xy = np.array([p[1], p[0]], dtype=np.float64)
    direction, tip = None, p_xy
    skip = int(round(width / 2))
    fit = pts[skip: skip + max(6, int(2 * width))] # Why the 6 ?
    if len(fit) < 4: # Why this ?
        fit = pts
    if len(fit) >= 4:
        xy = fit[:, ::-1]
        c = xy.mean(axis=0)
        _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
        t = vt[0]
        if np.dot(t, p_xy - c) < 0:
            t = -t
        direction = t / (np.linalg.norm(t) + 1e-12)

        # march the fitted centre-line outward until it leaves the stroke
        s_p = float(np.dot(direction, p_xy - c))
        s_last = 0.0
        s = 0.0
        smax = s_p + 2 * width + 3
        while s <= smax:
            x = int(round(c[0] + s * direction[0]))
            y = int(round(c[1] + s * direction[1]))
            if not (0 <= x < w and 0 <= y < h) or not lines[y, x]:
                break
            s_last = s
            s += 0.5
        tip = c + max(s_last, s_p) * direction if s_last >= s_p else p_xy

    return {"pos": tip, "yx": p, "dir": direction, "width": width,
            "path": set(path)}


# drawing
def bezier(p0, p1, p2, p3, n=28):
    t = np.linspace(0, 1, n)[:, None]
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 \
        + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    return np.round(pts).astype(np.int32)


def draw_curve(img, pts, thickness):
    thickness = max(1, int(thickness))
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, 0, thickness, cv2.LINE_8)
    r = thickness // 2
    for q in (pts[0], pts[-1]):
        cv2.circle(img, (int(q[0]), int(q[1])), r, 0, -1)


# main algorithm
def complete_image(gray, max_gap=12, gap_scale=2.0, max_cost=1.6, max_angle=70.0,
                   tee=True, tee_ratio=0.8, spur_k=1.5, lat_width=1.5, lat_frac=0.0):
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    lines = (binary < 127).astype(np.uint8)
    out = binary.copy()
    if lines.sum() == 0:
        return out, 0

    skel = skeletonize(lines > 0).astype(np.uint8)
    dist = cv2.distanceTransform(lines, cv2.DIST_L2, 5)
    g_width = max(1.0, 2 * float(np.median(dist[skel > 0])) - 1)
    skel = prune_spurs_dt(skel, dist, k=spur_k)

    eps = [describe_endpoint(p, skel, lines, dist, g_width) for p in find_endpoints(skel)]
    if not eps:
        return out, 0

    def dmax_for(*widths):
        return max_gap + gap_scale * float(np.mean(widths))

    # ---- candidate pairs ------------------------------------------------
    cos_lim = np.cos(np.deg2rad(max_angle))
    cands = []
    if len(eps) >= 2:
        P = np.array([e["pos"] for e in eps])
        r_all = dmax_for(max(e["width"] for e in eps))
        for i, j in cKDTree(P).query_pairs(r_all):
            a, b = eps[i], eps[j]
            d = float(np.linalg.norm(b["pos"] - a["pos"]))
            wm = 0.5 * (a["width"] + b["width"])

            if d < 1 or d > dmax_for(a["width"], b["width"]):
                continue
            if b["yx"] in a["path"] or a["yx"] in b["path"]:
                continue
            v = (b["pos"] - a["pos"]) / d

            if a["dir"] is not None and b["dir"] is not None:
                ca, cb = float(a["dir"] @ v), float(b["dir"] @ -v)
                if ca < cos_lim or cb < cos_lim:
                    continue
                perp_a = d * np.sqrt(max(0.0, 1 - ca * ca))
                perp_b = d * np.sqrt(max(0.0, 1 - cb * cb))
                if max(perp_a, perp_b) > lat_width * wm + 2 + lat_frac * d:
                    continue                          # strokes do not line up
                cc = float(-(a["dir"] @ b["dir"]))
                lat = 0.5 * (perp_a + perp_b) / max(wm, 2.0)
            else:
                ca = cb = cc = 0.5
                lat = 0.5
            wdiff = abs(a["width"] - b["width"]) / max(a["width"], b["width"], 2.0)
            cost = (d / dmax_for(a["width"], b["width"])
                    + (1 - ca) + (1 - cb) + 0.5 * (1 - cc)
                    + 0.8 * lat + 0.5 * wdiff)
            if cost <= max_cost:
                cands.append((cost, i, j))

    # ---- greedy one-to-one matching -------------------------------------
    cands.sort(key=lambda c: c[0])
    used = set()
    n_bridges = 0
    for cost, i, j in cands:
        if i in used or j in used:
            continue
        used.update((i, j))
        a, b = eps[i], eps[j]
        width = 0.5 * (a["width"] + b["width"])
        if a["dir"] is not None and b["dir"] is not None:
            a0 = a["pos"] - a["dir"] * 0.5 * width       # start just inside the stroke
            b0 = b["pos"] - b["dir"] * 0.5 * width
            k = 0.35 * float(np.linalg.norm(b0 - a0))
            pts = bezier(a0, a0 + a["dir"] * k, b0 + b["dir"] * k, b0)
        else:
            pts = np.round(np.stack([a["pos"], b["pos"]])).astype(np.int32)
        draw_curve(out, pts, round(width))
        n_bridges += 1

    # ---- endpoint -> stroke ("T") bridging ------------------------------
    if tee:
        h, w = lines.shape
        for i, e in enumerate(eps):
            if i in used or e["dir"] is None:
                continue
            tmax = int(tee_ratio * dmax_for(e["width"]))
            best = None
            px, py = e["pos"]
            for ang in (0, 8, -8, 16, -16):
                rad = np.deg2rad(ang)
                c_, s_ = np.cos(rad), np.sin(rad)
                dx = e["dir"][0] * c_ - e["dir"][1] * s_
                dy = e["dir"][0] * s_ + e["dir"][1] * c_
                left_own = False
                for t in range(1, tmax + 1):
                    x, y = int(round(px + t * dx)), int(round(py + t * dy))
                    if not (0 <= x < w and 0 <= y < h):
                        break
                    if not lines[y, x]:
                        left_own = True
                    elif left_own:
                        score = t + 0.25 * abs(ang)
                        if best is None or score < best[0]:
                            best = (score, t, dx, dy)
                        break
            if best is not None:
                _, t, dx, dy = best
                tgt = np.array([px + (t + 0.5 * e["width"]) * dx, py + (t + 0.5 * e["width"]) * dy])
                start = e["pos"] - e["dir"] * 0.5 * e["width"]
                pts = np.round(np.stack([start, tgt])).astype(np.int32)
                draw_curve(out, pts, round(e["width"]))
                n_bridges += 1

    return out, n_bridges


def run(input_dir, output_dir, **kw):
    in_path, out_path = Path(input_dir), Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ok = err = total = 0
    for f in sorted(in_path.glob("*.png")):
        try:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            res, nb = complete_image(img, **kw)
            cv2.imwrite(str(out_path / f.name), res)
            ok += 1
            total += nb
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            err += 1
    print("-" * 40)
    print("algo2 done!")
    print(f"Images saved: {ok} | Errors: {err} | Bridges drawn: {total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Direction-aware line completion")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--max-gap", type=float, default=12)
    ap.add_argument("--gap-scale", type=float, default=2.0)
    ap.add_argument("--max-cost", type=float, default=1.6)
    ap.add_argument("--max-angle", type=float, default=70.0)
    ap.add_argument("--no-tee", action="store_true")
    ap.add_argument("--tee-ratio", type=float, default=0.8)
    ap.add_argument("--spur-k", type=float, default=1.5)
    ap.add_argument("--lat-width", type=float, default=1.5)
    ap.add_argument("--lat-frac", type=float, default=0.0)
    a = ap.parse_args()
    run(a.input_dir, a.output_dir, max_gap=a.max_gap, gap_scale=a.gap_scale,
        max_cost=a.max_cost, max_angle=a.max_angle, tee=not a.no_tee,
        tee_ratio=a.tee_ratio, spur_k=a.spur_k,
        lat_width=a.lat_width, lat_frac=a.lat_frac)

"""
Usage:
    python3 algo2.py ./Inputs ./Results
"""