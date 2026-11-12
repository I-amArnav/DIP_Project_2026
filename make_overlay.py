import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from evaluate import components, dist_to, is_two_sided

GREEN = (60, 170, 40)    # BGR
RED = (40, 40, 230)
BLUE = (230, 120, 30)


def analyze(gt_img, in_img, res_img, tau, min_px, close_frac):
    gt_skel = skeletonize(gt_img < 128)
    in_skel = skeletonize(in_img < 128)
    out_skel = skeletonize(res_img < 128)
    d_gt, d_in, d_out = dist_to(gt_skel), dist_to(in_skel), dist_to(out_skel)

    missing = gt_skel & (d_in > tau)
    added = out_skel & (d_in > tau)

    bridges = []
    for ys, xs in components(added, min_px):
        frac = float(np.mean(d_gt[ys, xs] <= tau))
        bridges.append({"len": ys.size, "frac": frac, "ok": frac >= close_frac,
                        "med_d": float(np.median(d_gt[ys, xs]))})
    gaps = []
    for ys, xs in components(missing, min_px):
        closed = float(np.mean(d_out[ys, xs] <= tau)) >= close_frac
        gaps.append({"len": ys.size, "closed": closed,
                     "two_sided": is_two_sided(ys, xs, in_skel, tau)})
    return {
        "in_skel": in_skel,
        "missed_px": missing & (d_out > tau),
        "correct_px": added & (d_gt <= tau),
        "false_px": added & (d_gt > tau),
        "bridges": bridges,
        "gaps": gaps,
    }


def render(gt_img, in_img, res_img, a, name, scale):
    h, w = gt_img.shape
    canvas = np.full((h, w, 3), 255, np.uint8)
    canvas[gt_img < 128] = (228, 228, 228)
    canvas[a["in_skel"]] = (120, 120, 120)
    k = np.ones((3, 3), np.uint8)
    for mask, color in ((a["missed_px"], BLUE), (a["correct_px"], GREEN), (a["false_px"], RED)):
        canvas[cv2.dilate(mask.astype(np.uint8), k).astype(bool)] = color

    panels = [cv2.cvtColor(in_img, cv2.COLOR_GRAY2BGR), canvas, cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)]
    titles = ["input", "overlay (green ok, red false, blue missed)", "result"]
    out = []
    for p, t in zip(panels, titles):
        if scale != 1:
            p = cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        bar = np.full((22, p.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, t, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        out.append(np.vstack([bar, p]))
        out.append(np.full((out[-1].shape[0], 4, 3), 160, np.uint8))
    panel = np.hstack(out[:-1])

    n_false = sum(not b["ok"] for b in a["bridges"])
    foot = np.full((22, panel.shape[1], 3), 255, np.uint8)
    text = (f"{name} | correct {int(a['correct_px'].sum())}px, false {int(a['false_px'].sum())}px, "
            f"missed {int(a['missed_px'].sum())}px | false bridges {n_false}")
    cv2.putText(foot, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([panel, foot])


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Failure overlays + diagnosis for line completion")
    ap.add_argument("gt_dir")
    ap.add_argument("results_dir")
    ap.add_argument("input_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--n", type=int, default=20, help="Number of overlay panels to save")
    ap.add_argument("--select", default="random",
                    choices=["worst_precision", "worst_recall", "random", "all"])
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--gt-threshold", type=int, default=200)
    ap.add_argument("--min-event-px", type=int, default=3)
    ap.add_argument("--close-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gt_dir, res_dir, in_dir = Path(args.gt_dir), Path(args.results_dir), Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load(path):
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None

    items = []
    for gt_file in sorted(gt_dir.glob("*.png")) or sorted(gt_dir.glob("*.jpg")):
        gt = load(gt_file)
        inp = load(in_dir / gt_file.name)
        res = load(res_dir / gt_file.name)
        if gt is None or inp is None or res is None:
            continue
        h, w = gt.shape
        if inp.shape != (h, w):
            inp = cv2.resize(inp, (w, h))
        if res.shape != (h, w):
            res = cv2.resize(res, (w, h))
        if args.gt_threshold:
            gt = np.where(gt > args.gt_threshold, 255, 0).astype(np.uint8)
        a = analyze(gt, inp, res, args.tau, args.min_event_px, args.close_frac)
        items.append((gt_file.name, gt, inp, res, a))

    if not items:
        print("No matching images found.")
        return

    # ---------------- pooled diagnosis over all images ----------------------
    bridges = [b for it in items for b in it[4]["bridges"]]
    gaps = [g for it in items for g in it[4]["gaps"]]
    ok_b = [b for b in bridges if b["ok"]]
    bad_b = [b for b in bridges if not b["ok"]]
    misaligned = [b for b in bad_b if b["med_d"] <= 3 * args.tau]
    spurious = [b for b in bad_b if b["med_d"] > 3 * args.tau]
    closed = [g for g in gaps if g["closed"]]
    missed = [g for g in gaps if not g["closed"]]
    missed_bridgeable = [g for g in missed if g["two_sided"]]
    missed_onesided = [g for g in missed if not g["two_sided"]]

    print("=" * 78)
    print(f"DIAGNOSIS over {len(items)} images (tau = {args.tau}px)")
    print("=" * 78)
    print(f"Gaps: {len(gaps)} total | closed {len(closed)} | missed {len(missed)}")
    print(f"  missed & one-sided (tip/short stroke erased; unrecoverable): {len(missed_onesided)}")
    print(f"  missed & bridgeable (strokes on both sides; algorithm failures): {len(missed_bridgeable)}")
    print(f"  median gap size (GT skeleton px): closed {med([g['len'] for g in closed]):.1f} | "
          f"missed-bridgeable {med([g['len'] for g in missed_bridgeable]):.1f} | "
          f"missed-onesided {med([g['len'] for g in missed_onesided]):.1f}")
    print(f"Bridges: {len(bridges)} total | correct {len(ok_b)} | false {len(bad_b)}")
    print(f"  false bridges that are misaligned (near GT, but off): {len(misaligned)}")
    print(f"  false bridges that are spurious (far from any GT stroke): {len(spurious)}")
    print(f"  median bridge length (px): correct {med([b['len'] for b in ok_b]):.1f} | "
          f"false {med([b['len'] for b in bad_b]):.1f}")

    print("-" * 78)
    print("Observations")
    hints = []
    if bad_b and ok_b and med([b["len"] for b in bad_b]) > 1.3 * med([b["len"] for b in ok_b]):
        hints.append("False bridges are noticeably longer than correct ones -> lower --max-gap / --gap-scale.")
    if bad_b and len(spurious) > len(misaligned):
        hints.append("Most false bridges are spurious (join things that should stay apart) -> lower "
                     "--max-cost, lower --max-angle, or try --no-tee.")
    if bad_b and len(misaligned) >= len(spurious):
        hints.append("Most false bridges are misaligned, not spurious -> matching is fine, the CURVE is off: "
                     "improve tangent estimation / Bezier handle length in algo.py.")
    if missed and len(missed_bridgeable) > 0.3 * len(missed):
        hints.append("Many missed gaps were bridgeable -> raise --max-gap, or relax --max-angle / --max-cost.")
    if missed and len(missed_onesided) > 0.5 * len(missed):
        hints.append("Most misses are one-sided (no stroke on the far side): no endpoint method can fix these; "
                     "consider reporting the bridgeable close rate as the fair number.")
    if not hints:
        hints.append("No dominant failure pattern; inspect the overlays.")
    for h_ in hints:
        print(" *", h_)

    # ---------------- choose and render overlays ----------------------------
    def false_px(it): return int(it[4]["false_px"].sum())
    def missed_px(it): return int(it[4]["missed_px"].sum())

    if args.select == "worst_precision":
        chosen = sorted(items, key=false_px, reverse=True)
        chosen = [c for c in chosen if false_px(c) > 0]
    elif args.select == "worst_recall":
        chosen = sorted(items, key=missed_px, reverse=True)
        chosen = [c for c in chosen if missed_px(c) > 0]
    elif args.select == "random":
        random.Random(args.seed).shuffle(items)
        chosen = list(items)
    else:
        chosen = list(items)
    if args.select != "all":
        chosen = chosen[:args.n]

    for rank, (name, gt, inp, res, a) in enumerate(chosen, start=1):
        cv2.imwrite(str(out_dir / f"{rank:03d}_{Path(name).stem}.png"),
                    render(gt, inp, res, a, name, args.scale))
    print("-" * 78)
    print(f"Saved {len(chosen)} overlay panels ({args.select}) to {out_dir.resolve()}")


if __name__ == "__main__":
    main()

"""
Usage:
    python make_overlay.py ./Outputs ./Results ./Inputs ./Overlays
"""