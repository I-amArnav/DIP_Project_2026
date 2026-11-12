import argparse
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.ndimage import label as nd_label
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize

NAN = float("nan")
STRUCT8 = np.ones((3, 3), dtype=int)


def safe_mean(x):
    x = np.asarray(x)
    return float(x.mean()) if x.size else NAN


def dist_to(mask):
    """Distance from every pixel to the nearest True pixel of `mask`."""
    if mask.any():
        return distance_transform_edt(~mask)
    return np.full(mask.shape, np.inf)


def count_components(lines):
    n, _ = cv2.connectedComponents(lines.astype(np.uint8), connectivity=8)
    return n-1

def f_score(p, r):
    if np.isnan(p) or np.isnan(r):
        return NAN
    return 2*p*r / (p+r) if (p+r) > 0 else 0.0

def components(mask, min_px=3):
    """8-connected components of a boolean mask -> list of (ys, xs) index arrays."""
    lab, _ = nd_label(mask, structure=STRUCT8)
    out = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == i)
        if ys.size >= min_px:
            out.append((ys + sl[0].start, xs + sl[1].start))
    return out


def is_two_sided(ys, xs, in_skel, tau):
    """True if existing (input) strokes touch this gap piece at >= 2 separate places."""
    pad = int(tau) + 3
    h, w = in_skel.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    comp = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    comp[ys - y0, xs - x0] = True
    near = in_skel[y0:y1, x0:x1] & (distance_transform_edt(~comp) <= tau + 2)
    _, k = nd_label(near, structure=STRUCT8)
    return k >= 2


# metric registry: (key, label, higher_is_better (True/False/None), group)
G_STD = "STANDARD PIXEL-WISE METRICS (kept; very strict, see notes)"
G_TOL = "TOLERANT SKELETON METRICS (match within tau px)"
G_GAP = "GAP-FOCUSED PIXEL METRICS (gap / added pixels defined per tau)"
G_EVT = "GAP-EVENT METRICS (per-image averages; each gap / bridge counts once)"
G_CNT = "GAP-EVENT TOTALS (pooled over all images)"
G_DIST = "DISTANCE / TOPOLOGY / PERCEPTUAL"

# Metrics where the corrupted input is 1.0 by construction (it never adds or
# removes anything beyond the damage), so "Better?" is meaningless for them.
def is_reference_only(key):
    return key == "precision" or key.startswith(("added_precision", "preservation", "bridge_precision"))


def build_specs(taus, event_tau):
    specs = [
        ("psnr", "PSNR (dB)", True, G_STD),
        ("ssim", "SSIM", True, G_STD),
        ("mse", "MSE", False, G_STD),
        ("accuracy", "Accuracy", True, G_STD),
        ("precision", "Precision", True, G_STD),
        ("recall", "Recall", True, G_STD),
        ("f1", "F1-Score", True, G_STD),
    ]
    for t in taus:
        specs += [
            (f"tol_precision@{t}", f"Tol. Precision @{t}px", True, G_TOL),
            (f"tol_recall@{t}", f"Tol. Recall @{t}px", True, G_TOL),
            (f"tol_f1@{t}", f"Tol. F1 @{t}px", True, G_TOL),
        ]
    for t in taus:
        specs += [
            (f"gap_recall@{t}", f"Gap Recall @{t}px", True, G_GAP),
            (f"added_precision@{t}", f"Added Precision @{t}px", True, G_GAP),
            (f"gap_f1@{t}", f"Gap F1 @{t}px", True, G_GAP),
            (f"preservation@{t}", f"Preservation @{t}px", True, G_GAP),
        ]
    e = event_tau
    specs += [
        ("gap_close_rate", f"Gap close rate @{e}px", True, G_EVT),
        ("bridgeable_close_rate", f"Bridgeable-gap close @{e}", True, G_EVT),
        ("bridge_precision", f"Bridge precision @{e}px", True, G_EVT),
        ("event_f1", f"Event F1 @{e}px", True, G_EVT),
        ("false_bridges", "False bridges / image", False, G_EVT),
        ("missed_gaps", "Missed gaps / image", False, G_EVT),
        ("n_gaps", "Gaps (total)", None, G_CNT),
        ("n_gaps_closed", "Gaps closed", None, G_CNT),
        ("n_bridgeable", "Bridgeable gaps", None, G_CNT),
        ("n_bridgeable_closed", "Bridgeable gaps closed", None, G_CNT),
        ("n_bridges", "Bridges drawn", None, G_CNT),
        ("n_bridges_correct", "Bridges correct", None, G_CNT),
    ]
    specs += [
        ("chamfer", "Chamfer dist (px)", False, G_DIST),
        ("hd95", "HD95 (px)", False, G_DIST),
        ("cc_error", "Component-count |err|", False, G_DIST),
        ("cc_signed", "Component-count diff (+frag/-merge)", None, G_DIST),
        ("blur_ssim", "Blurred SSIM", True, G_DIST),
    ]
    return specs


# per-image computation
def prepare_reference(gt_img, in_img, blur_sigma):
    gt_lines = gt_img < 128
    in_lines = in_img < 128
    gt_skel = skeletonize(gt_lines)
    in_skel = skeletonize(in_lines)
    return {
        "gt_img": gt_img,
        "gt_skel": gt_skel,
        "d_gt": dist_to(gt_skel),
        "in_skel": in_skel,
        "d_in": dist_to(in_skel),
        "gt_cc": count_components(gt_lines),
        "blur_gt": cv2.GaussianBlur(gt_img, (0, 0), blur_sigma),
        "blur_sigma": blur_sigma,
    }


def standard_metrics(gt_img, tgt_img):
    total = gt_img.size
    mse = float(np.mean((gt_img.astype(np.float64) - tgt_img.astype(np.float64)) ** 2))
    p = psnr(gt_img, tgt_img, data_range=255) if mse > 0 else 100.0
    s = ssim(gt_img, tgt_img, data_range=255)

    gt_lines = gt_img < 128
    tg_lines = tgt_img < 128
    tp = np.sum(gt_lines & tg_lines)
    fp = np.sum(~gt_lines & tg_lines)
    fn = np.sum(gt_lines & ~tg_lines)
    tn = np.sum(~gt_lines & ~tg_lines)

    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"psnr": p, "ssim": s, "mse": mse, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


def event_metrics(tgt_skel, d_out, ref, tau, min_px, close_frac):
    missing = ref["gt_skel"] & (ref["d_in"] > tau)
    added = tgt_skel & (ref["d_in"] > tau)
    gaps = components(missing, min_px)
    bridges = components(added, min_px)

    closed = bridgeable = bridgeable_closed = 0
    for ys, xs in gaps:
        is_closed = np.mean(d_out[ys, xs] <= tau) >= close_frac
        closed += is_closed
        if is_two_sided(ys, xs, ref["in_skel"], tau):
            bridgeable += 1
            bridgeable_closed += is_closed
    correct = sum(1 for ys, xs in bridges if np.mean(ref["d_gt"][ys, xs] <= tau) >= close_frac)

    n_g, n_b = len(gaps), len(bridges)
    rate = closed / n_g if n_g else NAN
    bprec = correct / n_b if n_b else 1.0
    return {
        "gap_close_rate": rate,
        "bridgeable_close_rate": bridgeable_closed / bridgeable if bridgeable else NAN,
        "bridge_precision": bprec,
        "event_f1": f_score(bprec, rate),
        "false_bridges": n_b - correct,
        "missed_gaps": n_g - closed,
        "n_gaps": n_g, "n_gaps_closed": closed,
        "n_bridgeable": bridgeable, "n_bridgeable_closed": bridgeable_closed,
        "n_bridges": n_b, "n_bridges_correct": correct,
    }


def tolerant_metrics(tgt_img, ref, taus, event_tau, min_px, close_frac):
    m = {}
    tgt_lines = tgt_img < 128
    tgt_skel = skeletonize(tgt_lines)
    d_out = dist_to(tgt_skel)

    gt_skel, d_gt, d_in, in_skel = ref["gt_skel"], ref["d_gt"], ref["d_in"], ref["in_skel"]
    n_out = int(tgt_skel.sum())
    gt_ok = bool(gt_skel.any())

    for t in taus:
        if gt_ok:
            p = safe_mean(d_gt[tgt_skel] <= t) if n_out else 1.0
            r = safe_mean(d_out[gt_skel] <= t)
        else:
            p = r = NAN
        m[f"tol_precision@{t}"] = p
        m[f"tol_recall@{t}"] = r
        m[f"tol_f1@{t}"] = f_score(p, r)

        # gap / added pixel sets are defined per tau -> input scores exactly 0 recall
        missing = gt_skel & (d_in > t)
        added = tgt_skel & (d_in > t)
        gap_r = safe_mean(d_out[missing] <= t) if missing.any() else NAN
        add_p = safe_mean(d_gt[added] <= t) if (added.any() and gt_ok) else 1.0
        m[f"gap_recall@{t}"] = gap_r
        m[f"added_precision@{t}"] = add_p
        m[f"gap_f1@{t}"] = f_score(add_p, gap_r)
        m[f"preservation@{t}"] = safe_mean(d_out[in_skel] <= t) if in_skel.any() else NAN

    m.update(event_metrics(tgt_skel, d_out, ref, event_tau, min_px, close_frac))

    if gt_ok and n_out:
        a, b = d_gt[tgt_skel], d_out[gt_skel]
        m["chamfer"] = (a.mean() + b.mean()) / 2
        m["hd95"] = max(np.percentile(a, 95), np.percentile(b, 95))
    else:
        m["chamfer"] = m["hd95"] = NAN

    diff = count_components(tgt_lines) - ref["gt_cc"]
    m["cc_signed"] = diff
    m["cc_error"] = abs(diff)
    blur_t = cv2.GaussianBlur(tgt_img, (0, 0), ref["blur_sigma"])
    m["blur_ssim"] = ssim(ref["blur_gt"], blur_t, data_range=255)
    return m


def all_metrics(tgt_img, ref, taus, event_tau, min_px, close_frac):
    m = standard_metrics(ref["gt_img"], tgt_img)
    m.update(tolerant_metrics(tgt_img, ref, taus, event_tau, min_px, close_frac))
    return m


# main evaluation loop
def evaluate_line_completion(gt_dir, results_dir, input_dir, log_file,
                             taus=(1, 2, 3), event_tau=2, gt_threshold=200,
                             blur_sigma=2.0, min_event_px=3, close_frac=0.8):
    gt_path, res_path, in_path = Path(gt_dir), Path(results_dir), Path(input_dir)
    for path, name in [(gt_path, "Ground Truth"), (res_path, "Results"), (in_path, "Inputs")]:
        if not path.exists():
            print(f"Error: {name} directory '{path}' does not exist.")
            return

    gt_files = sorted(gt_path.glob("*.png")) or sorted(gt_path.glob("*.jpg"))
    if not gt_files:
        print(f"No image files found in ground truth directory: {gt_path}")
        return

    taus = list(taus)
    specs = build_specs(taus, event_tau)
    keys = [s[0] for s in specs]
    store = {"input": {k: [] for k in keys}, "output": {k: [] for k in keys}}
    ok = missing_files = 0

    print(f"Evaluating {len(gt_files)} images (tolerances: {taus} px, event tau: {event_tau}, "
          f"GT threshold: {gt_threshold or 'off'}) ...")

    with open(log_file, "w") as log:
        log.write("Filename," + ",".join(f"In_{k}" for k in keys) + ","
                  + ",".join(f"Out_{k}" for k in keys) + "\n")

        for gt_file in gt_files:
            res_file = res_path / gt_file.name
            in_file = in_path / gt_file.name
            if not res_file.exists():
                res_file = res_path / gt_file.with_suffix(".jpg").name
            if not in_file.exists():
                in_file = in_path / gt_file.with_suffix(".jpg").name
            if not res_file.exists() or not in_file.exists():
                missing_files += 1
                continue

            gt_img = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            res_img = cv2.imread(str(res_file), cv2.IMREAD_GRAYSCALE)
            in_img = cv2.imread(str(in_file), cv2.IMREAD_GRAYSCALE)
            if gt_img is None or res_img is None or in_img is None:
                continue

            h, w = gt_img.shape
            if res_img.shape != (h, w):
                res_img = cv2.resize(res_img, (w, h))
            if in_img.shape != (h, w):
                in_img = cv2.resize(in_img, (w, h))

            # Same rule as generate_inputs.py: pixel <= threshold -> line.
            if gt_threshold:
                gt_img = np.where(gt_img > gt_threshold, 255, 0).astype(np.uint8)

            ref = prepare_reference(gt_img, in_img, blur_sigma)
            m_in = all_metrics(in_img, ref, taus, event_tau, min_event_px, close_frac)
            m_out = all_metrics(res_img, ref, taus, event_tau, min_event_px, close_frac)

            for k in keys:
                store["input"][k].append(m_in[k])
                store["output"][k].append(m_out[k])
            log.write(gt_file.name + ","
                      + ",".join(f"{m_in[k]:.4f}" for k in keys) + ","
                      + ",".join(f"{m_out[k]:.4f}" for k in keys) + "\n")
            ok += 1

    # ---- summary ----------------------------------------------------------
    line = "=" * 96
    print(line)
    print(" CUMULATIVE EVALUATION SUMMARY: INPUT vs. ALGORITHM OUTPUT")
    print(line)
    print(f"Successfully evaluated: {ok} images")
    if missing_files:
        print(f"Missing result/input files skipped: {missing_files}")

    current_group = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key, label, higher_better, group in specs:
            if group != current_group:
                current_group = group
                print("-" * 96)
                print(group)
                print(f"{'Metric':<36} | {'Corrupted Input':>14} | {'Algorithm Output':>16} | "
                      f"{'Delta':>10} | Better?")
            if group == G_CNT:
                a, b = np.nansum(store["input"][key]), np.nansum(store["output"][key])
            else:
                a, b = np.nanmean(store["input"][key]), np.nanmean(store["output"][key])
            d = b - a
            if is_reference_only(key):
                tag = "(ref)"
            elif higher_better is None:
                tag = "-"
            else:
                improved = (d > 1e-9) if higher_better else (d < -1e-9)
                tag = "yes" if improved else ("same" if abs(d) <= 1e-9 else "NO")
            fmt = ".0f" if group == G_CNT else ".4f"
            print(f"{label:<36} | {a:>14{fmt}} | {b:>16{fmt}} | {d:>+10{fmt}} | {tag}")

        # pooled headline numbers
        tg, tc = np.nansum(store["output"]["n_gaps"]), np.nansum(store["output"]["n_gaps_closed"])
        tb, tbc = np.nansum(store["output"]["n_bridgeable"]), np.nansum(store["output"]["n_bridgeable_closed"])
        nb, nbc = np.nansum(store["output"]["n_bridges"]), np.nansum(store["output"]["n_bridges_correct"])
        print("-" * 96)
        print("POOLED HEADLINE (algorithm output)")
        if tg:
            print(f"  Gaps closed            : {int(tc)}/{int(tg)} = {tc / tg:.1%}")
        if tb:
            print(f"  Bridgeable gaps closed : {int(tbc)}/{int(tb)} = {tbc / tb:.1%}"
                  f"   (gaps with strokes on both sides; ceiling for endpoint methods)")
        if nb:
            print(f"  Bridges correct        : {int(nbc)}/{int(nb)} = {nbc / nb:.1%}"
                  f"   ({int(nb - nbc)} false bridges)")
        elif tg:
            print("  Bridges drawn          : 0")
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate line completion results")
    ap.add_argument("gt_dir")
    ap.add_argument("results_dir")
    ap.add_argument("input_dir")
    ap.add_argument("--log-file", default="evaluation_comparison_log.txt")
    ap.add_argument("--tol", default="1,2,3")
    ap.add_argument("--event-tau", type=int, default=2)
    ap.add_argument("--gt-threshold", type=int, default=200)
    ap.add_argument("--blur-sigma", type=float, default=2.0)
    ap.add_argument("--min-event-px", type=int, default=3)
    ap.add_argument("--close-frac", type=float, default=0.8)
    args = ap.parse_args()

    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    evaluate_line_completion(
        args.gt_dir, args.results_dir, args.input_dir, args.log_file,
        taus=[int(t) for t in args.tol.split(",")], event_tau=args.event_tau,
        gt_threshold=args.gt_threshold, blur_sigma=args.blur_sigma,
        min_event_px=args.min_event_px, close_frac=args.close_frac,
    )

"""
Usage:
    python evaluate.py ./Outputs ./Results ./Inputs
    python evaluate.py ./Outputs ./Results ./Inputs --tol 1,2,3 --event-tau 2
"""