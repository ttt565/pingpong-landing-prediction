#!/usr/bin/env python3
"""Phase-1 EXIT TEST on real video: measure within-arc heteroscedasticity (H)
and confidence-vs-precision calibration for a ball detector on OpenTTGames
(120 fps static-camera table tennis, per-frame ball labels).

Data (gitignored, ~226 MB; CC BY-NC-SA 4.0, research use):
    mkdir -p real/data && cd real/data
    curl -L -O https://lab.osai.ai/datasets/openttgames/data/test_2.mp4
    curl -L -o test_2.zip https://lab.osai.ai/datasets/openttgames/data/test_2.zip
    unzip test_2.zip -d test_2_markup

Detector: temporal-median background subtraction + small-blob selection with
temporal continuity — the same class as the Gazebo Route-B detector, i.e. a
lightweight stand-in for TrackNet. THE HARNESS IS DETECTOR-AGNOSTIC: score any
(frame,u,v,conf) CSV with --detections to run the same analysis on TrackNet
or any other model's output.

Fair protocol: detection runs without labels except a disclosed one-time
anchor at each arc's first labeled frame (a real system anchors on serve
detection). Per labeled frame: error = |detection − label| px. Analysis:
confidence-decile calibration curve sigma(conf), gamma-hat + conf-noise-hat
under the synthetic model conf=(1/sigma)^gamma*exp(N(0,cn)), per-arc H, and
dropout rate — the exact quantities the fig5 thresholds (run_miscalibration)
need for the go/no-go.

Outputs: results_exit_test.md + fig_exit_test.png (in real/)
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
DIFF_THR = 60          # summed |RGB| distance to background: blob membership
AREA_RANGE = (3, 600)  # ball blob size window (px) at 1920x1080
MAX_WH = 60
MIN_EXTENT = 0.25
R_SEARCH0, R_GROW, R_MAX = 60.0, 40.0, 220.0
N_BG = 36              # frames sampled across the video for the median bg
MIN_ARC = 12           # min labeled frames for an arc to count
N_BINS = 8             # confidence bins for the calibration curve


def load_labels(markup_dir):
    with open(os.path.join(markup_dir, "ball_markup.json")) as f:
        raw = json.load(f)
    lab = {int(k): (v["x"], v["y"]) for k, v in raw.items()
           if v["x"] >= 0 and v["y"] >= 0}
    frames = sorted(lab)
    arcs, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b - a <= 2:
            cur.append(b)
        else:
            arcs.append(cur)
            cur = [b]
    arcs.append(cur)
    return lab, [a for a in arcs if len(a) >= MIN_ARC]


def median_background(cap, n_frames):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, N_BG).astype(int)
    samples = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            samples.append(fr)
    stack = np.stack(samples)
    H = stack.shape[1]
    bg = np.empty(stack.shape[1:], np.float32)
    for r in range(0, H, 90):
        bg[r:r + 90] = np.median(stack[:, r:r + 90], axis=0)
    return bg


def detect(frame, bg, pred, radius):
    """Best ball-candidate blob near the predicted position.
    Returns (u, v, conf) or None."""
    diff = np.abs(frame.astype(np.float32) - bg).sum(axis=2)
    mask = (diff > DIFF_THR).astype(np.uint8)
    n, cc, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best, best_d = None, np.inf
    for j in range(1, n):
        area = stats[j, cv2.CC_STAT_AREA]
        w, h = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        if not (AREA_RANGE[0] <= area <= AREA_RANGE[1]) or w > MAX_WH or h > MAX_WH:
            continue
        extent = area / max(w * h, 1)
        if extent < MIN_EXTENT:
            continue
        u, v = cents[j]
        d = np.hypot(u - pred[0], v - pred[1])
        if d < radius and d < best_d:
            ys, xs = np.nonzero(cc == j)
            conf = float(diff[ys, xs].mean()) * extent
            best, best_d = (float(u), float(v), conf), d
    return best


def run_detector(video, labels, arcs):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    print("building median background ...")
    bg = median_background(cap, N_BG)

    rows = []          # (arc_id, frame, u, v, conf, err_px)
    dropped = 0
    labeled_total = 0
    for ai, arc in enumerate(arcs):
        f0, f1 = arc[0], arc[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        pos = np.array(labels[f0], float)     # disclosed one-time anchor
        vel = np.zeros(2)
        radius = R_SEARCH0
        for fr_idx in range(f0, f1 + 1):
            ok, frame = cap.read()
            if not ok:
                break
            det = detect(frame, bg, pos + vel, radius)
            has_label = fr_idx in labels
            labeled_total += has_label
            if det is None:
                dropped += has_label
                pos = pos + vel
                radius = min(radius + R_GROW, R_MAX)
                continue
            u, v, conf = det
            new = np.array([u, v])
            vel = new - pos
            pos = new
            radius = R_SEARCH0
            if has_label:
                err = float(np.hypot(u - labels[fr_idx][0],
                                     v - labels[fr_idx][1]))
                rows.append((ai, fr_idx, u, v, conf, err))
        print(f"arc {ai}: frames {f0}-{f1}, "
              f"{sum(1 for r in rows if r[0] == ai)} scored")
    cap.release()
    return rows, dropped, labeled_total


def analyze(rows, dropped, labeled_total, out_md, out_fig):
    arr = np.array([(r[0], r[4], r[5]) for r in rows])
    arc_id, conf, err = arr[:, 0], arr[:, 1], arr[:, 2]

    # gross-failure split: identity errors (grabbed something else entirely)
    ok = err < 60
    gross = 1.0 - ok.mean()

    qs = np.quantile(conf[ok], np.linspace(0, 1, N_BINS + 1))
    bins = np.clip(np.searchsorted(qs, conf, side="right") - 1, 0, N_BINS - 1)
    sig_bin, conf_bin = np.full(N_BINS, np.nan), np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        m = ok & (bins == b)
        if m.sum() >= 8:
            sig_bin[b] = np.median(err[m]) / 1.177     # per-axis sigma, Rayleigh
            conf_bin[b] = np.median(conf[m])

    good = np.isfinite(sig_bin)
    gamma, icpt = np.polyfit(np.log(1.0 / sig_bin[good]),
                             np.log(conf_bin[good]), 1)
    # per-frame conf noise under conf=(1/sigma)^gamma*e^eps
    sig_frame = sig_bin[bins]
    m = ok & np.isfinite(sig_frame)
    eps = np.log(conf[m]) - (gamma * np.log(1.0 / sig_frame[m]) + icpt)
    cn_hat = float(eps.std())

    Hs = []
    for a in np.unique(arc_id):
        s = sig_frame[(arc_id == a) & m]
        if len(s) >= 8:
            Hs.append(float(np.std(s) / np.mean(s)))
    H_mean = float(np.mean(Hs))

    drop_rate = dropped / max(labeled_total, 1)
    lines = [
        "# Real-video exit test (OpenTTGames test_2, 120 fps)\n",
        f"{len(rows)} scored detections over {len(np.unique(arc_id))} arcs; "
        f"dropout {100 * drop_rate:.1f}% of labeled frames, gross failures "
        f"(err>60 px) {100 * gross:.1f}%. Detector: median-background + "
        "temporal continuity (lightweight TrackNet stand-in; harness accepts "
        "any detector CSV).\n",
        "## Confidence-decile calibration\n",
        "| bin | median conf | sigma (px) |", "|---|---|---|"]
    for b in range(N_BINS):
        if np.isfinite(sig_bin[b]):
            lines.append(f"| {b} | {conf_bin[b]:.1f} | {sig_bin[b]:.2f} |")
    lines += [
        "\n## Exit-test numbers vs the fig5 thresholds\n",
        f"- sigma dynamic range across confidence bins: "
        f"{np.nanmax(sig_bin) / np.nanmin(sig_bin):.2f}x",
        f"- calibration exponent gamma-hat = {gamma:.2f} "
        f"(threshold from run_miscalibration: >= 0.5)",
        f"- confidence log-noise cn-hat = {cn_hat:.2f} "
        f"(threshold: <= ~0.6; at >= 1.0 confidence weighting HURTS)",
        f"- within-arc heteroscedasticity H = {H_mean:.2f} "
        f"(mean over arcs; sweep says the M3-vs-robust gap only opens at "
        f"high bad-frame rates even when H ~ 1)",
        f"- dropout rate {100 * drop_rate:.1f}% — failures are dropouts, "
        f"as in the rendered Route B",
        "\n## Verdict",
        f"gamma-hat {'passes' if gamma >= 0.5 else 'FAILS'} and cn-hat "
        f"{'passes' if cn_hat <= 0.6 else 'FAILS'} the fig5 thresholds for "
        "this detector on this footage. Per the robust-baseline result, the "
        "deployable default remains a robust loss; confidence weighting is "
        "justified only if BOTH pass on the production detector.",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(out_md, "w") as f:
        f.write(text)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(conf_bin[good], sig_bin[good], "o-")
    a1.set_xlabel("confidence (median per decile)")
    a1.set_ylabel("empirical per-axis sigma (px)")
    a1.set_title(f"Calibration: gamma={gamma:.2f}, log-noise={cn_hat:.2f}")
    a1.grid(alpha=.3)
    a2.hist(Hs, bins=10, color="#1f77b4", alpha=.8)
    a2.set_xlabel("within-arc H"); a2.set_ylabel("arcs")
    a2.set_title(f"H per arc (mean {H_mean:.2f})")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=130)
    print(f"wrote {out_md} + {out_fig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(HERE, "data", "test_2.mp4"))
    ap.add_argument("--markup", default=os.path.join(HERE, "data", "test_2_markup"))
    ap.add_argument("--detections", default=None,
                    help="skip the built-in detector; score this CSV "
                         "(columns: frame,u,v,conf) against the labels")
    ap.add_argument("--save-detections",
                    default=os.path.join(HERE, "detections.csv"))
    a = ap.parse_args()

    labels, arcs = load_labels(a.markup)
    print(f"{sum(len(x) for x in arcs)} labeled frames in {len(arcs)} arcs")

    if a.detections:
        with open(a.detections) as f:
            det = {int(r["frame"]): (float(r["u"]), float(r["v"]),
                                     float(r["conf"]))
                   for r in csv.DictReader(f)}
        rows, dropped, labeled_total = [], 0, 0
        for ai, arc in enumerate(arcs):
            for fr in arc:
                labeled_total += 1
                if fr in det:
                    u, v, c = det[fr]
                    err = float(np.hypot(u - labels[fr][0], v - labels[fr][1]))
                    rows.append((ai, fr, u, v, c, err))
                else:
                    dropped += 1
    else:
        rows, dropped, labeled_total = run_detector(a.video, labels, arcs)
        with open(a.save_detections, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arc", "frame", "u", "v", "conf", "err_px"])
            w.writerows(rows)

    analyze(rows, dropped, labeled_total,
            os.path.join(HERE, "results_exit_test.md"),
            os.path.join(HERE, "fig_exit_test.png"))


if __name__ == "__main__":
    main()
