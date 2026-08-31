#!/usr/bin/env python3
"""
run_pipeline_invariance.py  —  Backbone-invariance of the DyRA-IIoT risk pipeline
                               (fills Table 12b in the main paper, Section 4.12).

It re-runs the *same* risk pipeline used in the Section 4.12 replay
    R(t) = P(t) * Impact(A) * gamma   ->   alert = K-consecutive( R(t) > tau )
on the identical set of replay windows, but with the probability trace P(t)
taken from a *different* backbone (e.g. Vanilla-Transformer or 1D-CNN) instead
of the reference TiDE. It reports, per backbone:
    - per-class detection rate (classes with >= K windows)
    - K-consecutive false-alarm rate on Normal windows
    - attack-to-alert delay (in windows and ms)
    - structural suppression check on low-criticality nodes (max R vs tau)
so you can confirm that only P(t)-calibration-dependent metrics move, while the
structural behavior is identical.

--------------------------------------------------------------------------
WHAT YOU NEED (reuse your Section 4.12 replay harness)
--------------------------------------------------------------------------
The Section 4.12 replay already builds, for each of the 58,833 windows, the
tuple (P(t), routed-node Impact, class label, benign/attack). To make this
script work you only need to DUMP those per-window arrays once per backbone.
The *metadata* (Impact, class, is_attack, order) is identical across backbones;
only P(t) changes. Save one file per backbone:

    replay/TiDE.npz
    replay/VanillaTransformer.npz
    replay/1D-CNN.npz            (optional)

Each .npz must contain aligned 1-D arrays, in the SAME window order that the
replay streams them (temporal / per-class-block order matters for the
K-consecutive rule):

    p        float [N]   P(t) for this backbone
    impact   float [N]   Impact(A) of the node each window is routed to
    is_attack int  [N]   1 for attack windows, 0 for Normal
    cls      str   [N]   class label ('Normal', 'Ransomware', 'DDoS_UDP', ...)
    gamma    float [N]   operational context factor (1.0 throughout the replay)
                         -- optional; defaults to 1.0 if absent

Example dump inside your existing replay loop:

    np.savez('replay/VanillaTransformer.npz',
             p=P_vt, impact=impact_per_window, is_attack=y,
             cls=np.array(class_per_window), gamma=np.full(N, 1.0))

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python run_pipeline_invariance.py --root replay/ --tau 0.5 --K 3 --window-ms 66.7
    python run_pipeline_invariance.py --root replay/ --md          # markdown out
    python run_pipeline_invariance.py --root replay/ --ref TiDE    # sanity-anchor

Only numpy is required.
"""

import argparse
import glob
import os
import numpy as np


def load_backbone(path):
    d = np.load(path, allow_pickle=True)
    p = np.asarray(d["p"], dtype=np.float64).ravel()
    impact = np.asarray(d["impact"], dtype=np.float64).ravel()
    is_attack = np.asarray(d["is_attack"]).astype(int).ravel()
    cls = np.asarray(d["cls"]).astype(str).ravel()
    gamma = (np.asarray(d["gamma"], dtype=np.float64).ravel()
             if "gamma" in d.files else np.full_like(p, 1.0))
    n = len(p)
    assert all(len(a) == n for a in (impact, is_attack, cls, gamma)), \
        f"{path}: arrays not aligned"
    if p.min() < 0 or p.max() > 1:          # logits -> sigmoid
        p = 1.0 / (1.0 + np.exp(-p))
    return dict(p=p, impact=impact, is_attack=is_attack, cls=cls, gamma=gamma)


def evaluate(data, tau, K, window_ms):
    p, impact, gamma = data["p"], data["impact"], data["gamma"]
    cls, is_attack = data["cls"], data["is_attack"]
    R = p * impact * gamma
    over = R > tau

    # ---- K-consecutive alert onsets, reset at class-block boundaries ----
    # In the Section 4.12 replay each attack class is a separate replay segment
    # routed to its target node, and Normal windows form their own stream, so
    # the K-consecutive run must reset whenever the class label changes; a
    # global run would leave the pipeline "latched" after the first alert and
    # spuriously mark later class blocks as missed. An alert is raised at window
    # t if over[t-K+1 .. t] are all True within the current block.
    alert_onset = np.zeros(len(over), dtype=bool)
    run = 0
    for t in range(len(over)):
        if t > 0 and cls[t] != cls[t - 1]:
            run = 0                        # new segment -> reset persistence
        run = run + 1 if over[t] else 0
        if run == K:                       # onset exactly when run first hits K
            alert_onset[t] = True

    # ---- false-alarm rate on Normal windows ----
    normal_mask = is_attack == 0
    n_normal = int(normal_mask.sum())
    false_alerts = int(alert_onset[normal_mask].sum())
    far = false_alerts / n_normal if n_normal else float("nan")

    # ---- per-class detection (classes with >= K windows) ----
    detection = {}
    for c in sorted(set(cls[is_attack == 1])):
        idx = np.where((cls == c) & (is_attack == 1))[0]
        if len(idx) < K:
            detection[c] = ("n/a (<K windows)", None)
            continue
        # detected if any K-consecutive alert onset occurs within this class block
        detected = bool(alert_onset[idx].any())
        # attack-to-alert delay: windows from block start to first onset
        onsets = np.where(alert_onset[idx])[0]
        delay = int(onsets[0]) if len(onsets) else None
        detection[c] = ("detected" if detected else "missed", delay)

    detected_classes = [c for c, (s, _) in detection.items() if s == "detected"]
    eligible = [c for c, (s, _) in detection.items() if not s.startswith("n/a")]
    det_rate = len(detected_classes) / len(eligible) if eligible else float("nan")

    # median attack-to-alert delay across detected classes
    delays = [d for _, (s, d) in detection.items() if s == "detected" and d is not None]
    med_delay = int(np.median(delays)) if delays else None

    # ---- structural suppression: low-criticality nodes ----
    # a node/window is "structurally suppressed" if impact * gamma_max < tau,
    # i.e. it can never alert regardless of P(t). Use gamma max seen (or 1.5).
    gamma_max = max(float(gamma.max()), 1.5)
    low_mask = impact * gamma_max < tau
    max_R_low = float(R[low_mask].max()) if low_mask.any() else float("nan")

    return dict(
        far=far, false_alerts=false_alerts, n_normal=n_normal,
        det_rate=det_rate, detected=len(detected_classes), eligible=len(eligible),
        med_delay=med_delay, med_delay_ms=(med_delay * window_ms if med_delay else None),
        detection=detection,
        suppressed_fraction=float(low_mask.mean()),
        max_R_low=max_R_low, gamma_max=gamma_max,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with per-backbone .npz replay dumps")
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--window-ms", type=float, default=66.7)
    ap.add_argument("--ref", default="TiDE", help="reference backbone name for anchoring")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, args.glob)))
    if not files:
        raise SystemExit(f"No files matched {os.path.join(args.root, args.glob)}")

    results = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        results[name] = evaluate(load_backbone(f), args.tau, args.K, args.window_ms)

    # order: reference first, then the rest
    names = ([n for n in results if n.lower() == args.ref.lower()]
             + [n for n in results if n.lower() != args.ref.lower()])

    def fmt(r):
        far = f"{r['far']*100:.3f}%"
        det = f"{r['detected']}/{r['eligible']}"
        delay = (f"{r['med_delay']} win (~{r['med_delay_ms']:.0f} ms)"
                 if r['med_delay'] is not None else "n/a")
        supp = f"max R={r['max_R_low']:.3f} < tau={args.tau} (OK)" \
            if r['max_R_low'] < args.tau else f"max R={r['max_R_low']:.3f} !!"
        return far, det, delay, supp

    header = ["Metric"] + names
    metrics = ["K-consec. false-alarm rate", "Detected classes (>=K)",
               "Attack-to-alert delay (median)", "Low-criticality suppression"]
    table = {m: [] for m in metrics}
    for n in names:
        far, det, delay, supp = fmt(results[n])
        table[metrics[0]].append(far)
        table[metrics[1]].append(det)
        table[metrics[2]].append(delay)
        table[metrics[3]].append(supp)

    # console
    print("\nBackbone-invariance of the Section 4.12 replay "
          f"(tau={args.tau}, K={args.K}, window={args.window_ms} ms)\n")
    colw = max(30, *(len(n) for n in names)) if names else 30
    print("  " + "Metric".ljust(32) + "".join(n.ljust(colw + 2) for n in names))
    print("  " + "-" * (32 + (colw + 2) * len(names)))
    for m in metrics:
        print("  " + m.ljust(32) + "".join(v.ljust(colw + 2) for v in table[m]))

    if args.md:
        print("\nMarkdown (Table 12b — paste, then convert to a Word table):\n")
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for m in metrics:
            print("| " + m + " | " + " | ".join(table[m]) + " |")

    # per-class detail
    print("\nPer-class detection detail:")
    for n in names:
        print(f"  [{n}]")
        for c, (s, d) in results[n]["detection"].items():
            extra = f", onset +{d} win" if d is not None else ""
            print(f"    {c:<16} {s}{extra}")

    print("\nInterpretation: the false-alarm rate and detection should match the "
          "TiDE reference within noise; the low-criticality suppression row is "
          "identical by construction. Fill these values into Table 12b.")


if __name__ == "__main__":
    main()
