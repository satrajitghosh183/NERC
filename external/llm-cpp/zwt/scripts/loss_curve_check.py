#!/usr/bin/env python3
"""Loss-curve quality gate for zwt vs a published baseline.

Reads the metrics CSV emitted by `zwt_pretrain --metrics-csv <path>`
(columns: step,loss,lr,grad_norm,tokens_seen,wall_secs,tok_per_s) and
compares it against a reference CSV that must supply at least two
columns: an x-axis (step or tokens_seen) and a loss column.

The alignment axis is chosen via --axis {step,tokens}.  Reference rows
are linearly interpolated onto the zwt x-axis so the two curves do not
need to share sampling cadence.

Quality gate semantics:

    abs_delta_i     = zwt_loss_i - ref_loss_i
    rel_delta_i     = abs_delta_i / max(ref_loss_i, 1e-6)

The run PASSES when the p95 of |rel_delta| over the overlapping range
stays under --rel-tol (default 3%) AND the last-window mean of abs_delta
stays under --abs-tol (default 0.05 nats).  Anything else FAILS with a
nonzero exit code — wire this into CI once an H100 run is producing a
CSV daily.

Optional plot via matplotlib is written to --plot <png>.  If matplotlib
isn't installed, plotting is silently skipped; the numerical gate still
runs and sets the exit code.

Example:

    loss_curve_check.py \\
        --zwt runs/olmo2_1b/metrics.csv \\
        --ref references/olmo2_1b_published.csv \\
        --axis tokens \\
        --ref-x tokens_seen --ref-y train_loss \\
        --rel-tol 0.03 --abs-tol 0.05 \\
        --plot runs/olmo2_1b/loss_curve.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass


@dataclass
class Curve:
    x: list[float]
    y: list[float]
    label: str


def _read_csv(path: str, x_col: str, y_col: str, label: str) -> Curve:
    xs: list[float] = []
    ys: list[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if x_col not in reader.fieldnames or y_col not in reader.fieldnames:
            raise SystemExit(
                f"{path}: missing required column "
                f"(have {reader.fieldnames}, need {x_col!r} and {y_col!r})"
            )
        for row in reader:
            try:
                xv = float(row[x_col])
                yv = float(row[y_col])
            except ValueError:
                continue
            xs.append(xv)
            ys.append(yv)
    if not xs:
        raise SystemExit(f"{path}: no usable rows")
    # Enforce monotonic-increasing x by sorting (some runs restart and
    # emit duplicate step numbers; keep the last value per x).
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    dedup_x: list[float] = []
    dedup_y: list[float] = []
    for i in order:
        if dedup_x and xs[i] == dedup_x[-1]:
            dedup_y[-1] = ys[i]
        else:
            dedup_x.append(xs[i])
            dedup_y.append(ys[i])
    return Curve(x=dedup_x, y=dedup_y, label=label)


def _interp(ref: Curve, xs: list[float]) -> list[float | None]:
    """Linear interp of ref onto xs; None outside the ref range."""
    out: list[float | None] = []
    rx, ry = ref.x, ref.y
    j = 0
    for x in xs:
        if x < rx[0] or x > rx[-1]:
            out.append(None)
            continue
        while j + 1 < len(rx) and rx[j + 1] < x:
            j += 1
        if rx[j] == x or j + 1 >= len(rx):
            out.append(ry[j])
            continue
        x0, x1 = rx[j], rx[j + 1]
        y0, y1 = ry[j], ry[j + 1]
        t = (x - x0) / (x1 - x0)
        out.append(y0 + t * (y1 - y0))
    return out


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zwt", required=True, help="zwt metrics CSV")
    ap.add_argument("--ref", required=True, help="reference curve CSV")
    ap.add_argument("--axis", choices=("step", "tokens"), default="tokens")
    ap.add_argument("--zwt-x", default=None,
                    help="override zwt x column (defaults follow --axis)")
    ap.add_argument("--zwt-y", default="loss")
    ap.add_argument("--ref-x", default=None,
                    help="override ref x column (defaults follow --axis)")
    ap.add_argument("--ref-y", default="loss")
    ap.add_argument("--rel-tol", type=float, default=0.03,
                    help="p95 |rel_delta| must stay below this (default 3%%)")
    ap.add_argument("--abs-tol", type=float, default=0.05,
                    help="last-window mean abs_delta must stay below this")
    ap.add_argument("--window", type=int, default=20,
                    help="number of tail points for the abs-tol check")
    ap.add_argument("--plot", default=None, help="optional PNG output")
    args = ap.parse_args(argv)

    default_x = "step" if args.axis == "step" else "tokens_seen"
    zwt = _read_csv(args.zwt, args.zwt_x or default_x, args.zwt_y, "zwt")
    ref = _read_csv(args.ref, args.ref_x or default_x, args.ref_y, "ref")

    ref_on_zwt = _interp(ref, zwt.x)
    abs_delta: list[float] = []
    rel_delta: list[float] = []
    overlap_x: list[float] = []
    for x, y, ry in zip(zwt.x, zwt.y, ref_on_zwt):
        if ry is None:
            continue
        ad = y - ry
        rd = ad / max(abs(ry), 1e-6)
        abs_delta.append(ad)
        rel_delta.append(rd)
        overlap_x.append(x)

    if not abs_delta:
        print("no overlap between zwt and ref x-ranges", file=sys.stderr)
        return 2

    p95_rel = _percentile([abs(r) for r in rel_delta], 0.95)
    median_rel = _percentile([abs(r) for r in rel_delta], 0.50)
    window = min(args.window, len(abs_delta))
    tail_mean_abs = sum(abs_delta[-window:]) / window

    print(f"overlap points: {len(abs_delta)} / {len(zwt.x)}")
    print(f"median |rel_delta|: {median_rel:.4f}")
    print(f"p95    |rel_delta|: {p95_rel:.4f}   (tol {args.rel_tol})")
    print(f"tail-{window} mean abs_delta: {tail_mean_abs:+.4f}"
          f"   (tol {args.abs_tol})")

    rel_ok = p95_rel <= args.rel_tol
    abs_ok = abs(tail_mean_abs) <= args.abs_tol

    if args.plot:
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
            ax[0].plot(zwt.x, zwt.y, label="zwt", lw=1.2)
            ax[0].plot(ref.x, ref.y, label="reference", lw=1.2, alpha=0.8)
            ax[0].set_ylabel("train loss")
            ax[0].legend()
            ax[0].set_title(f"loss curve vs baseline (axis={args.axis})")
            ax[1].plot(overlap_x, abs_delta, color="tab:red", lw=1.0)
            ax[1].axhline(0, color="k", lw=0.5, alpha=0.5)
            ax[1].set_ylabel("zwt − ref")
            ax[1].set_xlabel(args.axis)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            print(f"plot: {args.plot}")
        except ImportError:
            print("matplotlib not installed; skipping plot", file=sys.stderr)

    if rel_ok and abs_ok:
        print("PASS")
        return 0
    print("FAIL:", "rel" if not rel_ok else "", "abs" if not abs_ok else "")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
