#!/usr/bin/env python3
"""Aggregate stable_center_audit.csv into the summary tables. Read-only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
CSV = OUT_DIR / "stable_center_audit.csv"

WINDOWS = (5, 10, 15, 20, 25, 30)
PCTS = (10, 25, 50, 75, 90, 95)
RATIO_TOL = 0.02

FRAME_BUCKETS = [("3-4", 3, 4), ("5-6", 5, 6), ("7-10", 7, 10),
                 ("11-20", 11, 20), ("21-50", 21, 50), ("51+", 51, 10**9)]
DUR_BUCKETS = [("<0.10s", -np.inf, 0.10), ("0.10-0.20s", 0.10, 0.20),
               ("0.20-0.40s", 0.20, 0.40), ("0.40-0.80s", 0.40, 0.80),
               (">=0.80s", 0.80, np.inf)]


def load():
    import csv as _csv
    with CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if k in ("song_id", "split"):
            cols[k] = np.array(vals, dtype=object)
        else:
            cols[k] = np.array([np.nan if v == "" else float(v) for v in vals], dtype=float)
    return cols, len(rows)


def dist(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    o = {"n": int(x.size), "min": float(x.min())}
    for p in PCTS:
        o[f"p{p}"] = float(np.percentile(x, p))
    o["max"] = float(x.max())
    o["mean"] = float(x.mean())
    return o


def fmt(d, prec=2):
    if d.get("n", 0) == 0:
        return "| n=0 | — | — | — | — | — | — | — | — |"
    f = lambda v: f"{v:.{prec}f}"
    return (f"| {d['n']:,} | {f(d['min'])} | {f(d['p10'])} | {f(d['p25'])} | {f(d['p50'])} "
            f"| {f(d['p75'])} | {f(d['p90'])} | {f(d['p95'])} | {f(d['max'])} |")


def main():
    c, n = load()
    R = {}
    R["clean_notes"] = n

    nmid = c["n_frames_middle50"]
    elig_mid = np.isfinite(c["middle50_median_cents"])
    R["eligible_middle50"] = int(elig_mid.sum())
    R["ineligible_middle50"] = int((~elig_mid).sum())
    R["eligible_q1"] = int(np.isfinite(c["first_quarter_median_cents"]).sum())
    R["eligible_q4"] = int(np.isfinite(c["last_quarter_median_cents"]).sum())
    R["eligible_mid_third"] = int(np.isfinite(c["middle_third_median_cents"]).sum())
    R["eligible_first_third"] = int(np.isfinite(c["first_third_median_cents"]).sum())
    R["eligible_last_third"] = int(np.isfinite(c["last_third_median_cents"]).sum())
    R["eligible_mid_drift"] = int(np.isfinite(c["middle50_drift_cents"]).sum())
    R["eligible_full_drift"] = int(np.isfinite(c["full_drift_cents"]).sum())
    R["eligible_onset_center"] = int(np.isfinite(c["onset_to_center"]).sum())
    R["eligible_release_center"] = int(np.isfinite(c["release_from_center"]).sum())
    R["all_three_regions"] = int((np.isfinite(c["first_quarter_median_cents"])
                                 & elig_mid & np.isfinite(c["last_quarter_median_cents"])).sum())
    R["notes_with_time_gaps"] = int(np.nansum(c["has_time_gaps"]))

    # frame count / duration overview
    R["frames"] = dist(c["num_frames"])
    R["duration"] = dist(c["duration_s"])

    # §2 center differences
    R["diff"] = {k: dist(c[k]) for k in
                 ("abs_full_minus_middle50", "abs_onset_to_center", "abs_release_from_center")}
    # §2 raw region medians
    R["medians"] = {k: dist(c[k]) for k in
                    ("full_median_cents", "middle50_median_cents", "middle_third_median_cents",
                     "first_quarter_median_cents", "last_quarter_median_cents")}

    # §3 within-region stability
    R["stab"] = {k: dist(c[k]) for k in
                 ("full_iqr_cents", "full_mad_cents", "full_std_cents", "full_p90_p10_cents",
                  "middle50_iqr_cents", "middle50_mad_cents", "middle50_std_cents",
                  "middle50_p90_p10_cents")}

    # §3/§9 ratios
    R["ratio"] = {}
    for name in ("iqr", "mad", "std"):
        r = c[f"ratio_{name}"]
        v = r[np.isfinite(r)]
        R["ratio"][name] = {
            "dist": dist(v),
            "pct_more_stable": float(100 * np.count_nonzero(v < 1 - RATIO_TOL) / v.size),
            "pct_approx_equal": float(100 * np.count_nonzero(np.abs(v - 1) <= RATIO_TOL) / v.size),
            "pct_less_stable": float(100 * np.count_nonzero(v > 1 + RATIO_TOL) / v.size),
            "pct_strictly_lower": float(100 * np.count_nonzero(v < 1.0) / v.size),
            "pct_exactly_equal": float(100 * np.count_nonzero(v == 1.0) / v.size),
        }
        mid = c[f"middle50_{name}_cents"]; full = c[f"full_{name}_cents"]
        m = np.isfinite(mid) & np.isfinite(full)
        R["ratio"][name]["median_full"] = float(np.median(full[m]))
        R["ratio"][name]["median_mid"] = float(np.median(mid[m]))
        R["ratio"][name]["rel_reduction_pct"] = float(
            100 * (np.median(full[m]) - np.median(mid[m])) / np.median(full[m])
        ) if np.median(full[m]) > 0 else None
        R["ratio"][name]["median_abs_improvement"] = float(np.median(full[m] - mid[m]))

    # §4 within-window fractions
    R["within"] = {}
    for w in WINDOWS:
        col = c[f"mid_frac_within_{w}c"]
        v = col[np.isfinite(col)]
        R["within"][w] = {
            "dist": dist(v),
            "pct_notes_ge_75": float(100 * np.count_nonzero(v >= 0.75) / v.size),
            "pct_notes_ge_90": float(100 * np.count_nonzero(v >= 0.90) / v.size),
            "pct_notes_eq_100": float(100 * np.count_nonzero(v >= 0.999999) / v.size),
            "mean_frac": float(v.mean()),
        }
    R["within_full"] = {}
    for w in WINDOWS:
        col = c[f"full_frac_within_{w}c"]
        v = col[np.isfinite(col)]
        R["within_full"][w] = {"pct_notes_ge_75": float(100 * np.count_nonzero(v >= 0.75) / v.size),
                               "median_frac": float(np.median(v))}

    # §5 drift
    R["drift"] = {
        "mid_abs": dist(c["abs_middle50_drift_cents"]),
        "full_abs": dist(c["abs_full_drift_cents"]),
        "mid_signed": dist(c["middle50_drift_cents"]),
        "mid_slope": dist(c["middle50_slope_cents_per_dur"]),
        "full_slope": dist(c["full_slope_cents_per_dur"]),
    }
    md = c["abs_middle50_drift_cents"]; mdv = md[np.isfinite(md)]
    R["drift"]["mid_frac_under"] = {t: float(100 * np.count_nonzero(mdv < t) / mdv.size)
                                    for t in (5, 10, 15, 20, 30)}
    fd = c["abs_full_drift_cents"]; fdv = fd[np.isfinite(fd)]
    R["drift"]["full_frac_under"] = {t: float(100 * np.count_nonzero(fdv < t) / fdv.size)
                                     for t in (5, 10, 15, 20, 30)}

    # §6 onset/release signed
    R["onset_release"] = {
        "onset_signed": dist(c["onset_to_center"]),
        "release_signed": dist(c["release_from_center"]),
        "onset_abs": dist(c["abs_onset_to_center"]),
        "release_abs": dist(c["abs_release_from_center"]),
    }
    o = c["onset_to_center"]; ov = o[np.isfinite(o)]
    r_ = c["release_from_center"]; rv = r_[np.isfinite(r_)]
    R["onset_release"]["pct_onset_below_center"] = float(100 * np.count_nonzero(ov > 0) / ov.size)
    R["onset_release"]["pct_onset_above_center"] = float(100 * np.count_nonzero(ov < 0) / ov.size)
    R["onset_release"]["pct_release_above_center"] = float(100 * np.count_nonzero(rv > 0) / rv.size)
    R["onset_release"]["pct_release_below_center"] = float(100 * np.count_nonzero(rv < 0) / rv.size)
    # is the middle a plateau? compare |onset diff| and |release diff| to middle IQR
    mi = c["middle50_iqr_cents"]
    m = np.isfinite(c["abs_onset_to_center"]) & np.isfinite(mi)
    R["onset_release"]["pct_onset_diff_exceeds_mid_iqr"] = float(
        100 * np.count_nonzero(c["abs_onset_to_center"][m] > mi[m]) / m.sum())
    m2 = np.isfinite(c["abs_release_from_center"]) & np.isfinite(mi)
    R["onset_release"]["pct_release_diff_exceeds_mid_iqr"] = float(
        100 * np.count_nonzero(c["abs_release_from_center"][m2] > mi[m2]) / m2.sum())

    # §7 stratification
    def bucket_stats(mask):
        sub = {k: c[k][mask] for k in c if k not in ("song_id", "split")}
        g = lambda k: float(np.nanmedian(sub[k])) if np.isfinite(sub[k]).any() else None
        elig = int(np.isfinite(sub["middle50_median_cents"]).sum())
        fr = sub["mid_frac_within_20c"]
        rat = sub["ratio_iqr"]; ratv = rat[np.isfinite(rat)]
        return {
            "n": int(mask.sum()),
            "eligible_mid": elig,
            "pct_eligible_mid": float(100 * elig / mask.sum()) if mask.sum() else None,
            "med_full_iqr": g("full_iqr_cents"),
            "med_mid_iqr": g("middle50_iqr_cents"),
            "med_mid_mad": g("middle50_mad_cents"),
            "med_abs_mid_drift": g("abs_middle50_drift_cents"),
            "med_frac_within_20": float(np.nanmedian(fr)) if np.isfinite(fr).any() else None,
            "med_ratio_iqr": float(np.median(ratv)) if ratv.size else None,
            "pct_mid_more_stable": float(100 * np.count_nonzero(ratv < 1 - RATIO_TOL) / ratv.size) if ratv.size else None,
            "med_duration": g("duration_s"),
            "med_full_mad": g("full_mad_cents"),
        }

    R["by_frames"] = {}
    for name, lo, hi in FRAME_BUCKETS:
        R["by_frames"][name] = bucket_stats((c["num_frames"] >= lo) & (c["num_frames"] <= hi))
    R["by_duration"] = {}
    for name, lo, hi in DUR_BUCKETS:
        m = (c["duration_s"] >= lo) & (c["duration_s"] < hi) if np.isfinite(hi) else (c["duration_s"] >= lo)
        if name == "<0.10s":
            m = c["duration_s"] < hi
        R["by_duration"][name] = bucket_stats(m)

    # §8 candidate criteria
    R["criteria"] = {}
    base = elig_mid
    nb = int(base.sum())
    for label, w in (("A_loose_30c", 30), ("B_moderate_20c", 20), ("C_strict_10c", 10)):
        f = c[f"mid_frac_within_{w}c"]
        ok = base & (f >= 0.75)
        okd = ok & np.isfinite(c["abs_middle50_drift_cents"]) & (c["abs_middle50_drift_cents"] < 20)
        # notes with no drift estimate cannot satisfy the drift-augmented version
        R["criteria"][label] = {
            "n_eligible": nb,
            "count": int(ok.sum()),
            "pct_of_eligible": float(100 * ok.sum() / nb),
            "pct_of_all_clean": float(100 * ok.sum() / n),
            "count_with_drift20": int(okd.sum()),
            "pct_of_eligible_with_drift20": float(100 * okd.sum() / nb),
        }

    # split breakdown
    R["by_split"] = {}
    for sp in ("train", "validation", "test"):
        m = c["split"] == sp
        st = bucket_stats(m)
        rat = c["ratio_iqr"][m]; ratv = rat[np.isfinite(rat)]
        f20 = c["mid_frac_within_20c"][m]
        st["pct_ge75_within20"] = float(100 * np.count_nonzero(f20[np.isfinite(f20)] >= 0.75) / np.isfinite(f20).sum())
        f10 = c["mid_frac_within_10c"][m]
        st["pct_ge75_within10"] = float(100 * np.count_nonzero(f10[np.isfinite(f10)] >= 0.75) / np.isfinite(f10).sum())
        f30 = c["mid_frac_within_30c"][m]
        st["pct_ge75_within30"] = float(100 * np.count_nonzero(f30[np.isfinite(f30)] >= 0.75) / np.isfinite(f30).sum())
        st["n_songs"] = int(len(set(c["song_id"][m].tolist())))
        R["by_split"][sp] = st

    # stored core-median agreement: how close is production core-trim to middle50?
    sm = c["stored_core_median_cents"]; mm = c["middle50_median_cents"]
    m = np.isfinite(sm) & np.isfinite(mm)
    R["core_vs_mid50"] = dist(np.abs(sm[m] - mm[m]))
    fm = c["stored_median_cents"]
    R["storedfull_vs_mid50"] = dist(np.abs(fm[m] - mm[m]))
    R["core_vs_storedfull"] = dist(np.abs(sm - fm))
    # production core trim only engages at >=10 voiced frames
    lt10 = c["num_frames"] < 10
    R["pct_notes_under10_frames"] = float(100 * lt10.sum() / n)
    R["n_notes_under10_frames"] = int(lt10.sum())
    R["core_equals_full_when_short"] = float(
        100 * np.count_nonzero(np.abs(sm[lt10] - fm[lt10]) < 1e-12) / lt10.sum())

    (OUT_DIR / "_summary_stats.json").write_text(json.dumps(R, indent=2, default=float) + "\n",
                                                 encoding="utf-8")

    # ---- console report ----
    P = print
    P(f"clean notes: {R['clean_notes']:,}")
    P(f"eligible middle50 (>=3 frames in 0.25-0.75): {R['eligible_middle50']:,} "
      f"({100*R['eligible_middle50']/n:.2f}%)   ineligible: {R['ineligible_middle50']:,}")
    P(f"eligible q1 {R['eligible_q1']:,}  q4 {R['eligible_q4']:,}  midthird {R['eligible_mid_third']:,}"
      f"  all3regions {R['all_three_regions']:,}")
    P(f"eligible mid drift (>=4 frames) {R['eligible_mid_drift']:,}  full drift {R['eligible_full_drift']:,}")
    P(f"notes with interleaved unvoiced gaps: {R['notes_with_time_gaps']:,} ({100*R['notes_with_time_gaps']/n:.1f}%)")
    P(f"frames: median {R['frames']['p50']:.0f} p10 {R['frames']['p10']:.0f} p90 {R['frames']['p90']:.0f} max {R['frames']['max']:.0f}")
    P()
    P("STABILITY (cents)            |      n |   min |   P10 |   P25 |   med |   P75 |   P90 |   P95 |    max")
    for k in ("full_iqr_cents", "middle50_iqr_cents", "full_mad_cents", "middle50_mad_cents",
              "full_std_cents", "middle50_std_cents", "full_p90_p10_cents", "middle50_p90_p10_cents"):
        P(f"{k:28s} {fmt(R['stab'][k])}")
    P()
    P("RATIO mid/full")
    for k, v in R["ratio"].items():
        P(f"  {k}: median ratio {v['dist']['p50']:.4f}  P25 {v['dist']['p25']:.4f}  P75 {v['dist']['p75']:.4f}")
        P(f"       more stable {v['pct_more_stable']:.1f}%  approx equal {v['pct_approx_equal']:.1f}%  "
          f"less stable {v['pct_less_stable']:.1f}%  (strictly<1: {v['pct_strictly_lower']:.1f}%, ==1: {v['pct_exactly_equal']:.1f}%)")
        P(f"       median full {v['median_full']:.3f} -> mid {v['median_mid']:.3f}  "
          f"reduction {v['rel_reduction_pct']:.1f}%  median abs improvement {v['median_abs_improvement']:.3f}")
    P()
    P("CENTER DIFFERENCES")
    for k, v in R["diff"].items():
        P(f"{k:28s} {fmt(v)}")
    P()
    P("WITHIN-WINDOW (middle 50%)")
    for w in WINDOWS:
        v = R["within"][w]
        P(f"  +-{w:2d}c: median frac {v['dist']['p50']:.3f} mean {v['mean_frac']:.3f} | "
          f">=75% of frames: {v['pct_notes_ge_75']:.1f}% of notes | >=90%: {v['pct_notes_ge_90']:.1f}% | 100%: {v['pct_notes_eq_100']:.1f}%")
    P()
    P("DRIFT (abs, cents across region)")
    P(f"{'middle50':28s} {fmt(R['drift']['mid_abs'])}")
    P(f"{'full':28s} {fmt(R['drift']['full_abs'])}")
    P(f"  mid under: " + "  ".join(f"<{t}c {v:.1f}%" for t, v in R["drift"]["mid_frac_under"].items()))
    P(f"  full under: " + "  ".join(f"<{t}c {v:.1f}%" for t, v in R["drift"]["full_frac_under"].items()))
    P()
    P("ONSET / RELEASE")
    for k, v in R["onset_release"].items():
        if isinstance(v, dict):
            P(f"{k:28s} {fmt(v)}")
    P(f"  onset below center (mid>q1): {R['onset_release']['pct_onset_below_center']:.1f}%  above: {R['onset_release']['pct_onset_above_center']:.1f}%")
    P(f"  release above center: {R['onset_release']['pct_release_above_center']:.1f}%  below: {R['onset_release']['pct_release_below_center']:.1f}%")
    P(f"  |onset diff| > mid IQR: {R['onset_release']['pct_onset_diff_exceeds_mid_iqr']:.1f}%   "
      f"|release diff| > mid IQR: {R['onset_release']['pct_release_diff_exceeds_mid_iqr']:.1f}%")
    P()
    P("BY FRAME COUNT")
    P(f"{'bucket':>10} {'n':>7} {'%elig':>7} {'fullIQR':>8} {'midIQR':>8} {'midMAD':>8} {'|drift|':>8} {'fr<=20c':>8} {'ratio':>7} {'%better':>8}")
    for name, v in R["by_frames"].items():
        g = lambda x, f="8.2f": ("     —" if x is None else f"{x:{f}}")
        P(f"{name:>10} {v['n']:>7,} {g(v['pct_eligible_mid'],'7.1f')} {g(v['med_full_iqr'])} {g(v['med_mid_iqr'])} "
          f"{g(v['med_mid_mad'])} {g(v['med_abs_mid_drift'])} {g(v['med_frac_within_20'])} "
          f"{g(v['med_ratio_iqr'],'7.3f')} {g(v['pct_mid_more_stable'],'8.1f')}")
    P()
    P("BY DURATION")
    for name, v in R["by_duration"].items():
        g = lambda x, f="8.2f": ("     —" if x is None else f"{x:{f}}")
        P(f"{name:>12} {v['n']:>7,} {g(v['pct_eligible_mid'],'7.1f')} {g(v['med_full_iqr'])} {g(v['med_mid_iqr'])} "
          f"{g(v['med_mid_mad'])} {g(v['med_abs_mid_drift'])} {g(v['med_frac_within_20'])}")
    P()
    P("CRITERIA")
    for k, v in R["criteria"].items():
        P(f"  {k}: {v['count']:,} / {v['n_eligible']:,} = {v['pct_of_eligible']:.1f}% of eligible "
          f"({v['pct_of_all_clean']:.1f}% of all clean) | +drift<20c: {v['count_with_drift20']:,} = {v['pct_of_eligible_with_drift20']:.1f}%")
    P()
    P("BY SPLIT")
    for sp, v in R["by_split"].items():
        P(f"  {sp:11s} n={v['n']:>6,} songs={v['n_songs']:>2}  fullIQR {v['med_full_iqr']:.2f} midIQR {v['med_mid_iqr']:.2f} "
          f"ratio {v['med_ratio_iqr']:.3f} %better {v['pct_mid_more_stable']:.1f}  "
          f">=75%within10 {v['pct_ge75_within10']:.1f}%  within20 {v['pct_ge75_within20']:.1f}%  within30 {v['pct_ge75_within30']:.1f}%")
    P()
    P("PRODUCTION CORE-TRIM vs MIDDLE50")
    P(f"  |stored_core_median - middle50_median|: {fmt(R['core_vs_mid50'])}")
    P(f"  |stored_full_median - middle50_median|: {fmt(R['storedfull_vs_mid50'])}")
    P(f"  |stored_core - stored_full|:            {fmt(R['core_vs_storedfull'])}")
    P(f"  notes <10 frames (core trim disabled): {R['n_notes_under10_frames']:,} ({R['pct_notes_under10_frames']:.1f}%); "
      f"core==full there: {R['core_equals_full_when_short']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
