#!/usr/bin/env python3
"""Pick example contours and render the five diagnostic plots. Read-only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
CSV_PATH = OUT_DIR / "stable_center_audit.csv"
CANONICAL_DIR = REPO_ROOT / "data" / "main_data"

N_EX = 10


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("song_id", "split"):
                continue
            r[k] = np.nan if v == "" else float(v)
    return rows


def contour(song_id: str, note_index: int):
    c = np.load(CANONICAL_DIR / song_id / "notes.npz", allow_pickle=False)
    off = c["contour_offsets"]
    lo, hi = int(off[note_index]), int(off[note_index + 1])
    return (c["contour_cents_deviation"][lo:hi],
            c["contour_crepe_confidence"][lo:hi],
            c["contour_time_s"][lo:hi])


def main():
    rows = load_rows()
    elig = [r for r in rows if np.isfinite(r["middle50_median_cents"])]
    # require a real middle region and a drift estimate for the ranked picks
    ranked = [r for r in elig if np.isfinite(r["abs_middle50_drift_cents"])
              and r["n_frames_middle50"] >= 6]

    groups = {
        "extremely_stable": sorted(
            ranked, key=lambda r: (r["middle50_iqr_cents"], r["abs_middle50_drift_cents"]))[:N_EX],
        "highly_unstable_drifting": sorted(
            ranked, key=lambda r: -r["abs_middle50_drift_cents"])[:N_EX],
        "onset_release_differ": sorted(
            [r for r in ranked if np.isfinite(r["abs_onset_to_center"])
             and np.isfinite(r["abs_release_from_center"])],
            key=lambda r: -(r["abs_onset_to_center"] + r["abs_release_from_center"]))[:N_EX],
        "full_vs_middle_median_differ": sorted(
            ranked, key=lambda r: -r["abs_full_minus_middle50"])[:N_EX],
    }

    fields = ["group", "song_id", "source_note_index", "split", "duration_s", "num_frames",
              "target_midi", "median_confidence", "full_median_cents", "middle50_median_cents",
              "full_iqr_cents", "middle50_iqr_cents", "middle50_mad_cents",
              "middle50_drift_cents", "abs_middle50_drift_cents", "onset_to_center",
              "release_from_center", "abs_full_minus_middle50", "has_time_gaps"]

    ex_rows = []
    contour_rows = []
    for gname, items in groups.items():
        for r in items:
            ex_rows.append({"group": gname, **{k: r[k] for k in fields if k != "group"}})
            cents, conf, t = contour(r["song_id"], int(r["source_note_index"]))
            for j, (cv, cf, tv) in enumerate(zip(cents, conf, t)):
                contour_rows.append({
                    "group": gname, "song_id": r["song_id"],
                    "source_note_index": int(r["source_note_index"]),
                    "frame": j, "time_s": float(tv),
                    "cents_deviation": float(cv), "crepe_confidence": float(cf),
                })

    with (OUT_DIR / "example_notes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(ex_rows)

    with (OUT_DIR / "example_contours.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(contour_rows[0].keys()))
        w.writeheader()
        w.writerows(contour_rows)

    print(f"example_notes.csv: {len(ex_rows)} rows")
    print(f"example_contours.csv: {len(contour_rows)} frames")

    # ---------- plots ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_iqr = np.array([r["full_iqr_cents"] for r in rows])
    mid_iqr = np.array([r["middle50_iqr_cents"] for r in elig])
    drift = np.array([r["abs_middle50_drift_cents"] for r in rows
                      if np.isfinite(r["abs_middle50_drift_cents"])])
    f20 = np.array([r["mid_frac_within_20c"] for r in elig])
    nf = np.array([r["num_frames"] for r in elig])
    mi = np.array([r["middle50_iqr_cents"] for r in elig])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(0, 150, 61)
    ax[0].hist(np.clip(full_iqr, 0, 150), bins=bins, alpha=0.6, label="full note", color="#4C78A8")
    ax[0].hist(np.clip(mid_iqr, 0, 150), bins=bins, alpha=0.6, label="middle 50%", color="#F58518")
    ax[0].axvline(np.median(full_iqr), color="#4C78A8", ls="--")
    ax[0].axvline(np.median(mid_iqr), color="#F58518", ls="--")
    ax[0].set_xlabel("IQR (cents, clipped at 150)"); ax[0].set_ylabel("notes")
    ax[0].set_title("Full-note vs middle-50% IQR"); ax[0].legend()
    ax[1].hist(np.clip(drift, 0, 200), bins=np.linspace(0, 200, 61), color="#54A24B")
    ax[1].axvline(np.median(drift), color="k", ls="--", label=f"median {np.median(drift):.1f}c")
    ax[1].set_xlabel("|fitted drift| across middle 50% (cents, clipped at 200)")
    ax[1].set_ylabel("notes"); ax[1].set_title("Middle-region drift"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "iqr_and_drift.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(f20, bins=np.linspace(0, 1, 41), color="#B279A2")
    ax[0].axvline(0.75, color="r", ls="--", label="75% of frames")
    ax[0].set_xlabel("fraction of middle frames within ±20c of middle median")
    ax[0].set_ylabel("notes"); ax[0].set_title("Concentration around the middle center")
    ax[0].legend()
    buckets = [("3-4", 3, 4), ("5-6", 5, 6), ("7-10", 7, 10), ("11-20", 11, 20),
               ("21-50", 21, 50), ("51+", 51, 10**9)]
    labels, med_mid, med_full = [], [], []
    allnf = np.array([r["num_frames"] for r in rows])
    allfull = np.array([r["full_iqr_cents"] for r in rows])
    for name, lo, hi in buckets:
        m_e = (nf >= lo) & (nf <= hi)
        m_a = (allnf >= lo) & (allnf <= hi)
        labels.append(name)
        med_mid.append(np.median(mi[m_e]) if m_e.sum() else np.nan)
        med_full.append(np.median(allfull[m_a]) if m_a.sum() else np.nan)
    x = np.arange(len(labels))
    ax[1].plot(x, med_full, "o-", label="full note", color="#4C78A8")
    ax[1].plot(x, med_mid, "s-", label="middle 50%", color="#F58518")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_xlabel("frame count bucket"); ax[1].set_ylabel("median IQR (cents)")
    ax[1].set_title("Stability vs note length"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "concentration_and_length.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(17, 6), sharey=False)
    picks = groups["extremely_stable"][:5] + groups["highly_unstable_drifting"][:5]
    for k, (r, axx) in enumerate(zip(picks, axes.ravel())):
        cents, conf, t = contour(r["song_id"], int(r["source_note_index"]))
        tn = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.zeros_like(t)
        axx.plot(tn, cents, ".-", ms=3, lw=1, color="#333")
        axx.axvspan(0.25, 0.75, color="#F58518", alpha=0.15)
        axx.axhline(r["middle50_median_cents"], color="#F58518", ls="--", lw=1)
        axx.set_title(f"{r['song_id'][:16]}#{int(r['source_note_index'])}\n"
                      f"n={int(r['num_frames'])} midIQR={r['middle50_iqr_cents']:.0f}c",
                      fontsize=7)
        axx.tick_params(labelsize=6)
        if k == 0:
            axx.set_ylabel("stable\ncents", fontsize=8)
        if k == 5:
            axx.set_ylabel("drifting\ncents", fontsize=8)
    fig.suptitle("Top: most stable middle regions.  Bottom: largest middle-region drift.  "
                 "Shaded = middle 50%", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT_DIR / "example_contours.png", dpi=130); plt.close(fig)

    print("plots: iqr_and_drift.png, concentration_and_length.png, example_contours.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
