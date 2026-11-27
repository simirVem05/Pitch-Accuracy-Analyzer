"""
Post-hoc checks for axis1_hybrid_chroma_chords — written AFTER seeing results.json. Not pre-registered.

Questions the main run raised:
  1. Real-song excess-fit distribution vs synthetic: why does gamma = 0.15 accept only ~27% of real beats?
  2. Gamma sensitivity of D (alpha = 0.5) on synthetic tests and real songs.
  3. Confidence-weighted CHORD-ONLY variants (D's eligibility + excess-fit weights, alpha = 0), because the
     chroma term penalised compatible-but-silent pitch classes. Three fallbacks when no chord qualifies:
       Dchord_chromafb : chroma ratio (as D)
       Dchord_Bfb      : experiment-2 posterior chords (flagged LOW_EVIDENCE), abstain only if chroma uninformative
       Dchord_abstain  : abstain
Reference tier for H@Tn here is the best-fitting template (top r) of the dominant beat, identical for every variant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import axis1_hybrid as E3  # noqa: E402
import hybrid_model as HM  # noqa: E402

_orig = HM.segment_methods
EXTRA = ["Dchord_chromafb", "Dchord_Bfb", "Dchord_abstain"]


def patched(v, ev, kappa, gamma, delta):
    out, diag = _orig(v, ev, kappa, gamma, delta)
    has, inf, C = diag["has_chord"], diag["informative"], diag["C"]
    FD = out["Fchord_D"]
    out["Dchord_chromafb"] = np.where(has[:, None], FD, np.where(inf[:, None], C, np.nan))
    out["Dchord_Bfb"] = np.where(has[:, None], FD, np.where(inf[:, None], out["B"], np.nan))
    out["Dchord_abstain"] = np.where(has[:, None], FD, np.nan)
    return out, diag


def main() -> int:
    R = json.loads((HERE / "results.json").read_text())
    kappa = R["threshold_study"]["kappa"]
    HM.segment_methods = patched
    E3.ALL_METHODS = E3.ALL_METHODS + EXTRA
    out = {"note": "post hoc; not pre-registered", "kappa": kappa}
    songs = sorted(json.loads((E3.REPO / "data" / "dataset_selections.json").read_text())["songs"])
    # 1. excess-fit distributions
    real_c, real_cb_by_status = [], []
    for s in songs:
        h, n = E3.X2.load_song(s)
        edges = E3.M.beat_segments(h["chroma_frame_times_s"], h["beat_times_s"])
        seg = E3.M.segment_mean_chroma(h["other_chroma_cens"].astype(float), h["chroma_frame_times_s"], edges)
        ev = HM.evidence(seg)
        real_c.append((ev.r - ev.m[:, None]).max(1))
    rc = np.concatenate(real_c)
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)}  # noqa: E731
    out["excess_fit_best"] = {"real_beat_segments": q(rc), "synthetic_medians": {k: v["excess_best_median"] for k, v in R["threshold_study"]["components"].items()},
                              "real_share_ge": {g: float(np.mean(rc >= g)) for g in (0.05, 0.10, 0.15)}}
    # 2/3. per gamma: synthetic pass counts and real metrics
    methods = ["A", "B", "C5w_a0.5", "D_a0.5"] + EXTRA
    out["by_gamma"] = {}
    for gamma in (0.05, 0.10, 0.15):
        syn = E3.synthetic_tests(kappa, gamma, 10.0)
        fails = {m: sorted({f"{c['layer']}{c['case']}" for c in syn["checks"] if c["method"] == m and not c["pass"]}) for m in methods}
        rng = np.random.default_rng(20261128)
        V = {m: [] for m in methods}
        Vn = {m: [] for m in methods}
        pcs, refs = [], []
        status = []
        for s in songs:
            h, n = E3.X2.load_song(s)
            cens = h["other_chroma_cens"].astype(float)
            t, b = h["chroma_frame_times_s"].astype(float), h["beat_times_s"].astype(float)
            st, en = n["start_time_s"].astype(float), n["end_time_s"].astype(float)
            P = E3.run_pipeline(cens, t, b, st, en, kappa, gamma, 10.0)
            noisy = cens * np.exp(rng.normal(0, E3.NOISE_SIGMA, cens.shape))
            noisy /= np.clip(np.linalg.norm(noisy, axis=0, keepdims=True), 1e-12, None)
            PN = E3.run_pipeline(noisy, t, b, st, en, kappa, gamma, 10.0)
            for m in methods:
                V[m].append(P["notes"][m])
                Vn[m].append(PN["notes"][m])
            pcs.append(n["pitch_class"].astype(int))
            refs.append(E3.tier_index_matrix()[P["diag"]["top_r"][P["dom"]]])
            status.append(P["diag"]["status_D"][P["dom"]])
        pc_all, ref_all, st_all = np.concatenate(pcs), np.vstack(refs), np.concatenate(status)
        res = {"synthetic_passed": {m: syn["passed"][m] for m in methods}, "synthetic_failures": fails,
               "note_status": {k: float(np.mean(st_all == k)) for k in ("CHORD", "CHROMA_ONLY", "INSUFFICIENT")}, "real": {}}
        for m in methods:
            A_, B_ = np.vstack(V[m]), np.vstack(Vn[m])
            met = E3.metrics(A_, pc_all, ref_all)
            both = ~np.isnan(A_[:, 0]) & ~np.isnan(B_[:, 0])
            met["noise_mean_abs_dH"] = float(np.mean(np.abs(A_[both] - B_[both])))
            res["real"][m] = {k: met.get(k) for k in ("coverage", "breadth_ge_0.75", "breadth_ge_0.5", "share_notes_ge10_pcs_ge_0.5", "graded_share_in_0.25_0.75",
                                                      "mean_H_T1_chord_tone", "mean_H_T2_extension", "mean_H_T3_tension", "mean_H_T4_clash",
                                                      "mean_H_sung", "mean_H_alternatives", "win_rate_sung_vs_alt", "noise_mean_abs_dH")}
        out["by_gamma"][str(gamma)] = res
        print(f"gamma={gamma}: status {res['note_status']}")
        for m in methods:
            r = res["real"][m]
            print(f"  {m:16} syn T/S {syn['passed'][m]['T']}/{syn['passed'][m]['S']} fails {fails[m]} | cov {r['coverage']:.3f} pcs>=.75 {r['breadth_ge_0.75']:.2f} "
                  f"pcs>=.5 {r['breadth_ge_0.5']:.2f} >=10 {r['share_notes_ge10_pcs_ge_0.5']:.3f} H@T1-4 {r['mean_H_T1_chord_tone']:.2f}/{r['mean_H_T2_extension']:.2f}/"
                  f"{r['mean_H_T3_tension']:.2f}/{r['mean_H_T4_clash']:.2f} noise {r['noise_mean_abs_dH']:.4f}")
    (HERE / "posthoc_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["excess_fit_best"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
