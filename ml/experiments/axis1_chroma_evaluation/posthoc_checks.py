"""
Post-hoc checks for axis1_chroma_evaluation — written AFTER seeing results.json.

Not part of the pre-registered protocol; nothing here selects a method. It
answers questions the main tables leave open:

  1. combining vs global alone: paired song-level win-rate differences against `global`
  2. slack false positives: alternatives lifted above the original only because of slack
  3. argsort tie-order bias: how stored rank depends on pitch-class index within ties
  4. headline spread: between-song spread of the production calibrated score vs perturbation drops

Reuses compute_song() from the main script so every number comes from the same
code path. Writes posthoc_checks.json next to this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import axis1_chroma_evaluation as A  # noqa: E402

SEED = 20261127


def boot_ci(x: np.ndarray, seed: int) -> list:
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, x.size, size=(A.BOOTSTRAP_RESAMPLES, x.size))].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> int:
    splits = json.loads((HERE / "axis1_splits.json").read_text())
    sel = json.loads((A.REPO / "data" / "dataset_selections.json").read_text())["songs"]
    group_of = {s: g for g in ("dev", "holdout", "robustness") for s in splits[g]}
    songs = [A.compute_song(s, group_of[s], sel[s]) for s in sorted(sel)]
    groups = {"dev": ["dev"], "holdout": ["holdout"], "robustness": ["robustness"], "all_63": ["dev", "holdout", "robustness"]}
    out: dict = {"note": "post hoc; not pre-registered; no selection is made from these numbers"}

    # 1. paired win-rate difference vs `global`
    def song_win(r, m):
        return float(A.per_note_metrics(r["S"][m], r["notes"]["pc"])["win"].mean())

    comp = ["prod_geo_w25", "geo_w50", "arith_w50", "chord_geo_w25", "chord_local_slack", "rank_local_slack"]
    out["paired_vs_global"] = {}
    for g, members in groups.items():
        gs = [r for r in songs if r["group"] in members]
        base = np.array([song_win(r, "global") for r in gs])
        out["paired_vs_global"][g] = {}
        for i, m in enumerate(comp):
            x = np.array([song_win(r, m) for r in gs]) - base
            out["paired_vs_global"][g][m] = {"diff": float(x.mean()), "ci95": boot_ci(x, SEED + i),
                                             "songs_better": int((x > A.TOL).sum()), "songs": len(gs)}

    # 2. slack: how often does slack lift an alternative above the original that the no-slack span had below it?
    for pair in (("rank_local", "rank_local_slack"), ("geo_w25_noslack", "prod_geo_w25")):
        lifted_alt = flipped_orig = pairs = 0
        orig_lift, alt_lift = [], []
        for r in songs:
            a, b, pc = r["S"][pair[0]], r["S"][pair[1]], r["notes"]["pc"]
            idx = np.arange(pc.size)
            oa, ob = a[idx, pc], b[idx, pc]
            mask = np.ones_like(a, dtype=bool)
            mask[idx, pc] = False
            below_before = (a < oa[:, None] - A.TOL) & mask
            above_after = (b > ob[:, None] + A.TOL) & mask
            above_before = (a > oa[:, None] + A.TOL) & mask
            below_after = (b < ob[:, None] - A.TOL) & mask
            lifted_alt += int((below_before & above_after).sum())
            flipped_orig += int((above_before & below_after).sum())
            pairs += int(mask.sum())
            orig_lift.append(ob - oa)
            alt_lift.append((b - a)[mask])
        out[f"slack_{pair[0]}_to_{pair[1]}"] = {
            "pairs": pairs, "alternative_overtakes_original_due_to_slack": lifted_alt,
            "original_overtakes_alternative_due_to_slack": flipped_orig,
            "net_pairs_gained": flipped_orig - lifted_alt,
            "mean_lift_original": float(np.concatenate(orig_lift).mean()),
            "mean_lift_alternatives": float(np.concatenate(alt_lift).mean())}
    # calibrated saturation with and without slack
    sat = {}
    for m in ("geo_w25_noslack", "prod_geo_w25"):
        o, al = [], []
        for r in songs:
            S, pc = r["S"][m], r["notes"]["pc"]
            idx = np.arange(pc.size)
            mask = np.ones_like(S, dtype=bool)
            mask[idx, pc] = False
            o.append(A.calibrated(S[idx, pc]))
            al.append(A.calibrated(S[mask]))
        o, al = np.concatenate(o), np.concatenate(al)
        sat[m] = {"orig_at_100pct": float((o >= 1).mean()), "alternatives_at_100pct": float((al >= 1).mean()),
                  "orig_mean": float(o.mean()), "alternatives_mean": float(al.mean()), "gap": float(o.mean() - al.mean())}
    out["calibrated_saturation_slack_vs_noslack_all_63"] = sat

    # 3. tie-order bias: stored (argsort) rank minus average-tie rank, by pitch-class index, over tied entries
    bias_sum, bias_n = np.zeros(12), np.zeros(12)
    note_diff = []
    for s in sorted(sel):
        d = A.load_song(s)
        cens = d["h"]["other_chroma_cens"].astype(np.float64)
        rank = np.round(d["h"]["other_chroma_rank"].astype(np.float64) * 11) / 11
        avg = (A.rankdata(cens, axis=0, method="average") - 1) / 11
        diff = rank - avg
        tied = ~np.isclose(diff, 0, atol=1e-9)
        bias_sum += np.where(tied, diff, 0).sum(axis=1)
        bias_n += tied.sum(axis=1)
    r_all = {s["song"]: s for s in songs}
    for r in songs:
        pc = r["notes"]["pc"]
        idx = np.arange(pc.size)
        note_diff.append(r["S"]["prod_geo_w25"][idx, pc] - r["S"]["geo_w25_avgties"][idx, pc])
    nd = np.concatenate(note_diff)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    out["tie_order_bias"] = {
        "mean_rank_shift_over_all_entries_by_pc": {names[i]: float(bias_sum[i] / max(sum(bias_n) / 12, 1)) for i in range(12)},
        "mean_rank_shift_within_tied_entries_by_pc": {names[i]: float(bias_sum[i] / max(bias_n[i], 1)) for i in range(12)},
        "production_orig_score_minus_avgties": {"mean": float(nd.mean()), "mean_abs": float(np.abs(nd).mean()),
                                                "p99_abs": float(np.percentile(np.abs(nd), 99)), "max_abs": float(np.abs(nd).max())},
    }

    # 4. between-song spread of the production calibrated headline (duration-weighted, original notes)
    hl = {}
    for g, members in groups.items():
        gs = [r for r in songs if r["group"] in members]
        v = []
        for r in gs:
            pc, w = r["notes"]["pc"], r["notes"]["dur"]
            v.append(100 * np.average(A.calibrated(r["S"]["prod_geo_w25"][np.arange(pc.size), pc]), weights=w))
        v = np.array(v)
        hl[g] = {"mean_pct": float(v.mean()), "sd_pct": float(v.std(ddof=1)), "min_pct": float(v.min()), "max_pct": float(v.max())}
    out["production_headline_between_song_spread"] = hl

    (HERE / "posthoc_checks.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
