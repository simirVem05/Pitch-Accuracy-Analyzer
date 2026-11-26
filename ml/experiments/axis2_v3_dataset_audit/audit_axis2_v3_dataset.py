#!/usr/bin/env python3
"""
Structural audit of data/ml_dataset/axis2_v3 against axis2_v2 and the canonical notes.

Descriptive only. No model is trained or evaluated, nothing is predicted, and
no split's performance is measured; the test portion is read only for structural
counts and shift statistics, exactly like train and validation. Nothing here fed
back into generation: the corruption rule was fixed before this audit existed.

Every check is re-derived from the files themselves rather than trusted from
the builder: core medians are recomputed with production `_core_slice`, the
constant-shift property is measured frame by frame, and V2/V3 are matched by
(song_id, source_note_index, variant), not by row order.

  python ml/experiments/axis2_v3_dataset_audit/audit_axis2_v3_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "backend"))
from note_segmentation import _core_slice  # noqa: E402

V2 = REPO / "data" / "ml_dataset" / "axis2_v2"
V3 = REPO / "data" / "ml_dataset" / "axis2_v3"
V1_SPLITS = REPO / "data" / "ml_dataset" / "axis2_v1" / "splits.json"
MAIN = REPO / "data" / "main_data"
SELECTIONS = REPO / "data" / "dataset_selections.json"
LIMIT = 49.999
BANDS = {"corruption_1": (5.0, 15.0), "corruption_2": (15.0, 25.0), "corruption_3": (25.0, 35.0), "corruption_4": (35.0, 50.0)}
VARIANTS = ("clean", *BANDS)
SPLITS = ("train", "validation", "test")
PROTECTED = ("data/main_data", "data/ml_dataset/axis2_v1", "data/ml_dataset/axis2_v2", "data/dataset_selections.json",
             "ml/results/axis2_v1", "ml/results/axis2_v2", "ml/experiments/axis2_v2_natural_offset",
             "ml/experiments/axis2_v2_sequence_natural_offset", "ml/experiments/axis2_v2_hybrid",
             "ml/experiments/axis2_v2_dynamic_hybrid")
N_BUCKETS = (("N < -15", lambda x: x < -15.0), ("-15 <= N <= 15", lambda x: (x >= -15.0) & (x <= 15.0)),
             ("N > 15", lambda x: x > 15.0))
OUTPUTS = ("axis2_v3_dataset_results.json", "axis2_v3_dataset_summary.md", "axis2_v3_shift_distribution.csv")


def load(ds: Path) -> Dict[str, np.ndarray]:
    with np.load(ds / "examples.npz", allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def keys_of(d) -> List[tuple]:
    songs = d["song_ids"]
    return [(str(songs[s]), int(k), str(d["variant_names"][v]))
            for s, k, v in zip(d["song_index"], d["source_note_index"], d["variant_index"])]


def recomputed_cores(d) -> np.ndarray:
    off = d["contour_offsets"]
    return np.array([float(np.median(_core_slice(d["cents_flat"][off[i]:off[i + 1]]))) for i in range(off.size - 1)])


def shift_stats(s: np.ndarray) -> Dict[str, Any]:
    a = np.abs(s)
    return {"count": int(s.size), "positive": int((s > 0).sum()), "negative": int((s < 0).sum()),
            "percent_positive": 100 * float((s > 0).mean()), "percent_negative": 100 * float((s < 0).mean()),
            "mean_shift": float(s.mean()), "mean_abs_shift": float(a.mean()), "median_abs_shift": float(np.median(a)),
            "min_abs_shift": float(a.min()), "max_abs_shift": float(a.max())}


def manifest_hashes() -> Dict[str, str]:
    out = {}
    for rel in PROTECTED:
        root = REPO / rel
        for p in ([root] if root.is_file() else sorted(q for q in root.rglob("*") if q.is_file())):
            out[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main() -> int:
    for name in OUTPUTS:
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; refusing to overwrite")
    d2, d3 = load(V2), load(V3)
    s2 = json.loads((V2 / "summary.json").read_text())
    s3 = json.loads((V3 / "summary.json").read_text())
    sp2, sp3 = json.loads((V2 / "splits.json").read_text()), json.loads((V3 / "splits.json").read_text())
    sp1 = json.loads(V1_SPLITS.read_text())
    sel = json.loads(SELECTIONS.read_text())["songs"]
    checks: Dict[str, Any] = {}

    # ---- identity of the clean notes and splits
    k2, k3 = keys_of(d2), keys_of(d3)
    if len(set(k3)) != len(k3) or len(set(k2)) != len(k2):
        raise RuntimeError("duplicate (song, note, variant) key")
    checks["example_keys_identical_v2_v3"] = set(k2) == set(k3)
    pos2 = {k: i for i, k in enumerate(k2)}
    perm = np.array([pos2[k] for k in k3])          # row in V2 for each V3 row
    checks["row_order_identical_v2_v3"] = bool(np.array_equal(perm, np.arange(perm.size)))
    clean3 = d3["variant_index"] == 0
    checks["clean_note_ids_identical"] = ({k[:2] for k, c in zip(k3, clean3) if c}
                                          == {k[:2] for k, c in zip(k2, d2["variant_index"] == 0) if c})
    for arr in ("target_midi", "duration_s", "start_s", "end_s", "num_frames", "split_index", "song_index",
                "source_note_index", "variant_index"):
        checks[f"{arr}_identical_v2_v3"] = bool(np.array_equal(d3[arr], d2[arr][perm]))
    for arr in ("song_ids", "song_source_filenames", "song_split_index", "variant_names", "split_names", "contour_offsets"):
        checks[f"{arr}_identical_v2_v3"] = bool(np.array_equal(d3[arr], d2[arr]))
    checks["confidence_flat_identical_v2_v3"] = bool(np.array_equal(d3["confidence_flat"], d2["confidence_flat"]))
    off = d3["contour_offsets"]
    clean_rows = np.flatnonzero(clean3)
    checks["clean_cents_contours_identical_v2_v3"] = all(
        np.array_equal(d3["cents_flat"][off[r]:off[r + 1]], d2["cents_flat"][off[r]:off[r + 1]]) for r in clean_rows)
    checks["splits_song_lists_identical_v2_v3_v1"] = all(sorted(sp3[k]) == sorted(sp2[k]) == sorted(sp1[k]) for k in SPLITS)
    checks["splits_match_dataset_selections"] = all(
        sel[s]["axis2_split"] == k and sel[s]["axis2_ml"] for k in SPLITS for s in sp3[k])
    checks["split_assignment_identical_per_song"] = bool(np.array_equal(d3["song_split_index"], d2["song_split_index"]))

    # ---- clean contours verbatim from canonical notes
    canon_ok, canon_target_ok = True, True
    songs = [str(s) for s in d3["song_ids"]]
    for si, song in enumerate(songs):
        with np.load(MAIN / song / "notes.npz", allow_pickle=False) as z:
            nn = {k: np.asarray(z[k]) for k in ("contour_offsets", "contour_cents_deviation", "contour_crepe_confidence", "target_midi_note")}
        no = nn["contour_offsets"]
        for r in clean_rows[d3["song_index"][clean_rows] == si]:
            k = int(d3["source_note_index"][r])
            canon_ok &= bool(np.array_equal(d3["cents_flat"][off[r]:off[r + 1]], nn["contour_cents_deviation"][no[k]:no[k + 1]])
                             and np.array_equal(d3["confidence_flat"][off[r]:off[r + 1]], nn["contour_crepe_confidence"][no[k]:no[k + 1]]))
            canon_target_ok &= bool(d3["target_midi"][r] == nn["target_midi_note"][k])
    checks["clean_contours_verbatim_from_canonical_notes"] = canon_ok
    checks["clean_target_midi_equals_canonical"] = canon_target_ok

    # ---- per-note structure
    n = d3["variant_index"].size
    vi = d3["variant_index"].reshape(-1, 5)
    note_key = d3["song_index"].astype(np.int64) * 1_000_000 + d3["source_note_index"]
    checks["exactly_five_variants_per_note"] = bool(n % 5 == 0 and np.all(vi == np.arange(5))
                                                    and np.all(note_key.reshape(-1, 5) == note_key.reshape(-1, 5)[:, :1])
                                                    and np.unique(note_key).size == n // 5)
    tm = d3["target_midi"].reshape(-1, 5)
    target_mismatches = int(np.sum(tm != tm[:, :1]))
    S3 = d3["injected_shift_cents"].reshape(-1, 5)
    S2 = d2["injected_shift_cents"].reshape(-1, 5)   # row order verified identical above
    if not checks["row_order_identical_v2_v3"]:
        raise RuntimeError("row order differs; reshaped comparison below would be invalid")

    # ---- constant shift, frame by frame
    max_dev = 0.0
    conf_same = True
    for r in clean_rows:
        base = d3["cents_flat"][off[r]:off[r + 1]]
        cb = d3["confidence_flat"][off[r]:off[r + 1]]
        for v in range(1, 5):
            lo, hi = off[r + v], off[r + v + 1]
            max_dev = max(max_dev, float(np.max(np.abs((d3["cents_flat"][lo:hi] - base) - d3["injected_shift_cents"][r + v]))))
            conf_same &= bool(np.array_equal(d3["confidence_flat"][lo:hi], cb))
    checks["confidence_identical_across_variants"] = conf_same

    # ---- core medians
    core3 = recomputed_cores(d3)
    core2 = d2["core_median_cents"]
    N = core3.reshape(-1, 5)[:, 0]
    final = np.repeat(N, 5) + d3["injected_shift_cents"]
    corr = d3["variant_index"] > 0
    core_block = {
        "stored_vs_recomputed_max_abs_diff": float(np.max(np.abs(core3 - d3["core_median_cents"]))),
        "observed_vs_clean_plus_shift_max_abs_diff": float(np.max(np.abs(core3 - final))),
        "final_offset_max_abs_corrupted": float(np.abs(final[corr]).max()),
        "observed_core_max_abs_all": float(np.abs(core3).max()),
        "observed_core_max_abs_corrupted": float(np.abs(core3[corr]).max()),
        "observed_core_max_abs_clean": float(np.abs(core3[~corr]).max()),
        "examples_abs_core_ge_50": int(np.sum(np.abs(core3) >= 50.0)),
        "examples_abs_core_ge_49_999": int(np.sum(np.abs(core3) >= LIMIT)),
        "corrupted_violating_limit": int(np.sum(np.abs(final[corr]) >= LIMIT) + np.sum(np.abs(core3[corr]) >= LIMIT + 1e-9)),
        "clean_notes_abs_N_ge_49_999": int(np.sum(np.abs(N) >= LIMIT)),
    }
    checks["all_corrupted_within_limit"] = core_block["corrupted_violating_limit"] == 0
    checks["zero_examples_abs_core_ge_50"] = core_block["examples_abs_core_ge_50"] == 0

    # ---- V2 -> V3 shift changes (corrupted only)
    a2, a3 = S2[:, 1:], S3[:, 1:]
    changed = np.abs(a2 - a3) > 0
    sign_changed = np.sign(a2) != np.sign(a3)
    mag_changed = np.abs(np.abs(a2) - np.abs(a3)) > 1e-12
    v2_off_target = np.abs(N[:, None] + a2) >= LIMIT
    change = {"corrupted_examples": int(a2.size), "shift_changed": int(changed.sum()),
              "shift_changed_percent": 100 * float(changed.mean()), "sign_changed": int(sign_changed.sum()),
              "magnitude_changed": int(mag_changed.sum()),
              "changed_exactly_where_v2_left_target": bool(np.array_equal(changed, v2_off_target)),
              "v2_examples_abs_core_ge_50": int(np.sum(np.abs(core2) >= 50.0)),
              "notes_with_any_change": int(changed.any(axis=1).sum()),
              "by_band": {b: {"changed": int(changed[:, i].sum()), "percent": 100 * float(changed[:, i].mean()),
                              "positive_to_negative": int(np.sum((a2[:, i] > 0) & (a3[:, i] < 0))),
                              "negative_to_positive": int(np.sum((a2[:, i] < 0) & (a3[:, i] > 0)))}
                          for i, b in enumerate(BANDS)}}

    # ---- feasibility (from N and the drawn magnitude, independent of the builder)
    M = np.abs(a3)
    pos_ok = np.abs(N[:, None] + M) < LIMIT
    neg_ok = np.abs(N[:, None] - M) < LIMIT
    feas = {b: {"both": int(np.sum(pos_ok[:, i] & neg_ok[:, i])), "positive_only": int(np.sum(pos_ok[:, i] & ~neg_ok[:, i])),
                "negative_only": int(np.sum(~pos_ok[:, i] & neg_ok[:, i])), "neither": int(np.sum(~pos_ok[:, i] & ~neg_ok[:, i]))}
            for i, b in enumerate(BANDS)}
    for b in feas:
        feas[b]["percent_both"] = 100 * feas[b]["both"] / a3.shape[0]
    checks["feasibility_matches_builder_summary"] = all(
        {k: feas[b][k] for k in ("both", "positive_only", "negative_only")} == s3["sign_feasibility_by_band"][b] for b in BANDS)
    checks["no_infeasible_band"] = all(feas[b]["neither"] == 0 for b in BANDS)

    # ---- shift distributions
    split_note = d3["split_index"].reshape(-1, 5)[:, 0]
    dist = {"v2": {"all": shift_stats(a2.ravel()), **{b: shift_stats(a2[:, i]) for i, b in enumerate(BANDS)}},
            "v3": {"all": shift_stats(a3.ravel()), **{b: shift_stats(a3[:, i]) for i, b in enumerate(BANDS)}}}
    by_split = {sp: {"v2": shift_stats(a2[split_note == j].ravel()), "v3": shift_stats(a3[split_note == j].ravel())}
                for j, sp in enumerate(SPLITS)}
    per_note_pos = (a3 > 0).sum(axis=1)
    per_note = {str(k): int(np.sum(per_note_pos == k)) for k in range(5)}
    # Descriptive: how predictable the sign becomes from where the clean note sits.
    sign_vs_N = {}
    for i, b in enumerate(BANDS):
        sign_vs_N[b] = {}
        for lab, fn in N_BUCKETS:
            m = fn(N)
            sign_vs_N[b][lab] = {"notes": int(m.sum()), "percent_positive_v2": 100 * float((a2[m, i] > 0).mean()),
                                 "percent_positive_v3": 100 * float((a3[m, i] > 0).mean())}

    def counts(d, s):
        sc = {sp: int(np.sum(d["song_split_index"] == j)) for j, sp in enumerate(SPLITS)}
        return {"songs": int(d["song_ids"].size), "clean_notes": int(np.sum(d["variant_index"] == 0)),
                "examples": int(d["variant_index"].size),
                **{f"{sp}_songs": sc[sp] for sp in SPLITS},
                **{f"{sp}_clean_notes": int(np.sum((d["split_index"] == j) & (d["variant_index"] == 0))) for j, sp in enumerate(SPLITS)},
                **{f"{sp}_examples": int(np.sum(d["split_index"] == j)) for j, sp in enumerate(SPLITS)},
                "summary_json_agrees": all(s[f"{sp}_example_count"] == int(np.sum(d["split_index"] == j)) for j, sp in enumerate(SPLITS))}

    cnt2, cnt3 = counts(d2, s2), counts(d3, s3)
    comparison = {
        "v2": {**cnt2, "positive_corruptions": dist["v2"]["all"]["positive"], "negative_corruptions": dist["v2"]["all"]["negative"],
               "examples_abs_core_ge_50": int(np.sum(np.abs(core2) >= 50.0)), "max_abs_core": float(np.abs(core2).max())},
        "v3": {**cnt3, "positive_corruptions": dist["v3"]["all"]["positive"], "negative_corruptions": dist["v3"]["all"]["negative"],
               "examples_abs_core_ge_50": core_block["examples_abs_core_ge_50"], "max_abs_core": core_block["observed_core_max_abs_all"]},
    }

    # ---- protected files
    before = {}
    for line in (OUT / "protected_sha256_before.txt").read_text().splitlines():
        h, p = line.split(None, 1)
        before[p.strip()] = h
    after = manifest_hashes()
    protected = {"files": len(after), "unchanged": before == after,
                 "missing_after": sorted(set(before) - set(after)), "new_after": sorted(set(after) - set(before)),
                 "changed": sorted(p for p in before.keys() & after.keys() if before[p] != after[p]),
                 "before_manifest": "protected_sha256_before.txt (shasum -a 256, taken before the builder was edited)"}
    checks["protected_files_byte_identical"] = protected["unchanged"]

    build_report = json.loads((V3 / "_build_report.json").read_text())
    logs = {name: (OUT / "build_logs" / name).read_text() for name in sorted(p.name for p in (OUT / "build_logs").iterdir())}
    repro = {name: '"passed": true' in txt for name, txt in logs.items() if "reproduce" in name or "determinism" in name}

    results = {
        "dataset": "axis2_v3", "test_evaluated": False, "models_run": False,
        "dataset_path": str(V3.relative_to(REPO)),
        "builder": "scripts/build_axis2_ml_dataset.py --dataset-name axis2_v3 --notes-file notes.npz "
                   f"--corruption-policy target_preserving --created-at {s3['created_at_utc']}",
        "source": {"notes": "data/main_data/<song>/notes.npz (v2 canonical, schema 2)", "membership": "data/dataset_selections.json (axis2_ml)",
                   "splits": s3["split_assignment_source"], "source_notes_sha256_identical_to_v2": s3["source_notes_sha256"] == s2["source_notes_sha256"]},
        "rule": {"constraint": s3["target_preserving_constraint"], "sign_policy": s3["corruption_sign_policy"],
                 "global_seed": s3["global_seed"], "seed_derivation": s3["seed_derivation"],
                 "magnitude_sampling": s3["corruption_magnitude_sampling"], "bands": s3["corruption_magnitude_bands_cents"],
                 "band_endpoints": "numpy Generator.uniform draws from [lo, hi), unchanged from axis2_v2"},
        "counts": comparison, "core_median": core_block,
        "constant_shift_max_within_note_deviation_cents": max_dev,
        "target_midi_mismatches_across_siblings": target_mismatches,
        "v2_to_v3_changes": change, "sign_feasibility_by_band": feas,
        "shift_distribution": dist, "shift_distribution_by_split": by_split,
        "positive_corruptions_per_note_v3": per_note, "sign_vs_clean_core_v2_v3": sign_vs_N,
        "checks": checks, "all_checks_passed": all(bool(v) for v in checks.values()),
        "builder_validation": build_report, "reproduce_and_determinism_logs_passed": repro,
        "protected": protected,
    }
    (OUT / "axis2_v3_dataset_results.json").write_text(json.dumps(results, indent=2) + "\n")

    with (OUT / "axis2_v3_shift_distribution.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "band", "count", "positive", "negative", "percent_positive", "mean_shift", "mean_abs_shift",
                    "min_abs_shift", "max_abs_shift", "feasible_both", "feasible_positive_only", "feasible_negative_only",
                    "changed_from_v2"])
        for ds in ("v2", "v3"):
            for b in ("all", *BANDS):
                x = dist[ds][b]
                f_ = feas.get(b, {k: sum(feas[bb][k] for bb in BANDS) for k in ("both", "positive_only", "negative_only")})
                ch = change["by_band"][b]["changed"] if b in BANDS else change["shift_changed"]
                w.writerow([ds, b, x["count"], x["positive"], x["negative"], f"{x['percent_positive']:.4f}", f"{x['mean_shift']:.6f}",
                            f"{x['mean_abs_shift']:.6f}", f"{x['min_abs_shift']:.6f}", f"{x['max_abs_shift']:.6f}",
                            f_["both"], f_["positive_only"], f_["negative_only"], ch if ds == "v3" else ""])

    (OUT / "axis2_v3_dataset_summary.md").write_text(summary_md(results))
    print(json.dumps({"all_checks_passed": results["all_checks_passed"], "failed": [k for k, v in checks.items() if not v],
                      "protected_unchanged": protected["unchanged"]}, indent=1))
    return 0 if results["all_checks_passed"] else 1


def summary_md(r: Dict[str, Any]) -> str:
    c2, c3 = r["counts"]["v2"], r["counts"]["v3"]
    cb, ch, fe = r["core_median"], r["v2_to_v3_changes"], r["sign_feasibility_by_band"]
    L = ["# Axis 2 V3 dataset audit", "",
         "Generated by `audit_axis2_v3_dataset.py`. Structural audit only — **no model run, test not evaluated**.", "",
         f"Rule: `{r['rule']['constraint']}`. {r['rule']['sign_policy']}.", "",
         "## V2 vs V3", "", "| | V2 | V3 |", "|---|---|---|"]
    for lab, k in (("songs", "songs"), ("clean notes", "clean_notes"), ("total examples", "examples"),
                   ("train clean notes / examples", None), ("validation clean notes / examples", None), ("test clean notes / examples", None),
                   ("positive corruptions", "positive_corruptions"), ("negative corruptions", "negative_corruptions"),
                   ("examples \\|core\\| ≥ 50", "examples_abs_core_ge_50"), ("max \\|core\\| (¢)", "max_abs_core")):
        if k is None:
            sp = lab.split()[0]
            L.append(f"| {lab} | {c2[f'{sp}_clean_notes']} / {c2[f'{sp}_examples']} | {c3[f'{sp}_clean_notes']} / {c3[f'{sp}_examples']} |")
        else:
            fmt = (lambda x: f"{x:.4f}") if k == "max_abs_core" else str
            L.append(f"| {lab} | {fmt(c2[k])} | {fmt(c3[k])} |")
    L += ["", f"Corruption shifts changed V2 → V3: **{ch['shift_changed']} of {ch['corrupted_examples']} "
              f"({ch['shift_changed_percent']:.2f}%)**; sign changed {ch['sign_changed']}, magnitude changed {ch['magnitude_changed']}; "
              f"{ch['notes_with_any_change']} notes affected. Changed exactly where V2 left the target: {ch['changed_exactly_where_v2_left_target']}.", "",
          "## Shift distribution (V3)", "", "| band | + | − | % + | mean S | mean \\|S\\| | min \\|S\\| | max \\|S\\| | changed | + → − | − → + |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in ("all", *BANDS):
        x = r["shift_distribution"]["v3"][b]
        cc = ch["by_band"].get(b)
        L.append(f"| {b} | {x['positive']} | {x['negative']} | {x['percent_positive']:.2f} | {x['mean_shift']:+.3f} | {x['mean_abs_shift']:.3f} | "
                 f"{x['min_abs_shift']:.3f} | {x['max_abs_shift']:.3f} | "
                 + (f"{cc['changed']} | {cc['positive_to_negative']} | {cc['negative_to_positive']} |" if cc else f"{ch['shift_changed']} | | |"))
    L += ["", "V2 for reference: " + ", ".join(f"{b} {r['shift_distribution']['v2'][b]['mean_abs_shift']:.3f}" for b in BANDS)
          + " mean |S| (identical magnitudes; only signs differ).", "",
          "**Positive corruptions per note (V3):** " + ", ".join(f"{k}: {v}" for k, v in r["positive_corruptions_per_note_v3"].items()), "",
          "**By split (V3):** " + "; ".join(f"{sp} {x['v3']['positive']}+ / {x['v3']['negative']}− ({x['v3']['percent_positive']:.2f}% +)"
                                            for sp, x in r["shift_distribution_by_split"].items()), "",
          "## Sign feasibility (per clean note, per band)", "", "| band | both | positive only | negative only | neither | % both |",
          "|---|---|---|---|---|---|"]
    for b in BANDS:
        x = fe[b]
        L.append(f"| {b} | {x['both']} | {x['positive_only']} | {x['negative_only']} | {x['neither']} | {x['percent_both']:.2f} |")
    L += ["", "## Sign vs clean core N (% positive, descriptive)", "", "| band | " + " | ".join(lab for lab, _ in N_BUCKETS) + " |",
          "|---|" + "---|" * len(N_BUCKETS)]
    for b in BANDS:
        L.append(f"| {b} | " + " | ".join(f"{x['percent_positive_v2']:.1f} → {x['percent_positive_v3']:.1f}" for x in r["sign_vs_clean_core_v2_v3"][b].values()) + " |")
    L += ["", "## Integrity", "",
          f"- Observed core − (clean core + S), max: {cb['observed_vs_clean_plus_shift_max_abs_diff']:.2e} ¢; stored vs recomputed core: "
          f"{cb['stored_vs_recomputed_max_abs_diff']:.2e} ¢.",
          f"- Max \\|core\\|: all {cb['observed_core_max_abs_all']:.4f}, corrupted {cb['observed_core_max_abs_corrupted']:.4f}, "
          f"clean {cb['observed_core_max_abs_clean']:.4f}; examples ≥ 50 ¢: {cb['examples_abs_core_ge_50']}; corrupted violating ±49.999: "
          f"{cb['corrupted_violating_limit']}.",
          f"- Constant-shift max within-note deviation: {r['constant_shift_max_within_note_deviation_cents']:.2e} ¢.",
          f"- Target-MIDI mismatches across siblings: {r['target_midi_mismatches_across_siblings']}.",
          f"- Checks: {sum(bool(v) for v in r['checks'].values())}/{len(r['checks'])} passed"
          + ("" if r["all_checks_passed"] else f" — FAILED: {[k for k, v in r['checks'].items() if not v]}") + ".",
          f"- Protected files byte-identical: {r['protected']['unchanged']} ({r['protected']['files']} files).",
          "- test_evaluated: false; no models run.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
