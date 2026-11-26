#!/usr/bin/env python3
"""
Frozen handcrafted natural-offset model on an Axis 2 dataset (controlled rerun).

The model is NOT re-selected. The feature set (ALL, 71 shift-invariant features)
and HGB configuration (hgb_leaf31_lr05) are the ones selected on axis2_v2,
read from the stored V2 result and asserted against the V2 experiment's code,
which is imported, not copied:

  N        = clean production core median (target, train clean notes only)
  N_hat    = HGB(shift-invariant features of the example's own contour + neighbour context)
  S_hat    = observed core median of the example - N_hat

Grouped 5-fold CV by song on train clean notes is run for the frozen config only
(an in-train estimate; nothing is selected from it). The final model is fit on
all train clean notes and validation is predicted once.

  python natural_offset_frozen.py --dataset axis2_v2 --out /tmp/x   # must reproduce the stored V2 result
  python natural_offset_frozen.py --dataset axis2_v3 --out ml/experiments/axis2_v3_natural_offset

The test split is sealed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axis2_v3_shared"))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import frozen_experts as FE  # noqa: E402

NAT, H = FE.NAT, FE.H
OUTPUTS = ("natural_offset_results.json", "natural_offset_summary.md", "validation_predictions.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=FE.DATASETS)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    t0 = time.time()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; refusing to overwrite")
    protected = FE.protected_paths()
    hashes_before = NAT._hash_tree(protected)

    _, nat_cfg = H.frozen_configs()                     # frozen V2 selection, asserted against code
    param, cols = nat_cfg["param"], nat_cfg["columns"]
    d = FE.load(args.dataset)
    D, vc = d["D"], d["variant_codes"]
    tr_clean = d["tr"][D["variant_index"][d["tr"]] == vc["clean"]]
    va_all = d["va"]
    assert tr_clean.size == d["splits"]["train_clean_note_count"]

    print(f"[{args.dataset}] featurizing", flush=True)
    X_tr_full, N_tr = FE.featurize_natural(d, tr_clean)
    X_va_full, core_va = FE.featurize_natural(d, va_all)
    X_tr, X_va = X_tr_full[:, cols], X_va_full[:, cols]
    groups_tr = D["song_index"][tr_clean]
    va_var = D["variant_index"][va_all]

    # Shift invariance of the extractor across each validation note's five variants.
    order = np.lexsort((va_var, D["source_note_index"][va_all], D["song_index"][va_all]))
    Xg = X_va_full[order].reshape(-1, 5, X_va_full.shape[1])
    spread = np.max(np.abs(Xg - Xg[:, :1, :]), axis=(0, 1))

    # Grouped CV of the frozen config only (train clean notes, by song).
    folds = list(GroupKFold(n_splits=FE.CV_FOLDS).split(X_tr, N_tr, groups_tr))
    assert all(not (set(groups_tr[a]) & set(groups_tr[b])) for a, b in folds)
    maes, rmses = [], []
    for a, b in folds:
        e = NAT.fit_predict("hgb", param, X_tr[a], N_tr[a], X_tr[b]) - N_tr[b]
        maes.append(float(np.mean(np.abs(e)))); rmses.append(float(np.sqrt(np.mean(e ** 2))))
    cv_row = {"feature_set": nat_cfg["feature_set"], "model": "hgb", "param": param, "n_features": len(cols),
              "cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)), "cv_rmse": float(np.mean(rmses)), "fold_mae": maes}
    v2_cv = next(r for r in json.loads(H.NATURAL_RESULTS.read_text())["cv_results"]
                 if (r["feature_set"], r["model"], r["param"]) == (nat_cfg["feature_set"], "hgb", param))
    cv_vs_v2 = float(np.max(np.abs(np.array(maes) - np.array(v2_cv["fold_mae"]))))

    # Final fit on all train clean notes; validation predicted once.
    print(f"[{args.dataset}] final fit and single validation pass", flush=True)
    Nhat = NAT.fit_predict("hgb", param, X_tr, N_tr, X_va)
    clean = va_var == vc["clean"]
    N_va = core_va[clean]
    e_N = Nhat[clean] - N_va
    N_mean = float(N_tr.mean())
    natural = {"selected": NAT.metrics(e_N) | {"tail": NAT._tail(np.abs(e_N))},
               "baseline_N0": NAT.metrics(-N_va), "baseline_train_mean_N": NAT.metrics(N_mean - N_va) | {"train_mean_N": N_mean}}
    natural["improvement_vs_N0"] = {"mae": natural["baseline_N0"]["mae"] - natural["selected"]["mae"],
                                    "percent": 100 * (natural["baseline_N0"]["mae"] - natural["selected"]["mae"]) / natural["baseline_N0"]["mae"],
                                    "rmse": natural["baseline_N0"]["rmse"] - natural["selected"]["rmse"]}

    y_va = d["y"][va_all]
    S_hat = core_va - Nhat
    e_S = S_hat - y_va
    Nh_g, e_g = Nhat[order].reshape(-1, 5), e_S[order].reshape(-1, 5)
    corr = va_var != vc["clean"]
    shift_val = {
        "overall": NAT.metrics(e_S) | {"tail": NAT._tail(np.abs(e_S))},
        "tail_full": FE.metrics(e_S),
        "by_variant": {n: NAT.metrics(e_S[va_var == c]) for n, c in vc.items()},
        "by_sign": {"positive": NAT.metrics(e_S[corr & (y_va > 0)]), "negative": NAT.metrics(e_S[corr & (y_va < 0)])},
        "prediction_slope": {"corrupted_only": NAT.slope_stats(y_va[corr], S_hat[corr]), "all_validation_examples": NAT.slope_stats(y_va, S_hat)},
        "structural_identity": {
            "max_within_note_N_hat_spread": float(np.max(Nh_g.max(1) - Nh_g.min(1))),
            "max_within_note_signed_error_spread": float(np.max(e_g.max(1) - e_g.min(1))),
            "groups_violating_1e-6": int(np.count_nonzero((e_g.max(1) - e_g.min(1)) > 1e-6)),
            "note_groups": int(e_g.shape[0]),
            "note": "N_hat comes from each example's own features; equality across siblings is measured, not imposed"},
    }

    with (out / "validation_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["song_id", "source_note_index", "variant", "frame_count", "duration_s", "target_midi",
                    "observed_core_median", "predicted_natural_offset", "predicted_shift", "injected_shift", "shift_error"])
        for j, i in enumerate(va_all):
            w.writerow([str(d["song_ids"][D["song_index"][i]]), int(D["source_note_index"][i]), FE.EXPECTED_VARIANT_ORDER[int(va_var[j])],
                        int(D["num_frames"][i]), f"{D['duration_s'][i]:.4f}", int(D["target_midi"][i]),
                        f"{core_va[j]:.6f}", f"{Nhat[j]:.6f}", f"{S_hat[j]:.6f}", f"{y_va[j]:.6f}", f"{e_S[j]:.6f}"])

    hashes_after = NAT._hash_tree(protected)
    results = {
        "experiment": f"{args.dataset}_natural_offset_frozen",
        "test_evaluated": False,
        "dataset": FE.rel(d["dir"]),
        "frozen_definition": {"source": "ml/experiments/axis2_v2_natural_offset (selection on axis2_v2; not re-selected)",
                              "feature_set": nat_cfg["feature_set"], "n_features": len(cols), "features": nat_cfg["features"],
                              "model": nat_cfg["model"], "param": param, "hyperparameters": nat_cfg["hyperparameters"],
                              "fit_rows": "train clean notes only", "target": "clean core median N",
                              "shift_estimate": "observed core median - N_hat"},
        "selected_model": {"feature_set": nat_cfg["feature_set"], "model": "hgb", "param": param, "cv_mae": cv_row["cv_mae"],
                           "cv_mae_std": cv_row["cv_mae_std"], "cv_rmse": cv_row["cv_rmse"], "n_features": len(cols),
                           "features": nat_cfg["features"], "selection": "frozen from axis2_v2"},
        "cv_results": [cv_row],
        "cv_vs_v2_stored_max_fold_mae_diff": cv_vs_v2,
        "cv_note": "train clean notes are identical across axis2_v2 and axis2_v3, so this CV is expected to equal V2's exactly",
        "counts": {"train_clean_notes": int(tr_clean.size), "validation_clean_notes": int(clean.sum()), "validation_examples": int(va_all.size),
                   "sealed_test_metadata_only": {"examples": d["splits"]["test_example_count"], "clean_notes": d["splits"]["test_clean_note_count"]}},
        "natural_offset_validation": natural,
        "synthetic_shift_validation": shift_val,
        "integrity": {"test_examples_featurized": 0, "test_song_note_files_opened": FE.test_songs_opened(d),
                      "validation_used_for_selection": False, "validation_evaluations": 1,
                      "shift_invariance_max_feature_difference_vs_clean_sibling": float(spread.max()),
                      "protected_paths": [FE.rel(x) for x in protected],
                      "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
                      "protected_unchanged": hashes_before == hashes_after, "runtime_s": time.time() - t0},
    }
    if results["integrity"]["test_song_note_files_opened"]:
        raise RuntimeError("a test song's notes were opened")
    (out / "natural_offset_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    (out / "natural_offset_summary.md").write_text(summary_md(results))
    print(json.dumps({"N_hat_clean": natural["selected"], "shift": {k: shift_val["overall"][k] for k in ("mae", "rmse")},
                      "slope": shift_val["prediction_slope"]["corrupted_only"]["slope"], "cv_vs_v2": cv_vs_v2,
                      "protected_unchanged": results["integrity"]["protected_unchanged"]}, indent=1, default=float))
    return 0


def summary_md(r) -> str:
    f = lambda x, k=4: f"{x:.{k}f}"  # noqa: E731
    nv, sv = r["natural_offset_validation"], r["synthetic_shift_validation"]
    L = [f"# Natural-offset model (frozen) on {r['dataset'].split('/')[-1]}", "",
         "Generated by `natural_offset_frozen.py`. Feature set and HGB config frozen from the axis2_v2 selection; "
         "validation evaluated once; **test not evaluated**.", "",
         f"Frozen: {r['frozen_definition']['feature_set']} ({r['frozen_definition']['n_features']} features), "
         f"{r['frozen_definition']['param']}. Train-CV MAE of N̂ {f(r['selected_model']['cv_mae'])} "
         f"(max fold diff vs V2 {r['cv_vs_v2_stored_max_fold_mae_diff']:.1e}).", "",
         "## N̂ on clean validation notes", "", "| | MAE | RMSE | median abs | mean signed |", "|---|---|---|---|---|"]
    for k, lab in (("selected", "N̂ (HGB)"), ("baseline_N0", "N̂ = 0 (core median)"), ("baseline_train_mean_N", "N̂ = train mean")):
        x = nv[k]
        L.append(f"| {lab} | {f(x['mae'])} | {f(x['rmse'])} | {f(x['median_absolute_error'])} | {x['mean_signed_error']:+.4f} |")
    o = sv["overall"]
    L += ["", "## Shift estimate Ŝ = observed core − N̂ (all validation examples)", "",
          f"MAE {f(o['mae'])}, RMSE {f(o['rmse'])}, median abs {f(o['median_absolute_error'])}, mean signed {o['mean_signed_error']:+.4f}.", "",
          "| variant | MAE | RMSE | mean signed |", "|---|---|---|---|"]
    for n, x in sv["by_variant"].items():
        L.append(f"| {n} | {f(x['mae'])} | {f(x['rmse'])} | {x['mean_signed_error']:+.4f} |")
    L += ["", "| sign | n | MAE | RMSE | mean signed |", "|---|---|---|---|---|"]
    for s, x in sv["by_sign"].items():
        L.append(f"| {s} | {x['count']} | {f(x['mae'])} | {f(x['rmse'])} | {x['mean_signed_error']:+.4f} |")
    sl = sv["prediction_slope"]["corrupted_only"]
    si = sv["structural_identity"]
    L += ["", f"Slope (corrupted): {sl['slope']:.4f}, intercept {sl['intercept']:+.4f}, R² {sl['r2']:.4f}.", "",
          f"Sibling N̂ spread max {si['max_within_note_N_hat_spread']:.2e} c; signed-error spread max {si['max_within_note_signed_error_spread']:.2e} c "
          f"({si['groups_violating_1e-6']} of {si['note_groups']} notes above 1e-6).", "",
          f"Protected files unchanged: {r['integrity']['protected_unchanged']}. Test songs opened: {r['integrity']['test_song_note_files_opened']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
