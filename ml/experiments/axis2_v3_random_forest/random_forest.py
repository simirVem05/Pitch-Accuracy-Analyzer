#!/usr/bin/env python3
"""
Axis 2 V3: direct Random Forest regression of the injected shift, on exactly Ridge's features.

    y = injected_shift_cents
    X = the 24 note-level features of ml/baselines/ridge_baseline.py, computed by its own
        build_matrix / extract_features (imported, not re-implemented); raw, unscaled
    model = sklearn RandomForestRegressor(n_estimators=300, random_state=20260824, n_jobs=-1)

The only change from the V3 Ridge baseline is the model class: linear Ridge vs a
nonlinear forest on identical inputs. The forest never sees song/artist identity,
variant, band, sign, shift magnitude, siblings, clean references, natural-offset
predictions, or chroma; song identity is used only as the CV grouping variable.

Selection (train only): GroupKFold(5) by song over all train examples, the same
partition as the V3 Ridge run; 36 configurations (max_depth x min_samples_leaf x
max_features); lowest mean CV MAE, and among candidates within 0.05 c of it prefer
larger min_samples_leaf, then smaller max_depth, then fewer features per split.
The selected model is refit on all train examples and validation is predicted once.
A center-only diagnostic (the 4 pitch-center features) goes through the identical
procedure. Validation permutation importance is interpretation only.

Comparators (Ridge, natural offset, fixed and dynamic hybrids) are read from the
stored V3 dynamic-hybrid validation predictions and checked against their stored
metrics; none is refit or modified.

The test split is sealed: test indices never enter a featurized set.

  python ml/experiments/axis2_v3_random_forest/random_forest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axis2_v3_shared"))

import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import frozen_experts as FE  # noqa: E402

RB, NAT = FE.RB, FE.NAT
REPO = FE.REPO
OUT = Path(__file__).resolve().parent
CACHE = Path("/tmp/axis2_v3_rf_cv_cache")      # resumable CV cache, outside the repo
DATASET = "axis2_v3"
RIDGE_RESULTS = REPO / "ml/results/axis2_v3/ridge_baseline.json"
DYN_PREDICTIONS = REPO / "ml/experiments/axis2_v3_dynamic_hybrid/gate_validation_predictions.csv"
DYN_RESULTS = REPO / "ml/experiments/axis2_v3_dynamic_hybrid/dynamic_hybrid_results.json"
HYB_RESULTS = REPO / "ml/experiments/axis2_v3_hybrid/hybrid_results.json"
NAT_RESULTS = REPO / "ml/experiments/axis2_v3_natural_offset/natural_offset_results.json"
BASELINES = REPO / "ml/results/axis2_v3"

SEED = 20260824
N_ESTIMATORS = 300
MAX_DEPTHS = (None, 12, 20)
MIN_LEAVES = (1, 2, 5, 10)
MAX_FEATURES = (1.0, "sqrt", 0.5)
TIE_TOLERANCE_CENTS = 0.05
NOTE_TIE_CENTS = 0.5
PERM_REPEATS = 5
SAVE_MODEL_MAX_NODES = 5_000_000

FRAME_BUCKETS = (("3-4", 3, 5), ("5-6", 5, 7), ("7-9", 7, 10), ("10-20", 10, 21), ("21-50", 21, 51), ("51+", 51, None))
DURATION_BUCKETS = (("<0.10", None, 0.10), ("0.10-0.20", 0.10, 0.20), ("0.20-0.40", 0.20, 0.40), ("0.40-0.80", 0.40, 0.80), (">=0.80", 0.80, None))
CORE_BUCKETS = (("0-10", 0, 10), ("10-20", 10, 20), ("20-30", 20, 30), ("30-40", 30, 40), ("40-49.999", 40, None))
COMPARE = ("ridge", "rf", "natural", "fixed", "dynamic")
LABEL = {"zero": "Zero", "core": "Core median", "ridge": "Ridge", "rf": "Random Forest", "rf_center": "RF center-only",
         "natural": "Natural offset", "fixed": "Fixed hybrid", "dynamic": "Dynamic hybrid"}
OUTPUTS = ("random_forest_results.json", "random_forest_summary.md", "cv_results.csv", "validation_predictions.csv",
           "feature_importance.csv", "random_forest_config.json")
FORBIDDEN = ("song", "artist", "filename", "clean_note_id", "variant", "severity", "band", "sign", "shift", "sibling",
             "natural", "chroma", "harmonic")


def in_range(v, lo, hi):
    m = np.ones(v.shape, bool)
    if lo is not None:
        m &= v >= lo
    if hi is not None:
        m &= v < hi
    return m


def short(e: np.ndarray) -> Dict[str, Any]:
    a = np.abs(e)
    return {"count": int(e.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "median_absolute_error": float(np.median(a)), "mean_signed_error": float(e.mean())}


def eff_features(mf, n: int) -> int:
    if mf == "sqrt":
        return max(1, int(math.sqrt(n)))
    return max(1, int(mf * n)) if isinstance(mf, float) and mf < 1.0 else n


def label(cfg) -> str:
    return f"depth={cfg['max_depth']}|leaf={cfg['min_samples_leaf']}|mf={cfg['max_features']}"


def make(cfg) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1, **cfg)


def grouped_search(name: str, X: np.ndarray, y: np.ndarray, folds) -> List[Dict[str, Any]]:
    """Every grid configuration over the shared song folds. Train only; results cached per (feature set, config, fold)."""
    CACHE.mkdir(exist_ok=True)
    rows = []
    grid = [dict(max_depth=d, min_samples_leaf=l, max_features=m) for d, l, m in itertools.product(MAX_DEPTHS, MIN_LEAVES, MAX_FEATURES)]
    for ci, cfg in enumerate(grid):
        oof = np.full(y.size, np.nan)
        maes, rmses, secs = [], [], []
        for k, (fit, held) in enumerate(folds):
            cf = CACHE / f"{name}__{label(cfg).replace('|', '_').replace('=', '-')}__fold{k}.npz"
            if cf.exists():
                with np.load(cf) as z:
                    p, sec = z["pred"], float(z["seconds"])
                if p.size != held.size:
                    raise RuntimeError(f"stale cache {cf}")
            else:
                t = time.time()
                p = make(cfg).fit(X[fit], y[fit]).predict(X[held])
                sec = time.time() - t
                np.savez(cf, pred=p, seconds=sec)
            oof[held] = p
            e = p - y[held]
            maes.append(float(np.mean(np.abs(e)))); rmses.append(float(np.sqrt(np.mean(e ** 2)))); secs.append(sec)
        e = oof - y
        rows.append({"feature_set": name, "config": label(cfg), **cfg, "n_estimators": N_ESTIMATORS,
                     "effective_max_features": eff_features(cfg["max_features"], X.shape[1]),
                     "fold_mae": maes, "fold_rmse": rmses, "mean_cv_mae": float(np.mean(maes)), "std_cv_mae": float(np.std(maes)),
                     "mean_cv_rmse": float(np.mean(rmses)), "pooled_oof_mae": float(np.mean(np.abs(e))),
                     "pooled_oof_rmse": float(np.sqrt(np.mean(e ** 2))), "fit_predict_seconds_total": float(sum(secs)), "_oof": oof})
        print(f"[{name} {ci + 1:2d}/{len(grid)}] {label(cfg):34s} CV MAE {np.mean(maes):.4f} RMSE {np.mean(rmses):.4f} "
              f"({sum(secs):.0f}s)", flush=True)
    return rows


def select(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = min(r["mean_cv_mae"] for r in rows)
    within = [r for r in rows if r["mean_cv_mae"] - best <= TIE_TOLERANCE_CENTS]
    depth_rank = lambda d: math.inf if d is None else d  # noqa: E731
    chosen = sorted(within, key=lambda r: (-r["min_samples_leaf"], depth_rank(r["max_depth"]), r["effective_max_features"], r["mean_cv_mae"]))[0]
    within_ids = {id(r) for r in within}          # identity, not ==: rows hold numpy arrays
    for r in rows:
        r["within_tolerance"] = id(r) in within_ids
        r["selected"] = r is chosen
    return {"best_mean_cv_mae": best, "tolerance_cents": TIE_TOLERANCE_CENTS, "candidates_within_tolerance": [r["config"] for r in within],
            "rule": "among candidates within tolerance of the best mean CV MAE: larger min_samples_leaf, then smaller max_depth "
                    "(None = unlimited), then fewer features per split, then lower CV MAE",
            "selected": chosen}


def main() -> int:
    t0 = time.time()
    for name in OUTPUTS:
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; refusing to overwrite")
    for f in RB.FEATURES:
        if any(bad in f for bad in FORBIDDEN):
            raise RuntimeError(f"forbidden input {f}")
    protected = FE.protected_paths(("ml/results/axis2_v3", "ml/experiments/axis2_v3_natural_offset", "ml/experiments/axis2_v3_hybrid",
                                    "ml/experiments/axis2_v3_dynamic_hybrid", "ml/experiments/axis2_v3_shared"))
    hashes_before = NAT._hash_tree(protected)

    rid = json.loads(RIDGE_RESULTS.read_text())
    if tuple(rid["features"]) != RB.FEATURES or rid["test_evaluated"] is not False:
        raise RuntimeError("V3 Ridge result does not use ridge_baseline.FEATURES or is not test-sealed")

    d = FE.load(DATASET)
    D, vc = d["D"], d["variant_codes"]
    tr, va, y = d["tr"], d["va"], d["y"]
    print("featurizing with ridge_baseline.build_matrix", flush=True)
    X_tr, X_va = FE.featurize_ridge(d, tr), FE.featurize_ridge(d, va)
    fi = RB.FEATURES.index
    feature_checks = {
        "full_median_matches_stored_max_abs_diff": float(np.max(np.abs(np.r_[X_tr[:, fi("full_median_cents")], X_va[:, fi("full_median_cents")]]
                                                                        - D["median_cents"][np.r_[tr, va]]))),
        "core_median_matches_stored_max_abs_diff": float(np.max(np.abs(np.r_[X_tr[:, fi("core_median_cents")], X_va[:, fi("core_median_cents")]]
                                                                        - D["core_median_cents"][np.r_[tr, va]]))),
        "frame_count_matches_num_frames": bool(np.array_equal(np.r_[X_tr[:, fi("frame_count")], X_va[:, fi("frame_count")]], D["num_frames"][np.r_[tr, va]])),
    }
    if max(feature_checks["full_median_matches_stored_max_abs_diff"], feature_checks["core_median_matches_stored_max_abs_diff"]) > 1e-9 \
            or not feature_checks["frame_count_matches_num_frames"]:
        raise RuntimeError(f"features disagree with stored values: {feature_checks}")
    sanity = RB.shift_sanity(X_tr[:RB.SHIFT_CHECK_NOTES], d["cents_flat"], d["conf_flat"], d["off"], d["duration"], d["target_midi"],
                             tr[:RB.SHIFT_CHECK_NOTES])
    if not sanity["passed"]:
        raise RuntimeError(f"feature shift behaviour wrong: {sanity}")

    y_tr, y_va = y[tr], y[va]
    groups = D["song_index"][tr]
    folds = list(GroupKFold(n_splits=FE.CV_FOLDS).split(X_tr, y_tr, groups))
    for fit, held in folds:
        if set(groups[fit]) & set(groups[held]):
            raise RuntimeError("a song appears on both sides of a fold")
    # Same song partition as the V3 Ridge CV, so per-fold MAEs are directly comparable.
    ridge_cv = next(r for r in rid["hyperparameter_selection"]["cv_results"] if r["alpha"] == rid["hyperparameter_selection"]["selected_alpha"])
    gk = D["song_index"][tr].astype(np.int64) * 1_000_000 + D["source_note_index"][tr]
    siblings_split = int(sum(np.unique(gk[fit]).size + np.unique(gk[held]).size - np.unique(gk).size for fit, held in folds))

    center_cols = [fi(f) for f in RB.CENTER_FEATURES]
    rows_full = grouped_search("full24", X_tr, y_tr, folds)
    rows_center = grouped_search("center4", X_tr[:, center_cols], y_tr, folds)
    sel_full, sel_center = select(rows_full), select(rows_center)
    cfg_full = {k: sel_full["selected"][k] for k in ("max_depth", "min_samples_leaf", "max_features")}
    cfg_center = {k: sel_center["selected"][k] for k in ("max_depth", "min_samples_leaf", "max_features")}
    print(f"selected full: {label(cfg_full)}  center: {label(cfg_center)}", flush=True)

    # ---- final fits on all train; validation predicted once per model
    model = make(cfg_full).fit(X_tr, y_tr)
    model_c = make(cfg_center).fit(X_tr[:, center_cols], y_tr)
    p_va, pc_va = model.predict(X_va), model_c.predict(X_va[:, center_cols])
    p_tr = model.predict(X_tr)
    if not (np.all(np.isfinite(p_va)) and np.all(np.isfinite(pc_va))):
        raise RuntimeError("non-finite predictions")

    # ---- comparators from stored V3 predictions (not refit)
    keys = FE.keys(d, va)
    stored = {}
    with DYN_PREDICTIONS.open() as fh:
        for r in csv.DictReader(fh):
            stored[(r["song_id"], int(r["source_note_index"]), r["variant"])] = r
    if set(keys) != stored.keys() or len(set(keys)) != len(keys):
        raise RuntimeError("validation keys differ from the stored V3 predictions")
    col = lambda c: np.array([float(stored[k][c]) for k in keys])  # noqa: E731
    if np.max(np.abs(col("true_shift") - y_va)) > 2e-6:
        raise RuntimeError("stored targets differ")
    preds = {"ridge": col("ridge_prediction"), "rf": p_va, "rf_center": pc_va, "natural": col("natural_prediction"),
             "fixed": col("fixed_hybrid_prediction"), "dynamic": col("dynamic_gate_prediction")}
    errs = {k: p - y_va for k, p in preds.items()}
    dyn = json.loads(DYN_RESULTS.read_text())["validation"]["overall"]
    stored_check = {k: abs(float(np.mean(np.abs(errs[k]))) - dyn[k2]["mae"]) for k, k2 in
                    (("ridge", "ridge"), ("natural", "natural"), ("fixed", "fixed"), ("dynamic", "dynamic"))}
    stored_check["ridge_vs_ridge_baseline_json"] = abs(float(np.mean(np.abs(errs["ridge"]))) - rid["validation"]["overall"]["mae"])
    if max(stored_check.values()) > 1e-5:
        raise RuntimeError(f"stored comparator predictions do not reproduce their metrics: {stored_check}")

    var_va = D["variant_index"][va]
    corr = var_va != vc["clean"]
    clean = ~corr
    core_va = X_va[:, fi("core_median_cents")]
    overall = {k: FE.metrics(e) for k, e in errs.items()}
    by_variant = {v: {k: short(errs[k][var_va == c]) for k in errs} for v, c in vc.items()}
    slopes = {k: NAT.slope_stats(y_va[corr], preds[k][corr]) for k in errs}
    by_sign = {s: {k: short(errs[k][m]) for k in errs} for s, m in (("positive", corr & (y_va > 0)), ("negative", corr & (y_va < 0)))}
    nf, du = D["num_frames"][va].astype(float), d["duration"][va]
    clean_frames = {lab: {k: short(errs[k][clean & in_range(nf, lo, hi)]) for k in ("ridge", "rf")} for lab, lo, hi in FRAME_BUCKETS}
    clean_duration = {lab: {k: short(errs[k][clean & in_range(du, lo, hi)]) for k in ("ridge", "rf")} for lab, lo, hi in DURATION_BUCKETS}
    if np.any(np.abs(core_va) >= 49.999):
        raise RuntimeError("V3 validation example at or beyond the target boundary")
    by_core = {lab: {"count": int(in_range(np.abs(core_va), lo, hi).sum()),
                     **{f"{k}_mae": float(np.mean(np.abs(errs[k][in_range(np.abs(core_va), lo, hi)]))) for k in COMPARE}}
               for lab, lo, hi in CORE_BUCKETS}

    gkey = D["song_index"][va].astype(np.int64) * 1_000_000 + D["source_note_index"][va]
    uniq, inv, counts = np.unique(gkey, return_inverse=True, return_counts=True)
    if not np.all(counts == 5):
        raise RuntimeError("a validation note group does not have five variants")
    gmae = {k: np.bincount(inv, weights=np.abs(e)) / counts for k, e in errs.items()}

    def group_cmp(other: str) -> Dict[str, Any]:
        gain = gmae[other] - gmae["rf"]
        win, loss = gain > NOTE_TIE_CENTS, gain < -NOTE_TIE_CENTS
        return {"rf_better_percent": 100 * float(win.mean()), "tie_percent": 100 * float((~win & ~loss).mean()),
                f"{other}_better_percent": 100 * float(loss.mean()),
                "median_gain_when_rf_wins": float(np.median(gain[win])) if win.any() else None,
                "median_loss_when_rf_loses": float(np.median(-gain[loss])) if loss.any() else None, "mean_gain": float(gain.mean())}

    note_groups = {"note_groups": int(uniq.size), "tie_tolerance_cents": NOTE_TIE_CENTS,
                   **{f"rf_vs_{o}": group_cmp(o) for o in ("ridge", "natural", "dynamic")}}

    # ---- importance (interpretation only)
    print("permutation importance on validation (interpretation only)", flush=True)
    perm = permutation_importance(model, X_va, y_va, scoring="neg_mean_absolute_error", n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
    imp = sorted(({"feature": f, "impurity_importance": float(model.feature_importances_[j]),
                   "permutation_mae_increase_mean": float(perm.importances_mean[j]), "permutation_mae_increase_std": float(perm.importances_std[j]),
                   "is_center_feature": f in RB.CENTER_FEATURES} for j, f in enumerate(RB.FEATURES)), key=lambda r: -r["impurity_importance"])
    for i, r in enumerate(imp):
        r["impurity_rank"] = i + 1
    for i, r in enumerate(sorted(imp, key=lambda r: -r["permutation_mae_increase_mean"])):
        r["permutation_rank"] = i + 1
    with (OUT / "feature_importance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "is_center_feature", "impurity_importance", "impurity_rank",
                                           "permutation_mae_increase_mean", "permutation_mae_increase_std", "permutation_rank"])
        w.writeheader()
        for r in imp:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---- outputs
    with (OUT / "cv_results.csv").open("w", newline="") as fh:
        cols = ["feature_set", "config", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "effective_max_features",
                "mean_cv_mae", "std_cv_mae", "mean_cv_rmse", "pooled_oof_mae", "pooled_oof_rmse",
                *[f"fold{k}_mae" for k in range(FE.CV_FOLDS)], "within_tolerance", "selected", "fit_predict_seconds_total"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows_full + rows_center:
            w.writerow({**{c: r[c] for c in cols if c in r and not c.startswith("fold")},
                        **{f"fold{k}_mae": f"{r['fold_mae'][k]:.6f}" for k in range(FE.CV_FOLDS)},
                        **{c: f"{r[c]:.6f}" for c in ("mean_cv_mae", "std_cv_mae", "mean_cv_rmse", "pooled_oof_mae", "pooled_oof_rmse")}})
    with (OUT / "validation_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["song_id", "source_note_index", "variant", "true_shift", "observed_core_median", "ridge_prediction",
                    "random_forest_prediction", "random_forest_center_only_prediction", "natural_prediction", "dynamic_prediction",
                    "abs_random_forest_error"])
        for j, k in enumerate(keys):
            w.writerow([k[0], k[1], k[2], f"{y_va[j]:.6f}", f"{core_va[j]:.6f}", f"{preds['ridge'][j]:.6f}", f"{p_va[j]:.6f}",
                        f"{pc_va[j]:.6f}", f"{preds['natural'][j]:.6f}", f"{preds['dynamic'][j]:.6f}", f"{abs(errs['rf'][j]):.6f}"])

    nodes = int(sum(e.tree_.node_count for e in model.estimators_))
    saved = nodes <= SAVE_MODEL_MAX_NODES
    if saved:
        joblib.dump(model, OUT / "random_forest_model.joblib", compress=3)
    config = {"model": "sklearn.ensemble.RandomForestRegressor", "n_estimators": N_ESTIMATORS, **cfg_full, "random_state": SEED,
              "n_jobs": -1, "bootstrap": True, "criterion": "squared_error", "features": list(RB.FEATURES),
              "feature_source": "ml/baselines/ridge_baseline.py build_matrix / extract_features", "scaling": "none",
              "total_tree_nodes": nodes, "model_file_saved": saved,
              "model_file_note": None if saved else f"{nodes:,} nodes exceeds {SAVE_MODEL_MAX_NODES:,}; refit deterministically from this config",
              "center_only_diagnostic": {**cfg_center, "features": list(RB.CENTER_FEATURES)}}
    (OUT / "random_forest_config.json").write_text(json.dumps(config, indent=2) + "\n")

    zero = json.loads((BASELINES / "zero_baseline.json").read_text())["validation"]
    core = json.loads((BASELINES / "core_median_baseline.json").read_text())["validation"]
    ladder = {"zero": {"mae": zero["overall"]["mae"], "rmse": zero["overall"]["rmse"]},
              "core": {"mae": core["overall"]["mae"], "rmse": core["overall"]["rmse"]},
              **{k: {"mae": overall[k]["mae"], "rmse": overall[k]["rmse"]} for k in ("ridge", "rf", "natural", "fixed", "dynamic")}}
    rf_mae = overall["rf"]["mae"]
    band = ("very strong (< 13.46)" if rf_mae < 13.4557 else "strong (13.5-14.5)" if rf_mae < 14.5 else
            "useful nonlinear direct model (14.5-16.1)" if rf_mae < 16.1049 else "weak / negative (>= 16.10)")
    def strip(rows):
        return [{k: v for k, v in r.items() if k != "_oof"} for r in rows]
    sel_full_out = {**sel_full, "selected": {k: v for k, v in sel_full["selected"].items() if k != "_oof"}}
    sel_center_out = {**sel_center, "selected": {k: v for k, v in sel_center["selected"].items() if k != "_oof"}}
    hashes_after = NAT._hash_tree(protected)
    results = {
        "experiment": "axis2_v3_random_forest_direct", "test_evaluated": False, "dataset": "data/ml_dataset/axis2_v3",
        "question": "does a nonlinear Random Forest on exactly Ridge's 24 features beat the direct Ridge model?",
        "features": list(RB.FEATURES), "feature_source": "ml/baselines/ridge_baseline.py (build_matrix -> extract_features), imported",
        "feature_checks": feature_checks, "constant_shift_feature_sanity": sanity, "scaling": "none (raw features)",
        "excluded_inputs": ["song_id", "artist", "filename", "clean_note_id", "variant", "severity / band", "sign", "true shift magnitude",
                            "sibling information", "clean references", "natural-offset prediction", "harmonic fit / chroma"],
        "grid": {"n_estimators": N_ESTIMATORS, "max_depth": [str(x) for x in MAX_DEPTHS], "min_samples_leaf": list(MIN_LEAVES),
                 "max_features": [str(x) for x in MAX_FEATURES], "random_state": SEED, "n_jobs": -1, "configs_per_feature_set": len(rows_full)},
        "cv_protocol": {"method": "GroupKFold(5) by song over all train examples (all five variants of a note share its song, so siblings never straddle a fold)",
                        "folds": [{"fold": k, "fit_examples": int(a.size), "heldout_examples": int(b.size), "heldout_songs": int(np.unique(groups[b]).size)}
                                  for k, (a, b) in enumerate(folds)],
                        "note_groups_split_across_fold_sides": siblings_split,
                        "same_partition_as_v3_ridge": True, "validation_used_for_selection": False},
        "selection": {"full24": sel_full_out, "center4": sel_center_out},
        "cv_full24_vs_ridge_by_fold": {"ridge_fold_mae": ridge_cv["fold_mae"], "rf_fold_mae": sel_full["selected"]["fold_mae"],
                                       "ridge_mean_cv_mae": ridge_cv["mean_cv_mae"], "rf_mean_cv_mae": sel_full["selected"]["mean_cv_mae"]},
        "train_in_sample": short(p_tr - y_tr) | {"note": "in-sample on the fitted rows; optimistic for a forest"},
        "validation": {"overall": overall, "by_variant": by_variant, "prediction_slope_corrupted": slopes, "by_sign_corrupted": by_sign,
                       "clean_by_frame_count": clean_frames, "clean_by_duration": clean_duration, "by_abs_observed_core": by_core,
                       "note_groups": note_groups, "post_hoc_only": "all breakdowns computed after the model was fixed"},
        "center_only_diagnostic": {"features": list(RB.CENTER_FEATURES), "validation": FE.metrics(errs["rf_center"]),
                                   "by_variant_mae": {v: by_variant[v]["rf_center"]["mae"] for v in vc},
                                   "full_minus_center_mae": overall["rf"]["mae"] - overall["rf_center"]["mae"],
                                   "full_minus_center_rmse": overall["rf"]["rmse"] - overall["rf_center"]["rmse"]},
        "feature_importance": {"impurity_top10": [{"feature": r["feature"], "importance": r["impurity_importance"]} for r in imp[:10]],
                               "permutation_top10": [{"feature": r["feature"], "mae_increase": r["permutation_mae_increase_mean"],
                                                      "std": r["permutation_mae_increase_std"]}
                                                     for r in sorted(imp, key=lambda r: -r["permutation_mae_increase_mean"])[:10]],
                               "permutation_method": f"sklearn permutation_importance on validation, MAE, {PERM_REPEATS} repeats; interpretation only",
                               "caveat": "the four center features (and the percentiles) are strongly correlated; a forest splits credit among "
                                         "correlated features, and permuting one leaves its correlates intact, so individual importances understate "
                                         "the group and should not be read as unique contributions"},
        "ladder": ladder, "interpretation_band": band,
        "benchmarks": {"beats_ridge_16_1049": rf_mae < 16.1049, "beats_natural_14_0508": rf_mae < 14.0508, "beats_dynamic_13_4557": rf_mae < 13.4557},
        "comparator_sources": {"predictions": FE.rel(DYN_PREDICTIONS), "metrics_reproduced_max_diff": stored_check},
        "integrity": {"test_examples_featurized": 0, "test_predictions": 0, "validation_prediction_passes": {"rf": 1, "rf_center": 1},
                      "protected_paths": [FE.rel(p) for p in protected], "protected_hashes_before": hashes_before,
                      "protected_hashes_after": hashes_after, "protected_unchanged": hashes_before == hashes_after, "runtime_s": time.time() - t0},
    }
    results["cv_all"] = {"full24": strip(rows_full), "center4": strip(rows_center)}
    (OUT / "random_forest_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (OUT / "random_forest_summary.md").write_text(summary_md(results))
    print(json.dumps({"selected": label(cfg_full), "validation": {k: overall["rf"][k] for k in ("mae", "rmse")},
                      "center_only": {k: overall["rf_center"][k] for k in ("mae", "rmse")}, "band": band,
                      "protected_unchanged": results["integrity"]["protected_unchanged"]}, indent=1))
    return 0


def summary_md(r) -> str:
    f = lambda x, k=3: "—" if x is None else f"{x:.{k}f}"  # noqa: E731
    v, s = r["validation"], r["selection"]["full24"]["selected"]
    sc = r["selection"]["center4"]["selected"]
    o = v["overall"]
    L = ["# Axis 2 V3: direct Random Forest on Ridge's 24 features", "",
         "Generated by `random_forest.py`. Same features as the V3 Ridge baseline (its extractor, imported), raw/unscaled; "
         "hyperparameters chosen by GroupKFold(5)-by-song CV on train only; validation evaluated once; **test not evaluated**.", "",
         f"**Selected:** {s['config']} (300 trees). CV MAE {f(s['mean_cv_mae'], 4)} ± {f(s['std_cv_mae'], 4)}, CV RMSE {f(s['mean_cv_rmse'], 4)}, "
         f"pooled OOF MAE {f(s['pooled_oof_mae'], 4)}. Candidates within 0.05 c: {', '.join(r['selection']['full24']['candidates_within_tolerance'])}.", "",
         f"Interpretation band: **{r['interpretation_band']}**.", "",
         "## Ladder (validation)", "", "| model | MAE | RMSE |", "|---|---|---|"]
    for k, x in r["ladder"].items():
        L.append(f"| {LABEL[k]} | {f(x['mae'], 4)} | {f(x['rmse'], 4)} |")
    L += ["", "## Random Forest overall", "",
          f"MAE {f(o['rf']['mae'], 4)}, RMSE {f(o['rf']['rmse'], 4)}, median abs {f(o['rf']['median_absolute_error'])}, "
          f"mean signed {o['rf']['mean_signed_error']:+.3f}.", "",
          "| model | P75 | P90 | P95 | P99 | P99.5 | P99.9 | max | > 25 ¢ | > 50 ¢ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("ridge", "rf", "natural", "dynamic"):
        x = o[k]
        L.append(f"| {LABEL[k]} | " + " | ".join(f(x["quantiles"][q], 2) for q in ("p75", "p90", "p95", "p99", "p99.5", "p99.9"))
                 + f" | {f(x['max'], 2)} | {x['exceeding']['gt_25_cents']['count']} ({x['exceeding']['gt_25_cents']['percent']:.2f}%) | "
                   f"{x['exceeding']['gt_50_cents']['count']} ({x['exceeding']['gt_50_cents']['percent']:.2f}%) |")
    L += ["", "## MAE by variant", "", "| variant | Ridge | Random Forest | Natural | Fixed | Dynamic | RF center-only |", "|---|---|---|---|---|---|---|"]
    for n, x in v["by_variant"].items():
        L.append(f"| {n} | " + " | ".join(f(x[k]["mae"], 2) for k in ("ridge", "rf", "natural", "fixed", "dynamic", "rf_center")) + " |")
    L += ["", "**Random Forest by variant**", "", "| variant | n | MAE | RMSE | median abs | mean signed |", "|---|---|---|---|---|---|"]
    for n, x in v["by_variant"].items():
        y = x["rf"]
        L.append(f"| {n} | {y['count']} | {f(y['mae'])} | {f(y['rmse'])} | {f(y['median_absolute_error'])} | {y['mean_signed_error']:+.3f} |")
    L += ["", "## Prediction slope (corrupted validation; diagnostic only — V3 sign depends partly on the clean offset)", "",
          "| model | slope | intercept | R² |", "|---|---|---|---|"]
    for k in ("ridge", "rf", "natural", "dynamic"):
        x = v["prediction_slope_corrupted"][k]
        L.append(f"| {LABEL[k]} | {f(x['slope'])} | {x['intercept']:+.3f} | {f(x['r2'])} |")
    L += ["", "## Sign (corrupted validation)", "", "| sign | model | n | MAE | RMSE | mean signed |", "|---|---|---|---|---|---|"]
    for sgn, x in v["by_sign_corrupted"].items():
        for k in ("ridge", "rf", "natural", "dynamic"):
            L.append(f"| {sgn} | {LABEL[k]} | {x[k]['count']} | {f(x[k]['mae'])} | {f(x[k]['rmse'])} | {x[k]['mean_signed_error']:+.3f} |")
    c = v["by_variant"]["clean"]
    L += ["", "## Clean notes (3,740)", "", f"Ridge MAE {f(c['ridge']['mae'])} / RMSE {f(c['ridge']['rmse'])}; "
          f"Random Forest MAE {f(c['rf']['mae'])} / RMSE {f(c['rf']['rmse'])}.", "",
          "| frames | n | Ridge MAE | RF MAE |", "|---|---|---|---|"]
    for b, x in v["clean_by_frame_count"].items():
        L.append(f"| {b} | {x['rf']['count']} | {f(x['ridge']['mae'])} | {f(x['rf']['mae'])} |")
    L += ["", "| duration (s) | n | Ridge MAE | RF MAE |", "|---|---|---|---|"]
    for b, x in v["clean_by_duration"].items():
        L.append(f"| {b} | {x['rf']['count']} | {f(x['ridge']['mae'])} | {f(x['rf']['mae'])} |")
    L += ["", "## By |observed core| (post hoc)", "", "| bucket | n | Ridge | RF | Natural | Fixed | Dynamic |", "|---|---|---|---|---|---|---|"]
    for b, x in v["by_abs_observed_core"].items():
        L.append(f"| {b} | {x['count']} | " + " | ".join(f(x[f'{k}_mae']) for k in COMPARE) + " |")
    ng = v["note_groups"]
    L += ["", f"## Note groups ({ng['note_groups']} notes; mean |error| over five variants; tie within 0.5 c)", "",
          "| comparison | RF better | tie | other better | median gain (RF wins) | median loss (RF loses) |", "|---|---|---|---|---|---|"]
    for o_ in ("ridge", "natural", "dynamic"):
        x = ng[f"rf_vs_{o_}"]
        L.append(f"| RF vs {LABEL[o_]} | {x['rf_better_percent']:.1f}% | {x['tie_percent']:.1f}% | {x[f'{o_}_better_percent']:.1f}% | "
                 f"{f(x['median_gain_when_rf_wins'])} | {f(x['median_loss_when_rf_loses'])} |")
    cd = r["center_only_diagnostic"]
    L += ["", "## Center-only diagnostic", "", f"Selected {sc['config']}, CV MAE {f(sc['mean_cv_mae'], 4)}. Validation MAE "
          f"{f(cd['validation']['mae'], 4)} / RMSE {f(cd['validation']['rmse'], 4)}; full − center = {cd['full_minus_center_mae']:+.4f} c MAE "
          f"({cd['full_minus_center_rmse']:+.4f} RMSE).", "",
          "## Feature importance (interpretation only)", "", "| rank | impurity | importance | permutation | MAE increase |", "|---|---|---|---|---|"]
    fi_ = r["feature_importance"]
    for i in range(10):
        a, b = fi_["impurity_top10"][i], fi_["permutation_top10"][i]
        L.append(f"| {i + 1} | {a['feature']} | {f(a['importance'], 4)} | {b['feature']} | {f(b['mae_increase'], 3)} ± {f(b['std'], 3)} |")
    L += ["", f"_{fi_['caveat']}._", "", "## Integrity", "",
          f"- Features equal Ridge's: full/core median vs stored max diff {r['feature_checks']['full_median_matches_stored_max_abs_diff']:.1e} / "
          f"{r['feature_checks']['core_median_matches_stored_max_abs_diff']:.1e}; constant-shift sanity passed: {r['constant_shift_feature_sanity']['passed']}.",
          f"- Note groups split across a fold's fit/held-out sides: {r['cv_protocol']['note_groups_split_across_fold_sides']}.",
          f"- Comparator predictions reproduce their stored metrics (max diff {max(r['comparator_sources']['metrics_reproduced_max_diff'].values()):.1e}).",
          f"- Test examples featurized: 0. Protected files unchanged: {r['integrity']['protected_unchanged']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
