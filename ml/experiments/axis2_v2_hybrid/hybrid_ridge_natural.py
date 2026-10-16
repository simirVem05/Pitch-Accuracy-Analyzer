#!/usr/bin/env python3
"""
Axis 2 V2: fixed convex blend of the two existing shift estimators.

    hybrid = w * ridge + (1 - w) * natural,   0 <= w <= 1, ONE global scalar

Components are frozen, not re-tuned. Their feature extractors, model classes,
and hyperparameters are imported from the scripts that produced them, and the
selected settings are read from their stored results and asserted:

  ridge    ml/baselines/ridge_baseline.py; StandardScaler -> Ridge(alpha) on all
           24 features, fit on every train example (all five variants),
           target = injected shift.
  natural  ml/experiments/axis2_v2_natural_offset/; HGB on the 71 shift-invariant
           features, fit on train CLEAN notes, target N = clean core median;
           shift estimate = own observed core median - N_hat(own features).

w is chosen on train only, from out-of-fold predictions. One song-level
partition (GroupKFold(5) by song) is used for both components, so no train
example is ever predicted by a model that saw its song. As a freeze check, the
per-fold MAEs each component produced in its original CV are reproduced exactly
before any blending happens. Validation is predicted once, after w is fixed.

The test split is sealed: no test contour is featurized, no test song's
notes.npz is opened, and no test prediction exists.

  python ml/experiments/axis2_v2_hybrid/hybrid_ridge_natural.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
# Importing the natural-offset experiment must not drop a __pycache__ into its
# protected directory.
sys.dont_write_bytecode = True
NATURAL_DIR = REPO / "ml" / "experiments" / "axis2_v2_natural_offset"
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "ml" / "baselines"))
sys.path.insert(0, str(NATURAL_DIR))

import natural_offset_feature_experiment as NAT  # noqa: E402
import ridge_baseline as RB  # noqa: E402
from zero_baseline import EXPECTED_SPLIT_ORDER, EXPECTED_VARIANT_ORDER, _resolve_lookup, _verify_counts  # noqa: E402

DATASET = REPO / "data" / "ml_dataset" / "axis2_v2"
RIDGE_RESULTS = REPO / "ml" / "results" / "axis2_v2" / "ridge_baseline.json"
NATURAL_RESULTS = NATURAL_DIR / "natural_offset_results.json"
NATURAL_VAL_PREDICTIONS = NATURAL_DIR / "validation_predictions.csv"

CV_FOLDS = 5
WEIGHTS = tuple(round(k / 100, 2) for k in range(101))
# Two weights whose OOF MAE differs by less than this are treated as tied; the
# tie goes to the weight nearest 0.5 (the most central, least extreme blend).
TIE_TOLERANCE_CENTS = 1e-9
NOTE_TIE_CENTS = 0.5
# Freeze checks: reproduced fold MAEs and validation results must match the
# stored ones to this tolerance; the natural CSV stores 6 decimals.
FREEZE_TOLERANCE = 1e-9
CSV_TOLERANCE = 2e-6

QUANTILES = (50, 75, 90, 95, 99, 99.5, 99.9)
THRESHOLDS = (25, 50, 100)
MAGNITUDE_BANDS = (("clean (0)", 0.0, 0.0), ("5-15", 5.0, 15.0), ("15-25", 15.0, 25.0),
                   ("25-35", 25.0, 35.0), ("35-50", 35.0, 50.0))
FINE_BANDS = tuple((f"{lo}-{lo + 5}", float(lo), float(lo + 5)) for lo in range(5, 50, 5))
FRAME_BUCKETS = (("3-4", 3, 5), ("5-6", 5, 7), ("7-9", 7, 10), ("10-20", 10, 21), ("21-50", 21, 51), ("51+", 51, None))
DURATION_BUCKETS = (("<0.10", None, 0.10), ("0.10-0.20", 0.10, 0.20), ("0.20-0.40", 0.20, 0.40),
                    ("0.40-0.80", 0.40, 0.80), (">=0.80", 0.80, None))
ESTIMATORS = ("ridge", "natural", "hybrid")
OUTPUTS = ("hybrid_results.json", "hybrid_summary.md", "weight_sweep.csv", "validation_predictions.csv")


# ---------------------------------------------------------------- metrics


def metrics(err: np.ndarray) -> Dict[str, Any]:
    a = np.abs(err)
    return {"count": int(err.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median_absolute_error": float(np.median(a)), "mean_signed_error": float(err.mean())}


def tail(err: np.ndarray) -> Dict[str, Any]:
    a = np.abs(err)
    return {"quantiles": {f"p{q:g}": float(np.percentile(a, q)) for q in QUANTILES},
            "max": float(a.max()),
            "exceeding": {f"gt_{t}_cents": {"count": int(np.count_nonzero(a > t)),
                                            "percent": 100.0 * float(np.count_nonzero(a > t)) / a.size}
                          for t in THRESHOLDS}}


def in_band(mag: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi == 0.0:
        return mag == 0.0
    return (mag >= lo) & (mag <= hi) if hi == 50.0 else (mag >= lo) & (mag < hi)


def in_range(v: np.ndarray, lo, hi) -> np.ndarray:
    m = np.ones(v.shape, bool)
    if lo is not None:
        m &= v >= lo
    if hi is not None:
        m &= v < hi
    return m


# ---------------------------------------------------------------- frozen components


def frozen_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Read the selected settings from the stored results and assert they are what the code will run."""
    rid = json.loads(RIDGE_RESULTS.read_text())
    nat = json.loads(NATURAL_RESULTS.read_text())
    hp = rid["hyperparameter_selection"]
    if tuple(rid["features"]) != RB.FEATURES or hp["feature_count"] != len(RB.FEATURES):
        raise RuntimeError("stored Ridge feature list differs from ridge_baseline.FEATURES")
    if rid["test_evaluated"] is not False or nat["test_evaluated"] is not False:
        raise RuntimeError("a component result is not test-sealed")
    sel = nat["selected_model"]
    nat_cols = NAT.cols_for(sel["feature_set"])
    if sel["model"] != "hgb" or [NAT.ALL_NAMES[c] for c in nat_cols] != sel["features"]:
        raise RuntimeError("stored natural-offset selection differs from the experiment's feature definitions")
    if nat["cv_design"]["hgb_configs"][sel["param"]] != NAT.HGB_CONFIGS[sel["param"]]:
        raise RuntimeError("stored HGB hyperparameters differ from the experiment's HGB_CONFIGS")
    alpha = hp["selected_alpha"]
    ridge_cv = next(r for r in hp["cv_results"] if r["alpha"] == alpha)
    nat_cv = next(r for r in nat["cv_results"]
                  if (r["feature_set"], r["model"], r["param"]) == (sel["feature_set"], sel["model"], sel["param"]))
    ridge = {"source": "ml/baselines/ridge_baseline.py", "result": "ml/results/axis2_v2/ridge_baseline.json",
             "model": "StandardScaler -> sklearn Ridge", "alpha": alpha, "features": list(RB.FEATURES),
             "n_features": len(RB.FEATURES), "fit_rows": "every train example (all five variants)",
             "target": "injected_shift_cents", "stored_cv_fold_mae": ridge_cv["fold_mae"],
             "stored_validation": {k: rid["validation"]["overall"][k] for k in ("mae", "rmse")},
             "stored_validation_by_variant_mae": {v: rid["validation"]["by_variant"][v]["mae"] for v in EXPECTED_VARIANT_ORDER}}
    natural = {"source": "ml/experiments/axis2_v2_natural_offset/natural_offset_feature_experiment.py",
               "result": "ml/experiments/axis2_v2_natural_offset/natural_offset_results.json",
               "model": "sklearn HistGradientBoostingRegressor(random_state=0, early_stopping=False)",
               "feature_set": sel["feature_set"], "param": sel["param"], "hyperparameters": NAT.HGB_CONFIGS[sel["param"]],
               "features": sel["features"], "n_features": len(nat_cols), "columns": nat_cols,
               "fit_rows": "train clean notes only", "target": "clean core median N",
               "shift_estimate": "observed core median of the example - N_hat(example's own features)",
               "stored_cv_fold_mae": nat_cv["fold_mae"],
               "stored_validation": {k: nat["synthetic_shift_validation"]["overall"][k] for k in ("mae", "rmse")}}
    return ridge, natural


def ridge_fit_predict(alpha: float, Xfit: np.ndarray, yfit: np.ndarray, Xpred: np.ndarray) -> np.ndarray:
    model, warns = RB._fit(alpha, Xfit, yfit)
    p, chk = RB._predict(model, Xpred)
    if warns or not chk["ok"]:
        raise RuntimeError(f"Ridge fit/predict check failed: {warns} {chk}")
    return p


def natural_fit_predict(param: str, Xfit: np.ndarray, Nfit: np.ndarray, Xpred: np.ndarray) -> np.ndarray:
    return NAT.fit_predict("hgb", param, Xfit, Nfit, Xpred)


# ---------------------------------------------------------------- main


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; refusing to overwrite")
    protected = [REPO / "data" / "main_data", REPO / "data" / "ml_dataset" / "axis2_v1", DATASET,
                 REPO / "data" / "dataset_selections.json", REPO / "ml" / "results" / "axis2_v1",
                 REPO / "ml" / "results" / "axis2_v2", NATURAL_DIR,
                 REPO / "ml" / "experiments" / "axis2_v2_sequence_natural_offset"]
    hashes_before = NAT._hash_tree(protected)

    ridge_cfg, nat_cfg = frozen_configs()
    alpha, nat_param, nat_cols = ridge_cfg["alpha"], nat_cfg["param"], nat_cfg["columns"]

    splits = json.loads((DATASET / "splits.json").read_text())
    with np.load(DATASET / "examples.npz", allow_pickle=False) as z:
        D = {k: np.asarray(z[k]) for k in z.files if k not in ("cents_flat", "confidence_flat")}
        cents_flat = np.asarray(z["cents_flat"], np.float64)
        conf_flat = np.asarray(z["confidence_flat"], np.float64)
    split_codes = _resolve_lookup(D["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
    variant_codes = _resolve_lookup(D["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")
    _verify_counts(D["split_index"], split_codes, splits)
    off = np.asarray(D["contour_offsets"], np.int64)
    song_ids = D["song_ids"]
    duration = np.asarray(D["duration_s"], np.float64)
    target_midi = D["target_midi"]
    y_all = np.asarray(D["injected_shift_cents"], np.float64)

    # Test examples never enter these index sets.
    tr = np.flatnonzero(D["split_index"] == split_codes["train"])
    va = np.flatnonzero(D["split_index"] == split_codes["validation"])
    assert tr.size == splits["train_example_count"] and va.size == splits["validation_example_count"]

    # ---- features, each component with its own extractor
    print("featurizing (Ridge)", flush=True)
    Xr_tr = RB.build_matrix(cents_flat, conf_flat, off, duration, target_midi, tr)
    Xr_va = RB.build_matrix(cents_flat, conf_flat, off, duration, target_midi, va)

    print("featurizing (natural offset)", flush=True)
    ctx_cache: Dict[str, Dict[str, np.ndarray]] = {}
    ctx_by_song: Dict[int, np.ndarray] = {}
    songs_opened: set = set()

    def nat_featurize(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.empty((idx.size, len(NAT.ALL_NAMES)))
        core = np.empty(idx.size)
        for r, i in enumerate(idx):
            a, b = int(off[i]), int(off[i + 1])
            f, core[r] = NAT.contour_features(cents_flat[a:b], conf_flat[a:b], float(duration[i]), int(target_midi[i]))
            s = int(D["song_index"][i])
            if s not in ctx_by_song:
                if D["song_split_index"][s] == split_codes["test"]:
                    raise RuntimeError("attempted to read a test song")
                ctx_by_song[s] = NAT.context_features(str(song_ids[s]), ctx_cache)
                songs_opened.add(str(song_ids[s]))
            X[r, :f.size] = f
            X[r, f.size:] = ctx_by_song[s][int(D["source_note_index"][i])]
        return X[:, nat_cols], core

    Xn_tr, core_tr = nat_featurize(tr)
    Xn_va, core_va = nat_featurize(va)
    for X in (Xr_tr, Xr_va, Xn_tr, Xn_va):
        if not np.all(np.isfinite(X)):
            raise RuntimeError("non-finite features")
    core_diff = float(max(np.max(np.abs(core_tr - D["core_median_cents"][tr])),
                          np.max(np.abs(core_va - D["core_median_cents"][va]))))
    if core_diff > 1e-9:
        raise RuntimeError(f"recomputed core median differs from stored by {core_diff}")

    var_tr, var_va = D["variant_index"][tr], D["variant_index"][va]
    clean_tr = var_tr == variant_codes["clean"]
    y_tr, y_va = y_all[tr], y_all[va]
    groups_tr = D["song_index"][tr]

    # ---- one song-level partition for both components
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(Xr_tr, y_tr, groups_tr))
    song_fold = {}
    for k, (_, held) in enumerate(folds):
        for s in np.unique(groups_tr[held]):
            song_fold[int(s)] = k
    fold_of = np.array([song_fold[int(s)] for s in groups_tr])
    for k, (fit, held) in enumerate(folds):
        if set(groups_tr[fit]) & set(groups_tr[held]):
            raise RuntimeError("a song appears on both sides of a fold")
    # The natural experiment ran GroupKFold on clean notes alone; confirm it yields the same song partition.
    clean_pos = np.flatnonzero(clean_tr)
    nat_native = list(GroupKFold(n_splits=CV_FOLDS).split(Xn_tr[clean_pos], core_tr[clean_pos], groups_tr[clean_pos]))
    native_partition = [frozenset(np.unique(groups_tr[clean_pos][h]).tolist()) for _, h in nat_native]
    shared_partition = [frozenset(np.unique(groups_tr[h]).tolist()) for _, h in folds]
    if native_partition != shared_partition:
        raise RuntimeError("the natural experiment's native folds partition songs differently from Ridge's")

    # ---- out-of-fold predictions on every train example
    print("out-of-fold predictions", flush=True)
    ridge_oof = np.full(tr.size, np.nan)
    nhat_oof = np.full(tr.size, np.nan)
    ridge_fold_mae, nat_fold_mae = [], []
    for k in range(CV_FOLDS):
        held, fit = fold_of == k, fold_of != k
        ridge_oof[held] = ridge_fit_predict(alpha, Xr_tr[fit], y_tr[fit], Xr_tr[held])
        ridge_fold_mae.append(float(np.mean(np.abs(ridge_oof[held] - y_tr[held]))))
        fit_clean = fit & clean_tr
        nhat_oof[held] = natural_fit_predict(nat_param, Xn_tr[fit_clean], core_tr[fit_clean], Xn_tr[held])
        held_clean = held & clean_tr
        nat_fold_mae.append(float(np.mean(np.abs(nhat_oof[held_clean] - core_tr[held_clean]))))
    natural_oof = core_tr - nhat_oof
    if np.isnan(ridge_oof).any() or np.isnan(natural_oof).any():
        raise RuntimeError("missing OOF predictions")

    # Freeze check: the original CV fold MAEs must reproduce exactly.
    ridge_fold_diff = float(np.max(np.abs(np.array(ridge_fold_mae) - np.array(ridge_cfg["stored_cv_fold_mae"]))))
    nat_fold_diff = float(np.max(np.abs(np.array(nat_fold_mae) - np.array(nat_cfg["stored_cv_fold_mae"]))))
    if ridge_fold_diff > FREEZE_TOLERANCE or nat_fold_diff > FREEZE_TOLERANCE:
        raise RuntimeError(f"component CV does not reproduce: ridge {ridge_fold_diff}, natural {nat_fold_diff}")

    # ---- alignment by stable key, not row order
    def keyed(idx: np.ndarray, pred: np.ndarray) -> Dict[Tuple[str, int, str], Tuple[float, float]]:
        out = {}
        for j, i in enumerate(idx):
            key = (str(song_ids[D["song_index"][i]]), int(D["source_note_index"][i]), EXPECTED_VARIANT_ORDER[int(D["variant_index"][i])])
            if key in out:
                raise RuntimeError(f"duplicate example key {key}")
            out[key] = (float(y_all[i]), float(pred[j]))
        return out

    def align(ridge_pred, natural_pred, idx) -> Dict[str, Any]:
        kr, kn = keyed(idx, ridge_pred), keyed(idx, natural_pred)
        if kr.keys() != kn.keys():
            raise RuntimeError("component prediction keys differ")
        y_diff = max(abs(kr[k][0] - kn[k][0]) for k in kr)
        if y_diff != 0.0:
            raise RuntimeError("component targets differ for the same key")
        return {"examples": len(kr), "ridge_examples": len(kr), "natural_examples": len(kn), "unique_keys": len(kr),
                "missing_in_either": len(kr.keys() ^ kn.keys()), "max_target_difference": y_diff,
                "key": "(song_id, source_note_index, variant)"}

    oof_alignment = align(ridge_oof, natural_oof, tr)

    # ---- weight sweep on train OOF
    corr_tr = ~clean_tr
    sweep = []
    for w in WEIGHTS:
        p = w * ridge_oof + (1.0 - w) * natural_oof
        m = metrics(p - y_tr)
        sl = NAT.slope_stats(y_tr[corr_tr], p[corr_tr])
        sweep.append({"weight_ridge": w, "weight_natural": round(1.0 - w, 2), "oof_mae": m["mae"], "oof_rmse": m["rmse"],
                      "median_absolute_error": m["median_absolute_error"], "mean_signed_error": m["mean_signed_error"],
                      "prediction_slope": sl["slope"]})
    best_mae = min(r["oof_mae"] for r in sweep)
    tied = [r for r in sweep if r["oof_mae"] - best_mae <= TIE_TOLERANCE_CENTS]
    chosen = min(tied, key=lambda r: (abs(r["weight_ridge"] - 0.5), r["weight_ridge"]))
    w = chosen["weight_ridge"]
    print(f"selected w_ridge = {w}", flush=True)
    with (OUT / "weight_sweep.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        wr.writeheader()
        for r in sweep:
            wr.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k not in ("weight_ridge", "weight_natural") else v)
                         for k, v in r.items()})
    hybrid_oof = w * ridge_oof + (1.0 - w) * natural_oof
    oof = {name: metrics(p - y_tr) | {"slope_corrupted": NAT.slope_stats(y_tr[corr_tr], p[corr_tr])["slope"]}
           for name, p in (("ridge", ridge_oof), ("natural", natural_oof), ("hybrid", hybrid_oof))}

    # ---- final fits on all train; validation predicted once
    print("final fit and single validation prediction", flush=True)
    ridge_va = ridge_fit_predict(alpha, Xr_tr, y_tr, Xr_va)
    nhat_va = natural_fit_predict(nat_param, Xn_tr[clean_tr], core_tr[clean_tr], Xn_va)
    natural_va = core_va - nhat_va
    hybrid_va = w * ridge_va + (1.0 - w) * natural_va
    val_alignment = align(ridge_va, natural_va, va)

    # Freeze check against the stored validation results.
    ridge_val_diff = max(abs(metrics(ridge_va - y_va)[k] - ridge_cfg["stored_validation"][k]) for k in ("mae", "rmse"))
    nat_val_diff = max(abs(metrics(natural_va - y_va)[k] - nat_cfg["stored_validation"][k]) for k in ("mae", "rmse"))
    stored_nat = {}
    with NATURAL_VAL_PREDICTIONS.open() as fh:
        for row in csv.DictReader(fh):
            stored_nat[(row["song_id"], int(row["source_note_index"]), row["variant"])] = float(row["predicted_shift"])
    kn_va = keyed(va, natural_va)
    if stored_nat.keys() != kn_va.keys():
        raise RuntimeError("validation keys differ from the natural experiment's stored predictions")
    nat_csv_diff = max(abs(stored_nat[k] - kn_va[k][1]) for k in kn_va)
    if ridge_val_diff > FREEZE_TOLERANCE or nat_val_diff > FREEZE_TOLERANCE or nat_csv_diff > CSV_TOLERANCE:
        raise RuntimeError(f"validation does not reproduce: ridge {ridge_val_diff}, natural {nat_val_diff}, csv {nat_csv_diff}")

    preds = {"ridge": ridge_va, "natural": natural_va, "hybrid": hybrid_va}
    errs = {k: p - y_va for k, p in preds.items()}
    corr_va = var_va != variant_codes["clean"]
    clean_va = ~corr_va
    mag = np.abs(y_va)

    def per_estimator(mask: np.ndarray) -> Dict[str, Any]:
        return {k: metrics(e[mask]) for k, e in errs.items()}

    overall = {k: metrics(e) | {"tail": tail(e)} for k, e in errs.items()}
    by_variant = {v: per_estimator(var_va == c) for v, c in variant_codes.items()}
    by_sign = {"positive": per_estimator(corr_va & (y_va > 0)), "negative": per_estimator(corr_va & (y_va < 0))}
    slopes = {k: NAT.slope_stats(y_va[corr_va], p[corr_va]) for k, p in preds.items()}
    by_magnitude = {lab: per_estimator(in_band(mag, lo, hi)) for lab, lo, hi in MAGNITUDE_BANDS}
    by_fine = {lab: per_estimator(corr_va & in_band(mag, lo, hi)) for lab, lo, hi in FINE_BANDS}
    nframes = D["num_frames"][va].astype(np.float64)
    by_frames = {lab: per_estimator(clean_va & in_range(nframes, lo, hi)) for lab, lo, hi in FRAME_BUCKETS}
    by_duration = {lab: per_estimator(clean_va & in_range(duration[va], lo, hi)) for lab, lo, hi in DURATION_BUCKETS}

    # ---- note-group comparison: mean |error| over each note's five siblings
    gkey = D["song_index"][va].astype(np.int64) * 1_000_000 + D["source_note_index"][va].astype(np.int64)
    uniq, inv, counts = np.unique(gkey, return_inverse=True, return_counts=True)
    if not np.all(counts == 5):
        raise RuntimeError("a validation note group does not have exactly five variants")
    gmae = {k: np.bincount(inv, weights=np.abs(e)) / counts for k, e in errs.items()}
    gain = gmae["ridge"] - gmae["hybrid"]                       # > 0 means the hybrid is better
    wins, losses = gain > NOTE_TIE_CENTS, gain < -NOTE_TIE_CENTS
    ties = ~(wins | losses)
    note_groups = {
        "note_groups": int(uniq.size), "tie_tolerance_cents": NOTE_TIE_CENTS,
        "metric": "per note: mean absolute error over its five variants",
        "hybrid_wins_percent": 100.0 * float(wins.mean()), "tied_percent": 100.0 * float(ties.mean()),
        "ridge_wins_percent": 100.0 * float(losses.mean()),
        "median_improvement_when_hybrid_wins": float(np.median(gain[wins])) if wins.any() else None,
        "median_degradation_when_hybrid_loses": float(np.median(-gain[losses])) if losses.any() else None,
        "mean_note_mae": {k: float(v.mean()) for k, v in gmae.items()},
        "mean_per_note_gain_hybrid_vs_ridge": float(gain.mean()),
        "natural_vs_ridge": {"natural_wins_percent": 100.0 * float(np.mean(gmae["ridge"] - gmae["natural"] > NOTE_TIE_CENTS)),
                             "ridge_wins_percent": 100.0 * float(np.mean(gmae["natural"] - gmae["ridge"] > NOTE_TIE_CENTS))},
    }

    rmae, hmae = overall["ridge"]["mae"], overall["hybrid"]["mae"]
    rrmse, hrmse = overall["ridge"]["rmse"], overall["hybrid"]["rmse"]
    primary = {"ridge_mae_benchmark": rmae, "hybrid_mae": hmae, "hybrid_beats_ridge_mae": hmae < rmae,
               "mae_improvement": rmae - hmae, "mae_improvement_percent": 100.0 * (rmae - hmae) / rmae,
               "ridge_rmse": rrmse, "hybrid_rmse": hrmse,
               "rmse_improvement": rrmse - hrmse, "rmse_improvement_percent": 100.0 * (rrmse - hrmse) / rrmse}

    comparison = {k: {"mae": overall[k]["mae"], "rmse": overall[k]["rmse"],
                      "median_absolute_error": overall[k]["median_absolute_error"],
                      "p90": overall[k]["tail"]["quantiles"]["p90"], "p95": overall[k]["tail"]["quantiles"]["p95"],
                      "p99": overall[k]["tail"]["quantiles"]["p99"], "max": overall[k]["tail"]["max"],
                      "prediction_slope": slopes[k]["slope"],
                      "by_variant_mae": {v: by_variant[v][k]["mae"] for v in EXPECTED_VARIANT_ORDER}}
                  for k in ESTIMATORS}

    with (OUT / "validation_predictions.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["song_id", "source_note_index", "variant", "frame_count", "duration_s", "injected_shift",
                     "ridge_prediction", "natural_prediction", "hybrid_prediction"])
        for j, i in enumerate(va):
            wr.writerow([str(song_ids[D["song_index"][i]]), int(D["source_note_index"][i]), EXPECTED_VARIANT_ORDER[int(var_va[j])],
                         int(D["num_frames"][i]), f"{duration[i]:.4f}", f"{y_va[j]:.6f}",
                         f"{ridge_va[j]:.6f}", f"{natural_va[j]:.6f}", f"{hybrid_va[j]:.6f}"])

    test_songs = {str(song_ids[s]) for s in range(song_ids.size) if D["song_split_index"][s] == split_codes["test"]}
    hashes_after = NAT._hash_tree(protected)
    results = {
        "experiment": "axis2_v2_ridge_natural_fixed_blend",
        "test_evaluated": False,
        "components": {"ridge": {k: v for k, v in ridge_cfg.items() if not k.startswith("stored_")},
                       "natural_offset": {k: v for k, v in nat_cfg.items() if not k.startswith("stored_") and k != "columns"}},
        "blend": {"formula": "w * ridge + (1-w) * natural", "selected_weight_ridge": w,
                  "selected_weight_natural": round(1.0 - w, 2), "selection_metric": "train_oof_mae",
                  "grid": "w = 0.00, 0.01, ..., 1.00 (101 values)", "global_scalar": True,
                  "depends_on": "nothing (no per-note, per-variant, per-song, or magnitude-dependent weighting)"},
        "oof_weight_selection": {
            "method": "GroupKFold(5) by song over all train examples; one song partition shared by both components; "
                      "Ridge refit per fold on all variants of the fit songs, natural model refit per fold on the fit songs' clean notes",
            "folds": [{"fold": k, "heldout_songs": int(np.unique(groups_tr[fold_of == k]).size),
                       "heldout_examples": int(np.sum(fold_of == k)), "fit_examples": int(np.sum(fold_of != k)),
                       "natural_fit_clean_notes": int(np.sum((fold_of != k) & clean_tr))} for k in range(CV_FOLDS)],
            "natural_native_folds_same_song_partition": True,
            "oof_metrics": oof,
            "selected": chosen,
            "tied_within_tolerance": [r["weight_ridge"] for r in tied], "tie_tolerance_cents": TIE_TOLERANCE_CENTS,
            "tie_rule": "nearest to 0.5",
            "slope_definition": "OLS of OOF prediction on injected shift, corrupted train examples only",
        },
        "validation": {"overall": overall, "by_variant": by_variant, "by_sign": by_sign,
                       "by_true_magnitude_band": by_magnitude, "by_true_magnitude_5c_corrupted_only": by_fine,
                       "by_frame_count_clean": by_frames, "by_duration_clean": by_duration,
                       "prediction_slope": {"definition": "OLS predicted_shift ~ injected_shift on corrupted validation examples",
                                            **slopes},
                       "notes": {"magnitude_bands": "the bands coincide with the variant definitions, so they equal by_variant; "
                                                    "the 5-cent bins are the finer post-hoc view",
                                 "post_hoc_only": "magnitude and length breakdowns were computed after w was fixed and did not inform it"}},
        "comparison": comparison,
        "primary_question": primary,
        "note_group_comparison": note_groups,
        "integrity": {
            "component_cv_reproduced": {"ridge_max_fold_mae_diff": ridge_fold_diff, "natural_max_fold_mae_diff": nat_fold_diff},
            "component_validation_reproduced": {"ridge_max_mae_rmse_diff": ridge_val_diff, "natural_max_mae_rmse_diff": nat_val_diff,
                                                "natural_max_per_example_diff_vs_stored_csv": nat_csv_diff},
            "oof_alignment": oof_alignment, "validation_alignment": val_alignment,
            "recomputed_vs_stored_core_median_max_abs_diff": core_diff,
            "validation_used_for_weight_selection": False, "validation_prediction_passes": 1,
            "test_examples_featurized": 0,
            "test_song_note_files_opened": len(songs_opened & test_songs),
            "sealed_test_metadata_only": {"examples": splits["test_example_count"], "clean_notes": splits["test_clean_note_count"]},
            "protected_paths": [str(p.relative_to(REPO)) for p in protected],
            "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
            "protected_unchanged": hashes_before == hashes_after,
            "runtime_s": time.time() - t0,
        },
    }
    if results["integrity"]["test_song_note_files_opened"]:
        raise RuntimeError("a test song's notes were opened")
    (OUT / "hybrid_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    (OUT / "hybrid_summary.md").write_text(summary_md(results))
    print(json.dumps({"w_ridge": w, "primary": primary, "protected_unchanged": hashes_before == hashes_after}, indent=1))
    return 0


# ---------------------------------------------------------------- report


def summary_md(r: Mapping[str, Any]) -> str:
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"
    v, c, b = r["validation"], r["comparison"], r["blend"]
    o = r["oof_weight_selection"]
    pq, ng = r["primary_question"], r["note_group_comparison"]
    L = ["# Axis 2 V2: fixed Ridge + natural-offset blend", "",
         "Generated by `hybrid_ridge_natural.py`. `hybrid = w * ridge + (1 - w) * natural`, one global scalar chosen by "
         "train out-of-fold MAE (GroupKFold(5) by song). Components frozen; validation evaluated once; **test not evaluated**.", "",
         f"**Selected:** w_ridge = **{b['selected_weight_ridge']:.2f}**, w_natural = **{b['selected_weight_natural']:.2f}**.", "",
         "| train OOF | MAE | RMSE | slope |", "|---|---|---|---|"]
    for k in ESTIMATORS:
        L.append(f"| {k} | {f(o['oof_metrics'][k]['mae'], 4)} | {f(o['oof_metrics'][k]['rmse'], 4)} | {f(o['oof_metrics'][k]['slope_corrupted'])} |")
    L += ["", "## Validation", "", "| | Ridge | Natural | Hybrid |", "|---|---|---|---|"]
    for lab, key in (("MAE", "mae"), ("RMSE", "rmse"), ("median abs error", "median_absolute_error"), ("P90", "p90"),
                     ("P95", "p95"), ("P99", "p99"), ("max error", "max"), ("prediction slope", "prediction_slope")):
        L.append(f"| {lab} | " + " | ".join(f(c[k][key], 4 if key in ("mae", "rmse") else 3) for k in ESTIMATORS) + " |")
    L += ["", f"Hybrid vs Ridge: MAE {f(pq['ridge_mae_benchmark'], 4)} → {f(pq['hybrid_mae'], 4)} "
              f"({pq['mae_improvement']:+.4f} c, {pq['mae_improvement_percent']:+.2f}%); RMSE {f(pq['ridge_rmse'], 4)} → "
              f"{f(pq['hybrid_rmse'], 4)} ({pq['rmse_improvement']:+.4f} c, {pq['rmse_improvement_percent']:+.2f}%). "
              "Positive = hybrid better.", "",
          "**MAE by variant**", "", "| variant | Ridge MAE | Natural MAE | Hybrid MAE | Hybrid − Ridge |", "|---|---|---|---|---|"]
    for name in EXPECTED_VARIANT_ORDER:
        x = v["by_variant"][name]
        L.append(f"| {name} | {f(x['ridge']['mae'])} | {f(x['natural']['mae'])} | {f(x['hybrid']['mae'])} | "
                 f"{x['hybrid']['mae'] - x['ridge']['mae']:+.3f} |")
    L += ["", "**Hybrid by variant**", "", "| variant | n | MAE | RMSE | median abs | mean signed |", "|---|---|---|---|---|---|"]
    for name in EXPECTED_VARIANT_ORDER:
        x = v["by_variant"][name]["hybrid"]
        L.append(f"| {name} | {x['count']} | {f(x['mae'])} | {f(x['rmse'])} | {f(x['median_absolute_error'])} | {x['mean_signed_error']:+.3f} |")
    L += ["", "**Corrupted examples by sign**", "", "| sign | n | Ridge MAE / signed | Natural MAE / signed | Hybrid MAE / RMSE / signed |",
          "|---|---|---|---|---|"]
    for s in ("positive", "negative"):
        x = v["by_sign"][s]
        L.append(f"| {s} | {x['hybrid']['count']} | {f(x['ridge']['mae'])} / {x['ridge']['mean_signed_error']:+.3f} | "
                 f"{f(x['natural']['mae'])} / {x['natural']['mean_signed_error']:+.3f} | "
                 f"{f(x['hybrid']['mae'])} / {f(x['hybrid']['rmse'])} / {x['hybrid']['mean_signed_error']:+.3f} |")
    L += ["", "**Prediction slope** (predicted ~ injected, corrupted validation)", "", "| | slope | intercept | R² |", "|---|---|---|---|"]
    for k in ESTIMATORS:
        x = v["prediction_slope"][k]
        L.append(f"| {k} | {f(x['slope'])} | {x['intercept']:+.3f} | {f(x['r2'])} |")
    ht = v["overall"]["hybrid"]["tail"]
    L += ["", "**Hybrid absolute-error tail**", "", "| " + " | ".join(ht["quantiles"]) + " | max |",
          "|" + "---|" * (len(ht["quantiles"]) + 1),
          "| " + " | ".join(f(x, 2) for x in ht["quantiles"].values()) + f" | {f(ht['max'], 2)} |", "",
          " · ".join(f"> {t.split('_')[1]} ¢: {x['count']} ({x['percent']:.2f}%)" for t, x in ht["exceeding"].items()), "",
          "## Note-group comparison (Hybrid vs Ridge)", "",
          f"{ng['note_groups']} validation notes; per note, mean |error| over its five variants; tie = within {ng['tie_tolerance_cents']} c.", "",
          f"- Hybrid better: **{ng['hybrid_wins_percent']:.1f}%** (median gain {f(ng['median_improvement_when_hybrid_wins'])} c)",
          f"- Tied: {ng['tied_percent']:.1f}%",
          f"- Ridge better: **{ng['ridge_wins_percent']:.1f}%** (median loss {f(ng['median_degradation_when_hybrid_loses'])} c)", "",
          "## Post-hoc breakdowns (did not inform w)", "", "**By true injected magnitude, 5-cent bins (corrupted only)**", "",
          "| |shift| | n | Ridge | Natural | Hybrid |", "|---|---|---|---|---|"]
    for lab, x in v["by_true_magnitude_5c_corrupted_only"].items():
        L.append(f"| {lab} | {x['hybrid']['count']} | {f(x['ridge']['mae'])} | {f(x['natural']['mae'])} | {f(x['hybrid']['mae'])} |")
    for title, key in (("By frame count (clean validation notes, MAE)", "by_frame_count_clean"),
                       ("By duration (clean validation notes, MAE)", "by_duration_clean")):
        L += ["", f"**{title}**", "", "| bucket | n | Ridge | Natural | Hybrid |", "|---|---|---|---|---|"]
        for lab, x in v[key].items():
            L.append(f"| {lab} | {x['hybrid']['count']} | {f(x['ridge']['mae'])} | {f(x['natural']['mae'])} | {f(x['hybrid']['mae'])} |")
    ig = r["integrity"]
    L += ["", "## Integrity", "",
          f"- Component CV reproduced: Ridge fold-MAE max diff {ig['component_cv_reproduced']['ridge_max_fold_mae_diff']:.1e}, "
          f"natural {ig['component_cv_reproduced']['natural_max_fold_mae_diff']:.1e}.",
          f"- Component validation reproduced: Ridge {ig['component_validation_reproduced']['ridge_max_mae_rmse_diff']:.1e}, "
          f"natural {ig['component_validation_reproduced']['natural_max_mae_rmse_diff']:.1e} "
          f"(per-example vs stored CSV {ig['component_validation_reproduced']['natural_max_per_example_diff_vs_stored_csv']:.1e}).",
          f"- Alignment by (song, note, variant): {ig['oof_alignment']['examples']} train OOF, "
          f"{ig['validation_alignment']['examples']} validation; 0 missing, 0 duplicates, identical targets.",
          f"- Test examples featurized: {ig['test_examples_featurized']}; test notes files opened: {ig['test_song_note_files_opened']}.",
          f"- Protected files unchanged: {ig['protected_unchanged']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
