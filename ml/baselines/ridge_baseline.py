#!/usr/bin/env python3
"""
Ridge baseline (Baseline 4) for Axis 2 V2: the first learned statistical baseline.

    y = injected_shift_cents   (signed regression)
    X = hand-engineered note-level features of each example's OWN contour
    model = StandardScaler -> Ridge(alpha), alpha chosen by 5-fold GroupKFold
            inside the training set, grouped by song

Anti-leakage is structural. `extract_features` receives one example at a time:
its cents contour, its confidence contour, its duration and its target MIDI. It
is never given labels, variant codes, sibling indices, or song identity. Song
identity is used only as the CV grouping variable, never as a feature. The
validation split is touched exactly once, after alpha is fixed and the final
model is fit on all training examples.

The test split is sealed: test examples are excluded before feature
extraction, so no test contour is read and no test prediction exists.

  python ridge_baseline.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from median_cents_baseline import _evaluate
from zero_baseline import (
    EXPECTED_SPLIT_ORDER,
    EXPECTED_VARIANT_ORDER,
    REPO_ROOT,
    DatasetContractError,
    _metrics,
    _resolve_lookup,
    _tail,
    _verify_counts,
    dataset_paths,
)

sys.path.insert(0, str(REPO_ROOT / "backend"))
from note_segmentation import _core_slice  # noqa: E402


DATASET_VERSION = "axis2_v2"
EVALUATED_SPLITS = ("train", "validation")
SEALED_SPLIT = "test"
CANDIDATE_ALPHAS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
CV_FOLDS = 5
REGION_FRACTION = 0.20
FRAME_BUCKETS = ((3, 4), (5, 6), (7, 9), (10, 20), (21, 50), (51, None))
DURATION_BUCKETS_S = ((None, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.80), (0.80, None))
SHIFT_CHECK_NOTES = 200
SHIFT_CHECK_SYNTHETIC_CENTS = 37.5

CENTER_FEATURES = ("full_median_cents", "core_median_cents", "mean_cents", "confidence_weighted_mean_cents")
FEATURES = CENTER_FEATURES + (
    "std_cents", "mad_cents", "iqr_cents",
    "p10_cents", "p25_cents", "p75_cents", "p90_cents", "p90_minus_p10_cents",
    "onset_minus_middle_cents", "release_minus_middle_cents", "release_minus_onset_cents",
    "linear_slope_cents_per_note",
    "mean_confidence", "median_confidence", "p25_confidence", "min_confidence",
    "duration_s", "frame_count", "log1p_frame_count",
    "target_midi",
)
# Expected behaviour when every frame of the cents contour moves by +S.
SHIFT_EQUIVARIANT = CENTER_FEATURES + ("p10_cents", "p25_cents", "p75_cents", "p90_cents")
SHIFT_INVARIANT = tuple(f for f in FEATURES if f not in SHIFT_EQUIVARIANT)

FEATURE_DEFINITIONS = {
    "full_median_cents": "median of the contour",
    "core_median_cents": "median of backend/note_segmentation.py:_core_slice(contour) (production core)",
    "mean_cents": "mean of the contour",
    "confidence_weighted_mean_cents": "sum(cents*conf)/sum(conf); falls back to mean_cents if sum(conf) <= 0",
    "std_cents": "population standard deviation (ddof=0)",
    "mad_cents": "median(|cents - median(cents)|), unscaled",
    "iqr_cents": "P75 - P25",
    "p10_cents/p25_cents/p75_cents/p90_cents": "np.percentile, linear interpolation",
    "p90_minus_p10_cents": "P90 - P10",
    "onset/middle/release regions": ("k = max(1, floor(0.2*n)) frames by index; onset = first k, release = last k, "
                                     "middle = frames k..n-k (the whole contour if that is empty); each summarized by "
                                     "its median. Index-based, like production's core slice. Not the production core."),
    "onset_minus_middle_cents": "median(onset) - median(middle)",
    "release_minus_middle_cents": "median(release) - median(middle)",
    "release_minus_onset_cents": "median(release) - median(onset)",
    "linear_slope_cents_per_note": "OLS slope of cents on position linspace(0, 1, n); 0.0 if n < 2",
    "mean/median/p25/min_confidence": "CREPE confidence contour statistics; no filtering",
    "duration_s": "stored duration (last minus first frame time)",
    "frame_count": "contour length", "log1p_frame_count": "log(1 + contour length)",
    "target_midi": "stored target MIDI, numeric",
}


def extract_features(cents: np.ndarray, conf: np.ndarray, duration_s: float, target_midi: int) -> np.ndarray:
    """Features of ONE example from its own data only. No labels, siblings, or identity."""
    n = cents.size
    med = float(np.median(cents))
    p10, p25, p75, p90 = (float(v) for v in np.percentile(cents, (10, 25, 75, 90)))
    wsum = float(np.sum(conf))
    mean = float(np.mean(cents))
    wmean = float(np.sum(cents * conf) / wsum) if wsum > 0 else mean
    k = max(1, int(np.floor(REGION_FRACTION * n)))
    middle = cents[k:n - k] if n - 2 * k >= 1 else cents
    onset_m, mid_m, rel_m = float(np.median(cents[:k])), float(np.median(middle)), float(np.median(cents[n - k:]))
    if n >= 2:
        x = np.linspace(0.0, 1.0, n)
        xc = x - x.mean()
        slope = float(np.sum(xc * (cents - mean)) / np.sum(xc * xc))
    else:
        slope = 0.0
    return np.array([
        med, float(np.median(_core_slice(cents))), mean, wmean,
        float(np.std(cents)), float(np.median(np.abs(cents - med))), p75 - p25,
        p10, p25, p75, p90, p90 - p10,
        onset_m - mid_m, rel_m - mid_m, rel_m - onset_m,
        slope,
        float(np.mean(conf)), float(np.median(conf)), float(np.percentile(conf, 25)), float(np.min(conf)),
        float(duration_s), float(n), float(np.log1p(n)),
        float(target_midi),
    ], dtype=np.float64)


def build_matrix(cents_flat, conf_flat, offsets, duration, target_midi, indices) -> np.ndarray:
    X = np.empty((indices.size, len(FEATURES)), dtype=np.float64)
    for row, i in enumerate(indices):
        a, b = int(offsets[i]), int(offsets[i + 1])
        X[row] = extract_features(cents_flat[a:b], conf_flat[a:b], duration[i], int(target_midi[i]))
    return X


def _model(alpha: float) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


# NumPy 2.x on Apple Accelerate BLAS raises "divide by zero / overflow / invalid
# value encountered in matmul" for ordinary well-conditioned products (verified
# here: a random 1000x24 product raises all three and matches einsum to 4e-15).
# Those are counted, not trusted or ignored: every prediction is re-derived
# without BLAS and compared. Any other warning disqualifies a candidate.
_SPURIOUS = "encountered in matmul"
PREDICTION_CHECK_TOLERANCE = 1e-6


def _split_warnings(caught) -> Tuple[int, List[str]]:
    spurious = sum(1 for w in caught if issubclass(w.category, RuntimeWarning) and _SPURIOUS in str(w.message))
    other = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught
                    if not (issubclass(w.category, RuntimeWarning) and _SPURIOUS in str(w.message))})
    return spurious, other


def _fit(alpha: float, X: np.ndarray, y: np.ndarray) -> Tuple[Pipeline, List[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _model(alpha).fit(X, y)
    return model, _split_warnings(caught)[1]


def _predict(model: Pipeline, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pipeline prediction, cross-checked against a BLAS-free recomputation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = model.predict(X)
    spurious, other = _split_warnings(caught)
    sc, rg = model.named_steps["scaler"], model.named_steps["ridge"]
    manual = np.einsum("ij,j->i", (X - sc.mean_) / sc.scale_, rg.coef_) + rg.intercept_
    diff = float(np.max(np.abs(p - manual))) if p.size else 0.0
    ok = bool(np.all(np.isfinite(p)) and diff <= PREDICTION_CHECK_TOLERANCE and not other)
    return p, {"finite": bool(np.all(np.isfinite(p))), "max_abs_diff_vs_blas_free": diff,
               "spurious_matmul_warnings": spurious, "other_warnings": other, "ok": ok}


def grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, columns: Sequence[int]) -> Dict[str, Any]:
    """Alpha selection by GroupKFold over training songs. Validation is never seen here."""
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(X, y, groups))
    for tr, te in folds:
        if set(groups[tr]) & set(groups[te]):
            raise DatasetContractError("a song appears on both sides of a CV fold")
    Xs = StandardScaler().fit_transform(X[:, columns])
    rank = int(np.linalg.matrix_rank(Xs))
    rows = []
    for alpha in CANDIDATE_ALPHAS:
        maes, rmses, warns, checks_ok, maxdiff, spurious = [], [], set(), True, 0.0, 0
        for tr, te in folds:
            model, w = _fit(alpha, X[tr][:, columns], y[tr])
            warns.update(w)
            p, chk = _predict(model, X[te][:, columns])
            checks_ok &= chk["ok"]
            maxdiff = max(maxdiff, chk["max_abs_diff_vs_blas_free"]) if np.isfinite(chk["max_abs_diff_vs_blas_free"]) else float("inf")
            spurious += chk["spurious_matmul_warnings"]
            warns.update(chk["other_warnings"])
            e = p - y[te]
            maes.append(float(np.mean(np.abs(e))))
            rmses.append(float(np.sqrt(np.mean(e ** 2))))
        reasons = []
        if alpha == 0.0 and rank < len(columns):
            reasons.append(f"alpha=0 is ordinary least squares and the standardized training matrix has rank {rank} < {len(columns)} "
                           "(exactly collinear features), so the solution is not unique")
        if warns:
            reasons.append("numerical warnings: " + "; ".join(sorted(warns)))
        if not checks_ok:
            reasons.append("prediction cross-check failed (non-finite or BLAS-free recomputation disagrees)")
        rows.append({"alpha": alpha, "fold_mae": maes, "mean_cv_mae": float(np.mean(maes)),
                     "std_cv_mae": float(np.std(maes)), "fold_rmse": rmses, "mean_cv_rmse": float(np.mean(rmses)),
                     "max_abs_diff_vs_blas_free": maxdiff, "spurious_matmul_warnings": spurious,
                     "ineligible_reasons": reasons, "eligible": not reasons})
    eligible = [r for r in rows if r["eligible"]]
    best = min(eligible, key=lambda r: (r["mean_cv_mae"], -r["alpha"]))  # ties -> stronger regularization
    return {
        "method": f"{CV_FOLDS}_fold_grouped_cv_by_song (sklearn GroupKFold, train split only)",
        "selection_metric": "mean CV MAE; ties broken toward larger alpha; candidates with numerical warnings ineligible",
        "candidate_alphas": list(CANDIDATE_ALPHAS),
        "folds": [{"fold": i, "train_examples": int(tr.size), "heldout_examples": int(te.size),
                   "heldout_songs": int(np.unique(groups[te]).size)} for i, (tr, te) in enumerate(folds)],
        "standardized_train_matrix_rank": rank, "feature_count": len(columns),
        "cv_results": rows,
        "selected_alpha": best["alpha"],
        "selected_mean_cv_mae": best["mean_cv_mae"],
        "selected_mean_cv_rmse": best["mean_cv_rmse"],
    }


def _bucket(values: np.ndarray, lo, hi, upper_inclusive: bool) -> np.ndarray:
    m = np.ones(values.shape, bool)
    if lo is not None:
        m &= values >= lo
    if hi is not None:
        m &= (values <= hi) if upper_inclusive else (values < hi)
    return m


def _mae(e: np.ndarray) -> float | None:
    return float(np.mean(np.abs(e))) if e.size else None


def _rmse(e: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(e ** 2))) if e.size else None


def shift_sanity(X_rows: np.ndarray, cents_flat, conf_flat, offsets, duration, target_midi, idx) -> Dict[str, Any]:
    """Feature-implementation check: features of contour+S vs contour, on the first train examples."""
    eq_err, inv_err = 0.0, 0.0
    eq_i = [FEATURES.index(f) for f in SHIFT_EQUIVARIANT]
    inv_i = [FEATURES.index(f) for f in SHIFT_INVARIANT]
    for row, i in enumerate(idx):
        a, b = int(offsets[i]), int(offsets[i + 1])
        shifted = extract_features(cents_flat[a:b] + SHIFT_CHECK_SYNTHETIC_CENTS, conf_flat[a:b], duration[i], int(target_midi[i]))
        d = shifted - X_rows[row]
        eq_err = max(eq_err, float(np.max(np.abs(d[eq_i] - SHIFT_CHECK_SYNTHETIC_CENTS))))
        inv_err = max(inv_err, float(np.max(np.abs(d[inv_i]))))
    return {"examples_checked": int(len(idx)), "synthetic_shift_cents": SHIFT_CHECK_SYNTHETIC_CENTS,
            "shift_equivariant_features": list(SHIFT_EQUIVARIANT), "max_equivariance_error": eq_err,
            "shift_invariant_features": list(SHIFT_INVARIANT), "max_invariance_error": inv_err,
            "passed": eq_err <= 1e-9 and inv_err <= 1e-9, "sample": "first train examples in file order (train only)"}


def main() -> int:
    paths = dataset_paths(DATASET_VERSION)
    rdir = paths["results_dir"]
    out_json, out_md = rdir / "ridge_baseline.json", rdir / "ridge_baseline_comparison.md"
    for p in (out_json, out_md):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    prior = {name: json.loads((rdir / f"{name}.json").read_text(encoding="utf-8"))
             for name in ("zero_baseline", "median_cents_baseline", "core_median_baseline")}
    for name, r in prior.items():
        if r["dataset"]["dataset_name"] != DATASET_VERSION or r["test_evaluated"] is not False:
            raise DatasetContractError(f"{name} is not a sealed-test {DATASET_VERSION} result")

    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    with np.load(paths["examples"], allow_pickle=False) as d:
        split_codes = _resolve_lookup(d["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
        variant_codes = _resolve_lookup(d["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")
        split_index, variant_index = np.asarray(d["split_index"]), np.asarray(d["variant_index"])
        song_index, note_index = np.asarray(d["song_index"]), np.asarray(d["source_note_index"])
        song_ids = np.asarray(d["song_ids"])
        y_all = np.asarray(d["injected_shift_cents"], np.float64)
        duration, target_midi = np.asarray(d["duration_s"], np.float64), np.asarray(d["target_midi"])
        num_frames = np.asarray(d["num_frames"], np.int64)
        stored_median, stored_core = np.asarray(d["median_cents"], np.float64), np.asarray(d["core_median_cents"], np.float64)
        cents_flat, conf_flat = np.asarray(d["cents_flat"], np.float64), np.asarray(d["confidence_flat"], np.float64)
        offsets = np.asarray(d["contour_offsets"], np.int64)
    _verify_counts(split_index, split_codes, splits)
    if summary.get("dataset_name") != DATASET_VERSION:
        raise DatasetContractError("dataset name mismatch")

    # Test indices never enter either set: no test contour is read below.
    tr_idx = np.flatnonzero(split_index == split_codes["train"])
    va_idx = np.flatnonzero(split_index == split_codes["validation"])
    X_tr = build_matrix(cents_flat, conf_flat, offsets, duration, target_midi, tr_idx)
    X_va = build_matrix(cents_flat, conf_flat, offsets, duration, target_midi, va_idx)
    y_tr, y_va = y_all[tr_idx], y_all[va_idx]
    for X in (X_tr, X_va):
        if not np.all(np.isfinite(X)):
            raise DatasetContractError("non-finite feature values")
    fi = FEATURES.index
    feature_checks = {
        "full_median_matches_stored_max_abs_diff": float(np.max(np.abs(np.r_[X_tr[:, fi("full_median_cents")], X_va[:, fi("full_median_cents")]]
                                                                        - stored_median[np.r_[tr_idx, va_idx]]))),
        "core_median_matches_stored_max_abs_diff": float(np.max(np.abs(np.r_[X_tr[:, fi("core_median_cents")], X_va[:, fi("core_median_cents")]]
                                                                        - stored_core[np.r_[tr_idx, va_idx]]))),
        "frame_count_matches_num_frames": bool(np.array_equal(np.r_[X_tr[:, fi("frame_count")], X_va[:, fi("frame_count")]],
                                                              num_frames[np.r_[tr_idx, va_idx]])),
        "all_features_finite": True,
    }
    if max(feature_checks["full_median_matches_stored_max_abs_diff"], feature_checks["core_median_matches_stored_max_abs_diff"]) > 1e-9:
        raise DatasetContractError(f"feature medians disagree with stored medians: {feature_checks}")
    sanity = shift_sanity(X_tr[:SHIFT_CHECK_NOTES], cents_flat, conf_flat, offsets, duration, target_midi, tr_idx[:SHIFT_CHECK_NOTES])
    if not sanity["passed"]:
        raise DatasetContractError(f"feature shift behaviour wrong: {sanity}")

    groups = song_index[tr_idx]
    all_cols = list(range(len(FEATURES)))
    center_cols = [fi(f) for f in CENTER_FEATURES]

    def run(columns):
        hp = grouped_cv(X_tr, y_tr, groups, columns)
        model, w = _fit(hp["selected_alpha"], X_tr[:, columns], y_tr)
        return hp, model, w

    hp, model, fit_warn = run(all_cols)
    hp_c, model_c, fit_warn_c = run(center_cols)

    # ---- the single validation evaluation of each fixed model
    (p_tr, k1), (p_va, k2) = _predict(model, X_tr), _predict(model, X_va)
    (pc_tr, k3), (pc_va, k4) = _predict(model_c, X_tr[:, center_cols]), _predict(model_c, X_va[:, center_cols])
    prediction_checks = {"full_train": k1, "full_validation": k2, "center_train": k3, "center_validation": k4}
    if not all(k["ok"] for k in prediction_checks.values()):
        raise DatasetContractError(f"prediction checks failed: {prediction_checks}")
    e_va = p_va - y_va
    var_va, var_tr = variant_index[va_idx], variant_index[tr_idx]

    def section(y, p, var):
        s = _evaluate(y, p, var, variant_codes)
        return s

    results: Dict[str, Any] = {
        "experiment": f"{DATASET_VERSION}_ridge_baseline",
        "model": "ridge_regression (sklearn Pipeline: StandardScaler -> Ridge)",
        "target": "injected_shift_cents (signed)",
        "test_evaluated": False,
        "dataset": {"path": str(paths["examples"].relative_to(REPO_ROOT)), "dataset_name": DATASET_VERSION,
                    "split_encoding": split_codes, "variant_encoding": variant_codes, "sealed_split": SEALED_SPLIT,
                    "train_examples": int(tr_idx.size), "validation_examples": int(va_idx.size),
                    "sealed_split_metadata_only": {"example_count": int(splits["test_example_count"]),
                                                   "clean_note_count": int(splits["test_clean_note_count"])}},
        "features": list(FEATURES),
        "feature_definitions": FEATURE_DEFINITIONS,
        "excluded_from_features": ["song_id", "artist", "filename", "clean_note_id", "variant", "severity", "sign",
                                   "sibling information", "chroma / harmonic fit"],
        "hyperparameter_selection": hp,
        "final_fit_numerical_warnings": fit_warn,
        "train": section(y_tr, p_tr, var_tr),
        "validation": section(y_va, p_va, var_va),
    }
    results["train"]["tail"] = _tail(np.abs(p_tr - y_tr))

    # ---- descriptive breakdowns (validation)
    clean = var_va == variant_codes["clean"]
    nf, du = num_frames[va_idx], duration[va_idx]
    full_c, core_c = X_va[:, fi("full_median_cents")], X_va[:, fi("core_median_cents")]   # errors of Baselines 2/3 on clean (target 0)
    def bucket_rows(values, buckets, inclusive, label):
        out = {}
        for lo, hi in buckets:
            name = (f"{lo}+" if hi is None else f"{lo}-{hi}") if label == "frames" else \
                   (f"<{hi:.2f}s" if lo is None else f">={lo:.2f}s" if hi is None else f"{lo:.2f}-{hi:.2f}s")
            m_all = _bucket(values, lo, hi, inclusive)
            m = m_all & clean
            out[name] = {
                "clean_note_count": int(m.sum()),
                "ridge_clean_mae": _mae(e_va[m]), "ridge_clean_rmse": _rmse(e_va[m]),
                "core_median_clean_mae": _mae(core_c[m]), "full_median_clean_mae": _mae(full_c[m]),
                "ridge_minus_core_mae": (_mae(e_va[m]) - _mae(core_c[m])) if m.any() else None,
                "all_variants_example_count": int(m_all.sum()),
                "ridge_all_variants_mae": _mae(e_va[m_all]), "ridge_all_variants_rmse": _rmse(e_va[m_all]),
            }
        return out
    results["validation"]["by_frame_count"] = bucket_rows(nf, FRAME_BUCKETS, True, "frames")
    results["validation"]["by_duration"] = bucket_rows(du, DURATION_BUCKETS_S, False, "duration")

    by_song = {}
    for code in np.unique(song_index[va_idx]):
        m = song_index[va_idx] == code
        r = _metrics(y_va[m], p_va[m])
        by_song[str(song_ids[code])] = {"example_count": r["example_count"], "mae": r["mae"], "rmse": r["rmse"],
                                        "mean_signed_error": r["mean_signed_error"]}
    order = sorted(by_song, key=lambda s: by_song[s]["mae"])
    results["validation"]["by_song"] = by_song
    results["validation"]["song_extremes"] = {"best_by_mae": {"song_id": order[0], **by_song[order[0]]},
                                              "worst_by_mae": {"song_id": order[-1], **by_song[order[-1]]}}

    # ---- coefficients (standardized space)
    ridge = model.named_steps["ridge"]
    coefs = sorted(zip(FEATURES, ridge.coef_.tolist()), key=lambda kv: -abs(kv[1]))
    results["coefficients"] = {
        "space": "standardized features (StandardScaler fit on all training examples)",
        "intercept": float(ridge.intercept_), "selected_alpha": hp["selected_alpha"],
        "sorted_by_abs": [{"feature": f, "coefficient": c} for f, c in coefs],
        "caveat": "features are strongly correlated (several are exact linear combinations); coefficients show how "
                  "this fit distributes weight, not causal importance",
    }

    # ---- shrinkage diagnostic (fixed models, no re-fitting or tuning): how strongly
    # predictions follow the injected shift, and the raw-unit weight on pitch center
    def slope(x, yv):
        xc = x - x.mean()
        return float(np.sum(xc * (yv - yv.mean())) / np.sum(xc * xc))
    sc = model.named_steps["scaler"]
    raw = ridge.coef_ / sc.scale_
    corrupted = var_va != variant_codes["clean"]
    results["shrinkage_diagnostic"] = {
        "raw_unit_coefficients_center_features": {f: float(raw[fi(f)]) for f in CENTER_FEATURES},
        "sum_raw_center_coefficients": float(sum(raw[fi(f)] for f in CENTER_FEATURES)),
        "validation_slope_prediction_on_injected_shift_corrupted": slope(y_va[corrupted], p_va[corrupted]),
        "validation_slope_prediction_on_injected_shift_center_only_corrupted": slope(y_va[corrupted], pc_va[corrupted]),
        "validation_mean_prediction_by_variant": {n: float(np.mean(p_va[var_va == c])) for n, c in variant_codes.items()},
        "validation_mean_abs_prediction_by_variant": {n: float(np.mean(np.abs(p_va[var_va == c]))) for n, c in variant_codes.items()},
        "validation_mean_abs_target_by_variant": {n: float(np.mean(np.abs(y_va[var_va == c]))) for n, c in variant_codes.items()},
        "interpretation": ("a sum of raw center coefficients below 1 means predictions are a shrunk pitch-center estimate: "
                           "the model trades bias on large shifts for lower error on clean and small shifts, exploiting "
                           "the label distribution (20% zeros, |shift| <= 50 c) rather than new contour information"),
    }

    # ---- center-only ablation
    results["center_only_ablation"] = {
        "features": list(CENTER_FEATURES),
        "hyperparameter_selection": hp_c,
        "final_fit_numerical_warnings": fit_warn_c,
        "train": section(y_tr, pc_tr, var_tr)["overall"],
        "validation": section(y_va, pc_va, var_va),
        "coefficients": {"intercept": float(model_c.named_steps["ridge"].intercept_),
                         "by_feature": dict(zip(CENTER_FEATURES, model_c.named_steps["ridge"].coef_.tolist()))},
    }
    results["center_only_ablation"]["validation"].pop("tail", None)
    results["center_only_ablation"]["validation"]["tail"] = _tail(np.abs(pc_va - y_va))

    # ---- comparison to deterministic baselines (validation)
    pv = {"zero": prior["zero_baseline"]["validation"], "full_median": prior["median_cents_baseline"]["validation"],
          "core_median": prior["core_median_baseline"]["validation"], "ridge_center_only": results["center_only_ablation"]["validation"],
          "ridge": results["validation"]}
    def row(v):
        q = v["tail"]["absolute_error_quantiles"]
        return {"mae": v["overall"]["mae"], "rmse": v["overall"]["rmse"], "median_absolute_error": v["overall"]["median_absolute_error"],
                "p90": q["p90"], "p95": q["p95"], "p99": q["p99"]}
    core_o, ridge_o = pv["core_median"]["overall"], results["validation"]["overall"]
    band = clean & (np.abs(full_c) <= 50.0)
    results["comparison_to_baselines"] = {
        "split": "validation",
        "ladder": {k: row(v) for k, v in pv.items()},
        "by_variant": {n: {k: {"mae": v["by_variant"][n]["mae"], "rmse": v["by_variant"][n]["rmse"]} for k, v in pv.items()}
                       for n in EXPECTED_VARIANT_ORDER},
        "ridge_vs_core_median": {
            "mae": {"core_median": core_o["mae"], "ridge": ridge_o["mae"], "absolute_improvement": core_o["mae"] - ridge_o["mae"],
                    "percent_improvement": 100.0 * (core_o["mae"] - ridge_o["mae"]) / core_o["mae"]},
            "rmse": {"core_median": core_o["rmse"], "ridge": ridge_o["rmse"], "absolute_improvement": core_o["rmse"] - ridge_o["rmse"],
                     "percent_improvement": 100.0 * (core_o["rmse"] - ridge_o["rmse"]) / core_o["rmse"]},
        },
        "full_feature_vs_center_only": {
            "mae": {"center_only": pv["ridge_center_only"]["overall"]["mae"], "full": ridge_o["mae"],
                    "absolute_improvement": pv["ridge_center_only"]["overall"]["mae"] - ridge_o["mae"]},
            "rmse": {"center_only": pv["ridge_center_only"]["overall"]["rmse"], "full": ridge_o["rmse"],
                     "absolute_improvement": pv["ridge_center_only"]["overall"]["rmse"] - ridge_o["rmse"]},
        },
        "clean_examples": {k: {"mae": v["by_variant"]["clean"]["mae"], "rmse": v["by_variant"]["clean"]["rmse"]} for k, v in pv.items()},
        "clean_note": "zero is perfect on clean examples by construction (it always predicts 0); read clean together with corrupted",
        "secondary_diagnostic_full_median_within_50c": {
            "definition": "validation clean notes with |full median| <= 50 c; descriptive only, training unchanged",
            "note_count": int(band.sum()),
            "full_median": {"mae": _mae(full_c[band]), "rmse": _rmse(full_c[band])},
            "core_median": {"mae": _mae(core_c[band]), "rmse": _rmse(core_c[band])},
            "ridge": {"mae": _mae(e_va[band]), "rmse": _rmse(e_va[band])},
            "ridge_center_only": {"mae": _mae(pc_va[band] - y_va[band]), "rmse": _rmse(pc_va[band] - y_va[band])},
        },
        "core_median_caveat": ("target MIDI is chosen from the production core median, so core-median errors are bounded "
                               "to +-50 c by construction; its tail advantage is partly structural"),
    }

    tr_o = results["train"]["overall"]
    results["train_vs_validation"] = {
        "train_mae": tr_o["mae"], "validation_mae": ridge_o["mae"], "train_rmse": tr_o["rmse"], "validation_rmse": ridge_o["rmse"],
        "cv_mae": hp["selected_mean_cv_mae"],
        "interpretation": "learned model: train is optimistic; grouped CV MAE is the honest in-train estimate of held-out-song error",
    }
    results["integrity"] = {
        "feature_inputs_per_example": ["own cents contour", "own confidence contour", "own duration_s", "own target_midi"],
        "song_identity_used_as_feature": False, "song_identity_used_for": "GroupKFold grouping only",
        "sibling_information_used": False, "target_used_only_as_y": True,
        "validation_used_for_selection": False, "validation_evaluations_per_model": 1,
        "scaler_fit_inside_pipeline": True,
        "test_examples_featurized": 0, "test_predictions": 0,
        "feature_checks": feature_checks, "constant_shift_feature_sanity": sanity,
        "predictions_all_finite": True,
        "prediction_checks": prediction_checks,
        "blas_note": ("NumPy 2.x + Apple Accelerate emits spurious 'encountered in matmul' RuntimeWarnings for "
                      "well-conditioned products; they are counted, and every prediction is verified against a "
                      "BLAS-free einsum recomputation"),
    }

    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_markdown(results), encoding="utf-8")
    v = results["validation"]["overall"]
    print(f"alpha={hp['selected_alpha']} cvMAE={hp['selected_mean_cv_mae']:.4f}")
    print(f"train MAE {tr_o['mae']:.4f} RMSE {tr_o['rmse']:.4f} | validation MAE {v['mae']:.4f} RMSE {v['rmse']:.4f}")
    print(f"center-only validation MAE {pv['ridge_center_only']['overall']['mae']:.4f} RMSE {pv['ridge_center_only']['overall']['rmse']:.4f}")
    print("Test evaluated: NO")
    print(f"Results: {out_json}")
    return 0


def _markdown(r: Mapping[str, Any]) -> str:
    f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
    c = r["comparison_to_baselines"]
    L = ["# Baseline 4 (Ridge) on axis2_v2: validation", "",
         "Generated by `ml/baselines/ridge_baseline.py`. Alpha chosen by 5-fold GroupKFold (by song) inside train; "
         "validation evaluated once; **test not evaluated**.", "",
         f"Selected alpha **{r['hyperparameter_selection']['selected_alpha']:g}**, grouped-CV MAE "
         f"{r['hyperparameter_selection']['selected_mean_cv_mae']:.4f}.", "",
         "| validation | MAE | RMSE | median abs | P90 | P95 | P99 |", "|---|---|---|---|---|---|---|"]
    for k, x in c["ladder"].items():
        L.append(f"| {k} | {f(x['mae'])} | {f(x['rmse'])} | {f(x['median_absolute_error'])} | {f(x['p90'], 2)} | {f(x['p95'], 2)} | {f(x['p99'], 2)} |")
    L += ["", "**MAE / RMSE by variant**", "", "| variant | " + " | ".join(c["ladder"]) + " |", "|---|" + "---|" * len(c["ladder"])]
    for n, x in c["by_variant"].items():
        L.append(f"| {n} | " + " | ".join(f"{f(x[k]['mae'], 2)} / {f(x[k]['rmse'], 2)}" for k in c["ladder"]) + " |")
    s = r["validation"]["by_sign"]
    L += ["", "**Ridge by corruption sign**", "", "| sign | n | MAE | RMSE | mean signed |", "|---|---|---|---|---|"]
    for k in ("positive", "negative"):
        L.append(f"| {k} | {s[k]['example_count']} | {f(s[k]['mae'])} | {f(s[k]['rmse'])} | {s[k]['mean_signed_error']:+.4f} |")
    L += ["", "**Clean notes by frame count (MAE)**", "", "| frames | notes | full median | core median | Ridge | Ridge − core | Ridge RMSE |", "|---|---|---|---|---|---|---|"]
    for b, x in r["validation"]["by_frame_count"].items():
        L.append(f"| {b} | {x['clean_note_count']} | {f(x['full_median_clean_mae'])} | {f(x['core_median_clean_mae'])} | {f(x['ridge_clean_mae'])} | {x['ridge_minus_core_mae']:+.4f} | {f(x['ridge_clean_rmse'])} |")
    L += ["", "**Clean notes by duration (MAE)**", "", "| duration | notes | full median | core median | Ridge |", "|---|---|---|---|---|"]
    for b, x in r["validation"]["by_duration"].items():
        L.append(f"| {b} | {x['clean_note_count']} | {f(x['full_median_clean_mae'])} | {f(x['core_median_clean_mae'])} | {f(x['ridge_clean_mae'])} |")
    d = c["secondary_diagnostic_full_median_within_50c"]
    L += ["", f"**Secondary diagnostic: clean notes with |full median| ≤ 50 c (n={d['note_count']})**", "",
          "| | MAE | RMSE |", "|---|---|---|"]
    for k in ("full_median", "core_median", "ridge_center_only", "ridge"):
        L.append(f"| {k} | {f(d[k]['mae'])} | {f(d[k]['rmse'])} |")
    L += ["", "**Top standardized coefficients**", "", "| feature | coefficient |", "|---|---|"]
    for x in r["coefficients"]["sorted_by_abs"][:12]:
        L.append(f"| {x['feature']} | {x['coefficient']:+.4f} |")
    L += [f"", f"Intercept {r['coefficients']['intercept']:+.4f}. {r['coefficients']['caveat']}.", "",
          "**By song (validation; estimator fit per recording, not singer quality)**", "", "| song | n | MAE | RMSE | signed |", "|---|---|---|---|---|"]
    for sg, x in sorted(r["validation"]["by_song"].items(), key=lambda kv: kv[1]["mae"]):
        L.append(f"| {sg} | {x['example_count']} | {f(x['mae'])} | {f(x['rmse'])} | {x['mean_signed_error']:+.3f} |")
    L += ["", f"_{c['core_median_caveat']}._", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
