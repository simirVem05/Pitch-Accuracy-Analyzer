#!/usr/bin/env python3
"""
Axis 2 V2 feature discovery: can shift-invariant information predict a clean
note's natural pitch offset?

Target, per CLEAN note:
    N = median(_core_slice(clean cents contour))          (production core median)

Features never contain absolute pitch center. Contour features are computed on
    centered[t] = cents[t] - median(_core_slice(cents))
which removes the center by construction; the subtraction is the only use of N
inside feature construction, and it is exactly what makes every contour feature
invariant to a constant shift. Metadata (duration, length, target MIDI, pitch
class) and melodic context (neighbouring notes' target MIDI and timing from the
canonical v2 note sequence, never their cents) carry no center information.

Protocol: grouped 5-fold CV by song inside the training split's clean notes
selects feature group, model, and hyperparameters by mean CV MAE. The winner is
fit on all train clean notes and evaluated once on validation. It is then turned
into a shift estimator, S_hat = observed core median - N_hat, evaluated on every
validation example with features extracted from that example's own contour.

The test split is sealed: test examples are never featurized, and no test song's
notes.npz is opened.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "ml" / "baselines"))
from note_segmentation import _core_slice  # noqa: E402
from zero_baseline import EXPECTED_SPLIT_ORDER, EXPECTED_VARIANT_ORDER, _resolve_lookup, _tail, _verify_counts  # noqa: E402

DATASET = REPO / "data" / "ml_dataset" / "axis2_v2"
MAIN = REPO / "data" / "main_data"
RIDGE_RESULTS = REPO / "ml" / "results" / "axis2_v2" / "ridge_baseline.json"
CV_FOLDS = 5
SEED = 20261014
REGION = 0.20
AUTOCORR_MIN_FRAMES = 20
PERM_REPEATS = 3
CENTER_ROUND_DECIMALS = 9

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
HGB_CONFIGS = {
    "hgb_leaf15_lr05": dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0),
    "hgb_depth3_lr03": dict(max_iter=400, learning_rate=0.03, max_depth=3, min_samples_leaf=100, l2_regularization=1.0),
    "hgb_leaf31_lr05": dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
    "hgb_leaf15_lr05_absloss": dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
                                    l2_regularization=1.0, loss="absolute_error"),
}

# ---------------------------------------------------------------- features

GROUP_FEATURES: Dict[str, Tuple[str, ...]] = {
    "A_metadata": ("duration_s", "frame_count", "log1p_frame_count", "target_midi", "pitch_class_sin", "pitch_class_cos"),
    "B_centered_distribution": ("mean_c", "median_c", "std_c", "mad_c", "iqr_c", "p10_c", "p25_c", "p75_c", "p90_c",
                                "p90_minus_p10_c", "frac_within_5", "frac_within_10", "frac_within_20", "frac_within_30"),
    "C_temporal_shape": ("slope_c", "onset_median_c", "middle_median_c", "release_median_c", "onset_minus_middle",
                         "release_minus_middle", "release_minus_onset", "first_minus_second_half", "curvature"),
    "D_movement": ("mean_abs_d1", "median_abs_d1", "std_d1", "p90_abs_d1", "mean_abs_d2", "median_abs_d2",
                   "max_abs_d1", "frac_pos_steps", "frac_neg_steps", "direction_changes", "direction_change_rate"),
    "E_drift_oscillation": ("detrended_std", "detrended_mad", "detrended_iqr", "zero_crossings", "zero_crossing_rate",
                            "turning_points", "turning_point_rate", "acf_peak", "acf_peak_lag", "acf_eligible"),
    "F_confidence": ("conf_mean", "conf_median", "conf_std", "conf_p25", "conf_min", "conf_onset", "conf_middle",
                     "conf_release", "conf_onset_minus_middle", "conf_release_minus_middle", "conf_vs_abs_d1_corr"),
    "G_prev_context": ("interval_in", "abs_interval_in", "prev_duration", "gap_before", "has_previous"),
    "G_next_context": ("interval_out", "abs_interval_out", "next_duration", "gap_after", "has_next"),
}
CONTOUR_GROUPS = ("A_metadata", "B_centered_distribution", "C_temporal_shape", "D_movement", "E_drift_oscillation", "F_confidence")
ALL_NAMES = [f for g in GROUP_FEATURES.values() for f in g]
COL = {f: i for i, f in enumerate(ALL_NAMES)}

FEATURE_SETS = {
    "A": ("A_metadata",),
    "A+B": ("A_metadata", "B_centered_distribution"),
    "A+B+C": ("A_metadata", "B_centered_distribution", "C_temporal_shape"),
    "A+B+C+D": ("A_metadata", "B_centered_distribution", "C_temporal_shape", "D_movement"),
    "A+B+C+D+E": ("A_metadata", "B_centered_distribution", "C_temporal_shape", "D_movement", "E_drift_oscillation"),
    "A+B+C+D+E+F": CONTOUR_GROUPS,
    "ALL_prev_only": CONTOUR_GROUPS + ("G_prev_context",),
    "ALL": CONTOUR_GROUPS + ("G_prev_context", "G_next_context"),
    "shape_only": ("B_centered_distribution", "C_temporal_shape", "D_movement", "E_drift_oscillation"),
}
FORBIDDEN = ("full_median_cents", "core_median_cents", "mean_cents", "confidence_weighted_mean_cents",
             "absolute cents quantiles", "absolute region medians", "song_id", "artist", "filename",
             "clean_note_id", "variant", "severity", "sign")

DEFINITIONS = {
    "centering": "centered = round(cents - median(_core_slice(cents)), 9); the core median is the target N, used only for this subtraction; rounding to 1e-9 c removes ~1e-14 float noise that a constant shift introduces",
    "pitch_class": "circular: sin/cos(2*pi*(target_midi mod 12)/12); chosen over one-hot to keep the feature count small and the representation continuous",
    "regions": "by frame index: k = max(1, floor(0.2*n)); onset = first k, release = last k, middle = frames k..n-k (whole contour if empty)",
    "halves": "first floor(n/2) frames vs the rest",
    "curvature": "mean(middle) - (mean(onset) + mean(release)) / 2",
    "slope": "OLS slope of centered on linspace(0,1,n) (cents per note); 0 if n < 2",
    "d1/d2": "adjacent first/second differences of centered; n >= 3 for every note so both exist",
    "direction_changes": "sign changes between successive nonzero d1; rate = changes / max(1, nonzero_d1 - 1)",
    "zero_crossings": "sign changes of nonzero centered values around 0; rate = crossings / (n - 1)",
    "turning_points": "sign changes of successive nonzero differences of the linearly detrended contour; rate = / max(1, n - 2)",
    "autocorrelation": f"n >= {AUTOCORR_MIN_FRAMES}: largest positive normalized autocorrelation of the detrended contour over lags 2..n//2 and its lag in frames; otherwise 0, 0, acf_eligible = 0",
    "conf_vs_abs_d1_corr": "Pearson correlation of confidence[1:] with |d1| when n >= 5 and both vary; otherwise 0",
    "context": "neighbours are the previous/next note in the song's canonical v2 notes.npz; interval in semitones of target MIDI; gaps = start - previous end, next start - end; missing neighbour -> 0 with has_previous/has_next = 0; no neighbour cents ever used",
}


def _sign_changes(v: np.ndarray) -> int:
    s = np.sign(v[v != 0])
    return int(np.count_nonzero(s[1:] != s[:-1])) if s.size > 1 else 0


def contour_features(cents: np.ndarray, conf: np.ndarray, duration: float, midi: int) -> Tuple[np.ndarray, float]:
    """Groups A-F for one example from its own contour. Returns (features, its core median)."""
    n = cents.size
    core = float(np.median(_core_slice(cents)))
    # Rounded to 1e-9 c: (cents + S) - (core + S) differs from cents - core by ~1e-14,
    # and tree splits placed exactly at 0.0 (the centered median of every untrimmed
    # note) react to that noise. Rounding removes it without using any other example.
    c = np.round(cents - core, CENTER_ROUND_DECIMALS)
    x = np.linspace(0.0, 1.0, n)
    k = max(1, int(np.floor(REGION * n)))
    mid = slice(k, n - k) if n - 2 * k >= 1 else slice(0, n)
    on, rel = c[:k], c[n - k:]
    xc = x - x.mean()
    slope = float(np.sum(xc * (c - c.mean())) / np.sum(xc * xc)) if n >= 2 else 0.0
    trend = c.mean() + slope * xc
    det = c - trend
    d1 = np.diff(c)
    d2 = np.diff(d1)
    ad1 = np.abs(d1)
    nz = d1[d1 != 0]
    med = float(np.median(c))
    q10, q25, q75, q90 = np.percentile(c, (10, 25, 75, 90))
    dq25, dq75 = np.percentile(det, (25, 75))
    half = n // 2

    acf_peak, acf_lag, eligible = 0.0, 0.0, 0.0
    if n >= AUTOCORR_MIN_FRAMES:
        z = det - det.mean()
        den = float(np.sum(z * z))
        if den > 1e-12:
            eligible = 1.0
            lags = np.arange(2, n // 2 + 1)
            ac = np.array([np.sum(z[:-L] * z[L:]) / den for L in lags])
            j = int(np.argmax(ac))
            if ac[j] > 0:
                acf_peak, acf_lag = float(ac[j]), float(lags[j])

    corr = 0.0
    if n >= 5:
        a, b = conf[1:], ad1
        if np.std(a) > 0 and np.std(b) > 0:
            corr = float(np.corrcoef(a, b)[0, 1])

    pc = int(midi) % 12
    feats = [
        # A
        float(duration), float(n), float(np.log1p(n)), float(midi), np.sin(2 * np.pi * pc / 12), np.cos(2 * np.pi * pc / 12),
        # B
        float(c.mean()), med, float(np.std(c)), float(np.median(np.abs(c - med))), float(q75 - q25),
        float(q10), float(q25), float(q75), float(q90), float(q90 - q10),
        *(float(np.mean(np.abs(c) <= t)) for t in (5, 10, 20, 30)),
        # C
        slope, float(np.median(on)), float(np.median(c[mid])), float(np.median(rel)),
        float(np.median(on) - np.median(c[mid])), float(np.median(rel) - np.median(c[mid])), float(np.median(rel) - np.median(on)),
        float(np.median(c[:half]) - np.median(c[half:])) if half >= 1 else 0.0,
        float(np.mean(c[mid]) - (np.mean(on) + np.mean(rel)) / 2),
        # D
        float(ad1.mean()), float(np.median(ad1)), float(np.std(d1)), float(np.percentile(ad1, 90)),
        float(np.mean(np.abs(d2))) if d2.size else 0.0, float(np.median(np.abs(d2))) if d2.size else 0.0,
        float(ad1.max()), float(np.mean(d1 > 0)), float(np.mean(d1 < 0)),
        float(_sign_changes(d1)), float(_sign_changes(d1)) / max(1, nz.size - 1),
        # E
        float(np.std(det)), float(np.median(np.abs(det - np.median(det)))), float(dq75 - dq25),
        float(_sign_changes(c)), float(_sign_changes(c)) / (n - 1),
        float(_sign_changes(np.diff(det))), float(_sign_changes(np.diff(det))) / max(1, n - 2),
        acf_peak, acf_lag, eligible,
        # F
        float(conf.mean()), float(np.median(conf)), float(np.std(conf)), float(np.percentile(conf, 25)), float(conf.min()),
        float(conf[:k].mean()), float(conf[mid].mean()), float(conf[n - k:].mean()),
        float(conf[:k].mean() - conf[mid].mean()), float(conf[n - k:].mean() - conf[mid].mean()), corr,
    ]
    return np.asarray(feats, dtype=np.float64), core


def context_features(song: str, cache: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    """Group G for every note of one song, from its canonical notes.npz (target MIDI and timing only)."""
    if song not in cache:
        with np.load(MAIN / song / "notes.npz", allow_pickle=False) as n:
            cache[song] = {k: np.asarray(n[k]) for k in ("target_midi_note", "start_time_s", "end_time_s", "duration_s")}
    n = cache[song]
    m = n["target_midi_note"].astype(np.float64)
    st, en, du = n["start_time_s"], n["end_time_s"], n["duration_s"]
    count = m.size
    G = np.zeros((count, 10))
    G[1:, 0] = m[1:] - m[:-1]; G[1:, 1] = np.abs(G[1:, 0]); G[1:, 2] = du[:-1]; G[1:, 3] = st[1:] - en[:-1]; G[1:, 4] = 1
    G[:-1, 5] = m[1:] - m[:-1]; G[:-1, 6] = np.abs(G[:-1, 5]); G[:-1, 7] = du[1:]; G[:-1, 8] = st[1:] - en[:-1]; G[:-1, 9] = 1
    return G


# ---------------------------------------------------------------- models


def make_model(kind: str, param) -> Any:
    if kind == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=param))])
    return HistGradientBoostingRegressor(random_state=0, early_stopping=False, **HGB_CONFIGS[param])


def fit_predict(kind, param, Xtr, ytr, Xte) -> np.ndarray:
    # NumPy 2.x + Accelerate raises spurious "encountered in matmul" warnings (verified in
    # ridge_baseline.py); predictions are checked finite instead, and Ridge against einsum.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        model = make_model(kind, param).fit(Xtr, ytr)
        p = model.predict(Xte)
    if not np.all(np.isfinite(p)):
        raise RuntimeError("non-finite predictions")
    if kind == "ridge":
        sc, rg = model.named_steps["scaler"], model.named_steps["ridge"]
        manual = np.einsum("ij,j->i", (Xte - sc.mean_) / sc.scale_, rg.coef_) + rg.intercept_
        if np.max(np.abs(manual - p)) > 1e-6:
            raise RuntimeError("Ridge prediction disagrees with BLAS-free recomputation")
    return p


def cols_for(feature_set: str) -> List[int]:
    return [COL[f] for g in FEATURE_SETS[feature_set] for f in GROUP_FEATURES[g]]


def metrics(err: np.ndarray) -> Dict[str, Any]:
    a = np.abs(err)
    return {"count": int(err.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median_absolute_error": float(np.median(a)), "mean_signed_error": float(err.mean())}


def slope_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    xc = x - x.mean()
    b = float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))
    a = float(y.mean() - b * x.mean())
    r2 = 1.0 - float(np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"slope": b, "intercept": a, "r2": r2, "count": int(x.size)}


def _hash_tree(paths: Sequence[Path]) -> Dict[str, str]:
    out = {}
    for root in paths:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        h = hashlib.sha256()
        for p in files:
            h.update(str(p.relative_to(REPO)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
        out[str(root.relative_to(REPO))] = h.hexdigest()
    return out


# ---------------------------------------------------------------- main


def main() -> int:
    t0 = time.time()
    protected = [REPO / "data" / "main_data", REPO / "data" / "ml_dataset" / "axis2_v1", DATASET,
                 REPO / "data" / "dataset_selections.json", REPO / "ml" / "results" / "axis2_v1", REPO / "ml" / "results" / "axis2_v2"]
    hashes_before = _hash_tree(protected)
    for name in ("natural_offset_results.json", "feature_group_results.csv", "natural_offset_summary.md"):
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; refusing to overwrite")

    splits = json.loads((DATASET / "splits.json").read_text())
    with np.load(DATASET / "examples.npz", allow_pickle=False) as z:
        D = {k: np.asarray(z[k]) for k in z.files if k not in ("cents_flat", "confidence_flat")}
        cents_flat, conf_flat = np.asarray(z["cents_flat"]), np.asarray(z["confidence_flat"])
    split_codes = _resolve_lookup(D["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
    variant_codes = _resolve_lookup(D["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")
    _verify_counts(D["split_index"], split_codes, splits)
    off, song_ids = D["contour_offsets"], D["song_ids"]

    # Test examples never enter these index sets; test songs' notes.npz is never opened.
    tr_clean = np.flatnonzero((D["split_index"] == split_codes["train"]) & (D["variant_index"] == variant_codes["clean"]))
    va_all = np.flatnonzero(D["split_index"] == split_codes["validation"])
    assert tr_clean.size == splits["train_clean_note_count"] and va_all.size == splits["validation_example_count"]

    ctx_cache: Dict[str, Dict[str, np.ndarray]] = {}
    ctx_by_song: Dict[int, np.ndarray] = {}

    def featurize(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.empty((idx.size, len(ALL_NAMES)))
        core = np.empty(idx.size)
        for r, i in enumerate(idx):
            a, b = int(off[i]), int(off[i + 1])
            f, core[r] = contour_features(cents_flat[a:b], conf_flat[a:b], float(D["duration_s"][i]), int(D["target_midi"][i]))
            s = int(D["song_index"][i])
            if s not in ctx_by_song:
                if D["song_split_index"][s] == split_codes["test"]:
                    raise RuntimeError("attempted to read a test song")
                ctx_by_song[s] = context_features(str(song_ids[s]), ctx_cache)
            X[r, :f.size] = f
            X[r, f.size:] = ctx_by_song[s][int(D["source_note_index"][i])]
        return X, core

    X_tr, N_tr = featurize(tr_clean)
    X_va, core_va = featurize(va_all)
    if not (np.all(np.isfinite(X_tr)) and np.all(np.isfinite(X_va))):
        raise RuntimeError("non-finite features")
    stored_core_diff = float(max(np.max(np.abs(N_tr - D["core_median_cents"][tr_clean])),
                                 np.max(np.abs(core_va - D["core_median_cents"][va_all]))))
    if stored_core_diff > 1e-9:
        raise RuntimeError(f"recomputed core median differs from stored by {stored_core_diff}")
    groups_tr = D["song_index"][tr_clean]

    # shift-invariance of the extractor, measured on validation siblings (features of
    # each corrupted example's own contour vs its clean sibling's)
    va_var = D["variant_index"][va_all]
    order = np.lexsort((va_var, D["source_note_index"][va_all], D["song_index"][va_all]))
    Xg = X_va[order].reshape(-1, 5, X_va.shape[1])
    feat_spread = np.max(np.abs(Xg - Xg[:, :1, :]), axis=(0, 1))
    invariance = {"max_feature_difference_vs_clean_sibling": float(feat_spread.max()),
                  "worst_feature": ALL_NAMES[int(np.argmax(feat_spread))],
                  "per_feature_max_difference": dict(zip(ALL_NAMES, feat_spread.tolist()))}

    # ---------------- grouped CV on train clean notes
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(X_tr, N_tr, groups_tr))
    assert all(not (set(groups_tr[a]) & set(groups_tr[b])) for a, b in folds)
    fold_mae_n0 = [float(np.mean(np.abs(N_tr[b]))) for _, b in folds]
    fold_mae_mean = [float(np.mean(np.abs(N_tr[b] - N_tr[a].mean()))) for a, b in folds]
    configs = [("ridge", a) for a in RIDGE_ALPHAS] + [("hgb", k) for k in HGB_CONFIGS]
    rows = []
    for fs in FEATURE_SETS:
        cols = cols_for(fs)
        for kind, param in configs:
            maes, rmses = [], []
            for a, b in folds:
                p = fit_predict(kind, param, X_tr[a][:, cols], N_tr[a], X_tr[b][:, cols])
                e = p - N_tr[b]
                maes.append(float(np.mean(np.abs(e)))); rmses.append(float(np.sqrt(np.mean(e ** 2))))
            rows.append({"feature_set": fs, "model": kind, "param": param, "n_features": len(cols),
                         "cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)), "cv_rmse": float(np.mean(rmses)),
                         "fold_mae": maes})
            print(f"{fs:16s} {kind:5s} {str(param):24s} CV MAE {np.mean(maes):.4f}", flush=True)

    best_per = {}
    for r in rows:
        key = (r["feature_set"], r["model"])
        if key not in best_per or r["cv_mae"] < best_per[key]["cv_mae"]:
            best_per[key] = r
    mean_n0, mean_nmean = float(np.mean(fold_mae_n0)), float(np.mean(fold_mae_mean))
    meta_best = min(best_per[("A", m)]["cv_mae"] for m in ("ridge", "hgb"))
    table = []
    for (fs, m), r in best_per.items():
        table.append({"feature_set": fs, "model": m, "best_param": r["param"], "cv_mae": r["cv_mae"], "cv_mae_std": r["cv_mae_std"],
                      "cv_rmse": r["cv_rmse"], "n_features": r["n_features"],
                      "improvement_vs_metadata_only": meta_best - r["cv_mae"],
                      "improvement_vs_zero_N": mean_n0 - r["cv_mae"], "improvement_vs_train_mean_N": mean_nmean - r["cv_mae"]})
    table.sort(key=lambda r: (list(FEATURE_SETS).index(r["feature_set"]), r["model"]))
    with (OUT / "feature_group_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        for r in table:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    sel = min(rows, key=lambda r: r["cv_mae"])
    best_ridge = min((r for r in rows if r["model"] == "ridge"), key=lambda r: r["cv_mae"])
    best_hgb = min((r for r in rows if r["model"] == "hgb"), key=lambda r: r["cv_mae"])
    sel_cols = cols_for(sel["feature_set"])
    sel_names = [ALL_NAMES[c] for c in sel_cols]

    # ---------------- permutation importance (train, grouped folds; validation unused)
    rng = np.random.default_rng(SEED)
    feat_imp = {f: [] for f in sel_names}
    sel_groups = FEATURE_SETS[sel["feature_set"]]
    grp_imp = {g: [] for g in sel_groups}
    for a, b in folds:
        Xa, Xb = X_tr[a][:, sel_cols], X_tr[b][:, sel_cols]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            model = make_model(sel["model"], sel["param"]).fit(Xa, N_tr[a])
            base = float(np.mean(np.abs(model.predict(Xb) - N_tr[b])))
            for j, f in enumerate(sel_names):
                for _ in range(PERM_REPEATS):
                    Xp = Xb.copy(); Xp[:, j] = Xp[rng.permutation(Xp.shape[0]), j]
                    feat_imp[f].append(float(np.mean(np.abs(model.predict(Xp) - N_tr[b]))) - base)
            for g in sel_groups:
                gj = [sel_names.index(f) for f in GROUP_FEATURES[g]]
                for _ in range(PERM_REPEATS):
                    Xp = Xb.copy(); Xp[:, gj] = Xp[rng.permutation(Xp.shape[0])][:, gj]   # rows permuted jointly
                    grp_imp[g].append(float(np.mean(np.abs(model.predict(Xp) - N_tr[b]))) - base)
    feat_rank = sorted(({"feature": f, "group": next(g for g in sel_groups if f in GROUP_FEATURES[g]),
                         "mae_increase_mean": float(np.mean(v)), "mae_increase_std": float(np.std(v))}
                        for f, v in feat_imp.items()), key=lambda r: -r["mae_increase_mean"])
    with (OUT / "feature_importance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feat_rank[0].keys()))
        w.writeheader()
        for r in feat_rank:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---------------- final fit on all train clean notes; single validation evaluation
    N_mean = float(N_tr.mean())
    va_clean = va_var == variant_codes["clean"]
    N_va = core_va[va_clean]                                   # clean core median = target
    Nhat_all = fit_predict(sel["model"], sel["param"], X_tr[:, sel_cols], N_tr, X_va[:, sel_cols])
    Nhat_clean = Nhat_all[va_clean]
    e_N = Nhat_clean - N_va
    natural = {
        "selected": metrics(e_N) | {"tail": _tail(np.abs(e_N))},
        "baseline_N0": metrics(-N_va),
        "baseline_train_mean_N": metrics(N_mean - N_va) | {"train_mean_N": N_mean},
    }
    natural["improvement_vs_N0"] = {"mae": natural["baseline_N0"]["mae"] - natural["selected"]["mae"],
                                    "percent": 100 * (natural["baseline_N0"]["mae"] - natural["selected"]["mae"]) / natural["baseline_N0"]["mae"],
                                    "rmse": natural["baseline_N0"]["rmse"] - natural["selected"]["rmse"]}

    # descriptive breakdowns on clean validation notes
    Xc = X_va[va_clean]
    def bucket(values, edges, labels):
        out = {}
        for (lo, hi), lab in zip(edges, labels):
            m = np.ones(values.shape, bool)
            if lo is not None: m &= values >= lo
            if hi is not None: m &= values < hi
            out[lab] = {"count": int(m.sum()), "selected_mae": float(np.mean(np.abs(e_N[m]))) if m.any() else None,
                        "N0_mae": float(np.mean(np.abs(N_va[m]))) if m.any() else None}
        return out
    nf = Xc[:, COL["frame_count"]]
    breakdowns = {
        "frame_count": bucket(nf, [(3, 5), (5, 7), (7, 10), (10, 21), (21, 51), (51, None)], ["3-4", "5-6", "7-9", "10-20", "21-50", "51+"]),
        "duration_s": bucket(Xc[:, COL["duration_s"]], [(None, .1), (.1, .2), (.2, .4), (.4, .8), (.8, None)],
                             ["<0.10", "0.10-0.20", "0.20-0.40", "0.40-0.80", ">=0.80"]),
        "target_midi": bucket(Xc[:, COL["target_midi"]], [(None, 55), (55, 60), (60, 65), (65, 70), (70, None)],
                              ["<55", "55-59", "60-64", "65-69", ">=70"]),
        "abs_interval_in (no previous -> 0)": bucket(Xc[:, COL["abs_interval_in"]], [(None, 1), (1, 3), (3, 5), (5, 8), (8, None)],
                                                     ["0", "1-2", "3-4", "5-7", "8+"]),
        "abs_interval_out (no next -> 0)": bucket(Xc[:, COL["abs_interval_out"]], [(None, 1), (1, 3), (3, 5), (5, 8), (8, None)],
                                                  ["0", "1-2", "3-4", "5-7", "8+"]),
    }

    # ---------------- shift estimator on every validation example
    y_va = D["injected_shift_cents"][va_all]
    S_hat = core_va - Nhat_all                                 # own observed core median minus own N_hat
    e_S = S_hat - y_va
    Nh_g = Nhat_all[order].reshape(-1, 5)
    e_g = e_S[order].reshape(-1, 5)
    corrupted = va_var != variant_codes["clean"]
    shift_val = {
        "overall": metrics(e_S) | {"tail": _tail(np.abs(e_S))},
        "by_variant": {n: metrics(e_S[va_var == c]) for n, c in variant_codes.items()},
        "by_sign": {"positive": metrics(e_S[corrupted & (y_va > 0)]), "negative": metrics(e_S[corrupted & (y_va < 0)])},
        "prediction_slope": {"corrupted_only": slope_stats(y_va[corrupted], S_hat[corrupted]),
                             "all_validation_examples": slope_stats(y_va, S_hat)},
        "structural_identity": {
            "max_within_note_N_hat_spread": float(np.max(Nh_g.max(1) - Nh_g.min(1))),
            "max_within_note_signed_error_spread": float(np.max(e_g.max(1) - e_g.min(1))),
            "groups_violating_1e-6": int(np.count_nonzero((e_g.max(1) - e_g.min(1)) > 1e-6)),
            "note_groups": int(e_g.shape[0]),
            "note": "N_hat comes from each example's own features; equality across siblings is measured, not imposed",
        },
    }

    # ---------------- comparison to existing Ridge
    rid = json.loads(RIDGE_RESULTS.read_text())
    rv = rid["validation"]
    comparison = {
        "overall": {"ridge": {k: rv["overall"][k] for k in ("mae", "rmse", "median_absolute_error")},
                    "natural_offset_estimator": {k: shift_val["overall"][k] for k in ("mae", "rmse", "median_absolute_error")}},
        "by_variant_mae": {n: {"ridge": rv["by_variant"][n]["mae"], "natural_offset_estimator": shift_val["by_variant"][n]["mae"],
                               "difference_new_minus_ridge": shift_val["by_variant"][n]["mae"] - rv["by_variant"][n]["mae"]}
                           for n in EXPECTED_VARIANT_ORDER},
        "by_variant_rmse": {n: {"ridge": rv["by_variant"][n]["rmse"], "natural_offset_estimator": shift_val["by_variant"][n]["rmse"]}
                            for n in EXPECTED_VARIANT_ORDER},
        "by_sign": {s: {"ridge_mae": rv["by_sign"][s]["mae"], "ridge_mean_signed": rv["by_sign"][s]["mean_signed_error"],
                        "new_mae": shift_val["by_sign"][s]["mae"], "new_mean_signed": shift_val["by_sign"][s]["mean_signed_error"]}
                    for s in ("positive", "negative")},
        "slope_corrupted": {"ridge": rid["shrinkage_diagnostic"]["validation_slope_prediction_on_injected_shift_corrupted"],
                            "natural_offset_estimator": shift_val["prediction_slope"]["corrupted_only"]["slope"]},
        "core_median_baseline_equivalent": "with N_hat = 0 this estimator is exactly the core-median baseline (MAE 17.4255)",
    }

    val_rows = []
    for j, i in enumerate(va_all):
        val_rows.append([str(song_ids[D["song_index"][i]]), int(D["source_note_index"][i]), EXPECTED_VARIANT_ORDER[int(va_var[j])],
                         int(D["num_frames"][i]), f"{D['duration_s'][i]:.4f}", int(D["target_midi"][i]),
                         f"{core_va[j]:.6f}", f"{Nhat_all[j]:.6f}", f"{S_hat[j]:.6f}", f"{y_va[j]:.6f}", f"{e_S[j]:.6f}"])
    with (OUT / "validation_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["song_id", "source_note_index", "variant", "frame_count", "duration_s", "target_midi",
                    "observed_core_median", "predicted_natural_offset", "predicted_shift", "injected_shift", "shift_error"])
        w.writerows(val_rows)

    hashes_after = _hash_tree(protected)
    results = {
        "experiment": "axis2_v2_natural_offset_feature_discovery",
        "test_evaluated": False,
        "target": "clean_core_median_cents = median(_core_slice(clean cents contour)), recomputed and checked against stored core_median_cents",
        "absolute_pitch_center_removed": True,
        "forbidden_features": list(FORBIDDEN),
        "feature_groups": {g: list(f) for g, f in GROUP_FEATURES.items()},
        "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "feature_definitions": DEFINITIONS,
        "counts": {"train_clean_notes": int(tr_clean.size), "validation_clean_notes": int(va_clean.sum()),
                   "validation_examples": int(va_all.size), "train_songs": int(np.unique(groups_tr).size),
                   "sealed_test_metadata_only": {"examples": splits["test_example_count"], "clean_notes": splits["test_clean_note_count"]}},
        "cv_design": {"method": "GroupKFold(5) by song on train clean notes", "folds": [{"fit": int(a.size), "heldout": int(b.size),
                      "heldout_songs": int(np.unique(groups_tr[b]).size)} for a, b in folds],
                      "ridge_alphas": list(RIDGE_ALPHAS), "hgb_configs": HGB_CONFIGS,
                      "selection": "lowest mean CV MAE over every (feature set, model, hyperparameter)"},
        "cv_reference": {"N0_cv_mae": mean_n0, "train_mean_N_cv_mae": mean_nmean},
        "cv_results": rows,
        "feature_group_table": table,
        "best_ridge": {k: best_ridge[k] for k in ("feature_set", "param", "cv_mae", "cv_mae_std")},
        "best_hgb": {k: best_hgb[k] for k in ("feature_set", "param", "cv_mae", "cv_mae_std")},
        "selected_model": {k: sel[k] for k in ("feature_set", "model", "param", "cv_mae", "cv_mae_std", "cv_rmse", "n_features")}
                          | {"features": sel_names},
        "natural_offset_validation": natural,
        "natural_offset_breakdowns_validation_clean": breakdowns,
        "synthetic_shift_validation": shift_val,
        "comparison_to_ridge": comparison,
        "feature_importance": {"method": f"grouped-fold permutation on train (validation unused), {PERM_REPEATS} repeats; "
                                         "MAE increase when permuted; groups permuted jointly",
                               "top_features": feat_rank[:15],
                               "by_group": {g: {"mae_increase_mean": float(np.mean(v)), "mae_increase_std": float(np.std(v))}
                                            for g, v in grp_imp.items()}},
        "integrity": {
            "feature_inputs": "own cents contour (centered), own confidence, own duration, own target MIDI, neighbours' target MIDI + timing",
            "test_examples_featurized": 0, "test_song_note_files_opened": 0,
            "validation_used_for_selection": False, "validation_evaluations": 1,
            "recomputed_vs_stored_core_median_max_abs_diff": stored_core_diff,
            "shift_invariance_of_features": {k: v for k, v in invariance.items() if k != "per_feature_max_difference"},
            "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
            "protected_unchanged": hashes_before == hashes_after,
            "runtime_s": time.time() - t0,
        },
    }
    (OUT / "natural_offset_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    (OUT / "natural_offset_summary.md").write_text(summary_md(results))
    print(json.dumps({"selected": results["selected_model"] | {"features": len(sel_names)}, "natural": natural["selected"],
                      "N0": natural["baseline_N0"]["mae"], "shift_mae": shift_val["overall"]["mae"],
                      "slope": shift_val["prediction_slope"]["corrupted_only"]["slope"],
                      "protected_unchanged": hashes_before == hashes_after}, indent=1, default=float))
    return 0


def summary_md(r: Dict[str, Any]) -> str:
    f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
    L = ["# Axis 2 V2: natural-offset feature discovery", "",
         "Generated by `natural_offset_feature_experiment.py`. Target N = clean core median. No absolute pitch center in any feature. "
         "Selection by 5-fold GroupKFold (song) on train; validation evaluated once; **test not evaluated**.", "",
         f"CV references: N=0 → {f(r['cv_reference']['N0_cv_mae'])}, train-mean N → {f(r['cv_reference']['train_mean_N_cv_mae'])}.", "",
         "| feature set | model | best param | CV MAE | ± | features | vs metadata | vs N=0 |", "|---|---|---|---|---|---|---|---|"]
    for t in r["feature_group_table"]:
        L.append(f"| {t['feature_set']} | {t['model']} | {t['best_param']} | {f(t['cv_mae'])} | {f(t['cv_mae_std'], 3)} | {t['n_features']} | "
                 f"{t['improvement_vs_metadata_only']:+.3f} | {t['improvement_vs_zero_N']:+.3f} |")
    s, n = r["selected_model"], r["natural_offset_validation"]
    L += ["", f"**Selected:** {s['feature_set']} / {s['model']} ({s['param']}), CV MAE {f(s['cv_mae'])}.", "",
          "| natural offset, validation clean notes | MAE | RMSE | median abs | mean signed |", "|---|---|---|---|---|"]
    for k, lab in (("baseline_N0", "N = 0"), ("baseline_train_mean_N", "N = train mean"), ("selected", "selected model")):
        L.append(f"| {lab} | {f(n[k]['mae'])} | {f(n[k]['rmse'])} | {f(n[k]['median_absolute_error'])} | {n[k]['mean_signed_error']:+.3f} |")
    c = r["comparison_to_ridge"]
    L += ["", "| shift estimate, validation | Ridge MAE | new MAE | new − Ridge |", "|---|---|---|---|"]
    for v, x in c["by_variant_mae"].items():
        L.append(f"| {v} | {f(x['ridge'])} | {f(x['natural_offset_estimator'])} | {x['difference_new_minus_ridge']:+.3f} |")
    L.append(f"| **overall** | {f(c['overall']['ridge']['mae'])} | {f(c['overall']['natural_offset_estimator']['mae'])} | "
             f"{c['overall']['natural_offset_estimator']['mae'] - c['overall']['ridge']['mae']:+.3f} |")
    L += ["", f"Slope of prediction on true shift (corrupted): Ridge {f(c['slope_corrupted']['ridge'], 3)}, new "
              f"{f(c['slope_corrupted']['natural_offset_estimator'], 3)}.", "",
          "**Permutation importance by group (train folds, MAE increase)**", "", "| group | MAE increase |", "|---|---|"]
    for g, x in sorted(r["feature_importance"]["by_group"].items(), key=lambda kv: -kv[1]["mae_increase_mean"]):
        L.append(f"| {g} | {x['mae_increase_mean']:+.4f} |")
    L += ["", "**Top individual features**", "", "| feature | group | MAE increase |", "|---|---|---|"]
    for x in r["feature_importance"]["top_features"][:12]:
        L.append(f"| {x['feature']} | {x['group']} | {x['mae_increase_mean']:+.4f} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
