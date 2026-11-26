"""
Dataset-parameterized machinery for the controlled Axis 2 rerun on axis2_v3.

The V2 experiment scripts (natural offset, fixed hybrid, dynamic hybrid) live in
protected directories and hard-code data/ml_dataset/axis2_v2, so they are not
edited. Their model definitions are imported from them instead of copied:

  frozen definitions   hybrid_ridge_natural.frozen_configs(): Ridge alpha and
                       feature list, natural-offset feature set and HGB config,
                       read from the stored V2 selections and asserted against code
  Ridge features/fit   ml/baselines/ridge_baseline.py (via the hybrid helpers)
  natural features     natural_offset_feature_experiment.contour_features /
                       context_features / fit_predict

Only the dataset path is a parameter. Every consumer runs in axis2_v2 mode first
and must reproduce the stored V2 results before it is run on axis2_v3.

The test split is sealed: test indices never enter a featurized set and a test
song's notes.npz is never opened (enforced in featurize_natural).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # importing from protected directories must not write __pycache__ there

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
V2_HYBRID_DIR = REPO / "ml" / "experiments" / "axis2_v2_hybrid"
V2_DYNAMIC_DIR = REPO / "ml" / "experiments" / "axis2_v2_dynamic_hybrid"
sys.path.insert(0, str(V2_HYBRID_DIR))
sys.path.insert(0, str(V2_DYNAMIC_DIR))

import hybrid_ridge_natural as H  # noqa: E402  (adds backend/, ml/baselines/, the V2 natural dir to sys.path)

NAT, RB = H.NAT, H.RB
EXPECTED_SPLIT_ORDER, EXPECTED_VARIANT_ORDER = H.EXPECTED_SPLIT_ORDER, H.EXPECTED_VARIANT_ORDER

DATASETS = ("axis2_v2", "axis2_v3")
CV_FOLDS = 5
FREEZE_TOLERANCE = 1e-9
CSV_TOLERANCE = 2e-6

QUANTILES = (75, 90, 95, 99, 99.5, 99.9)
THRESHOLDS = (25, 50, 100)

PROTECTED_COMMON = ("data/main_data", "data/ml_dataset/axis2_v1", "data/ml_dataset/axis2_v2", "data/ml_dataset/axis2_v3",
                    "data/dataset_selections.json", "ml/results/axis2_v1", "ml/results/axis2_v2",
                    "ml/experiments/axis2_v2_natural_offset", "ml/experiments/axis2_v2_sequence_natural_offset",
                    "ml/experiments/axis2_v2_hybrid", "ml/experiments/axis2_v2_dynamic_hybrid",
                    "ml/experiments/axis2_v3_dataset_audit")


def dataset_dir(version: str) -> Path:
    if version not in DATASETS:
        raise SystemExit(f"unknown dataset {version!r}; expected one of {DATASETS}")
    return REPO / "data" / "ml_dataset" / version


def protected_paths(extra: Tuple[str, ...] = ()) -> List[Path]:
    return [REPO / p for p in PROTECTED_COMMON + extra]


def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


# ---------------------------------------------------------------- metrics


def metrics(err: np.ndarray) -> Dict[str, Any]:
    a = np.abs(err)
    return {"count": int(err.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median_absolute_error": float(np.median(a)), "mean_signed_error": float(err.mean()),
            "quantiles": {f"p{q:g}": float(np.percentile(a, q)) for q in QUANTILES}, "max": float(a.max()),
            "exceeding": {f"gt_{t}_cents": {"count": int(np.count_nonzero(a > t)), "percent": 100.0 * float(np.mean(a > t))}
                          for t in THRESHOLDS}}


# ---------------------------------------------------------------- data


def load(version: str) -> Dict[str, Any]:
    ds = dataset_dir(version)
    splits = json.loads((ds / "splits.json").read_text())
    summary = json.loads((ds / "summary.json").read_text())
    if summary.get("dataset_name") != version:
        raise RuntimeError(f"{ds} summary names {summary.get('dataset_name')!r}")
    with np.load(ds / "examples.npz", allow_pickle=False) as z:
        D = {k: np.asarray(z[k]) for k in z.files if k not in ("cents_flat", "confidence_flat")}
        cents_flat = np.asarray(z["cents_flat"], np.float64)
        conf_flat = np.asarray(z["confidence_flat"], np.float64)
    split_codes = H._resolve_lookup(D["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
    variant_codes = H._resolve_lookup(D["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")
    H._verify_counts(D["split_index"], split_codes, splits)
    tr = np.flatnonzero(D["split_index"] == split_codes["train"])
    va = np.flatnonzero(D["split_index"] == split_codes["validation"])
    assert tr.size == splits["train_example_count"] and va.size == splits["validation_example_count"]
    return {"version": version, "dir": ds, "D": D, "cents_flat": cents_flat, "conf_flat": conf_flat, "splits": splits,
            "split_codes": split_codes, "variant_codes": variant_codes, "tr": tr, "va": va,
            "off": np.asarray(D["contour_offsets"], np.int64), "song_ids": D["song_ids"],
            "duration": np.asarray(D["duration_s"], np.float64), "target_midi": D["target_midi"],
            "y": np.asarray(D["injected_shift_cents"], np.float64), "ctx_cache": {}, "ctx_by_song": {}, "songs_opened": set()}


def featurize_ridge(d: Mapping[str, Any], idx: np.ndarray) -> np.ndarray:
    X = RB.build_matrix(d["cents_flat"], d["conf_flat"], d["off"], d["duration"], d["target_midi"], idx)
    if not np.all(np.isfinite(X)):
        raise RuntimeError("non-finite Ridge features")
    return X


def featurize_natural(d: Dict[str, Any], idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """All 71 natural-offset columns (contour groups A-F + neighbour context G) and the observed core median."""
    D, off = d["D"], d["off"]
    X = np.empty((idx.size, len(NAT.ALL_NAMES)))
    core = np.empty(idx.size)
    for r, i in enumerate(idx):
        a, b = int(off[i]), int(off[i + 1])
        f, core[r] = NAT.contour_features(d["cents_flat"][a:b], d["conf_flat"][a:b], float(d["duration"][i]), int(d["target_midi"][i]))
        s = int(D["song_index"][i])
        if s not in d["ctx_by_song"]:
            if D["song_split_index"][s] == d["split_codes"]["test"]:
                raise RuntimeError("attempted to read a test song")
            d["ctx_by_song"][s] = NAT.context_features(str(d["song_ids"][s]), d["ctx_cache"])
            d["songs_opened"].add(str(d["song_ids"][s]))
        X[r, :f.size] = f
        X[r, f.size:] = d["ctx_by_song"][s][int(D["source_note_index"][i])]
    if not np.all(np.isfinite(X)):
        raise RuntimeError("non-finite natural-offset features")
    diff = float(np.max(np.abs(core - D["core_median_cents"][idx])))
    if diff > 1e-9:
        raise RuntimeError(f"recomputed core median differs from stored by {diff}")
    return X, core


def test_songs_opened(d: Mapping[str, Any]) -> int:
    D = d["D"]
    test = {str(d["song_ids"][s]) for s in range(d["song_ids"].size) if D["song_split_index"][s] == d["split_codes"]["test"]}
    return len(d["songs_opened"] & test)


def keys(d: Mapping[str, Any], idx: np.ndarray) -> List[Tuple[str, int, str]]:
    D = d["D"]
    return [(str(d["song_ids"][D["song_index"][i]]), int(D["source_note_index"][i]), EXPECTED_VARIANT_ORDER[int(D["variant_index"][i])])
            for i in idx]


# ---------------------------------------------------------------- frozen experts


def expert_references(ridge_results: Path, natural_results: Path, alpha: float, nat_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Stored per-fold CV MAEs and validation metrics of the experts ON THE DATASET BEING RUN, for freeze checks."""
    rid = json.loads(Path(ridge_results).read_text())
    nat = json.loads(Path(natural_results).read_text())
    if rid["test_evaluated"] is not False or nat["test_evaluated"] is not False:
        raise RuntimeError("an expert result is not test-sealed")
    if tuple(rid["features"]) != RB.FEATURES or rid["hyperparameter_selection"]["selected_alpha"] != alpha:
        raise RuntimeError(f"{ridge_results} was not produced with the frozen Ridge definition")
    sel = nat["selected_model"]
    if (sel["feature_set"], sel["model"], sel["param"]) != (nat_cfg["feature_set"], "hgb", nat_cfg["param"]):
        raise RuntimeError(f"{natural_results} was not produced with the frozen natural-offset definition")
    rcv = next(r for r in rid["hyperparameter_selection"]["cv_results"] if r["alpha"] == alpha)
    ncv = next(r for r in nat["cv_results"] if (r["feature_set"], r["model"], r["param"]) == (sel["feature_set"], "hgb", sel["param"]))
    return {"ridge_cv_fold_mae": rcv["fold_mae"], "natural_cv_fold_mae": ncv["fold_mae"],
            "ridge_validation": {k: rid["validation"]["overall"][k] for k in ("mae", "rmse")},
            "natural_validation": {k: nat["synthetic_shift_validation"]["overall"][k] for k in ("mae", "rmse")},
            "ridge_results": rel(ridge_results), "natural_results": rel(natural_results)}


def shared_folds(groups_tr: np.ndarray, n: int) -> Tuple[list, np.ndarray]:
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(np.zeros((n, 1)), np.zeros(n), groups_tr))
    song_fold = {int(s): k for k, (_, held) in enumerate(folds) for s in np.unique(groups_tr[held])}
    fold_of = np.array([song_fold[int(s)] for s in groups_tr])
    for fit, held in folds:
        if set(groups_tr[fit]) & set(groups_tr[held]):
            raise RuntimeError("a song appears on both sides of a fold")
    return folds, fold_of


def prepare_experts(d: Dict[str, Any], ridge_cfg: Mapping[str, Any], nat_cfg: Mapping[str, Any], refs: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Expert features, shared-partition OOF predictions on train, and the single validation prediction,
    with the freeze checks of the V2 hybrid: per-fold CV MAEs and validation metrics must reproduce
    the stored expert results for this dataset.
    """
    D, tr, va, y = d["D"], d["tr"], d["va"], d["y"]
    alpha, nat_param, nat_cols = ridge_cfg["alpha"], nat_cfg["param"], nat_cfg["columns"]
    print(f"[{d['version']}] featurizing experts", flush=True)
    Xr_tr, Xr_va = featurize_ridge(d, tr), featurize_ridge(d, va)
    Xfull_tr, core_tr = featurize_natural(d, tr)
    Xfull_va, core_va = featurize_natural(d, va)
    Xn_tr, Xn_va = Xfull_tr[:, nat_cols], Xfull_va[:, nat_cols]
    var_tr = D["variant_index"][tr]
    clean_tr = var_tr == d["variant_codes"]["clean"]
    y_tr, y_va = y[tr], y[va]
    groups_tr = D["song_index"][tr]
    folds, fold_of = shared_folds(groups_tr, tr.size)

    clean_pos = np.flatnonzero(clean_tr)
    native = list(GroupKFold(n_splits=CV_FOLDS).split(Xn_tr[clean_pos], core_tr[clean_pos], groups_tr[clean_pos]))
    if [frozenset(np.unique(groups_tr[clean_pos][h]).tolist()) for _, h in native] != \
       [frozenset(np.unique(groups_tr[h]).tolist()) for _, h in folds]:
        raise RuntimeError("the natural model's native folds partition songs differently from Ridge's")

    print(f"[{d['version']}] expert out-of-fold predictions", flush=True)
    ridge_oof, nhat_oof = np.full(tr.size, np.nan), np.full(tr.size, np.nan)
    ridge_fold_mae, nat_fold_mae = [], []
    for k in range(CV_FOLDS):
        held, fit = fold_of == k, fold_of != k
        ridge_oof[held] = H.ridge_fit_predict(alpha, Xr_tr[fit], y_tr[fit], Xr_tr[held])
        ridge_fold_mae.append(float(np.mean(np.abs(ridge_oof[held] - y_tr[held]))))
        fit_clean = fit & clean_tr
        nhat_oof[held] = H.natural_fit_predict(nat_param, Xn_tr[fit_clean], core_tr[fit_clean], Xn_tr[held])
        held_clean = held & clean_tr
        nat_fold_mae.append(float(np.mean(np.abs(nhat_oof[held_clean] - core_tr[held_clean]))))
    natural_oof = core_tr - nhat_oof
    if np.isnan(ridge_oof).any() or np.isnan(natural_oof).any():
        raise RuntimeError("missing OOF predictions")
    ridge_fold_diff = float(np.max(np.abs(np.array(ridge_fold_mae) - np.array(refs["ridge_cv_fold_mae"]))))
    nat_fold_diff = float(np.max(np.abs(np.array(nat_fold_mae) - np.array(refs["natural_cv_fold_mae"]))))
    if max(ridge_fold_diff, nat_fold_diff) > FREEZE_TOLERANCE:
        raise RuntimeError(f"expert CV does not reproduce the stored results: ridge {ridge_fold_diff}, natural {nat_fold_diff}")

    print(f"[{d['version']}] final expert fits and single validation prediction", flush=True)
    ridge_va = H.ridge_fit_predict(alpha, Xr_tr, y_tr, Xr_va)
    nhat_va = H.natural_fit_predict(nat_param, Xn_tr[clean_tr], core_tr[clean_tr], Xn_va)
    natural_va = core_va - nhat_va
    ridge_val_diff = max(abs(metrics(ridge_va - y_va)[k] - refs["ridge_validation"][k]) for k in ("mae", "rmse"))
    nat_val_diff = max(abs(metrics(natural_va - y_va)[k] - refs["natural_validation"][k]) for k in ("mae", "rmse"))
    if max(ridge_val_diff, nat_val_diff) > FREEZE_TOLERANCE:
        raise RuntimeError(f"expert validation does not reproduce: ridge {ridge_val_diff}, natural {nat_val_diff}")
    return {"Xr_tr": Xr_tr, "Xr_va": Xr_va, "Xfull_tr": Xfull_tr, "Xfull_va": Xfull_va, "core_tr": core_tr, "core_va": core_va,
            "var_tr": var_tr, "var_va": D["variant_index"][va], "clean_tr": clean_tr, "y_tr": y_tr, "y_va": y_va,
            "groups_tr": groups_tr, "folds": folds, "fold_of": fold_of,
            "ridge_oof": ridge_oof, "natural_oof": natural_oof, "ridge_va": ridge_va, "natural_va": natural_va,
            "ridge_fold_mae": ridge_fold_mae, "nat_fold_mae": nat_fold_mae,
            "freeze": {"ridge_max_fold_mae_diff": ridge_fold_diff, "natural_max_fold_mae_diff": nat_fold_diff,
                       "ridge_max_validation_mae_rmse_diff": ridge_val_diff, "natural_max_validation_mae_rmse_diff": nat_val_diff,
                       "reference_results": {"ridge": refs["ridge_results"], "natural": refs["natural_results"]}}}
