#!/usr/bin/env python3
"""
Axis 2 V2: dynamic (per-example) blend of the two existing shift estimators.

    prediction = w(x) * ridge + (1 - w(x)) * natural,   w(x) = sigmoid(gate(x)) in [0, 1]

The question: does learning how much to trust each expert per note beat the
fixed 0.56 / 0.44 blend (validation MAE 12.5828)?

Experts are frozen. They are the exact Ridge baseline and handcrafted
natural-offset HGB, reached through the fixed-hybrid script's helpers, which
assert their stored settings. Their per-fold CV MAEs, validation metrics, and
per-example validation predictions are reproduced and checked before any gate
is trained.

Three gates, nothing else:
  fixed      w = 0.56, read from the stored fixed-hybrid result (not refit)
  magnitude  w = sigmoid(a + b * |observed core median|), 2 parameters
  dynamic    20 inference-time features -> Linear(20,16) -> GELU -> Linear(16,8)
             -> GELU -> Linear(8,1) -> sigmoid

Protocol (train only until the single validation pass):
  1. One GroupKFold(5) partition by song over all train examples, shared by both
     experts, yields an out-of-fold Ridge and natural prediction for every train
     example from models that never saw that example's song.
  2. Learned gates are trained on those OOF rows to minimise L1 of the blended
     prediction. Weight decay {0, 1e-4} and the epoch count are chosen by the
     pooled out-of-fold MAE of the gate itself, using the same song folds
     (a gate fit on four folds' OOF rows predicts the fifth).
  3. Experts are refit on all of train; each learned gate is refit on all train
     OOF rows with the selected settings; validation is predicted once.

The gate never sees the injected shift, its magnitude, sign, band, variant, a
clean/corrupted flag, or song/artist identity. Target MIDI is the clean note's
(the dataset does not recompute it after shifting), so it is identical across a
note's five variants and carries no label information.

The test split is sealed: no test contour is featurized, no test song's
notes.npz is opened, and no test prediction exists.

  python ml/experiments/axis2_v2_dynamic_hybrid/dynamic_hybrid.py
"""

from __future__ import annotations

import sys

# Must precede every import from a protected directory, so no __pycache__ lands there.
sys.dont_write_bytecode = True

import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HYBRID_DIR = REPO / "ml" / "experiments" / "axis2_v2_hybrid"
sys.path.insert(0, str(HYBRID_DIR))

import hybrid_ridge_natural as H  # noqa: E402  (puts backend/, ml/baselines/, natural dir on sys.path)

NAT, RB = H.NAT, H.RB
EXPECTED_SPLIT_ORDER, EXPECTED_VARIANT_ORDER = H.EXPECTED_SPLIT_ORDER, H.EXPECTED_VARIANT_ORDER

DATASET = H.DATASET
HYBRID_RESULTS = HYBRID_DIR / "hybrid_results.json"
HYBRID_VAL_PREDICTIONS = HYBRID_DIR / "validation_predictions.csv"

CV_FOLDS = 5
FREEZE_TOLERANCE = 1e-9
CSV_TOLERANCE = 2e-6
NOTE_TIE_CENTS = 0.5
FIXED_BENCHMARK = {"mae": 12.5828, "rmse": 15.9575}   # as quoted; checked against the stored result below

# ---- gate training (minimal, fixed in advance)
LEARNING_RATE = 1e-3
WEIGHT_DECAYS = (0.0, 1e-4)
BATCH_SIZE = 512
MAX_EPOCHS = 80
SEED = 20261124
SEED_CHECK = (SEED + 1, SEED + 2)   # train-CV-only robustness check of the selected config
HIDDEN = (16, 8)

# Gate inputs. Names on the right are the natural-offset extractor's columns
# (same features under the experiment's own names; no new extraction).
PRED_FEATURES = ("ridge_prediction", "natural_prediction", "abs_ridge_prediction", "abs_natural_prediction",
                 "ridge_minus_natural", "abs_ridge_minus_natural", "observed_core_median", "abs_observed_core_median")
META_FEATURES = {"duration_seconds": "duration_s", "frame_count": "frame_count", "log1p_frame_count": "log1p_frame_count",
                 "target_midi": "target_midi", "previous_interval": "interval_in", "next_interval": "interval_out",
                 "abs_previous_interval": "abs_interval_in", "abs_next_interval": "abs_interval_out",
                 "gap_before": "gap_before", "gap_after": "gap_after", "has_previous": "has_previous", "has_next": "has_next"}
GATE_FEATURES = PRED_FEATURES + tuple(META_FEATURES)
MAGNITUDE_FEATURES = ("abs_observed_core_median",)
FORBIDDEN_GATE_INPUTS = ("injected_shift", "true_shift", "abs_true_shift", "variant", "variant_index", "corruption",
                         "severity", "sign", "clean", "song", "artist")

QUANTILES = (50, 75, 90, 95, 99, 99.5, 99.9)
THRESHOLDS = (25, 50, 100)
CORE_BUCKETS = (("0-10", 0, 10), ("10-20", 10, 20), ("20-30", 20, 30), ("30-40", 30, 40), ("40-50", 40, 50), ("50+", 50, None))
DISAGREE_BUCKETS = (("0-5", 0, 5), ("5-10", 5, 10), ("10-20", 10, 20), ("20+", 20, None))
W_QUANTILES = (10, 25, 50, 75, 90)
MODELS = ("ridge", "natural", "fixed", "magnitude", "dynamic")
LABEL = {"ridge": "Ridge", "natural": "Natural", "fixed": "Fixed 56/44", "magnitude": "Magnitude gate", "dynamic": "Dynamic gate"}
OUTPUTS = ("dynamic_hybrid_results.json", "dynamic_hybrid_summary.md", "gate_validation_predictions.csv",
           "gate_weight_analysis.csv", "gate_model.pt", "gate_config.json")


# ---------------------------------------------------------------- metrics


def metrics(err: np.ndarray) -> Dict[str, Any]:
    a = np.abs(err)
    out = {"count": int(err.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "median_absolute_error": float(np.median(a)), "mean_signed_error": float(err.mean())}
    out["quantiles"] = {f"p{q:g}": float(np.percentile(a, q)) for q in QUANTILES[1:]}
    out["max"] = float(a.max())
    out["exceeding"] = {f"gt_{t}_cents": int(np.count_nonzero(a > t)) for t in THRESHOLDS}
    return out


def short(err: np.ndarray) -> Dict[str, float]:
    a = np.abs(err)
    return {"count": int(err.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_signed_error": float(err.mean())}


def bucket(v: np.ndarray, lo, hi) -> np.ndarray:
    m = v >= lo
    if hi is not None:
        m &= v < hi
    return m


def wdist(w: np.ndarray) -> Dict[str, float]:
    return {"mean": float(w.mean()), "std": float(w.std()), **{f"p{q}": float(np.percentile(w, q)) for q in W_QUANTILES},
            "min": float(w.min()), "max": float(w.max())}


# ---------------------------------------------------------------- gate


class Gate(nn.Module):
    def __init__(self, n_in: int, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "dynamic":
            self.net = nn.Sequential(nn.Linear(n_in, HIDDEN[0]), nn.GELU(), nn.Linear(HIDDEN[0], HIDDEN[1]), nn.GELU(),
                                     nn.Linear(HIDDEN[1], 1))
        elif kind == "magnitude":
            self.net = nn.Linear(n_in, 1)
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))


def train_gate(kind: str, X: np.ndarray, r: np.ndarray, n: np.ndarray, y: np.ndarray, wd: float, epochs: int, seed: int,
               X_eval: np.ndarray | None = None) -> Tuple[Gate, Dict[str, np.ndarray], np.ndarray | None]:
    """L1 on the blended prediction. Standardisation is fit on the gate's own fit rows.
    If X_eval is given, returns its weights after every epoch, shape (epochs, n_eval)."""
    torch.manual_seed(seed)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    scaler = {"mean": mu, "std": sd}
    model = Gate(X.shape[1], kind).double()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=wd)
    Xt, rt, nt, yt = _t((X - mu) / sd), _t(r), _t(n), _t(y)
    Xe = _t((X_eval - mu) / sd) if X_eval is not None else None
    trace = np.empty((epochs, X_eval.shape[0]), np.float32) if X_eval is not None else None
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xt.shape[0], generator=gen)
        for s in range(0, perm.numel(), BATCH_SIZE):
            b = perm[s:s + BATCH_SIZE]
            w = model(Xt[b])
            loss = torch.mean(torch.abs(w * rt[b] + (1.0 - w) * nt[b] - yt[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xe is not None:
            model.eval()
            with torch.no_grad():
                trace[ep] = model(Xe).numpy()
    return model, scaler, trace


def apply_gate(model: Gate, scaler: Mapping[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        w = model(_t((X - scaler["mean"]) / scaler["std"])).numpy().astype(np.float64)
    if not np.all(np.isfinite(w)) or w.min() < 0 or w.max() > 1:
        raise RuntimeError("gate produced weights outside [0, 1]")
    return w


def gate_inputs(ridge: np.ndarray, natural: np.ndarray, core: np.ndarray, Xnat_full: np.ndarray) -> np.ndarray:
    pred = np.column_stack([ridge, natural, np.abs(ridge), np.abs(natural), ridge - natural, np.abs(ridge - natural),
                            core, np.abs(core)])
    meta = Xnat_full[:, [NAT.COL[c] for c in META_FEATURES.values()]]
    X = np.column_stack([pred, meta])
    if X.shape[1] != len(GATE_FEATURES) or not np.all(np.isfinite(X)):
        raise RuntimeError("bad gate feature matrix")
    return X


# ---------------------------------------------------------------- main


def main() -> int:
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; refusing to overwrite")
    for f in GATE_FEATURES:
        if any(bad in f for bad in FORBIDDEN_GATE_INPUTS):
            raise RuntimeError(f"forbidden gate input {f}")
    protected = [REPO / "data" / "main_data", REPO / "data" / "ml_dataset" / "axis2_v1", DATASET,
                 REPO / "data" / "dataset_selections.json", REPO / "ml" / "results" / "axis2_v1",
                 REPO / "ml" / "results" / "axis2_v2", H.NATURAL_DIR,
                 REPO / "ml" / "experiments" / "axis2_v2_sequence_natural_offset", HYBRID_DIR]
    hashes_before = NAT._hash_tree(protected)

    ridge_cfg, nat_cfg = H.frozen_configs()
    alpha, nat_param, nat_cols = ridge_cfg["alpha"], nat_cfg["param"], nat_cfg["columns"]
    hyb = json.loads(HYBRID_RESULTS.read_text())
    if hyb["test_evaluated"] is not False:
        raise RuntimeError("fixed hybrid result is not test-sealed")
    W_FIXED = float(hyb["blend"]["selected_weight_ridge"])
    stored_fixed = {k: hyb["validation"]["overall"]["hybrid"][k] for k in ("mae", "rmse")}
    stored_fixed_oof = hyb["oof_weight_selection"]["oof_metrics"]["hybrid"]["mae"]

    splits = json.loads((DATASET / "splits.json").read_text())
    with np.load(DATASET / "examples.npz", allow_pickle=False) as z:
        D = {k: np.asarray(z[k]) for k in z.files if k not in ("cents_flat", "confidence_flat")}
        cents_flat = np.asarray(z["cents_flat"], np.float64)
        conf_flat = np.asarray(z["confidence_flat"], np.float64)
    split_codes = H._resolve_lookup(D["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
    variant_codes = H._resolve_lookup(D["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")
    H._verify_counts(D["split_index"], split_codes, splits)
    off = np.asarray(D["contour_offsets"], np.int64)
    song_ids = D["song_ids"]
    duration = np.asarray(D["duration_s"], np.float64)
    target_midi = D["target_midi"]
    y_all = np.asarray(D["injected_shift_cents"], np.float64)

    # Test examples never enter these index sets.
    tr = np.flatnonzero(D["split_index"] == split_codes["train"])
    va = np.flatnonzero(D["split_index"] == split_codes["validation"])
    assert tr.size == splits["train_example_count"] and va.size == splits["validation_example_count"]

    # ---- expert features, each with its own extractor (identical to the fixed-hybrid run)
    print("featurizing", flush=True)
    Xr_tr = RB.build_matrix(cents_flat, conf_flat, off, duration, target_midi, tr)
    Xr_va = RB.build_matrix(cents_flat, conf_flat, off, duration, target_midi, va)
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
        return X, core

    Xfull_tr, core_tr = nat_featurize(tr)
    Xfull_va, core_va = nat_featurize(va)
    Xn_tr, Xn_va = Xfull_tr[:, nat_cols], Xfull_va[:, nat_cols]
    for X in (Xr_tr, Xr_va, Xfull_tr, Xfull_va):
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

    # ---- one song-level partition, used by both experts AND by the gate's own CV
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(Xr_tr, y_tr, groups_tr))
    song_fold = {int(s): k for k, (_, held) in enumerate(folds) for s in np.unique(groups_tr[held])}
    fold_of = np.array([song_fold[int(s)] for s in groups_tr])
    for fit, held in folds:
        if set(groups_tr[fit]) & set(groups_tr[held]):
            raise RuntimeError("a song appears on both sides of a fold")

    print("expert out-of-fold predictions", flush=True)
    ridge_oof = np.full(tr.size, np.nan)
    nhat_oof = np.full(tr.size, np.nan)
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
    ridge_fold_diff = float(np.max(np.abs(np.array(ridge_fold_mae) - np.array(ridge_cfg["stored_cv_fold_mae"]))))
    nat_fold_diff = float(np.max(np.abs(np.array(nat_fold_mae) - np.array(nat_cfg["stored_cv_fold_mae"]))))
    fixed_oof = W_FIXED * ridge_oof + (1.0 - W_FIXED) * natural_oof
    fixed_oof_diff = abs(float(np.mean(np.abs(fixed_oof - y_tr))) - stored_fixed_oof)
    if max(ridge_fold_diff, nat_fold_diff, fixed_oof_diff) > FREEZE_TOLERANCE:
        raise RuntimeError(f"expert CV does not reproduce: {ridge_fold_diff} {nat_fold_diff} {fixed_oof_diff}")

    G_tr = gate_inputs(ridge_oof, natural_oof, core_tr, Xfull_tr)
    mag_idx = [GATE_FEATURES.index(f) for f in MAGNITUDE_FEATURES]

    # Leakage guard: every metadata gate input must be identical across a note's five variants.
    gk_tr = D["song_index"][tr].astype(np.int64) * 1_000_000 + D["source_note_index"][tr].astype(np.int64)
    meta_cols = [GATE_FEATURES.index(f) for f in META_FEATURES]
    order = np.lexsort((var_tr, gk_tr))
    M = G_tr[order][:, meta_cols].reshape(-1, 5, len(meta_cols))
    meta_variant_spread = float(np.max(np.abs(M - M[:, :1, :])))
    if meta_variant_spread != 0.0:
        raise RuntimeError("a metadata gate input differs between variants of one note")

    # ---- gate selection on train OOF rows only
    gate_specs = {"magnitude": mag_idx, "dynamic": list(range(len(GATE_FEATURES)))}
    selection: Dict[str, Any] = {}
    oof_w: Dict[str, np.ndarray] = {}
    for kind, cols in gate_specs.items():
        curves = {}
        traces = {}
        for wd in WEIGHT_DECAYS:
            print(f"gate CV: {kind} wd={wd}", flush=True)
            tr_w = np.empty((MAX_EPOCHS, tr.size), np.float32)
            for k in range(CV_FOLDS):
                fit, held = fold_of != k, fold_of == k
                _, _, trace = train_gate(kind, G_tr[fit][:, cols], ridge_oof[fit], natural_oof[fit], y_tr[fit], wd,
                                         MAX_EPOCHS, SEED + k, X_eval=G_tr[held][:, cols])
                tr_w[:, held] = trace
            pred = tr_w.astype(np.float64) * ridge_oof + (1.0 - tr_w.astype(np.float64)) * natural_oof
            curves[wd] = np.mean(np.abs(pred - y_tr), axis=1)
            traces[wd] = tr_w
        cands = [(float(curves[wd][e]), e + 1, wd) for wd in WEIGHT_DECAYS for e in range(MAX_EPOCHS)]
        best_mae, best_ep, best_wd = min(cands)
        w_sel = traces[best_wd][best_ep - 1].astype(np.float64)
        oof_w[kind] = w_sel
        p_sel = w_sel * ridge_oof + (1.0 - w_sel) * natural_oof
        selection[kind] = {
            "selected_weight_decay": best_wd, "selected_epochs": best_ep, "oof_mae": best_mae,
            "oof_metrics": short(p_sel - y_tr) | {"slope_corrupted": NAT.slope_stats(y_tr[~clean_tr], p_sel[~clean_tr])["slope"]},
            "oof_mae_by_weight_decay_at_best_epoch": {str(wd): float(curves[wd].min()) for wd in WEIGHT_DECAYS},
            "oof_best_epoch_by_weight_decay": {str(wd): int(np.argmin(curves[wd]) + 1) for wd in WEIGHT_DECAYS},
            "oof_mae_curve": {str(wd): [float(x) for x in curves[wd]] for wd in WEIGHT_DECAYS},
            "oof_weight_distribution": wdist(w_sel),
            "note": "epochs and weight decay were selected on this same pooled OOF MAE, so it is mildly optimistic "
                    f"(best of {len(cands)} candidates)",
        }
        # Robustness of the selected config to initialisation, train CV only (no selection made from this).
        seed_maes = [best_mae]
        for sd in SEED_CHECK:
            ww = np.empty(tr.size)
            for k in range(CV_FOLDS):
                fit, held = fold_of != k, fold_of == k
                m, sc, _ = train_gate(kind, G_tr[fit][:, cols], ridge_oof[fit], natural_oof[fit], y_tr[fit], best_wd,
                                      best_ep, sd + k)
                ww[held] = apply_gate(m, sc, G_tr[held][:, cols])
            seed_maes.append(float(np.mean(np.abs(ww * ridge_oof + (1 - ww) * natural_oof - y_tr))))
        selection[kind]["seed_check_oof_mae"] = {"seeds": [SEED, *SEED_CHECK], "oof_mae": seed_maes,
                                                 "std": float(np.std(seed_maes)),
                                                 "note": "selected config re-run at fixed epochs with other seeds; report only"}
        print(f"  selected wd={best_wd} epochs={best_ep} OOF MAE={best_mae:.4f}", flush=True)

    # ---- final experts on all train; validation predicted once
    print("final fits and single validation pass", flush=True)
    ridge_va = H.ridge_fit_predict(alpha, Xr_tr, y_tr, Xr_va)
    natural_va = core_va - H.natural_fit_predict(nat_param, Xn_tr[clean_tr], core_tr[clean_tr], Xn_va)
    fixed_va = W_FIXED * ridge_va + (1.0 - W_FIXED) * natural_va
    G_va = gate_inputs(ridge_va, natural_va, core_va, Xfull_va)

    final_gates, final_scalers, w_va = {}, {}, {}
    for kind, cols in gate_specs.items():
        s = selection[kind]
        m, sc, _ = train_gate(kind, G_tr[:, cols], ridge_oof, natural_oof, y_tr, s["selected_weight_decay"],
                              s["selected_epochs"], SEED + CV_FOLDS)
        final_gates[kind], final_scalers[kind] = m, sc
        w_va[kind] = apply_gate(m, sc, G_va[:, cols])
        s["final_fit"] = {"rows": int(tr.size), "source": "all train OOF expert predictions",
                          "epochs": s["selected_epochs"], "weight_decay": s["selected_weight_decay"], "seed": SEED + CV_FOLDS,
                          "train_oof_mae_in_sample": float(np.mean(np.abs(
                              apply_gate(m, sc, G_tr[:, cols]) * ridge_oof
                              + (1 - apply_gate(m, sc, G_tr[:, cols])) * natural_oof - y_tr)))}

    preds = {"ridge": ridge_va, "natural": natural_va, "fixed": fixed_va,
             "magnitude": w_va["magnitude"] * ridge_va + (1.0 - w_va["magnitude"]) * natural_va,
             "dynamic": w_va["dynamic"] * ridge_va + (1.0 - w_va["dynamic"]) * natural_va}
    weights = {"ridge": np.ones(va.size), "natural": np.zeros(va.size), "fixed": np.full(va.size, W_FIXED),
               "magnitude": w_va["magnitude"], "dynamic": w_va["dynamic"]}
    errs = {k: p - y_va for k, p in preds.items()}

    # ---- freeze checks on validation (experts and fixed hybrid reproduce stored results)
    ridge_val_diff = max(abs(short(errs["ridge"])[k] - ridge_cfg["stored_validation"][k]) for k in ("mae", "rmse"))
    nat_val_diff = max(abs(short(errs["natural"])[k] - nat_cfg["stored_validation"][k]) for k in ("mae", "rmse"))
    fixed_val_diff = max(abs(short(errs["fixed"])[k] - stored_fixed[k]) for k in ("mae", "rmse"))
    stored_rows = {}
    with HYBRID_VAL_PREDICTIONS.open() as fh:
        for row in csv.DictReader(fh):
            stored_rows[(row["song_id"], int(row["source_note_index"]), row["variant"])] = (
                float(row["ridge_prediction"]), float(row["natural_prediction"]), float(row["hybrid_prediction"]),
                float(row["injected_shift"]))
    keys = [(str(song_ids[D["song_index"][i]]), int(D["source_note_index"][i]), EXPECTED_VARIANT_ORDER[int(D["variant_index"][i])])
            for i in va]
    if len(set(keys)) != len(keys) or set(keys) != stored_rows.keys():
        raise RuntimeError("validation keys differ from the stored fixed-hybrid predictions")
    csv_diff = max(max(abs(stored_rows[k][0] - ridge_va[j]), abs(stored_rows[k][1] - natural_va[j]),
                       abs(stored_rows[k][2] - fixed_va[j]), abs(stored_rows[k][3] - y_va[j])) for j, k in enumerate(keys))
    if max(ridge_val_diff, nat_val_diff, fixed_val_diff) > FREEZE_TOLERANCE or csv_diff > CSV_TOLERANCE:
        raise RuntimeError(f"validation does not reproduce: {ridge_val_diff} {nat_val_diff} {fixed_val_diff} csv {csv_diff}")

    # ---- validation analysis (all post hoc; nothing below feeds back into a gate)
    corr_va = var_va != variant_codes["clean"]
    absdiff = np.abs(ridge_va - natural_va)
    abscore = np.abs(core_va)

    overall = {k: metrics(e) for k, e in errs.items()}
    fx, dy, mg = overall["fixed"], overall["dynamic"], overall["magnitude"]

    def vs_fixed(m):
        return {"mae": m["mae"], "rmse": m["rmse"],
                "mae_improvement": fx["mae"] - m["mae"], "mae_improvement_percent": 100 * (fx["mae"] - m["mae"]) / fx["mae"],
                "rmse_improvement": fx["rmse"] - m["rmse"], "rmse_improvement_percent": 100 * (fx["rmse"] - m["rmse"]) / fx["rmse"],
                "beats_fixed_mae_benchmark_12_5828": m["mae"] < FIXED_BENCHMARK["mae"]}

    primary = {"fixed_hybrid": {"mae": fx["mae"], "rmse": fx["rmse"]}, "dynamic_gate": vs_fixed(dy), "magnitude_gate": vs_fixed(mg),
               "positive_improvement_means": "gate better than fixed"}
    by_variant = {v: {k: short(errs[k][var_va == c]) for k in MODELS} for v, c in variant_codes.items()}
    slopes = {k: NAT.slope_stats(y_va[corr_va], preds[k][corr_va]) for k in MODELS}
    by_sign = {}
    for sgn, mask in (("positive", corr_va & (y_va > 0)), ("negative", corr_va & (y_va < 0))):
        by_sign[sgn] = {k: short(errs[k][mask]) | {"mean_ridge_weight": float(weights[k][mask].mean())} for k in MODELS}
    weight_dist = {k: wdist(weights[k]) for k in ("magnitude", "dynamic")}
    w_by_variant = {k: {v: float(weights[k][var_va == c].mean()) for v, c in variant_codes.items()} for k in ("magnitude", "dynamic")}

    def bucket_table(values, buckets):
        out = {}
        for lab, lo, hi in buckets:
            m = bucket(values, lo, hi)
            out[lab] = {"count": int(m.sum()),
                        **({f"{k}_mae": float(np.mean(np.abs(errs[k][m]))) for k in MODELS} if m.any() else {}),
                        **({f"{k}_mean_ridge_weight": float(weights[k][m].mean()) for k in ("magnitude", "dynamic")} if m.any() else {})}
        return out

    by_core = bucket_table(abscore, CORE_BUCKETS)
    by_disagree = bucket_table(absdiff, DISAGREE_BUCKETS)

    gkey = D["song_index"][va].astype(np.int64) * 1_000_000 + D["source_note_index"][va].astype(np.int64)
    uniq, inv, counts = np.unique(gkey, return_inverse=True, return_counts=True)
    if not np.all(counts == 5):
        raise RuntimeError("a validation note group does not have exactly five variants")
    gmae = {k: np.bincount(inv, weights=np.abs(e)) / counts for k, e in errs.items()}

    def group_cmp(a: str, b: str) -> Dict[str, Any]:
        gain = gmae[b] - gmae[a]   # > 0: a is better
        win, loss = gain > NOTE_TIE_CENTS, gain < -NOTE_TIE_CENTS
        return {f"{a}_better_percent": 100 * float(win.mean()), "tie_percent": 100 * float((~win & ~loss).mean()),
                f"{b}_better_percent": 100 * float(loss.mean()),
                f"median_gain_when_{a}_wins": float(np.median(gain[win])) if win.any() else None,
                f"median_loss_when_{b}_wins": float(np.median(-gain[loss])) if loss.any() else None,
                "mean_gain": float(gain.mean())}

    note_groups = {"note_groups": int(uniq.size), "tie_tolerance_cents": NOTE_TIE_CENTS,
                   "metric": "per note: mean absolute error over its five variants",
                   "dynamic_vs_fixed": group_cmp("dynamic", "fixed"), "dynamic_vs_ridge": group_cmp("dynamic", "ridge"),
                   "magnitude_vs_fixed": group_cmp("magnitude", "fixed"),
                   "mean_note_mae": {k: float(v.mean()) for k, v in gmae.items()}}

    # ---- outputs
    with (OUT / "gate_validation_predictions.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["song_id", "source_note_index", "variant", "true_shift", "observed_core_median",
                     "ridge_prediction", "natural_prediction", "fixed_hybrid_prediction", "magnitude_gate_prediction",
                     "magnitude_ridge_weight", "dynamic_gate_prediction", "dynamic_ridge_weight"])
        for j, k in enumerate(keys):
            wr.writerow([k[0], k[1], k[2], f"{y_va[j]:.6f}", f"{core_va[j]:.6f}", f"{ridge_va[j]:.6f}", f"{natural_va[j]:.6f}",
                         f"{fixed_va[j]:.6f}", f"{preds['magnitude'][j]:.6f}", f"{w_va['magnitude'][j]:.6f}",
                         f"{preds['dynamic'][j]:.6f}", f"{w_va['dynamic'][j]:.6f}"])

    with (OUT / "gate_weight_analysis.csv").open("w", newline="") as fh:
        cols = ["analysis", "bucket", "count", *[f"{k}_mae" for k in MODELS], *[f"{k}_rmse" for k in MODELS],
                "magnitude_mean_ridge_weight", "dynamic_mean_ridge_weight"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        groups = [("variant", lab, var_va == c) for lab, c in variant_codes.items()]
        groups += [("abs_observed_core_median", lab, bucket(abscore, lo, hi)) for lab, lo, hi in CORE_BUCKETS]
        groups += [("abs_ridge_minus_natural", lab, bucket(absdiff, lo, hi)) for lab, lo, hi in DISAGREE_BUCKETS]
        groups += [("corrupted_sign", "positive", corr_va & (y_va > 0)), ("corrupted_sign", "negative", corr_va & (y_va < 0)),
                   ("all", "validation", np.ones(va.size, bool))]
        for analysis, lab, m in groups:
            row = {"analysis": analysis, "bucket": lab, "count": int(m.sum())}
            if m.any():
                for k in MODELS:
                    row[f"{k}_mae"] = f"{np.mean(np.abs(errs[k][m])):.6f}"
                    row[f"{k}_rmse"] = f"{np.sqrt(np.mean(errs[k][m] ** 2)):.6f}"
                row["magnitude_mean_ridge_weight"] = f"{weights['magnitude'][m].mean():.6f}"
                row["dynamic_mean_ridge_weight"] = f"{weights['dynamic'][m].mean():.6f}"
            wr.writerow(row)

    mag_lin = final_gates["magnitude"].net
    a_std, b_std = float(mag_lin.bias.item()), float(mag_lin.weight.item())
    mu, sd = float(final_scalers["magnitude"]["mean"][0]), float(final_scalers["magnitude"]["std"][0])
    magnitude_params = {"formula": "w_ridge = sigmoid(a + b * |observed_core_median|)", "a": a_std - b_std * mu / sd,
                        "b_per_cent": b_std / sd,
                        "w_at_abs_core": {str(c): float(1 / (1 + np.exp(-(a_std + b_std * (c - mu) / sd))))
                                          for c in (0, 10, 20, 30, 40, 50, 75)}}
    torch.save({k: {"state_dict": final_gates[k].state_dict(), "scaler_mean": torch.from_numpy(final_scalers[k]["mean"]),
                    "scaler_std": torch.from_numpy(final_scalers[k]["std"]),
                    "features": [GATE_FEATURES[i] for i in gate_specs[k]]} for k in gate_specs}, OUT / "gate_model.pt")
    gate_config = {
        "dynamic": {"features": list(GATE_FEATURES), "feature_sources": {**{f: "computed from expert predictions / observed core median" for f in PRED_FEATURES},
                                                                        **{f: f"natural-offset extractor column '{c}'" for f, c in META_FEATURES.items()}},
                    "architecture": f"Linear({len(GATE_FEATURES)},{HIDDEN[0]}) -> GELU -> Linear({HIDDEN[0]},{HIDDEN[1]}) -> GELU -> Linear({HIDDEN[1]},1) -> sigmoid",
                    "parameters": n_params(final_gates["dynamic"])},
        "magnitude": {"features": list(MAGNITUDE_FEATURES), "architecture": "Linear(1,1) -> sigmoid",
                      "parameters": n_params(final_gates["magnitude"]), "fitted": magnitude_params},
        "training": {"loss": "L1 of w*ridge + (1-w)*natural against injected shift", "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                     "weight_decay_grid": list(WEIGHT_DECAYS), "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                     "input_standardisation": "z-score fit on the gate's own fit rows", "dtype": "float64", "seed": SEED,
                     "selection": "pooled OOF MAE over the shared song folds; argmin over (weight decay, epoch)",
                     "selected": {k: {"weight_decay": selection[k]["selected_weight_decay"], "epochs": selection[k]["selected_epochs"]}
                                  for k in gate_specs}},
        "fixed": {"weight_ridge": W_FIXED, "source": "ml/experiments/axis2_v2_hybrid/hybrid_results.json"},
    }
    (OUT / "gate_config.json").write_text(json.dumps(gate_config, indent=2) + "\n")

    test_songs = {str(song_ids[s]) for s in range(song_ids.size) if D["song_split_index"][s] == split_codes["test"]}
    hashes_after = NAT._hash_tree(protected)
    results = {
        "experiment": "axis2_v2_ridge_natural_dynamic_gate",
        "test_evaluated": False,
        "question": "does a learned per-example blend weight beat the fixed 0.56/0.44 blend?",
        "experts": {"ridge": {k: v for k, v in ridge_cfg.items() if not k.startswith("stored_")},
                    "natural_offset": {k: v for k, v in nat_cfg.items() if not k.startswith("stored_") and k != "columns"},
                    "frozen": True, "third_expert": None, "sequence_model_used": False},
        "gates": gate_config,
        "oof_protocol": {
            "method": "GroupKFold(5) by song over all train examples; one partition shared by both experts and by the gate CV; "
                      "Ridge refit per fold on all variants of the fit songs, natural HGB refit per fold on the fit songs' clean notes",
            "folds": [{"fold": k, "heldout_songs": int(np.unique(groups_tr[fold_of == k]).size),
                       "heldout_examples": int(np.sum(fold_of == k))} for k in range(CV_FOLDS)],
            "gate_cv": "for each fold k, a gate is fit on the OOF rows of the other four folds and predicts fold k's OOF rows",
            "final_gate": "refit on all train OOF rows with the selected weight decay and epoch count; applied once to validation "
                          "expert predictions from experts refit on all of train",
            "oof_mae": {"ridge": float(np.mean(np.abs(ridge_oof - y_tr))), "natural": float(np.mean(np.abs(natural_oof - y_tr))),
                        "fixed": float(np.mean(np.abs(fixed_oof - y_tr))), "magnitude": selection["magnitude"]["oof_mae"],
                        "dynamic": selection["dynamic"]["oof_mae"]},
            "selection": selection,
        },
        "validation": {"overall": overall, "primary_question": primary, "by_variant": by_variant,
                       "prediction_slope": {"definition": "OLS predicted_shift ~ injected_shift, corrupted validation examples", **slopes},
                       "weight_distribution": weight_dist, "mean_ridge_weight_by_variant": w_by_variant,
                       "by_abs_observed_core_median": by_core, "by_abs_ridge_minus_natural": by_disagree,
                       "by_sign_corrupted": by_sign, "note_groups": note_groups,
                       "post_hoc_only": "variant, sign, magnitude and disagreement breakdowns were computed after the gates were fixed"},
        "integrity": {
            "expert_cv_reproduced": {"ridge_max_fold_mae_diff": ridge_fold_diff, "natural_max_fold_mae_diff": nat_fold_diff,
                                     "fixed_hybrid_oof_mae_diff": fixed_oof_diff},
            "validation_reproduced": {"ridge_max_mae_rmse_diff": ridge_val_diff, "natural_max_mae_rmse_diff": nat_val_diff,
                                      "fixed_max_mae_rmse_diff": fixed_val_diff,
                                      "max_per_example_diff_vs_stored_fixed_hybrid_csv": csv_diff},
            "gate_metadata_inputs_identical_across_variants_max_diff": meta_variant_spread,
            "forbidden_gate_inputs_checked": list(FORBIDDEN_GATE_INPUTS),
            "recomputed_vs_stored_core_median_max_abs_diff": core_diff,
            "validation_used_for_gate_selection": False, "validation_prediction_passes": 1,
            "test_examples_featurized": 0, "test_song_note_files_opened": len(songs_opened & test_songs),
            "sealed_test_metadata_only": {"examples": splits["test_example_count"], "clean_notes": splits["test_clean_note_count"]},
            "protected_paths": [str(p.relative_to(REPO)) for p in protected],
            "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
            "protected_unchanged": hashes_before == hashes_after,
            "runtime_s": time.time() - t0,
        },
    }
    if results["integrity"]["test_song_note_files_opened"]:
        raise RuntimeError("a test song's notes were opened")
    (OUT / "dynamic_hybrid_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    (OUT / "dynamic_hybrid_summary.md").write_text(summary_md(results))
    print(json.dumps({"primary": primary, "protected_unchanged": hashes_before == hashes_after}, indent=1))
    return 0


# ---------------------------------------------------------------- report


def summary_md(r: Mapping[str, Any]) -> str:
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    v, g, o = r["validation"], r["gates"], r["oof_protocol"]
    pq = v["primary_question"]
    L = ["# Axis 2 V2: dynamic Ridge / natural-offset gate", "",
         "Generated by `dynamic_hybrid.py`. `prediction = w(x)·ridge + (1 − w(x))·natural`, w ∈ [0, 1] per example. "
         "Experts frozen; gates trained on train out-of-fold expert predictions (GroupKFold(5) by song); validation "
         "evaluated once; **test not evaluated**.", "",
         f"- Dynamic gate: {len(g['dynamic']['features'])} features, `{g['dynamic']['architecture']}`, "
         f"**{g['dynamic']['parameters']} parameters**.",
         f"- Magnitude gate: `{g['magnitude']['fitted']['formula']}`, a = {g['magnitude']['fitted']['a']:+.4f}, "
         f"b = {g['magnitude']['fitted']['b_per_cent']:+.5f} per cent ({g['magnitude']['parameters']} parameters).",
         f"- Fixed: w = {g['fixed']['weight_ridge']:.2f}.", "",
         "## Train OOF (selection data)", "", "| model | OOF MAE | selected wd | epochs | seed-check MAEs |", "|---|---|---|---|---|"]
    for k in ("ridge", "natural", "fixed"):
        L.append(f"| {LABEL[k]} | {f(o['oof_mae'][k], 4)} | — | — | — |")
    for k in ("magnitude", "dynamic"):
        s = o["selection"][k]
        L.append(f"| {LABEL[k]} | {f(s['oof_mae'], 4)} | {s['selected_weight_decay']:g} | {s['selected_epochs']} | "
                 + ", ".join(f(x, 4) for x in s["seed_check_oof_mae"]["oof_mae"]) + " |")
    L += ["", "## Validation", "", "| | " + " | ".join(LABEL[k] for k in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    ov = v["overall"]
    rows = [("MAE", lambda m: m["mae"], 4), ("RMSE", lambda m: m["rmse"], 4), ("median abs", lambda m: m["median_absolute_error"], 3),
            ("mean signed", lambda m: m["mean_signed_error"], 3)]
    rows += [(q.upper(), (lambda q: lambda m: m["quantiles"][q])(q), 2) for q in ("p75", "p90", "p95", "p99", "p99.5", "p99.9")]
    rows += [("max", lambda m: m["max"], 2)]
    for lab, fn, d in rows:
        L.append(f"| {lab} | " + " | ".join(f(fn(ov[k]), d) for k in MODELS) + " |")
    for t in THRESHOLDS:
        L.append(f"| > {t} ¢ | " + " | ".join(str(ov[k]["exceeding"][f"gt_{t}_cents"]) for k in MODELS) + " |")
    sl = v["prediction_slope"]
    L.append("| slope | " + " | ".join(f(sl[k]["slope"]) for k in MODELS) + " |")
    L += [""]
    for k in ("dynamic", "magnitude"):
        x = pq[f"{k}_gate"]
        L.append(f"**{LABEL[k]} vs fixed:** MAE {f(pq['fixed_hybrid']['mae'], 4)} → {f(x['mae'], 4)} "
                 f"({x['mae_improvement']:+.4f} c, {x['mae_improvement_percent']:+.2f}%); RMSE {f(pq['fixed_hybrid']['rmse'], 4)} → "
                 f"{f(x['rmse'], 4)} ({x['rmse_improvement']:+.4f} c, {x['rmse_improvement_percent']:+.2f}%). "
                 f"Beats 12.5828: **{x['beats_fixed_mae_benchmark_12_5828']}**. Positive = gate better.")
        L.append("")
    L += ["## By variant (MAE / RMSE)", "", "| variant | " + " | ".join(LABEL[k] for k in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for name in EXPECTED_VARIANT_ORDER:
        x = v["by_variant"][name]
        L.append(f"| {name} | " + " | ".join(f"{f(x[k]['mae'])} / {f(x[k]['rmse'])}" for k in MODELS) + " |")
    L += ["", "## Prediction slope (corrupted validation)", "", "| | slope | intercept | R² |", "|---|---|---|---|"]
    for k in MODELS:
        L.append(f"| {LABEL[k]} | {f(sl[k]['slope'])} | {sl[k]['intercept']:+.3f} | {f(sl[k]['r2'])} |")
    L += ["", "## Ridge-weight distribution (validation)", "", "| gate | mean | std | P10 | P25 | P50 | P75 | P90 | min | max |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("magnitude", "dynamic"):
        w = v["weight_distribution"][k]
        L.append(f"| {LABEL[k]} | " + " | ".join(f(w[c]) for c in ("mean", "std", "p10", "p25", "p50", "p75", "p90", "min", "max")) + " |")
    L += ["", "**Mean Ridge weight by variant** (analysis only; variant is not a gate input)", "",
          "| variant | Magnitude gate | Dynamic gate |", "|---|---|---|"]
    for name in EXPECTED_VARIANT_ORDER:
        L.append(f"| {name} | {f(v['mean_ridge_weight_by_variant']['magnitude'][name])} | {f(v['mean_ridge_weight_by_variant']['dynamic'][name])} |")
    for title, key in (("By |observed core median| (post hoc)", "by_abs_observed_core_median"),
                       ("By expert disagreement |ridge − natural| (post hoc)", "by_abs_ridge_minus_natural")):
        L += ["", f"## {title}", "", "| bucket | n | Ridge | Natural | Fixed | Magnitude | Dynamic | w̄ magnitude | w̄ dynamic |",
              "|---|---|---|---|---|---|---|---|---|"]
        for lab, x in v[key].items():
            if not x["count"]:
                L.append(f"| {lab} | 0 | | | | | | | |")
                continue
            L.append(f"| {lab} | {x['count']} | " + " | ".join(f(x[f'{k}_mae']) for k in MODELS) + " | "
                     f"{f(x['magnitude_mean_ridge_weight'])} | {f(x['dynamic_mean_ridge_weight'])} |")
    L += ["", "## Corrupted examples by sign", "", "| sign | model | n | MAE | RMSE | mean signed | w̄ ridge |", "|---|---|---|---|---|---|---|"]
    for sgn in ("positive", "negative"):
        for k in MODELS:
            x = v["by_sign_corrupted"][sgn][k]
            L.append(f"| {sgn} | {LABEL[k]} | {x['count']} | {f(x['mae'])} | {f(x['rmse'])} | {x['mean_signed_error']:+.3f} | {f(x['mean_ridge_weight'])} |")
    ng = v["note_groups"]
    L += ["", "## Note groups", "", f"{ng['note_groups']} validation notes; per note, mean |error| over its five variants; "
          f"tie = within {ng['tie_tolerance_cents']} c.", "", "| comparison | A better | tie | B better | median gain (A wins) | median loss (B wins) |",
          "|---|---|---|---|---|---|"]
    for key, a, b in (("dynamic_vs_fixed", "dynamic", "fixed"), ("dynamic_vs_ridge", "dynamic", "ridge"),
                      ("magnitude_vs_fixed", "magnitude", "fixed")):
        x = ng[key]
        L.append(f"| {LABEL[a]} (A) vs {LABEL[b]} (B) | {x[f'{a}_better_percent']:.1f}% | {x['tie_percent']:.1f}% | "
                 f"{x[f'{b}_better_percent']:.1f}% | {f(x[f'median_gain_when_{a}_wins'])} | {f(x[f'median_loss_when_{b}_wins'])} |")
    ig = r["integrity"]
    L += ["", "## Integrity", "",
          f"- Experts frozen and reproduced: fold-MAE max diff Ridge {ig['expert_cv_reproduced']['ridge_max_fold_mae_diff']:.1e}, "
          f"natural {ig['expert_cv_reproduced']['natural_max_fold_mae_diff']:.1e}; fixed-hybrid OOF MAE diff "
          f"{ig['expert_cv_reproduced']['fixed_hybrid_oof_mae_diff']:.1e}.",
          f"- Validation reproduced: Ridge {ig['validation_reproduced']['ridge_max_mae_rmse_diff']:.1e}, natural "
          f"{ig['validation_reproduced']['natural_max_mae_rmse_diff']:.1e}, fixed {ig['validation_reproduced']['fixed_max_mae_rmse_diff']:.1e}; "
          f"per-example vs stored fixed-hybrid CSV {ig['validation_reproduced']['max_per_example_diff_vs_stored_fixed_hybrid_csv']:.1e}.",
          f"- Gate metadata inputs identical across a note's five variants (max diff {ig['gate_metadata_inputs_identical_across_variants_max_diff']}).",
          "- No sequence model, no third expert. Validation used once, never for selection.",
          f"- Test examples featurized: {ig['test_examples_featurized']}; test notes files opened: {ig['test_song_note_files_opened']}.",
          f"- Protected files unchanged: {ig['protected_unchanged']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
