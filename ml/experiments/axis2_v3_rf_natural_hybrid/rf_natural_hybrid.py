#!/usr/bin/env python3
"""
Axis 2 V3: Random Forest + natural-offset hybrids (Ridge replaced by the frozen Random Forest).

    A  fixed     prediction = w * RF + (1 - w) * natural,        w in {0.00, ..., 1.00} from train OOF MAE
    B  dynamic   prediction = w(x) * RF + (1 - w(x)) * natural,  w(x) = gate(x) in [0, 1]

Experts are frozen, not re-tuned:
  RF       the selected V3 Random Forest (ml/experiments/axis2_v3_random_forest): 300 trees, max_depth 20,
           min_samples_leaf 10, max_features 0.5, random_state 20260824, on Ridge's 24 features
           (ridge_baseline.build_matrix). The full-train model is the saved random_forest_model.joblib;
           its validation predictions are checked against that experiment's stored CSV.
  natural  the V3 natural-offset HGB (ALL 71 features, hgb_leaf31_lr05), fit on clean train notes;
           S_hat = observed core median - N_hat.

OOF protocol: one GroupKFold(5)-by-song partition over all train examples (the partition both
previous V3 experiments used), so a note's five variants always share a fold. Both experts'
OOF predictions are REGENERATED here; the RF's are then compared with the RF experiment's cached
fold predictions and stored per-fold MAEs, and the natural model's with its stored fold MAEs.

Gate: the V2 dynamic-gate module is imported unchanged (Gate, train_gate, apply_gate, gate_inputs,
weight-decay grid, epochs, seeds, batch size, learning rate, L1 loss). Its expert slot formerly
holding the Ridge prediction now holds the RF prediction; the 20 inputs are otherwise identical.
Weight decay and epochs are selected on pooled train-OOF MAE exactly as before.

Validation is predicted once, after every train-only decision. The test split is sealed.

  python ml/experiments/axis2_v3_rf_natural_hybrid/rf_natural_hybrid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axis2_v3_shared"))

import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

import frozen_experts as FE  # noqa: E402
import dynamic_hybrid as DH  # noqa: E402   (V2 gate module, imported unchanged)

H, NAT, RB = FE.H, FE.NAT, FE.RB
REPO = FE.REPO
OUT = Path(__file__).resolve().parent
CACHE = Path("/tmp/axis2_v3_rf_natural_hybrid_cache")      # resumable expert cache, outside the repo
RF_DIR = REPO / "ml/experiments/axis2_v3_random_forest"
RF_MODEL = RF_DIR / "random_forest_model.joblib"
RF_CV_CACHE = Path("/tmp/axis2_v3_rf_cv_cache")
NAT_RESULTS = REPO / "ml/experiments/axis2_v3_natural_offset/natural_offset_results.json"
NAT_PREDICTIONS = REPO / "ml/experiments/axis2_v3_natural_offset/validation_predictions.csv"
OLD_DYN_RESULTS = REPO / "ml/experiments/axis2_v3_dynamic_hybrid/dynamic_hybrid_results.json"
OLD_DYN_PREDICTIONS = REPO / "ml/experiments/axis2_v3_dynamic_hybrid/gate_validation_predictions.csv"

GATE_FEATURES = tuple(f.replace("ridge", "rf") for f in DH.GATE_FEATURES)   # same columns, same order
CORE_BUCKETS = (("0-10", 0, 10), ("10-20", 10, 20), ("20-30", 20, 30), ("30-40", 30, 40), ("40-49.999", 40, None))
DISAGREE_BUCKETS = DH.DISAGREE_BUCKETS
NOTE_TIE = 0.5
MODELS = ("rf", "natural", "ridge", "old_fixed", "old_dynamic", "new_fixed", "new_dynamic")
LABEL = {"rf": "Random Forest", "natural": "Natural offset", "ridge": "Ridge", "old_fixed": "Old Ridge+Natural fixed",
         "old_dynamic": "Old Ridge+Natural dynamic", "new_fixed": "New RF+Natural fixed", "new_dynamic": "New RF+Natural dynamic",
         "oracle": "Oracle (min of RF, Natural)"}
OUTPUTS = ("rf_natural_hybrid_results.json", "rf_natural_hybrid_summary.md", "train_oof_predictions.csv", "validation_predictions.csv",
           "fixed_weight_sweep.csv", "dynamic_gate_config.json", "latency_benchmark.json", "dynamic_gate.pt", "natural_offset_model.joblib")
FORBIDDEN = DH.FORBIDDEN_GATE_INPUTS + ("natural_offset_true", "N_true")
OLD_CHAMPION = {"mae": 13.4557, "rmse": 17.3976}


def cached(name: str, fn):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}.npy"
    if f.exists():
        return np.load(f)
    a = fn()
    np.save(f, a)
    return a


def short(e):
    return DH.short(e)


def main() -> int:
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    for name in OUTPUTS:
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; refusing to overwrite")
    for f in GATE_FEATURES:
        if any(bad in f for bad in FORBIDDEN):
            raise RuntimeError(f"forbidden gate input {f}")
    if len(GATE_FEATURES) != 20:
        raise RuntimeError("gate feature count differs from the V2 gate")
    protected = FE.protected_paths(("ml/results/axis2_v3", "ml/experiments/axis2_v3_natural_offset", "ml/experiments/axis2_v3_hybrid",
                                    "ml/experiments/axis2_v3_dynamic_hybrid", "ml/experiments/axis2_v3_shared",
                                    "ml/experiments/axis2_v3_random_forest"))
    hashes_before = NAT._hash_tree(protected)

    # ---- frozen definitions
    rf_cfg = json.loads((RF_DIR / "random_forest_config.json").read_text())
    rf_res = json.loads((RF_DIR / "random_forest_results.json").read_text())
    if rf_res["test_evaluated"] is not False or tuple(rf_cfg["features"]) != RB.FEATURES:
        raise RuntimeError("RF experiment is not test-sealed or used different features")
    RF_PARAMS = {k: rf_cfg[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "random_state")}
    rf_model = joblib.load(RF_MODEL)
    if {k: rf_model.get_params()[k] for k in RF_PARAMS} != RF_PARAMS:
        raise RuntimeError("saved RF model parameters differ from the selected configuration")
    rf_sel = rf_res["selection"]["full24"]["selected"]
    _, nat_cfg = H.frozen_configs()
    nat_res = json.loads(NAT_RESULTS.read_text())
    if nat_res["test_evaluated"] is not False or nat_res["selected_model"]["param"] != nat_cfg["param"]:
        raise RuntimeError("V3 natural result differs from the frozen definition")
    nat_cv = nat_res["cv_results"][0]["fold_mae"]

    d = FE.load("axis2_v3")
    D, vc = d["D"], d["variant_codes"]
    tr, va = d["tr"], d["va"]
    y_tr, y_va = d["y"][tr], d["y"][va]
    print("featurizing", flush=True)
    Xr_tr, Xr_va = FE.featurize_ridge(d, tr), FE.featurize_ridge(d, va)
    Xfull_tr, core_tr = FE.featurize_natural(d, tr)
    Xfull_va, core_va = FE.featurize_natural(d, va)
    cols = nat_cfg["columns"]
    Xn_tr, Xn_va = Xfull_tr[:, cols], Xfull_va[:, cols]
    var_tr, var_va = D["variant_index"][tr], D["variant_index"][va]
    clean_tr = var_tr == vc["clean"]
    groups = D["song_index"][tr]
    folds, fold_of = FE.shared_folds(groups, tr.size)
    gk_tr = groups.astype(np.int64) * 1_000_000 + D["source_note_index"][tr]
    siblings_split = int(sum(len(set(gk_tr[a]) & set(gk_tr[b])) for a, b in folds))
    if siblings_split:
        raise RuntimeError("a note's variants straddle a fold")

    # ---- OOF expert predictions, regenerated on the shared song partition
    print("expert OOF predictions (regenerated)", flush=True)
    rf_oof, nhat_oof = np.full(tr.size, np.nan), np.full(tr.size, np.nan)
    rf_fold_mae, nat_fold_mae, cache_match = [], [], []
    for k, (fit, held) in enumerate(folds):
        p = cached(f"rf_oof_fold{k}", lambda: RandomForestRegressor(n_jobs=-1, **RF_PARAMS).fit(Xr_tr[fit], y_tr[fit]).predict(Xr_tr[held]))
        rf_oof[held] = p
        rf_fold_mae.append(float(np.mean(np.abs(p - y_tr[held]))))
        cf = RF_CV_CACHE / f"full24__depth-20_leaf-10_mf-0.5__fold{k}.npz"
        if cf.exists():
            with np.load(cf) as z:
                cache_match.append(float(np.max(np.abs(z["pred"] - p))) if z["pred"].shape == p.shape else float("inf"))
        fc = np.zeros(tr.size, bool); fc[fit] = True; fc &= clean_tr
        hc = np.zeros(tr.size, bool); hc[held] = True
        nhat_oof[held] = cached(f"nat_oof_fold{k}", lambda: H.natural_fit_predict(nat_cfg["param"], Xn_tr[fc], core_tr[fc], Xn_tr[held]))
        nat_fold_mae.append(float(np.mean(np.abs(nhat_oof[hc & clean_tr] - core_tr[hc & clean_tr]))))
    nat_oof = core_tr - nhat_oof
    if np.isnan(rf_oof).any() or np.isnan(nat_oof).any():
        raise RuntimeError("missing OOF predictions")
    rf_fold_diff = float(np.max(np.abs(np.array(rf_fold_mae) - np.array(rf_sel["fold_mae"]))))
    nat_fold_diff = float(np.max(np.abs(np.array(nat_fold_mae) - np.array(nat_cv))))
    if max(rf_fold_diff, nat_fold_diff) > 1e-9:
        raise RuntimeError(f"OOF experts do not reproduce stored fold MAEs: RF {rf_fold_diff}, natural {nat_fold_diff}")

    # ---- validation expert predictions (frozen full-train models)
    print("validation expert predictions", flush=True)
    rf_va = rf_model.predict(Xr_va)
    nat_model = NAT.make_model("hgb", nat_cfg["param"]).fit(Xn_tr[clean_tr], core_tr[clean_tr])
    nat_va = core_va - nat_model.predict(Xn_va)
    keys_va = FE.keys(d, va)
    rf_stored = {(r["song_id"], int(r["source_note_index"]), r["variant"]): float(r["random_forest_prediction"])
                 for r in csv.DictReader((RF_DIR / "validation_predictions.csv").open())}
    nat_stored = {(r["song_id"], int(r["source_note_index"]), r["variant"]): float(r["predicted_shift"]) for r in csv.DictReader(NAT_PREDICTIONS.open())}
    old = {(r["song_id"], int(r["source_note_index"]), r["variant"]): r for r in csv.DictReader(OLD_DYN_PREDICTIONS.open())}
    if not (set(keys_va) == rf_stored.keys() == nat_stored.keys() == old.keys()):
        raise RuntimeError("validation keys differ from stored predictions")
    rf_csv_diff = max(abs(rf_stored[k] - rf_va[j]) for j, k in enumerate(keys_va))
    nat_csv_diff = max(abs(nat_stored[k] - nat_va[j]) for j, k in enumerate(keys_va))
    if max(rf_csv_diff, nat_csv_diff) > FE.CSV_TOLERANCE:
        raise RuntimeError(f"validation experts do not reproduce stored predictions: RF {rf_csv_diff}, natural {nat_csv_diff}")
    oc = lambda c: np.array([float(old[k][c]) for k in keys_va])  # noqa: E731

    # ---- A: fixed blend on train OOF
    sweep = []
    for w in H.WEIGHTS:
        pr = w * rf_oof + (1 - w) * nat_oof
        e = pr - y_tr
        sweep.append({"weight_rf": w, "weight_natural": round(1 - w, 2), "oof_mae": float(np.mean(np.abs(e))),
                      "oof_rmse": float(np.sqrt(np.mean(e ** 2))), "slope_corrupted": NAT.slope_stats(y_tr[~clean_tr], pr[~clean_tr])["slope"]})
    best = min(r["oof_mae"] for r in sweep)
    tied = [r for r in sweep if r["oof_mae"] - best <= H.TIE_TOLERANCE_CENTS]
    chosen = min(tied, key=lambda r: (abs(r["weight_rf"] - 0.5), r["weight_rf"]))
    W = chosen["weight_rf"]
    print(f"fixed: w_rf = {W}", flush=True)
    fixed_oof = W * rf_oof + (1 - W) * nat_oof
    with (OUT / "fixed_weight_sweep.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sweep[0]))
        wr.writeheader()
        for r in sweep:
            wr.writerow({k: (f"{v:.6f}" if k not in ("weight_rf", "weight_natural") else v) for k, v in r.items()})

    # ---- B: dynamic gate (V2 procedure, RF in the expert slot)
    G_tr = DH.gate_inputs(rf_oof, nat_oof, core_tr, Xfull_tr)
    meta_cols = [DH.GATE_FEATURES.index(f) for f in DH.META_FEATURES]
    M = G_tr[np.lexsort((var_tr, gk_tr))][:, meta_cols].reshape(-1, 5, len(meta_cols))
    meta_spread = float(np.max(np.abs(M - M[:, :1, :])))
    if meta_spread != 0.0:
        raise RuntimeError("metadata gate inputs differ across a note's variants")
    curves, traces = {}, {}
    for wd in DH.WEIGHT_DECAYS:
        print(f"gate CV wd={wd}", flush=True)
        tw = np.empty((DH.MAX_EPOCHS, tr.size), np.float32)
        for k in range(FE.CV_FOLDS):
            fit, held = fold_of != k, fold_of == k
            _, _, trace = DH.train_gate("dynamic", G_tr[fit], rf_oof[fit], nat_oof[fit], y_tr[fit], wd, DH.MAX_EPOCHS, DH.SEED + k, X_eval=G_tr[held])
            tw[:, held] = trace
        pr = tw.astype(np.float64) * rf_oof + (1 - tw.astype(np.float64)) * nat_oof
        curves[wd] = np.mean(np.abs(pr - y_tr), axis=1)
        traces[wd] = tw
    cands = [(float(curves[wd][e]), e + 1, wd) for wd in DH.WEIGHT_DECAYS for e in range(DH.MAX_EPOCHS)]
    g_mae, g_ep, g_wd = min(cands)
    w_oof = traces[g_wd][g_ep - 1].astype(np.float64)
    dyn_oof = w_oof * rf_oof + (1 - w_oof) * nat_oof
    seed_maes = [g_mae]
    for sd in DH.SEED_CHECK:
        ww = np.empty(tr.size)
        for k in range(FE.CV_FOLDS):
            fit, held = fold_of != k, fold_of == k
            m, sc, _ = DH.train_gate("dynamic", G_tr[fit], rf_oof[fit], nat_oof[fit], y_tr[fit], g_wd, g_ep, sd + k)
            ww[held] = DH.apply_gate(m, sc, G_tr[held])
        seed_maes.append(float(np.mean(np.abs(ww * rf_oof + (1 - ww) * nat_oof - y_tr))))
    print(f"gate: wd={g_wd} epochs={g_ep} OOF MAE {g_mae:.4f}", flush=True)
    gate, gsc, _ = DH.train_gate("dynamic", G_tr, rf_oof, nat_oof, y_tr, g_wd, g_ep, DH.SEED + FE.CV_FOLDS)

    # ---- single validation pass
    G_va = DH.gate_inputs(rf_va, nat_va, core_va, Xfull_va)
    w_va = DH.apply_gate(gate, gsc, G_va)
    preds = {"rf": rf_va, "natural": nat_va, "ridge": oc("ridge_prediction"), "old_fixed": oc("fixed_hybrid_prediction"),
             "old_dynamic": oc("dynamic_gate_prediction"), "new_fixed": W * rf_va + (1 - W) * nat_va,
             "new_dynamic": w_va * rf_va + (1 - w_va) * nat_va}
    errs = {k: p - y_va for k, p in preds.items()}
    old_res = json.loads(OLD_DYN_RESULTS.read_text())["validation"]["overall"]
    old_check = max(abs(float(np.mean(np.abs(errs[a]))) - old_res[b]["mae"]) for a, b in
                    (("ridge", "ridge"), ("old_fixed", "fixed"), ("old_dynamic", "dynamic")))
    if old_check > 1e-5:
        raise RuntimeError(f"stored old-hybrid predictions do not reproduce their metrics ({old_check})")

    # ---- analysis (post hoc; nothing below feeds back)
    corr = var_va != vc["clean"]
    abs_core = np.abs(core_va)
    if np.any(abs_core >= 49.999):
        raise RuntimeError("validation example at or beyond the target boundary")
    disagree = np.abs(rf_va - nat_va)
    ae_rf, ae_nat = np.abs(errs["rf"]), np.abs(errs["natural"])
    oracle = np.where(ae_rf <= ae_nat, rf_va, nat_va)
    errs_all = {**errs, "oracle": oracle - y_va}
    overall = {k: FE.metrics(e) for k, e in errs_all.items()}
    by_variant = {v: {k: short(errs_all[k][var_va == c]) for k in errs_all} for v, c in vc.items()}
    w_by_variant = {v: float(w_va[var_va == c].mean()) for v, c in vc.items()}
    slopes = {k: NAT.slope_stats(y_va[corr], (preds[k] if k in preds else oracle)[corr]) for k in ("rf", "natural", "ridge", "old_dynamic", "new_fixed", "new_dynamic")}
    by_sign = {s: {**{k: short(errs[k][m]) for k in ("rf", "natural", "old_dynamic", "new_fixed", "new_dynamic")},
                   "new_dynamic_mean_rf_weight": float(w_va[m].mean())}
               for s, m in (("positive", corr & (y_va > 0)), ("negative", corr & (y_va < 0)))}

    def bucket_table(values, buckets):
        out = {}
        for lab, lo, hi in buckets:
            m = DH.bucket(values, lo, hi)
            out[lab] = {"count": int(m.sum()), **{f"{k}_mae": float(np.mean(np.abs(errs[k][m]))) for k in ("rf", "natural", "new_fixed", "new_dynamic", "old_dynamic")},
                        "mean_rf_weight": float(w_va[m].mean())}
        return out

    by_disagree = bucket_table(disagree, DISAGREE_BUCKETS)
    by_core = bucket_table(abs_core, CORE_BUCKETS)
    rf_better = ae_rf < ae_nat
    nat_better = ae_nat < ae_rf
    pick_rf = w_va > 0.5
    specialization = {
        "examples_rf_lower_error": int(rf_better.sum()), "examples_natural_lower_error": int(nat_better.sum()),
        "mean_rf_weight_where_rf_lower_error": float(w_va[rf_better].mean()),
        "mean_rf_weight_where_natural_lower_error": float(w_va[nat_better].mean()),
        "higher_weighted_expert_is_lower_error_fraction": float(np.mean(np.where(pick_rf, ae_rf <= ae_nat, ae_nat <= ae_rf))),
        "chance_references": {"always_rf": float(np.mean(ae_rf <= ae_nat)), "always_natural": float(np.mean(ae_nat <= ae_rf)),
                              "random_with_gate_pick_rate": float(pick_rf.mean() * np.mean(ae_rf <= ae_nat) + (1 - pick_rf.mean()) * np.mean(ae_nat <= ae_rf))},
        "gate_picks_rf_fraction": float(pick_rf.mean()),
        "note": "uses true validation errors for interpretation only, after all predictions were frozen",
    }
    oracle_ref = {"mae": overall["oracle"]["mae"], "rmse": overall["oracle"]["rmse"],
                  "gap_dynamic_minus_oracle_mae": overall["new_dynamic"]["mae"] - overall["oracle"]["mae"],
                  "gap_fixed_minus_oracle_mae": overall["new_fixed"]["mae"] - overall["oracle"]["mae"],
                  "share_of_fixed_to_oracle_gap_closed_by_dynamic": (overall["new_fixed"]["mae"] - overall["new_dynamic"]["mae"])
                  / (overall["new_fixed"]["mae"] - overall["oracle"]["mae"]),
                  "note": "per-example choice of the lower-error expert using the true label; not deployable"}

    gkey = D["song_index"][va].astype(np.int64) * 1_000_000 + D["source_note_index"][va]
    _, inv, counts = np.unique(gkey, return_inverse=True, return_counts=True)
    if not np.all(counts == 5):
        raise RuntimeError("validation note groups are not all five variants")
    gmae = {k: np.bincount(inv, weights=np.abs(e)) / counts for k, e in errs.items()}

    def gcmp(other):
        gain = gmae[other] - gmae["new_dynamic"]
        win, loss = gain > NOTE_TIE, gain < -NOTE_TIE
        return {"new_dynamic_better_percent": 100 * float(win.mean()), "tie_percent": 100 * float((~win & ~loss).mean()),
                f"{other}_better_percent": 100 * float(loss.mean()),
                "median_gain_when_new_wins": float(np.median(gain[win])) if win.any() else None,
                "median_loss_when_other_wins": float(np.median(-gain[loss])) if loss.any() else None, "mean_gain": float(gain.mean())}

    note_groups = {"note_groups": int(counts.size), "tie_tolerance_cents": NOTE_TIE,
                   **{f"new_dynamic_vs_{o}": gcmp(o) for o in ("old_dynamic", "rf", "natural", "new_fixed")}}
    fx, dy = overall["new_fixed"], overall["new_dynamic"]
    decision = {"fixed_mae": fx["mae"], "dynamic_mae": dy["mae"], "absolute_improvement": fx["mae"] - dy["mae"],
                "percent_improvement": 100 * (fx["mae"] - dy["mae"]) / fx["mae"], "fixed_rmse": fx["rmse"], "dynamic_rmse": dy["rmse"],
                "rmse_improvement": fx["rmse"] - dy["rmse"], "rmse_percent_improvement": 100 * (fx["rmse"] - dy["rmse"]) / fx["rmse"]}
    vs_champion = {"old_champion": OLD_CHAMPION, "new_dynamic": {"mae": dy["mae"], "rmse": dy["rmse"]},
                   "mae_improvement": OLD_CHAMPION["mae"] - dy["mae"], "mae_percent": 100 * (OLD_CHAMPION["mae"] - dy["mae"]) / OLD_CHAMPION["mae"],
                   "rmse_improvement": OLD_CHAMPION["rmse"] - dy["rmse"], "rmse_percent": 100 * (OLD_CHAMPION["rmse"] - dy["rmse"]) / OLD_CHAMPION["rmse"],
                   "exact_old_champion_from_stored": {"mae": overall["old_dynamic"]["mae"], "rmse": overall["old_dynamic"]["rmse"]},
                   "beats_old_champion": dy["mae"] < overall["old_dynamic"]["mae"]}

    # ---- artifacts
    torch.save({"state_dict": gate.state_dict(), "scaler_mean": torch.from_numpy(gsc["mean"]), "scaler_std": torch.from_numpy(gsc["std"]),
                "features": list(GATE_FEATURES)}, OUT / "dynamic_gate.pt")
    joblib.dump(nat_model, OUT / "natural_offset_model.joblib", compress=3)
    gate_config = {"features": list(GATE_FEATURES), "v2_gate_feature_names": list(DH.GATE_FEATURES),
                   "substitution": "the V2 gate's ridge_* inputs carry the RF prediction; computation, order and metadata columns unchanged",
                   "architecture": f"Linear(20,{DH.HIDDEN[0]}) -> GELU -> Linear({DH.HIDDEN[0]},{DH.HIDDEN[1]}) -> GELU -> Linear({DH.HIDDEN[1]},1) -> sigmoid",
                   "parameters": DH.n_params(gate), "output": "w = weight on the RF prediction",
                   "training": {"loss": "L1 of w*rf + (1-w)*natural", "optimizer": "Adam", "learning_rate": DH.LEARNING_RATE,
                                "weight_decay_grid": list(DH.WEIGHT_DECAYS), "batch_size": DH.BATCH_SIZE, "max_epochs": DH.MAX_EPOCHS,
                                "seed": DH.SEED, "selected_weight_decay": g_wd, "selected_epochs": g_ep, "dtype": "float64",
                                "input_standardisation": "z-score fit on the gate's own fit rows"},
                   "changes_from_v2_gate": "none beyond the expert substitution"}
    (OUT / "dynamic_gate_config.json").write_text(json.dumps(gate_config, indent=2) + "\n")

    keys_tr = FE.keys(d, tr)
    with (OUT / "train_oof_predictions.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["song_id", "source_note_index", "variant", "fold", "true_shift", "observed_core_median", "rf_oof_prediction",
                     "natural_oof_prediction", "fixed_oof_prediction", "dynamic_oof_rf_weight", "dynamic_oof_prediction"])
        for j, k in enumerate(keys_tr):
            wr.writerow([k[0], k[1], k[2], int(fold_of[j]), f"{y_tr[j]:.6f}", f"{core_tr[j]:.6f}", f"{rf_oof[j]:.6f}", f"{nat_oof[j]:.6f}",
                         f"{fixed_oof[j]:.6f}", f"{w_oof[j]:.6f}", f"{dyn_oof[j]:.6f}"])
    with (OUT / "validation_predictions.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["song_id", "source_note_index", "variant", "true_shift", "observed_core_median", "rf_prediction", "natural_prediction",
                     "old_ridge_prediction", "old_fixed_prediction", "old_dynamic_prediction", "new_fixed_prediction", "new_dynamic_prediction",
                     "dynamic_rf_weight", "oracle_prediction"])
        for j, k in enumerate(keys_va):
            wr.writerow([k[0], k[1], k[2], f"{y_va[j]:.6f}", f"{core_va[j]:.6f}"] +
                        [f"{preds[m][j]:.6f}" for m in ("rf", "natural", "ridge", "old_fixed", "old_dynamic", "new_fixed", "new_dynamic")] +
                        [f"{w_va[j]:.6f}", f"{oracle[j]:.6f}"])

    # ---- latency (descriptive; trained models only, loading excluded)
    print("latency benchmark", flush=True)
    latency = benchmark(rf_model, nat_model, gate, gsc, W, Xr_va, Xn_va, Xfull_va, core_va, d, va)

    hashes_after = NAT._hash_tree(protected)
    results = {
        "experiment": "axis2_v3_rf_natural_hybrid", "test_evaluated": False, "dataset": "data/ml_dataset/axis2_v3",
        "experts": {"rf": {**RF_PARAMS, "features": list(RB.FEATURES), "source": FE.rel(RF_DIR), "full_train_model": FE.rel(RF_MODEL)},
                    "natural": {"feature_set": nat_cfg["feature_set"], "param": nat_cfg["param"], "n_features": len(cols),
                                "source": FE.rel(NAT_RESULTS.parent)},
                    "ridge_in_new_hybrids": False, "third_expert": None},
        "oof": {"protocol": "GroupKFold(5) by song over all train examples; RF refit per fold on all variants of the fit songs; "
                            "natural HGB refit per fold on the fit songs' clean notes",
                "rf": "regenerated", "natural": "regenerated",
                "rf_fold_mae": rf_fold_mae, "rf_fold_mae_max_diff_vs_rf_experiment": rf_fold_diff,
                "rf_max_abs_diff_vs_rf_experiment_cached_fold_predictions": max(cache_match) if cache_match else None,
                "rf_cached_folds_compared": len(cache_match),
                "natural_fold_mae": nat_fold_mae, "natural_fold_mae_max_diff_vs_natural_experiment": nat_fold_diff,
                "note_groups_straddling_a_fold": siblings_split, "train_rows": int(tr.size), "duplicates": 0,
                "validation_or_test_rows_in_oof": 0},
        "fixed": {"selected_weight_rf": W, "selected_weight_natural": round(1 - W, 2), "tie_rule": "within 1e-9 c -> nearest 0.5",
                  "tied": [r["weight_rf"] for r in tied],
                  "train_oof_mae": {"rf": float(np.mean(np.abs(rf_oof - y_tr))), "natural": float(np.mean(np.abs(nat_oof - y_tr))),
                                    "fixed": float(np.mean(np.abs(fixed_oof - y_tr))), "dynamic": g_mae},
                  "train_oof_rmse": {"rf": float(np.sqrt(np.mean((rf_oof - y_tr) ** 2))), "natural": float(np.sqrt(np.mean((nat_oof - y_tr) ** 2))),
                                     "fixed": float(np.sqrt(np.mean((fixed_oof - y_tr) ** 2))), "dynamic": float(np.sqrt(np.mean((dyn_oof - y_tr) ** 2)))}},
        "dynamic": {"selected_weight_decay": g_wd, "selected_epochs": g_ep, "oof_mae": g_mae, "seed_check_oof_mae": seed_maes,
                    "oof_mae_curve": {str(k): [float(x) for x in v] for k, v in curves.items()}, "parameters": DH.n_params(gate),
                    "metadata_inputs_identical_across_variants_max_diff": meta_spread},
        "validation": {"overall": overall, "by_variant": by_variant, "prediction_slope_corrupted": slopes, "by_sign_corrupted": by_sign,
                       "dynamic_rf_weight_distribution": DH.wdist(w_va), "dynamic_rf_weight_by_variant": w_by_variant,
                       "by_abs_rf_minus_natural": by_disagree, "by_abs_observed_core": by_core, "note_groups": note_groups,
                       "specialization": specialization, "oracle": oracle_ref, "fixed_vs_dynamic": decision, "vs_old_champion": vs_champion,
                       "post_hoc_only": True},
        "latency": latency,
        "integrity": {"validation_used_for_selection": False, "validation_prediction_passes": 1, "test_examples_featurized": 0,
                      "test_predictions": 0, "test_song_note_files_opened": FE.test_songs_opened(d),
                      "rf_validation_max_diff_vs_stored_csv": rf_csv_diff, "natural_validation_max_diff_vs_stored_csv": nat_csv_diff,
                      "old_hybrid_predictions_reproduce_metrics_max_diff": old_check,
                      "protected_paths": [FE.rel(p) for p in protected], "protected_hashes_before": hashes_before,
                      "protected_hashes_after": hashes_after, "protected_unchanged": hashes_before == hashes_after, "runtime_s": time.time() - t0},
    }
    if results["integrity"]["test_song_note_files_opened"]:
        raise RuntimeError("a test song's notes were opened")
    (OUT / "latency_benchmark.json").write_text(json.dumps(latency, indent=2) + "\n")
    (OUT / "rf_natural_hybrid_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (OUT / "rf_natural_hybrid_summary.md").write_text(summary_md(results))
    print(json.dumps({"w_rf": W, "fixed": {k: fx[k] for k in ("mae", "rmse")}, "dynamic": {k: dy[k] for k in ("mae", "rmse")},
                      "oracle": overall["oracle"]["mae"], "protected_unchanged": hashes_before == hashes_after}, indent=1))
    return 0


def _median_time(fn, reps: int) -> float:
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def benchmark(rf, nat, gate, gsc, W, Xr, Xn, Xfull, core, d, va) -> Dict[str, Any]:
    reps = {1: 200, 100: 50, 1000: 20}
    out: Dict[str, Any] = {"note": "median wall time of already-trained inference on precomputed features; model loading excluded; "
                                   "RF uses its stored n_jobs=-1 (process pool start-up dominates single-note calls). Descriptive only.",
                           "machine_cpu_count": __import__("os").cpu_count(), "model_inference_seconds": {}, "feature_extraction_seconds": {}}
    for n, r in reps.items():
        sl = slice(0, n)
        rf_only = lambda: rf.predict(Xr[sl])  # noqa: E731
        nat_only = lambda: core[sl] - nat.predict(Xn[sl])  # noqa: E731
        fixed = lambda: W * rf.predict(Xr[sl]) + (1 - W) * (core[sl] - nat.predict(Xn[sl]))  # noqa: E731

        def dyn():
            a, b = rf.predict(Xr[sl]), core[sl] - nat.predict(Xn[sl])
            w = DH.apply_gate(gate, gsc, DH.gate_inputs(a, b, core[sl], Xfull[sl]))
            return w * a + (1 - w) * b
        out["model_inference_seconds"][str(n)] = {"rf": _median_time(rf_only, r), "natural": _median_time(nat_only, r),
                                                  "fixed_hybrid": _median_time(fixed, r), "dynamic_hybrid": _median_time(dyn, r)}
        rf1 = rf.set_params(n_jobs=1)
        out["model_inference_seconds"][str(n)]["rf_single_thread"] = _median_time(lambda: rf1.predict(Xr[sl]), r)
        rf.set_params(n_jobs=-1)
        idx = va[:n]

        def feats():
            FE.featurize_ridge(d, idx); FE.featurize_natural(d, idx)
        out["feature_extraction_seconds"][str(n)] = _median_time(feats, max(5, r // 10))
    size = lambda p: Path(p).stat().st_size  # noqa: E731
    out["model_file_bytes"] = {"rf": size(RF_MODEL), "natural": size(OUT / "natural_offset_model.joblib"), "gate": size(OUT / "dynamic_gate.pt")}
    out["model_file_bytes"]["total"] = sum(out["model_file_bytes"].values())
    out["rf_file_note"] = "joblib compress=3; in-memory size is larger"
    return out


def summary_md(r) -> str:
    f = lambda x, k=3: "—" if x is None else f"{x:.{k}f}"  # noqa: E731
    v = r["validation"]
    o = v["overall"]
    order = ("rf", "natural", "old_fixed", "old_dynamic", "new_fixed", "new_dynamic")
    best = min(order, key=lambda k: o[k]["mae"])
    L = ["# Axis 2 V3: Random Forest + natural-offset hybrids", "",
         "Generated by `rf_natural_hybrid.py`. Frozen experts (V3 Random Forest, V3 natural-offset HGB); Ridge is not an expert. "
         "Fixed weight and gate settings chosen on train out-of-fold predictions (GroupKFold(5) by song); validation evaluated once; "
         "**test not evaluated**.", "",
         f"- Fixed: w_RF = **{r['fixed']['selected_weight_rf']:.2f}**, w_natural = {r['fixed']['selected_weight_natural']:.2f}.",
         f"- Dynamic gate: V2 architecture and training, {r['dynamic']['parameters']} parameters; wd {r['dynamic']['selected_weight_decay']:g}, "
         f"{r['dynamic']['selected_epochs']} epochs; seed-check OOF MAEs " + ", ".join(f(x, 4) for x in r["dynamic"]["seed_check_oof_mae"]) + ".",
         f"- OOF experts regenerated; RF fold MAEs match the RF experiment (max diff {r['oof']['rf_fold_mae_max_diff_vs_rf_experiment']:.1e}), "
         f"natural fold MAEs match (max diff {r['oof']['natural_fold_mae_max_diff_vs_natural_experiment']:.1e}).", "",
         "| train OOF | MAE | RMSE |", "|---|---|---|"]
    for k in ("rf", "natural", "fixed", "dynamic"):
        L.append(f"| {k} | {f(r['fixed']['train_oof_mae'][k], 4)} | {f(r['fixed']['train_oof_rmse'][k], 4)} |")
    L += ["", "## Validation", "", "| model | MAE | RMSE | median AE |", "|---|---|---|---|"]
    for k in order + ("oracle",):
        m = f"**{f(o[k]['mae'], 4)}**" if k == best else f(o[k]["mae"], 4)
        L.append(f"| {LABEL[k]} | {m} | {f(o[k]['rmse'], 4)} | {f(o[k]['median_absolute_error'])} |")
    fd, vc_ = v["fixed_vs_dynamic"], v["vs_old_champion"]
    L += ["", f"Dynamic vs fixed: {fd['absolute_improvement']:+.4f} c MAE ({fd['percent_improvement']:+.2f}%), {fd['rmse_improvement']:+.4f} c RMSE. "
              f"New dynamic vs old champion (13.4557 / 17.3976): {vc_['mae_improvement']:+.4f} c MAE ({vc_['mae_percent']:+.2f}%), "
              f"{vc_['rmse_improvement']:+.4f} c RMSE ({vc_['rmse_percent']:+.2f}%). Positive = new better.", "",
          "## MAE by variant", "", "| model | clean | c1 | c2 | c3 | c4 |", "|---|---|---|---|---|---|"]
    for k in ("rf", "natural", "old_dynamic", "new_fixed", "new_dynamic", "oracle"):
        L.append(f"| {LABEL[k]} | " + " | ".join(f(v["by_variant"][n][k]["mae"], 2) for n in v["by_variant"]) + " |")
    L.append("| dynamic w̄_RF | " + " | ".join(f(v["dynamic_rf_weight_by_variant"][n]) for n in v["by_variant"]) + " |")
    wd = v["dynamic_rf_weight_distribution"]
    L += ["", "## Dynamic RF weight", "", "| mean | std | P10 | P25 | P50 | P75 | P90 | min | max |", "|---|---|---|---|---|---|---|---|---|",
          "| " + " | ".join(f(wd[q]) for q in ("mean", "std", "p10", "p25", "p50", "p75", "p90", "min", "max")) + " |"]
    for title, key in (("By |RF − natural|", "by_abs_rf_minus_natural"), ("By |observed core|", "by_abs_observed_core")):
        L += ["", f"## {title}", "", "| bucket | n | RF | Natural | Fixed | Dynamic | Old dynamic | w̄_RF |", "|---|---|---|---|---|---|---|---|"]
        for b, x in v[key].items():
            L.append(f"| {b} | {x['count']} | {f(x['rf_mae'])} | {f(x['natural_mae'])} | {f(x['new_fixed_mae'])} | {f(x['new_dynamic_mae'])} | "
                     f"{f(x['old_dynamic_mae'])} | {f(x['mean_rf_weight'])} |")
    L += ["", "## Prediction slope (corrupted; secondary diagnostic)", "", "| model | slope | intercept | R² |", "|---|---|---|---|"]
    for k, x in v["prediction_slope_corrupted"].items():
        L.append(f"| {LABEL[k]} | {f(x['slope'])} | {x['intercept']:+.3f} | {f(x['r2'])} |")
    L += ["", "## Sign (corrupted)", "", "| sign | model | n | MAE | RMSE | mean signed |", "|---|---|---|---|---|---|"]
    for s, x in v["by_sign_corrupted"].items():
        for k in ("rf", "natural", "old_dynamic", "new_fixed", "new_dynamic"):
            L.append(f"| {s} | {LABEL[k]} | {x[k]['count']} | {f(x[k]['mae'])} | {f(x[k]['rmse'])} | {x[k]['mean_signed_error']:+.3f} |")
        L.append(f"| {s} | new dynamic w̄_RF | | {f(x['new_dynamic_mean_rf_weight'])} | | |")
    L += ["", "## Tail", "", "| model | P90 | P95 | P99 | P99.5 | P99.9 | max | > 25 ¢ | > 50 ¢ |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ("rf", "natural", "old_dynamic", "new_fixed", "new_dynamic"):
        x = o[k]
        L.append(f"| {LABEL[k]} | " + " | ".join(f(x["quantiles"][q], 2) for q in ("p90", "p95", "p99", "p99.5", "p99.9")) + f" | {f(x['max'], 2)} | "
                 f"{x['exceeding']['gt_25_cents']['count']} ({x['exceeding']['gt_25_cents']['percent']:.2f}%) | "
                 f"{x['exceeding']['gt_50_cents']['count']} ({x['exceeding']['gt_50_cents']['percent']:.2f}%) |")
    ng = v["note_groups"]
    L += ["", f"## Note groups ({ng['note_groups']} notes; tie within 0.5 c)", "",
          "| new dynamic vs | new better | tie | other better | median gain | median loss | mean gain |", "|---|---|---|---|---|---|---|"]
    for o_ in ("old_dynamic", "rf", "natural", "new_fixed"):
        x = ng[f"new_dynamic_vs_{o_}"]
        L.append(f"| {LABEL[o_]} | {x['new_dynamic_better_percent']:.1f}% | {x['tie_percent']:.1f}% | {x[f'{o_}_better_percent']:.1f}% | "
                 f"{f(x['median_gain_when_new_wins'])} | {f(x['median_loss_when_other_wins'])} | {x['mean_gain']:+.3f} |")
    sp, orc = v["specialization"], v["oracle"]
    L += ["", "## Specialization and oracle (post hoc, true errors)", "",
          f"- Mean w_RF where RF had lower error: {f(sp['mean_rf_weight_where_rf_lower_error'])}; where natural had lower error: "
          f"{f(sp['mean_rf_weight_where_natural_lower_error'])}.",
          f"- Higher-weighted expert was the lower-error one: {100 * sp['higher_weighted_expert_is_lower_error_fraction']:.1f}% "
          f"(always-RF {100 * sp['chance_references']['always_rf']:.1f}%, always-natural {100 * sp['chance_references']['always_natural']:.1f}%, "
          f"random at the gate's pick rate {100 * sp['chance_references']['random_with_gate_pick_rate']:.1f}%).",
          f"- Oracle MAE {f(orc['mae'], 4)} / RMSE {f(orc['rmse'], 4)}; dynamic − oracle {orc['gap_dynamic_minus_oracle_mae']:+.4f} c; "
          f"dynamic closes {100 * orc['share_of_fixed_to_oracle_gap_closed_by_dynamic']:.1f}% of the fixed→oracle gap.", "",
          "## Latency (descriptive)", "", "| notes | RF | RF 1-thread | Natural | Fixed | Dynamic | feature extraction |", "|---|---|---|---|---|---|---|"]
    lat = r["latency"]
    for n, x in lat["model_inference_seconds"].items():
        L.append(f"| {n} | {1000 * x['rf']:.1f} ms | {1000 * x['rf_single_thread']:.1f} ms | {1000 * x['natural']:.1f} ms | {1000 * x['fixed_hybrid']:.1f} ms | "
                 f"{1000 * x['dynamic_hybrid']:.1f} ms | {1000 * lat['feature_extraction_seconds'][n]:.1f} ms |")
    b = lat["model_file_bytes"]
    L += ["", f"Model files: RF {b['rf'] / 1e6:.1f} MB, natural {b['natural'] / 1e6:.2f} MB, gate {b['gate'] / 1e3:.1f} kB; total {b['total'] / 1e6:.1f} MB.", "",
          "## Integrity", "", f"- Test featurized: 0. Validation passes: 1. Protected unchanged: {r['integrity']['protected_unchanged']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
