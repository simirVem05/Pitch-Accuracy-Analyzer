#!/usr/bin/env python3
"""
Axis 2 V3: Random Forest tree-count compression (only n_estimators varies).

Frozen: the V3 RF (max_depth 20, min_samples_leaf 10, max_features 0.5, random_state 20260824, Ridge's
24 features via ridge_baseline.build_matrix), the V3 natural-offset HGB, and the V2 dynamic-gate module
(imported unchanged). Tree counts 300, 200, 150, 100, 75, 50 are fit independently on the same
GroupKFold(5)-by-song partition and on all of train.

Selection (train only): the smallest tree count whose pooled train-OOF MAE is within +0.10 c of the
300-tree forest's. Validation is predicted only for the 300-tree reference (the stored model) and the
selected forest. The fixed blend weight and the dynamic gate are then rebuilt from train OOF exactly as in
ml/experiments/axis2_v3_rf_natural_hybrid; as a reproduction check the same code is first run with the
300-tree forest and must reproduce that experiment's stored validation predictions.

Latency is measured on train feature rows, so unselected forests never predict validation.
The test split is sealed.

  python ml/experiments/axis2_v3_rf_compression/rf_compression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axis2_v3_shared"))

import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

import frozen_experts as FE  # noqa: E402
import dynamic_hybrid as DH  # noqa: E402

H, NAT, RB = FE.H, FE.NAT, FE.RB
REPO = FE.REPO
OUT = Path(__file__).resolve().parent
CACHE = Path("/tmp/axis2_v3_rf_compression_cache")
TMP_MODELS = Path("/tmp/axis2_v3_rf_compression_models")
RF_DIR = REPO / "ml/experiments/axis2_v3_random_forest"
RF_MODEL = RF_DIR / "random_forest_model.joblib"
HYB_DIR = REPO / "ml/experiments/axis2_v3_rf_natural_hybrid"
# Exact float arrays the stored RF+natural hybrid trained its gate on. RF predict with n_jobs=-1 sums trees in
# thread order, so re-predicting differs by ~1e-14, and L1 gate training is not bitwise stable under that noise.
HYB_CACHE = Path("/tmp/axis2_v3_rf_natural_hybrid_cache")
NAT_RESULTS = REPO / "ml/experiments/axis2_v3_natural_offset/natural_offset_results.json"

TREE_COUNTS = (300, 200, 150, 100, 75, 50)
REFERENCE = 300
SELECT_TOL, REPORT_TOL = 0.10, 0.05
NOTE_TIE = 0.5
LAT_REPS = {1: 200, 100: 50, 1000: 20}
PIPELINE_SIZES = (100, 300, 1000)
OUTPUTS = ("rf_compression_results.json", "rf_compression_summary.md", "tree_count_cv.csv", "model_sizes.csv", "latency_results.csv",
           "selected_rf_config.json", "selected_fixed_hybrid_results.json", "selected_dynamic_hybrid_results.json",
           "validation_predictions.csv", "selected_dynamic_gate.pt")


def cached(name: str, fn):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}.npy"
    if f.exists():
        return np.load(f)
    a = fn()
    np.save(f, a)
    return a


def predict_1t(model, X):
    """Single-threaded predict: fixed tree-summation order, so predictions are bitwise reproducible."""
    n_jobs = model.n_jobs
    model.set_params(n_jobs=1)
    try:
        return model.predict(X)
    finally:
        model.set_params(n_jobs=n_jobs)


def median_time(fn, reps: int) -> float:
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def build_hybrids(tag, rf_oof, rf_va, nat_oof, nat_va, y_tr, clean_tr, core_tr, core_va, Xfull_tr, Xfull_va, fold_of):
    """Exactly the fixed sweep and dynamic-gate procedure of rf_natural_hybrid.py (V2 gate module, RF in the expert slot)."""
    sweep = []
    for w in H.WEIGHTS:
        pr = w * rf_oof + (1 - w) * nat_oof
        e = pr - y_tr
        sweep.append({"weight_rf": w, "weight_natural": round(1 - w, 2), "oof_mae": float(np.mean(np.abs(e))),
                      "oof_rmse": float(np.sqrt(np.mean(e ** 2))), "slope_corrupted": NAT.slope_stats(y_tr[~clean_tr], pr[~clean_tr])["slope"]})
    best = min(r["oof_mae"] for r in sweep)
    tied = [r for r in sweep if r["oof_mae"] - best <= H.TIE_TOLERANCE_CENTS]
    W = min(tied, key=lambda r: (abs(r["weight_rf"] - 0.5), r["weight_rf"]))["weight_rf"]
    fixed_oof = W * rf_oof + (1 - W) * nat_oof

    G_tr = DH.gate_inputs(rf_oof, nat_oof, core_tr, Xfull_tr)
    curves, traces = {}, {}
    for wd in DH.WEIGHT_DECAYS:
        print(f"[{tag}] gate CV wd={wd}", flush=True)
        tw = np.empty((DH.MAX_EPOCHS, y_tr.size), np.float32)
        for k in range(FE.CV_FOLDS):
            fit, held = fold_of != k, fold_of == k
            _, _, trace = DH.train_gate("dynamic", G_tr[fit], rf_oof[fit], nat_oof[fit], y_tr[fit], wd, DH.MAX_EPOCHS, DH.SEED + k, X_eval=G_tr[held])
            tw[:, held] = trace
        pr = tw.astype(np.float64) * rf_oof + (1 - tw.astype(np.float64)) * nat_oof
        curves[wd] = np.mean(np.abs(pr - y_tr), axis=1)
        traces[wd] = tw
    g_mae, g_ep, g_wd = min((float(curves[wd][e]), e + 1, wd) for wd in DH.WEIGHT_DECAYS for e in range(DH.MAX_EPOCHS))
    w_oof = traces[g_wd][g_ep - 1].astype(np.float64)
    seed_maes = [g_mae]
    for sd in DH.SEED_CHECK:
        ww = np.empty(y_tr.size)
        for k in range(FE.CV_FOLDS):
            fit, held = fold_of != k, fold_of == k
            m, sc, _ = DH.train_gate("dynamic", G_tr[fit], rf_oof[fit], nat_oof[fit], y_tr[fit], g_wd, g_ep, sd + k)
            ww[held] = DH.apply_gate(m, sc, G_tr[held])
        seed_maes.append(float(np.mean(np.abs(ww * rf_oof + (1 - ww) * nat_oof - y_tr))))
    gate, gsc, _ = DH.train_gate("dynamic", G_tr, rf_oof, nat_oof, y_tr, g_wd, g_ep, DH.SEED + FE.CV_FOLDS)
    w_va = DH.apply_gate(gate, gsc, DH.gate_inputs(rf_va, nat_va, core_va, Xfull_va))
    print(f"[{tag}] w_rf={W} gate wd={g_wd} epochs={g_ep} OOF {g_mae:.4f}", flush=True)
    return {"W": W, "sweep": sweep, "tied": [r["weight_rf"] for r in tied], "fixed_oof": fixed_oof,
            "oof_mae": {"rf": float(np.mean(np.abs(rf_oof - y_tr))), "natural": float(np.mean(np.abs(nat_oof - y_tr))),
                        "fixed": float(np.mean(np.abs(fixed_oof - y_tr))), "dynamic": g_mae},
            "gate_wd": g_wd, "gate_epochs": g_ep, "seed_maes": seed_maes, "w_oof": w_oof, "gate": gate, "gate_scaler": gsc,
            "w_va": w_va, "fixed_va": W * rf_va + (1 - W) * nat_va, "dynamic_va": w_va * rf_va + (1 - w_va) * nat_va}


def main() -> int:
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    for name in OUTPUTS:
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; refusing to overwrite")
    protected = FE.protected_paths(("ml/results/axis2_v3", "ml/experiments/axis2_v3_natural_offset", "ml/experiments/axis2_v3_hybrid",
                                    "ml/experiments/axis2_v3_dynamic_hybrid", "ml/experiments/axis2_v3_shared",
                                    "ml/experiments/axis2_v3_random_forest", "ml/experiments/axis2_v3_rf_natural_hybrid"))
    hashes_before = NAT._hash_tree(protected)

    rf_cfg = json.loads((RF_DIR / "random_forest_config.json").read_text())
    FROZEN = {k: rf_cfg[k] for k in ("max_depth", "min_samples_leaf", "max_features", "random_state")}
    if tuple(rf_cfg["features"]) != RB.FEATURES or rf_cfg["n_estimators"] != REFERENCE:
        raise RuntimeError("RF experiment config differs from the frozen definition")
    rf300 = joblib.load(RF_MODEL)
    if {k: rf300.get_params()[k] for k in FROZEN} != FROZEN or rf300.n_estimators != REFERENCE:
        raise RuntimeError("stored 300-tree model differs from the frozen definition")
    rf_sel_res = json.loads((RF_DIR / "random_forest_results.json").read_text())["selection"]["full24"]["selected"]
    _, nat_cfg = H.frozen_configs()
    nat_cv = json.loads(NAT_RESULTS.read_text())["cv_results"][0]["fold_mae"]
    hyb_res = json.loads((HYB_DIR / "rf_natural_hybrid_results.json").read_text())
    if hyb_res["test_evaluated"] is not False:
        raise RuntimeError("stored RF+natural hybrid is not test-sealed")

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
    gk = groups.astype(np.int64) * 1_000_000 + D["source_note_index"][tr]
    if sum(len(set(gk[a]) & set(gk[b])) for a, b in folds):
        raise RuntimeError("a note straddles a fold")

    # ---- grouped CV per tree count (independent fits, identical folds and rows)
    cv_rows, oof = [], {}
    for n in TREE_COUNTS:
        p_all = np.full(tr.size, np.nan)
        fm, fr = [], []
        for k, (fit, held) in enumerate(folds):
            p = cached(f"rf{n}_fold{k}", lambda: predict_1t(RandomForestRegressor(n_estimators=n, n_jobs=-1, **FROZEN).fit(Xr_tr[fit], y_tr[fit]), Xr_tr[held]))
            p_all[held] = p
            e = p - y_tr[held]
            fm.append(float(np.mean(np.abs(e)))); fr.append(float(np.sqrt(np.mean(e ** 2))))
        oof[n] = p_all
        e = p_all - y_tr
        cv_rows.append({"trees": n, "fold_mae": fm, "mean_fold_mae": float(np.mean(fm)), "std_fold_mae": float(np.std(fm)),
                        "pooled_oof_mae": float(np.mean(np.abs(e))), "fold_rmse": fr, "mean_fold_rmse": float(np.mean(fr)),
                        "pooled_oof_rmse": float(np.sqrt(np.mean(e ** 2)))})
        print(f"trees {n:3d}: pooled OOF MAE {cv_rows[-1]['pooled_oof_mae']:.4f}", flush=True)
    ref = next(r for r in cv_rows if r["trees"] == REFERENCE)
    ref_fold_diff = float(np.max(np.abs(np.array(ref["fold_mae"]) - np.array(rf_sel_res["fold_mae"]))))
    if ref_fold_diff > 1e-9:
        raise RuntimeError(f"300-tree CV does not reproduce the RF experiment ({ref_fold_diff})")
    for r in cv_rows:
        r["delta_vs_300"] = r["pooled_oof_mae"] - ref["pooled_oof_mae"]
        r["within_0_05"] = r["delta_vs_300"] <= REPORT_TOL
        r["within_0_10"] = r["delta_vs_300"] <= SELECT_TOL
    ok = [r for r in cv_rows if r["within_0_10"]]
    N = min(r["trees"] for r in ok)
    for r in cv_rows:
        r["selected"] = r["trees"] == N
    print(f"selected tree count: {N}", flush=True)

    # ---- full-train forests for size/latency; prefix property vs the stored 300-tree model
    TMP_MODELS.mkdir(exist_ok=True)
    full, sizes, prefix = {}, [], {}
    probe = Xr_tr[:2000]
    per_tree = np.stack([t.predict(probe) for t in rf300.estimators_])
    for n in TREE_COUNTS:
        m = RandomForestRegressor(n_estimators=n, n_jobs=-1, **FROZEN).fit(Xr_tr, y_tr)
        full[n] = m
        prefix[str(n)] = float(np.max(np.abs(predict_1t(m, probe) - per_tree[:n].mean(axis=0))))
        f = TMP_MODELS / f"rf_{n}.joblib"
        joblib.dump(m, f, compress=3)
        sizes.append({"trees": n, "bytes": f.stat().st_size, "size_MB": f.stat().st_size / 1e6,
                      "nodes": int(sum(e.tree_.node_count for e in m.estimators_))})
    ref_size = next(s for s in sizes if s["trees"] == REFERENCE)["bytes"]
    for s in sizes:
        s["size_reduction_vs_300_percent"] = 100 * (1 - s["bytes"] / ref_size)
    stored_300_vs_refit = float(np.max(np.abs(predict_1t(full[REFERENCE], probe) - predict_1t(rf300, probe))))
    is_prefix = max(prefix.values()) <= 1e-9 and stored_300_vs_refit <= 1e-9

    # ---- validation: only the 300-tree reference and the selected forest
    rf300_va = predict_1t(rf300, Xr_va)
    rfN_va = predict_1t(full[N], Xr_va)
    keys_va = FE.keys(d, va)
    stored = {(r["song_id"], int(r["source_note_index"]), r["variant"]): r for r in csv.DictReader((HYB_DIR / "validation_predictions.csv").open())}
    if set(keys_va) != stored.keys():
        raise RuntimeError("validation keys differ from the stored RF+natural predictions")
    sc = lambda c: np.array([float(stored[k][c]) for k in keys_va])  # noqa: E731
    if np.max(np.abs(sc("rf_prediction") - rf300_va)) > FE.CSV_TOLERANCE:
        raise RuntimeError("stored 300-tree RF validation predictions do not reproduce")

    # natural OOF / validation (frozen; reproduces stored fold MAEs)
    nhat = np.full(tr.size, np.nan)
    nat_fold = []
    for k, (fit, held) in enumerate(folds):
        fc = np.zeros(tr.size, bool); fc[fit] = True; fc &= clean_tr
        hc = np.zeros(tr.size, bool); hc[held] = True
        nhat[held] = cached(f"nat_fold{k}", lambda: H.natural_fit_predict(nat_cfg["param"], Xn_tr[fc], core_tr[fc], Xn_tr[held]))
        nat_fold.append(float(np.mean(np.abs(nhat[hc & clean_tr] - core_tr[hc & clean_tr]))))
    nat_oof = core_tr - nhat
    if float(np.max(np.abs(np.array(nat_fold) - np.array(nat_cv)))) > 1e-9:
        raise RuntimeError("natural OOF does not reproduce")
    nat_model = NAT.make_model("hgb", nat_cfg["param"]).fit(Xn_tr[clean_tr], core_tr[clean_tr])
    nat_va = core_va - nat_model.predict(Xn_va)
    if np.max(np.abs(sc("natural_prediction") - nat_va)) > FE.CSV_TOLERANCE:
        raise RuntimeError("natural validation predictions do not reproduce")

    # ---- hybrids: 300-tree reproduction check, then the selected forest
    args = (nat_oof, nat_va, y_tr, clean_tr, core_tr, core_va, Xfull_tr, Xfull_va, fold_of)
    # Reproduction check on the stored hybrid's exact training inputs (its cache), after confirming they match this run.
    hc_rf, hc_nat = np.full(tr.size, np.nan), np.full(tr.size, np.nan)
    for k, (_, held) in enumerate(folds):
        hc_rf[held] = np.load(HYB_CACHE / f"rf_oof_fold{k}.npy")
        hc_nat[held] = core_tr[held] - np.load(HYB_CACHE / f"nat_oof_fold{k}.npy")
    hc_diff = {"rf_oof_max_abs_diff": float(np.max(np.abs(hc_rf - oof[REFERENCE]))), "natural_oof_max_abs_diff": float(np.max(np.abs(hc_nat - nat_oof)))}
    if max(hc_diff.values()) > 1e-9:
        raise RuntimeError(f"stored hybrid's training inputs differ from this run's: {hc_diff}")
    h300 = build_hybrids("300", hc_rf, rf300_va, hc_nat, *args[1:])
    repro = {"fixed_weight_equal": h300["W"] == hyb_res["fixed"]["selected_weight_rf"],
             "gate_settings_equal": (h300["gate_wd"], h300["gate_epochs"]) == (hyb_res["dynamic"]["selected_weight_decay"], hyb_res["dynamic"]["selected_epochs"]),
             "fixed_max_diff": float(np.max(np.abs(h300["fixed_va"] - sc("new_fixed_prediction")))),
             "dynamic_max_diff": float(np.max(np.abs(h300["dynamic_va"] - sc("new_dynamic_prediction")))),
             "weight_max_diff": float(np.max(np.abs(h300["w_va"] - sc("dynamic_rf_weight")))),
             "inputs": "the stored hybrid's exact OOF arrays (its /tmp cache)", "inputs_vs_this_run": hc_diff}
    repro["passed"] = repro["fixed_weight_equal"] and repro["gate_settings_equal"] and max(repro["fixed_max_diff"], repro["dynamic_max_diff"], repro["weight_max_diff"]) <= FE.CSV_TOLERANCE
    if not repro["passed"]:
        raise RuntimeError(f"hybrid procedure does not reproduce the stored 300-tree hybrid: {repro}")
    hN = h300 if N == REFERENCE else build_hybrids(str(N), oof[N], rfN_va, *args)

    # ---- validation analysis
    preds = {"rf300": rf300_va, "rfN": rfN_va, "fixed300": h300["fixed_va"], "fixedN": hN["fixed_va"],
             "dyn300": h300["dynamic_va"], "dynN": hN["dynamic_va"]}
    errs = {k: p - y_va for k, p in preds.items()}
    overall = {k: FE.metrics(e) for k, e in errs.items()}
    by_variant = {v: {k: DH.short(errs[k][var_va == c]) for k in errs} for v, c in vc.items()}
    corr = var_va != vc["clean"]
    slopes = {k: NAT.slope_stats(y_va[corr], preds[k][corr]) for k in preds}

    def stability(a, b):
        dd = np.abs(a - b)
        return {"mean_abs_diff": float(dd.mean()), "median_abs_diff": float(np.median(dd)), "p90_abs_diff": float(np.percentile(dd, 90)),
                "p99_abs_diff": float(np.percentile(dd, 99)), "max_abs_diff": float(dd.max()), "pearson_r": float(np.corrcoef(a, b)[0, 1])}

    def wstats(w):
        return {**DH.wdist(w), "by_variant": {v: float(w[var_va == c].mean()) for v, c in vc.items()}}

    gkey = D["song_index"][va].astype(np.int64) * 1_000_000 + D["source_note_index"][va]
    _, inv, counts = np.unique(gkey, return_inverse=True, return_counts=True)
    g300 = np.bincount(inv, weights=np.abs(errs["dyn300"])) / counts
    gN = np.bincount(inv, weights=np.abs(errs["dynN"])) / counts
    gain = g300 - gN
    win, loss = gain > NOTE_TIE, gain < -NOTE_TIE
    note_groups = {"note_groups": int(counts.size), "tie_tolerance_cents": NOTE_TIE,
                   "compressed_better_percent": 100 * float(win.mean()), "tie_percent": 100 * float((~win & ~loss).mean()),
                   "300_better_percent": 100 * float(loss.mean()),
                   "median_gain_when_compressed_wins": float(np.median(gain[win])) if win.any() else None,
                   "median_loss_when_compressed_loses": float(np.median(-gain[loss])) if loss.any() else None,
                   "mean_difference_300_minus_compressed": float(gain.mean())}

    # ---- latency (train feature rows; trained inference only)
    print("latency", flush=True)
    lat_rows = []
    for n in TREE_COUNTS:
        m = full[n]
        row = {"trees": n}
        for k, r in LAT_REPS.items():
            Xs = Xr_tr[:k]
            row[f"{k}_note_ms_njobs-1"] = 1000 * median_time(lambda: m.predict(Xs), r)
            m.set_params(n_jobs=1)
            row[f"{k}_note_ms_single_thread"] = 1000 * median_time(lambda: m.predict(Xs), r)
            m.set_params(n_jobs=-1)
        lat_rows.append(row)
    mN, W, gate, gsc = full[N], hN["W"], hN["gate"], hN["gate_scaler"]
    hyb_lat = {}
    for k, r in LAT_REPS.items():
        a_, b_, c_, f_ = Xr_tr[:k], Xn_tr[:k], core_tr[:k], Xfull_tr[:k]
        fixed = lambda: W * mN.predict(a_) + (1 - W) * (c_ - nat_model.predict(b_))  # noqa: E731

        def dyn():
            p, q = mN.predict(a_), c_ - nat_model.predict(b_)
            w = DH.apply_gate(gate, gsc, DH.gate_inputs(p, q, c_, f_))
            return w * p + (1 - w) * q
        hyb_lat[str(k)] = {"rf_ms": 1000 * median_time(lambda: mN.predict(a_), r), "natural_ms": 1000 * median_time(lambda: c_ - nat_model.predict(b_), r),
                           "fixed_hybrid_ms": 1000 * median_time(fixed, r), "dynamic_hybrid_ms": 1000 * median_time(dyn, r)}
    pipeline = {}
    for k in PIPELINE_SIZES:
        idx = tr[:k]

        def full_stage():
            xr = FE.featurize_ridge(d, idx)
            xf, cc = FE.featurize_natural(d, idx)
            p, q = mN.predict(xr), cc - nat_model.predict(xf[:, cols])
            w = DH.apply_gate(gate, gsc, DH.gate_inputs(p, q, cc, xf))
            return w * p + (1 - w) * q
        pipeline[str(k)] = {"feature_extraction_plus_dynamic_hybrid_ms": 1000 * median_time(full_stage, 10 if k < 1000 else 5)}

    # ---- outputs
    with (OUT / "tree_count_cv.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trees", "mean_fold_mae", "std_fold_mae", "pooled_oof_mae", "mean_fold_rmse", "pooled_oof_rmse", "delta_vs_300",
                    "within_0_05", "within_0_10", "selected", *[f"fold{k}_mae" for k in range(5)]])
        for r in cv_rows:
            w.writerow([r["trees"], *(f"{r[c]:.6f}" for c in ("mean_fold_mae", "std_fold_mae", "pooled_oof_mae", "mean_fold_rmse", "pooled_oof_rmse")),
                        f"{r['delta_vs_300']:+.6f}", r["within_0_05"], r["within_0_10"], r["selected"], *(f"{x:.6f}" for x in r["fold_mae"])])
    with (OUT / "model_sizes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sizes[0]))
        w.writeheader()
        for s in sizes:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in s.items()})
    with (OUT / "latency_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lat_rows[0]))
        w.writeheader()
        for r in lat_rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    if N != REFERENCE:
        joblib.dump(full[N], OUT / "selected_rf_model.joblib", compress=3)
    (OUT / "selected_rf_config.json").write_text(json.dumps({
        "model": "sklearn.ensemble.RandomForestRegressor", "n_estimators": N, **FROZEN, "n_jobs": -1, "features": list(RB.FEATURES),
        "feature_source": "ml/baselines/ridge_baseline.py build_matrix", "fit_rows": "all axis2_v3 train examples",
        "model_file": "selected_rf_model.joblib" if N != REFERENCE else FE.rel(RF_MODEL),
        "selection_rule": "smallest tree count with pooled train-OOF MAE within +0.10 c of the 300-tree forest",
        "is_prefix_of_300_tree_forest": is_prefix}, indent=2) + "\n")
    torch.save({"state_dict": gate.state_dict(), "scaler_mean": torch.from_numpy(gsc["mean"]), "scaler_std": torch.from_numpy(gsc["std"]),
                "features": [f.replace("ridge", "rf") for f in DH.GATE_FEATURES]}, OUT / "selected_dynamic_gate.pt")

    def hyb_summary(h, kind):
        base = {"test_evaluated": False, "trees": N, "fixed_weight_rf": h["W"], "fixed_weight_natural": round(1 - h["W"], 2),
                "train_oof_mae": h["oof_mae"], "tied_weights": h["tied"]}
        if kind == "fixed":
            return {**base, "validation": {"overall": overall["fixedN"], "by_variant": {v: by_variant[v]["fixedN"] for v in vc},
                                           "slope": slopes["fixedN"]}, "weight_sweep": h["sweep"]}
        return {**base, "gate": {"weight_decay": h["gate_wd"], "epochs": h["gate_epochs"], "seed_check_oof_mae": h["seed_maes"],
                                 "parameters": DH.n_params(h["gate"]), "architecture": "V2 gate module, unchanged; RF in the expert slot"},
                "validation": {"overall": overall["dynN"], "by_variant": {v: by_variant[v]["dynN"] for v in vc}, "slope": slopes["dynN"],
                               "rf_weight": wstats(h["w_va"])}}
    (OUT / "selected_fixed_hybrid_results.json").write_text(json.dumps(hyb_summary(hN, "fixed"), indent=2) + "\n")
    (OUT / "selected_dynamic_hybrid_results.json").write_text(json.dumps(hyb_summary(hN, "dynamic"), indent=2) + "\n")
    with (OUT / "validation_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["song_id", "source_note_index", "variant", "true_shift", "rf300_prediction", f"rf{N}_prediction", "natural_prediction",
                    "fixed300_prediction", f"fixed{N}_prediction", "dynamic300_prediction", f"dynamic{N}_prediction",
                    "dynamic300_rf_weight", f"dynamic{N}_rf_weight"])
        for j, k in enumerate(keys_va):
            w.writerow([k[0], k[1], k[2], f"{y_va[j]:.6f}", f"{rf300_va[j]:.6f}", f"{rfN_va[j]:.6f}", f"{nat_va[j]:.6f}",
                        f"{h300['fixed_va'][j]:.6f}", f"{hN['fixed_va'][j]:.6f}", f"{h300['dynamic_va'][j]:.6f}", f"{hN['dynamic_va'][j]:.6f}",
                        f"{h300['w_va'][j]:.6f}", f"{hN['w_va'][j]:.6f}"])

    sel_size = next(s for s in sizes if s["trees"] == N)
    label = ("excellent compression" if N <= 100 else "useful compression" if N <= 200 else "little compression headroom")
    hashes_after = NAT._hash_tree(protected)
    results = {
        "experiment": "axis2_v3_rf_tree_count_compression", "test_evaluated": False, "dataset": "data/ml_dataset/axis2_v3",
        "frozen": {**FROZEN, "features": "ridge_baseline.FEATURES (24)", "natural": nat_cfg["param"], "gate": "V2 module unchanged"},
        "tree_counts": list(TREE_COUNTS), "cv": cv_rows, "cv_300_reproduces_rf_experiment_max_fold_diff": ref_fold_diff,
        "selection": {"rule": f"smallest tree count with pooled train-OOF MAE within +{SELECT_TOL} c of 300 trees",
                      "reference_oof_mae": ref["pooled_oof_mae"], "within_0_05": [r["trees"] for r in cv_rows if r["within_0_05"]],
                      "within_0_10": [r["trees"] for r in cv_rows if r["within_0_10"]], "selected_trees": N, "label": label},
        "determinism": {"smaller_forests_equal_prefix_of_300": is_prefix, "max_abs_diff_vs_prefix_average_by_trees": prefix,
                        "refit_300_vs_stored_300_max_abs_diff": stored_300_vs_refit,
                        "explanation": "sklearn draws one seed per tree from random_state in order, so with the same random_state the first n "
                                       "trees of a larger forest are the trees of an n-tree forest; each model here was still fit independently"},
        "model_sizes": sizes, "selected_size": sel_size,
        "validation": {"evaluated_models": ["rf300 (stored model)", f"rf{N}", "their fixed and dynamic hybrids"],
                       "overall": overall, "by_variant": by_variant, "slope_corrupted": slopes,
                       "rf_prediction_stability": stability(rf300_va, rfN_va), "dynamic_prediction_stability": stability(h300["dynamic_va"], hN["dynamic_va"]),
                       "gate_weight_300": wstats(h300["w_va"]), "gate_weight_selected": wstats(hN["w_va"]),
                       "note_groups_dynamic_selected_vs_300": note_groups},
        "hybrids": {"300": {"fixed_weight_rf": h300["W"], "oof_mae": h300["oof_mae"], "gate_wd": h300["gate_wd"], "gate_epochs": h300["gate_epochs"]},
                    str(N): {"fixed_weight_rf": hN["W"], "oof_mae": hN["oof_mae"], "gate_wd": hN["gate_wd"], "gate_epochs": hN["gate_epochs"],
                             "seed_check_oof_mae": hN["seed_maes"]},
                    "reproduction_of_stored_300_tree_hybrid": repro},
        "latency": {"rf_by_tree_count": lat_rows, "selected_hybrid": hyb_lat, "practical_model_stage_including_feature_extraction": pipeline,
                    "note": "median wall time, trained models only, measured on train feature rows; model loading excluded",
                    "cpu_count": os.cpu_count()},
        "integrity": {"validation_used_for_selection": False, "test_examples_featurized": 0, "test_predictions": 0,
                      "test_song_note_files_opened": FE.test_songs_opened(d), "protected_paths": [FE.rel(p) for p in protected],
                      "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
                      "protected_unchanged": hashes_before == hashes_after, "runtime_s": time.time() - t0},
    }
    if results["integrity"]["test_song_note_files_opened"]:
        raise RuntimeError("a test song's notes were opened")
    (OUT / "rf_compression_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (OUT / "rf_compression_summary.md").write_text(summary_md(results, N))
    print(json.dumps({"selected": N, "rf": {k: overall[f"rf{x}"][k] for x in ("300", "N") for k in ("mae",)},
                      "dynN": {k: overall["dynN"][k] for k in ("mae", "rmse")}, "protected": hashes_before == hashes_after}, indent=1))
    return 0


def summary_md(r, N) -> str:
    f = lambda x, k=3: "—" if x is None else f"{x:.{k}f}"  # noqa: E731
    v, o = r["validation"], r["validation"]["overall"]
    lat = {x["trees"]: x for x in r["latency"]["rf_by_tree_count"]}
    size = {s["trees"]: s for s in r["model_sizes"]}
    L = ["# Axis 2 V3: Random Forest tree-count compression", "",
         "Generated by `rf_compression.py`. Only `n_estimators` varies; everything else frozen. Tree count chosen on pooled train-OOF "
         "MAE (GroupKFold(5) by song); validation only for the 300-tree reference and the selected forest; **test not evaluated**.", "",
         f"**Selected: {N} trees** ({r['selection']['label']}). Within +0.05 c: {r['selection']['within_0_05']}; within +0.10 c: "
         f"{r['selection']['within_0_10']}.", "",
         "| trees | pooled OOF MAE | Δ vs 300 | mean fold MAE ± std | OOF RMSE | val RF MAE | size MB | 100-note ms (n_jobs=-1 / 1) |",
         "|---|---|---|---|---|---|---|---|"]
    for c in r["cv"]:
        n = c["trees"]
        val = f(o["rf300"]["mae"], 4) if n == 300 else (f(o["rfN"]["mae"], 4) if n == N else "not evaluated")
        L.append(f"| {n} | {f(c['pooled_oof_mae'], 4)} | {c['delta_vs_300']:+.4f} | {f(c['mean_fold_mae'], 4)} ± {f(c['std_fold_mae'], 3)} | "
                 f"{f(c['pooled_oof_rmse'], 4)} | {val} | {size[n]['size_MB']:.1f} | {lat[n]['100_note_ms_njobs-1']:.1f} / {lat[n]['100_note_ms_single_thread']:.1f} |")
    L += ["", f"Determinism: smaller forests equal the first n trees of the 300-tree forest: **{r['determinism']['smaller_forests_equal_prefix_of_300']}** "
              f"(max diff {max(r['determinism']['max_abs_diff_vs_prefix_average_by_trees'].values()):.1e}); each was still fit independently.", "",
          "## Validation", "", "| model | MAE | RMSE | median AE | clean | c1 | c2 | c3 | c4 |", "|---|---|---|---|---|---|---|---|---|"]
    names = {"rf300": "300-tree RF", "rfN": f"{N}-tree RF", "fixed300": "300-tree RF+Natural fixed", "fixedN": f"{N}-tree RF+Natural fixed",
             "dyn300": "300-tree RF+Natural dynamic", "dynN": f"{N}-tree RF+Natural dynamic"}
    for k, lab in names.items():
        L.append(f"| {lab} | {f(o[k]['mae'], 4)} | {f(o[k]['rmse'], 4)} | {f(o[k]['median_absolute_error'])} | "
                 + " | ".join(f(v["by_variant"][n][k]["mae"], 2) for n in v["by_variant"]) + " |")
    L += ["", "| model | P90 | P95 | P99 | P99.9 | max | > 25 ¢ | > 50 ¢ |", "|---|---|---|---|---|---|---|---|"]
    for k, lab in names.items():
        x = o[k]
        L.append(f"| {lab} | " + " | ".join(f(x["quantiles"][q], 2) for q in ("p90", "p95", "p99", "p99.9")) + f" | {f(x['max'], 2)} | "
                 f"{x['exceeding']['gt_25_cents']['count']} | {x['exceeding']['gt_50_cents']['count']} |")
    hy = r["hybrids"]
    L += ["", f"Fixed weight w_RF: 300 trees {hy['300']['fixed_weight_rf']:.2f}, {N} trees {hy[str(N)]['fixed_weight_rf']:.2f}. "
              f"Gate (wd, epochs): 300 ({hy['300']['gate_wd']:g}, {hy['300']['gate_epochs']}), {N} ({hy[str(N)]['gate_wd']:g}, {hy[str(N)]['gate_epochs']}). "
              f"300-tree hybrid reproduction of the stored experiment passed: {hy['reproduction_of_stored_300_tree_hybrid']['passed']}.", "",
          "## Stability (validation)", "", "| | mean | median | P90 | P99 | max | Pearson r |", "|---|---|---|---|---|---|---|"]
    for lab, k in (("RF 300 vs selected", "rf_prediction_stability"), ("Dynamic 300 vs selected", "dynamic_prediction_stability")):
        x = v[k]
        L.append(f"| {lab} | {f(x['mean_abs_diff'])} | {f(x['median_abs_diff'])} | {f(x['p90_abs_diff'])} | {f(x['p99_abs_diff'])} | {f(x['max_abs_diff'])} | {x['pearson_r']:.5f} |")
    L += ["", "| gate | mean | std | P10 | P25 | P50 | P75 | P90 | clean | c1 | c2 | c3 | c4 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, k in (("300", "gate_weight_300"), (str(N), "gate_weight_selected")):
        x = v[k]
        L.append(f"| {lab} | " + " | ".join(f(x[q]) for q in ("mean", "std", "p10", "p25", "p50", "p75", "p90")) + " | "
                 + " | ".join(f(x["by_variant"][n]) for n in x["by_variant"]) + " |")
    ng = v["note_groups_dynamic_selected_vs_300"]
    L += ["", f"Note groups (dynamic {N} vs 300, tie 0.5 c): compressed better {ng['compressed_better_percent']:.1f}%, tie {ng['tie_percent']:.1f}%, "
              f"300 better {ng['300_better_percent']:.1f}%; median gain {f(ng['median_gain_when_compressed_wins'])}, median loss "
              f"{f(ng['median_loss_when_compressed_loses'])}, mean (300 − compressed) {ng['mean_difference_300_minus_compressed']:+.4f} c.", "",
          "## Latency (median ms; train feature rows)", "", "| trees | 1 note | 100 notes | 1000 notes | 1 note (1 thread) | 100 (1 thread) | 1000 (1 thread) |",
          "|---|---|---|---|---|---|---|"]
    for n, x in lat.items():
        L.append(f"| {n} | {x['1_note_ms_njobs-1']:.1f} | {x['100_note_ms_njobs-1']:.1f} | {x['1000_note_ms_njobs-1']:.1f} | "
                 f"{x['1_note_ms_single_thread']:.1f} | {x['100_note_ms_single_thread']:.1f} | {x['1000_note_ms_single_thread']:.1f} |")
    L += ["", f"Selected {N}-tree hybrid: " + "; ".join(f"{k} notes RF {x['rf_ms']:.1f} / fixed {x['fixed_hybrid_ms']:.1f} / dynamic {x['dynamic_hybrid_ms']:.1f} ms"
                                                     for k, x in r["latency"]["selected_hybrid"].items()) + ".",
          "Practical model stage incl. feature extraction: " + "; ".join(f"{k} notes {x['feature_extraction_plus_dynamic_hybrid_ms']:.0f} ms"
                                                                        for k, x in r["latency"]["practical_model_stage_including_feature_extraction"].items()) + ".",
          "", f"Protected files unchanged: {r['integrity']['protected_unchanged']}. Test featurized: 0.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
