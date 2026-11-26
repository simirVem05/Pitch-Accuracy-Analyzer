#!/usr/bin/env python3
"""
Axis 2 V3 final sealed test evaluation of the frozen model (one-time).

Frozen model (loaded, never refit):
  RF      ml/experiments/axis2_v3_rf_compression/selected_rf_model.joblib   (50 trees; inference n_jobs=1)
  natural ml/experiments/axis2_v3_rf_natural_hybrid/natural_offset_model.joblib
          (the frozen V3 natural-offset HGB fit on all train clean notes; the compression experiment refit the
          identical model in memory without saving it, and the validation dry run proves equality)
  gate    ml/experiments/axis2_v3_rf_compression/selected_dynamic_gate.pt   (V2 gate module, RF in the expert slot)

    final = w * rf + (1 - w) * natural,  w = gate(20 inference-time inputs)

Sequence in --split test:
  1. write final_model_manifest_before_test.json (artifact, config, dataset, code hashes; validation reference)
  2. create the TEST_OPENED sentinel; refuse to run if it or any output already exists
  3. inference WITHOUT labels: injected_shift_cents and variant_index are never read in this stage
  4. save and hash the unlabeled predictions
  5. only then load labels and evaluate; write final_model_manifest_after_test.json

--split validation is a dry run of the identical code to a scratch directory: it must reproduce the
compression experiment's stored validation predictions and MAE 12.9703. It never touches test.

  python final_test_evaluation.py --split validation --out /tmp/axis2_v3_final_dry_run
  python final_test_evaluation.py --split test
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axis2_v3_shared"))

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402

import frozen_experts as FE  # noqa: E402
import dynamic_hybrid as DH  # noqa: E402

H, NAT, RB = FE.H, FE.NAT, FE.RB
REPO = FE.REPO
DEFAULT_OUT = Path(__file__).resolve().parent
DATASET = REPO / "data/ml_dataset/axis2_v3"
COMP = REPO / "ml/experiments/axis2_v3_rf_compression"
RFNAT = REPO / "ml/experiments/axis2_v3_rf_natural_hybrid"
ARTIFACTS = {"rf_model": COMP / "selected_rf_model.joblib", "dynamic_gate": COMP / "selected_dynamic_gate.pt",
             "natural_model": RFNAT / "natural_offset_model.joblib"}
CONFIGS = {"rf_config": COMP / "selected_rf_config.json", "dynamic_gate_results": COMP / "selected_dynamic_hybrid_results.json",
           "dynamic_gate_config": RFNAT / "dynamic_gate_config.json", "compression_results": COMP / "rf_compression_results.json",
           "natural_results": REPO / "ml/experiments/axis2_v3_natural_offset/natural_offset_results.json"}
DATA_FILES = {"examples": DATASET / "examples.npz", "splits": DATASET / "splits.json", "summary": DATASET / "summary.json",
              "manifest": DATASET / "manifest.csv"}
CODE = [Path(__file__).resolve(), REPO / "ml/experiments/axis2_v3_shared/frozen_experts.py",
        REPO / "ml/experiments/axis2_v2_dynamic_hybrid/dynamic_hybrid.py", REPO / "ml/experiments/axis2_v2_hybrid/hybrid_ridge_natural.py",
        REPO / "ml/experiments/axis2_v2_natural_offset/natural_offset_feature_experiment.py", REPO / "ml/baselines/ridge_baseline.py",
        REPO / "ml/baselines/zero_baseline.py", REPO / "ml/baselines/median_cents_baseline.py", REPO / "backend/note_segmentation.py"]
VALIDATION_REFERENCE = {"mae": 12.9703, "rmse": 16.7909}
SEED = 20260824
BOOT = 10_000
NOTE_TIE = 0.5
DISAGREE = DH.DISAGREE_BUCKETS
GEN_BANDS = ((0.5, "excellent (within ~+0.5 c of validation)"), (1.0, "good (within ~+1.0 c)"),
             (2.0, "moderate distribution shift (+1-2 c)"), (float("inf"), "significant distribution shift (> +2 c)"))
UNLABELED_FIELDS = ("song_index", "source_note_index", "split_index", "target_midi", "duration_s", "num_frames", "contour_offsets",
                    "cents_flat", "confidence_flat", "song_ids", "song_split_index", "split_names")
MODELS = ("zero", "full_median", "core_median", "rf", "natural", "dynamic")
LABEL = {"zero": "Zero", "full_median": "Full median", "core_median": "Core median", "rf": "Random Forest (50)",
         "natural": "Natural offset", "dynamic": "Dynamic hybrid", "oracle": "Oracle (not deployable)"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(extra: Dict[str, Any]) -> Dict[str, Any]:
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    status = git("status", "--porcelain")
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {k: {"path": FE.rel(p), "sha256": sha(p), "bytes": p.stat().st_size} for k, p in ARTIFACTS.items()},
            "configs": {k: {"path": FE.rel(p), "sha256": sha(p)} for k, p in CONFIGS.items()},
            "dataset": {k: {"path": FE.rel(p), "sha256": sha(p)} for k, p in DATA_FILES.items()},
            "code": {FE.rel(p): sha(p) for p in CODE},
            "git": {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(status), "porcelain_lines": len(status.splitlines()) if status else 0,
                    "status": status.splitlines()},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                            "torch": torch.__version__, "platform": platform.platform()},
            **extra}


def load_models():
    rf = joblib.load(ARTIFACTS["rf_model"])
    cfg = json.loads(CONFIGS["rf_config"].read_text())
    p = rf.get_params()
    if any(p[k] != cfg[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "random_state")) or rf.n_estimators != 50:
        raise RuntimeError("RF artifact does not match the frozen config")
    if tuple(cfg["features"]) != RB.FEATURES or rf.n_features_in_ != len(RB.FEATURES):
        raise RuntimeError("RF features differ from ridge_baseline.FEATURES")
    rf.set_params(n_jobs=1)
    nat = joblib.load(ARTIFACTS["natural_model"])
    _, nat_cfg = H.frozen_configs()
    if nat.n_features_in_ != len(nat_cfg["columns"]):
        raise RuntimeError("natural model feature count differs from the frozen definition")
    ck = torch.load(ARTIFACTS["dynamic_gate"], weights_only=True)
    if [f.replace("rf", "ridge") for f in ck["features"]] != list(DH.GATE_FEATURES):
        raise RuntimeError("gate features differ from the V2 gate inputs")
    gate = DH.Gate(len(DH.GATE_FEATURES), "dynamic").double()
    gate.load_state_dict(ck["state_dict"])
    scaler = {"mean": ck["scaler_mean"].numpy(), "std": ck["scaler_std"].numpy()}
    return rf, nat, nat_cfg["columns"], gate, scaler


def inference(split: str, rf, nat, cols, gate, scaler) -> Dict[str, Any]:
    """Predictions for one split from inference-time inputs only. Labels and variant codes are not read."""
    z = np.load(DATA_FILES["examples"], allow_pickle=False)   # NpzFile is lazy: only accessed keys are read
    D = {k: np.asarray(z[k]) for k in UNLABELED_FIELDS}
    z.close()
    codes = {str(s): i for i, s in enumerate(D["split_names"])}
    idx = np.flatnonzero(D["split_index"] == codes[split])
    off, cents, conf = np.asarray(D["contour_offsets"], np.int64), D["cents_flat"].astype(np.float64), D["confidence_flat"].astype(np.float64)
    dur, midi = D["duration_s"].astype(np.float64), D["target_midi"]
    Xr = RB.build_matrix(cents, conf, off, dur, midi, idx)
    Xf = np.empty((idx.size, len(NAT.ALL_NAMES)))
    core = np.empty(idx.size)
    ctx_cache, ctx = {}, {}
    for r, i in enumerate(idx):
        a, b = int(off[i]), int(off[i + 1])
        f, core[r] = NAT.contour_features(cents[a:b], conf[a:b], float(dur[i]), int(midi[i]))
        s = int(D["song_index"][i])
        if s not in ctx:
            ctx[s] = NAT.context_features(str(D["song_ids"][s]), ctx_cache)
        Xf[r, :f.size] = f
        Xf[r, f.size:] = ctx[s][int(D["source_note_index"][i])]
    if not (np.all(np.isfinite(Xr)) and np.all(np.isfinite(Xf))):
        raise RuntimeError("non-finite inference features")
    rf_p = rf.predict(Xr)
    nat_p = core - nat.predict(Xf[:, cols])
    w = DH.apply_gate(gate, scaler, DH.gate_inputs(rf_p, nat_p, core, Xf))
    final = w * rf_p + (1 - w) * nat_p
    fi = RB.FEATURES.index
    return {"idx": idx, "song_index": D["song_index"][idx], "note": D["source_note_index"][idx], "song_ids": D["song_ids"],
            "num_frames": D["num_frames"][idx], "rf": rf_p, "natural": nat_p, "w": w, "dynamic": final, "core": core,
            "full_median": Xr[:, fi("full_median_cents")], "core_median": Xr[:, fi("core_median_cents")],
            "songs": sorted({str(D["song_ids"][s]) for s in np.unique(D["song_index"][idx])})}


def short(e):
    a = np.abs(e)
    return {"count": int(e.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "median_absolute_error": float(np.median(a)), "mean_signed_error": float(e.mean())}


def evaluate(P: Dict[str, Any], y: np.ndarray, var: np.ndarray, vnames) -> Dict[str, Any]:
    vc = {n: i for i, n in enumerate(vnames)}
    preds = {"zero": np.zeros_like(y), "full_median": P["full_median"], "core_median": P["core_median"],
             "rf": P["rf"], "natural": P["natural"], "dynamic": P["dynamic"]}
    ae_rf, ae_nat = np.abs(P["rf"] - y), np.abs(P["natural"] - y)
    oracle = np.where(ae_rf <= ae_nat, P["rf"], P["natural"])
    errs = {k: p - y for k, p in preds.items()} | {"oracle": oracle - y}
    corr = var != vc["clean"]
    w = P["w"]
    overall = {k: FE.metrics(e) for k, e in errs.items()}
    by_variant = {n: {k: short(errs[k][var == c]) for k in errs} for n, c in vc.items()}
    by_sign = {s: {k: short(errs[k][m]) for k in ("rf", "natural", "dynamic")} | {"dynamic_mean_rf_weight": float(w[m].mean())}
               for s, m in (("positive", corr & (y > 0)), ("negative", corr & (y < 0)))}
    slopes = {k: NAT.slope_stats(y[corr], preds[k][corr]) for k in ("rf", "natural", "dynamic")}
    dis = np.abs(P["rf"] - P["natural"])
    by_dis = {}
    for lab, lo, hi in DISAGREE:
        m = DH.bucket(dis, lo, hi)
        by_dis[lab] = {"count": int(m.sum()), **{f"{k}_mae": float(np.mean(np.abs(errs[k][m]))) for k in ("rf", "natural", "dynamic")},
                       "mean_rf_weight": float(w[m].mean())}
    rf_better, nat_better = ae_rf < ae_nat, ae_nat < ae_rf
    routing = {"rf_lower_error_fraction": float(rf_better.mean()), "natural_lower_error_fraction": float(nat_better.mean()),
               "mean_rf_weight_where_rf_better": float(w[rf_better].mean()), "mean_rf_weight_where_natural_better": float(w[nat_better].mean()),
               "higher_weighted_expert_is_lower_error_fraction": float(np.mean(np.where(w > 0.5, ae_rf <= ae_nat, ae_nat <= ae_rf))),
               "gate_picks_rf_fraction": float(np.mean(w > 0.5))}
    oracle_ref = {"mae": overall["oracle"]["mae"], "rmse": overall["oracle"]["rmse"], "dynamic_minus_oracle_mae": overall["dynamic"]["mae"] - overall["oracle"]["mae"]}

    gkey = P["song_index"].astype(np.int64) * 1_000_000 + P["note"]
    uniq, inv, counts = np.unique(gkey, return_inverse=True, return_counts=True)
    if not np.all(counts == 5):
        raise RuntimeError("not every note has exactly five variants")
    g_abs = {k: np.bincount(inv, weights=np.abs(errs[k])) for k in ("rf", "natural", "dynamic")}
    g_sq = {k: np.bincount(inv, weights=errs[k] ** 2) for k in ("rf", "natural", "dynamic")}
    gm = {k: v / 5 for k, v in g_abs.items()}
    note_dist = {"notes": int(uniq.size), "mean": float(gm["dynamic"].mean()), "median": float(np.median(gm["dynamic"])),
                 **{f"p{q:g}": float(np.percentile(gm["dynamic"], q)) for q in (75, 90, 95, 99)}, "max": float(gm["dynamic"].max())}

    def gcmp(o):
        gain = gm[o] - gm["dynamic"]
        win, loss = gain > NOTE_TIE, gain < -NOTE_TIE
        return {"dynamic_better_percent": 100 * float(win.mean()), "tie_percent": 100 * float((~win & ~loss).mean()),
                f"{o}_better_percent": 100 * float(loss.mean()), "median_gain_when_dynamic_wins": float(np.median(gain[win])) if win.any() else None,
                "median_loss_when_dynamic_loses": float(np.median(-gain[loss])) if loss.any() else None, "mean_gain": float(gain.mean())}

    note_cmp = {"dynamic_vs_rf": gcmp("rf"), "dynamic_vs_natural": gcmp("natural")}

    per_song = []
    for s in np.unique(P["song_index"]):
        m = P["song_index"] == s
        e = errs["dynamic"][m]
        per_song.append({"song_id": str(P["song_ids"][s]), "clean_notes": int(np.sum(m & ~corr)), "examples": int(m.sum()),
                         "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "median_absolute_error": float(np.median(np.abs(e))),
                         "mean_signed_error": float(e.mean()), "p95": float(np.percentile(np.abs(e), 95)), "max": float(np.abs(e).max()),
                         "rf_mae": float(np.mean(np.abs(errs["rf"][m]))), "natural_mae": float(np.mean(np.abs(errs["natural"][m])))})

    # note-group bootstrap (siblings carried together)
    rng = np.random.default_rng(SEED)
    G = uniq.size
    stats = {k: [] for k in ("dynamic_mae", "dynamic_rmse", "rf_mae", "natural_mae", "dynamic_minus_rf_mae", "dynamic_minus_natural_mae")}
    for start in range(0, BOOT, 1000):
        ix = rng.integers(0, G, size=(min(1000, BOOT - start), G))
        n = 5.0 * G
        d_abs, r_abs, n_abs = g_abs["dynamic"][ix].sum(1) / n, g_abs["rf"][ix].sum(1) / n, g_abs["natural"][ix].sum(1) / n
        stats["dynamic_mae"].append(d_abs); stats["rf_mae"].append(r_abs); stats["natural_mae"].append(n_abs)
        stats["dynamic_rmse"].append(np.sqrt(g_sq["dynamic"][ix].sum(1) / n))
        stats["dynamic_minus_rf_mae"].append(d_abs - r_abs); stats["dynamic_minus_natural_mae"].append(d_abs - n_abs)
    point = {"dynamic_mae": overall["dynamic"]["mae"], "dynamic_rmse": overall["dynamic"]["rmse"], "rf_mae": overall["rf"]["mae"],
             "natural_mae": overall["natural"]["mae"], "dynamic_minus_rf_mae": overall["dynamic"]["mae"] - overall["rf"]["mae"],
             "dynamic_minus_natural_mae": overall["dynamic"]["mae"] - overall["natural"]["mae"]}
    boot_note = {k: {"point": point[k], "ci95": [float(np.percentile(np.concatenate(v), 2.5)), float(np.percentile(np.concatenate(v), 97.5))],
                     "std": float(np.std(np.concatenate(v)))} for k, v in stats.items()}
    for k in ("dynamic_minus_rf_mae", "dynamic_minus_natural_mae"):
        allv = np.concatenate(stats[k])
        boot_note[k]["fraction_resamples_dynamic_better"] = float(np.mean(allv < 0))
        boot_note[k]["ci_excludes_zero"] = bool(boot_note[k]["ci95"][1] < 0 or boot_note[k]["ci95"][0] > 0)

    # song-cluster bootstrap (only a handful of songs: unstable, descriptive)
    songs = np.unique(P["song_index"])
    s_abs = {k: np.array([np.abs(errs[k][P["song_index"] == s]).sum() for s in songs]) for k in ("rf", "natural", "dynamic")}
    s_sq = np.array([(errs["dynamic"][P["song_index"] == s] ** 2).sum() for s in songs])
    s_n = np.array([np.sum(P["song_index"] == s) for s in songs], float)
    rng2 = np.random.default_rng(SEED)
    ix = rng2.integers(0, songs.size, size=(BOOT, songs.size))
    nn = s_n[ix].sum(1)
    sb = {"dynamic_mae": s_abs["dynamic"][ix].sum(1) / nn, "dynamic_rmse": np.sqrt(s_sq[ix].sum(1) / nn), "rf_mae": s_abs["rf"][ix].sum(1) / nn,
          "natural_mae": s_abs["natural"][ix].sum(1) / nn}
    sb["dynamic_minus_rf_mae"] = sb["dynamic_mae"] - sb["rf_mae"]
    sb["dynamic_minus_natural_mae"] = sb["dynamic_mae"] - sb["natural_mae"]
    boot_song = {k: {"point": point[k], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in sb.items()}

    return {"overall": overall, "by_variant": by_variant, "by_sign_corrupted": by_sign, "prediction_slope_corrupted": slopes,
            "rf_weight_distribution": DH.wdist(w), "rf_weight_by_variant": {n: float(w[var == c].mean()) for n, c in vc.items()},
            "by_abs_rf_minus_natural": by_dis, "routing": routing, "oracle": oracle_ref, "note_level_dynamic_mae_distribution": note_dist,
            "note_group_comparison": note_cmp, "per_song": per_song,
            "bootstrap": {"note_group": {"resamples": BOOT, "seed": SEED, "unit": "clean-note group (five variants carried together)",
                                         "notes": int(G), "statistics": boot_note},
                          "song_cluster": {"resamples": BOOT, "seed": SEED, "songs": int(songs.size),
                                           "warning": f"only {songs.size} songs: unstable, low-sample, descriptive only; not a substitute for the note-group bootstrap",
                                           "statistics": boot_song}},
            "_errs": errs, "_oracle": oracle}


def validation_reference() -> Dict[str, Any]:
    """Stored validation metrics of the same frozen models (for the val-vs-test tables)."""
    b = REPO / "ml/results/axis2_v3"
    comp = json.loads(CONFIGS["compression_results"].read_text())["validation"]
    nat = json.loads(CONFIGS["natural_results"].read_text())["synthetic_shift_validation"]
    ref = {}
    for k, f in (("zero", "zero_baseline.json"), ("full_median", "median_cents_baseline.json"), ("core_median", "core_median_baseline.json")):
        v = json.loads((b / f).read_text())["validation"]
        ref[k] = {"mae": v["overall"]["mae"], "rmse": v["overall"]["rmse"], "median_absolute_error": v["overall"]["median_absolute_error"],
                  "by_variant_mae": {n: v["by_variant"][n]["mae"] for n in v["by_variant"]}}
    for k, src, kk in (("rf", comp["overall"], "rfN"), ("dynamic", comp["overall"], "dynN")):
        o = src[kk]
        ref[k] = {"mae": o["mae"], "rmse": o["rmse"], "median_absolute_error": o["median_absolute_error"], "tail": o,
                  "by_variant_mae": {n: comp["by_variant"][n][kk]["mae"] for n in comp["by_variant"]}}
    ref["natural"] = {"mae": nat["overall"]["mae"], "rmse": nat["overall"]["rmse"], "median_absolute_error": nat["overall"]["median_absolute_error"],
                      "tail": nat["tail_full"], "by_variant_mae": {n: nat["by_variant"][n]["mae"] for n in nat["by_variant"]}}
    ref["dynamic_rf_weight"] = comp["gate_weight_selected"]
    # disagreement buckets / oracle for the same frozen models, from the stored compression validation predictions
    rows = list(csv.DictReader((COMP / "validation_predictions.csv").open()))
    g = lambda c: np.array([float(r[c]) for r in rows])  # noqa: E731
    y, rf, na, dy, w = g("true_shift"), g("rf50_prediction"), g("natural_prediction"), g("dynamic50_prediction"), g("dynamic50_rf_weight")
    dis = np.abs(rf - na)
    ref["by_abs_rf_minus_natural"] = {lab: {"count": int(DH.bucket(dis, lo, hi).sum()),
                                            **{f"{k}_mae": float(np.mean(np.abs(p - y)[DH.bucket(dis, lo, hi)])) for k, p in (("rf", rf), ("natural", na), ("dynamic", dy))},
                                            "mean_rf_weight": float(w[DH.bucket(dis, lo, hi)].mean())} for lab, lo, hi in DISAGREE}
    orc = np.where(np.abs(rf - y) <= np.abs(na - y), rf, na)
    ref["oracle"] = {"mae": float(np.mean(np.abs(orc - y))), "rmse": float(np.sqrt(np.mean((orc - y) ** 2)))}
    return ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("validation", "test"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    out = args.out.resolve()
    test = args.split == "test"
    if test and out != DEFAULT_OUT:
        raise SystemExit("the test evaluation writes only to ml/experiments/axis2_v3_final_test")
    if not test and out == DEFAULT_OUT:
        raise SystemExit("the validation dry run must write to a scratch directory")
    out.mkdir(parents=True, exist_ok=True)
    outputs = ("final_model_manifest_before_test.json", "final_model_manifest_after_test.json", "final_test_predictions_unlabeled.csv",
               "final_test_predictions.csv", "final_test_results.json", "final_test_summary.md", "bootstrap_results.json", "per_song_results.csv")
    sentinel = out / "TEST_OPENED"
    for name in outputs + ("TEST_OPENED",):
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; the final evaluation runs once")

    splits = json.loads(DATA_FILES["splits"].read_text())
    val_ref = validation_reference()
    before = manifest({"purpose": f"frozen-model manifest written before any {args.split} data was read",
                       "final_validation_mae": VALIDATION_REFERENCE["mae"], "final_validation_rmse": VALIDATION_REFERENCE["rmse"],
                       "split": args.split})
    (out / "final_model_manifest_before_test.json").write_text(json.dumps(before, indent=2) + "\n")
    if test:
        sentinel.write_text(f"test split opened {datetime.now(timezone.utc).isoformat()}\n")

    # ---- stage 1: inference without labels
    rf, nat, cols, gate, scaler = load_models()
    print(f"[{args.split}] inference (labels not loaded)", flush=True)
    P = inference(args.split, rf, nat, cols, gate, scaler)
    exp_examples = splits[f"{args.split}_example_count"]
    if P["idx"].size != exp_examples or sorted(P["songs"]) != sorted(splits[args.split]):
        raise RuntimeError("split rows/songs do not match splits.json")
    unl = out / "final_test_predictions_unlabeled.csv"
    with unl.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["example_index", "song_id", "source_note_index", "rf_prediction", "natural_prediction", "dynamic_rf_weight", "final_prediction"])
        for j, i in enumerate(P["idx"]):
            w.writerow([int(i), str(P["song_ids"][P["song_index"][j]]), int(P["note"][j]), f"{P['rf'][j]:.6f}", f"{P['natural'][j]:.6f}",
                        f"{P['w'][j]:.6f}", f"{P['dynamic'][j]:.6f}"])
    unl_hash = sha(unl)
    print(f"unlabeled predictions saved, sha256 {unl_hash}", flush=True)

    # ---- stage 2: labels
    with np.load(DATA_FILES["examples"], allow_pickle=False) as z:
        y = np.asarray(z["injected_shift_cents"], np.float64)[P["idx"]]
        var = np.asarray(z["variant_index"])[P["idx"]]
        vnames = [str(v) for v in z["variant_names"]]
    if np.sum(var == vnames.index("clean")) != splits[f"{args.split}_clean_note_count"]:
        raise RuntimeError("clean-note count differs from splits.json")
    R = evaluate(P, y, var, vnames)
    errs = R.pop("_errs")
    R.pop("_oracle")

    dry = None
    if not test:
        rows = {(r["song_id"], int(r["source_note_index"]), r["variant"]): r for r in csv.DictReader((COMP / "validation_predictions.csv").open())}
        keys = [(str(P["song_ids"][P["song_index"][j]]), int(P["note"][j]), vnames[int(var[j])]) for j in range(P["idx"].size)]
        g = lambda c: np.array([float(rows[k][c]) for k in keys])  # noqa: E731
        dry = {"rf_max_diff": float(np.max(np.abs(g("rf50_prediction") - P["rf"]))),
               "natural_max_diff": float(np.max(np.abs(g("natural_prediction") - P["natural"]))),
               "weight_max_diff": float(np.max(np.abs(g("dynamic50_rf_weight") - P["w"]))),
               "dynamic_max_diff": float(np.max(np.abs(g("dynamic50_prediction") - P["dynamic"]))),
               "dynamic_mae": R["overall"]["dynamic"]["mae"], "dynamic_rmse": R["overall"]["dynamic"]["rmse"]}
        dry["passed"] = max(dry["rf_max_diff"], dry["natural_max_diff"], dry["weight_max_diff"], dry["dynamic_max_diff"]) <= FE.CSV_TOLERANCE \
            and round(dry["dynamic_mae"], 4) == VALIDATION_REFERENCE["mae"] and round(dry["dynamic_rmse"], 4) == VALIDATION_REFERENCE["rmse"]
        print(json.dumps({"dry_run": dry}, indent=1))
        if not dry["passed"]:
            raise RuntimeError("dry run does not reproduce the frozen model's stored validation predictions")

    with (out / "final_test_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["example_index", "song_id", "source_note_index", "clean_note_id", "variant", "true_shift_cents", "rf_prediction",
                    "natural_prediction", "dynamic_rf_weight", "final_prediction", "absolute_error", "signed_error"])
        for j, i in enumerate(P["idx"]):
            s = str(P["song_ids"][P["song_index"][j]])
            e = errs["dynamic"][j]
            w.writerow([int(i), s, int(P["note"][j]), f"{s}:{int(P['note'][j])}", vnames[int(var[j])], f"{y[j]:.6f}", f"{P['rf'][j]:.6f}",
                        f"{P['natural'][j]:.6f}", f"{P['w'][j]:.6f}", f"{P['dynamic'][j]:.6f}", f"{abs(e):.6f}", f"{e:.6f}"])
    with (out / "per_song_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(R["per_song"][0]))
        w.writeheader()
        for r in R["per_song"]:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    (out / "bootstrap_results.json").write_text(json.dumps({"test_evaluated": test, **R["bootstrap"]}, indent=2) + "\n")

    after = manifest({"purpose": "manifest after evaluation", "split": args.split})
    changed = [f"{sec}:{k}" for sec in ("artifacts", "configs", "dataset") for k in before[sec] if before[sec][k]["sha256"] != after[sec][k]["sha256"]]
    changed += [f"code:{k}" for k in before["code"] if before["code"][k] != after["code"][k]]
    after["frozen_items_changed_after_evaluation"] = changed
    (out / "final_model_manifest_after_test.json").write_text(json.dumps(after, indent=2) + "\n")

    dm, vm = R["overall"]["dynamic"]["mae"], VALIDATION_REFERENCE["mae"]
    gen = next(lab for lim, lab in GEN_BANDS if dm - vm <= lim)
    val_vs = {k: {"validation_mae": val_ref[k]["mae"], "test_mae": R["overall"][k]["mae"], "test_minus_validation_mae": R["overall"][k]["mae"] - val_ref[k]["mae"],
                  "test_over_validation_mae": R["overall"][k]["mae"] / val_ref[k]["mae"], "validation_rmse": val_ref[k]["rmse"],
                  "test_rmse": R["overall"][k]["rmse"], "test_minus_validation_rmse": R["overall"][k]["rmse"] - val_ref[k]["rmse"]} for k in MODELS}
    results = {
        "experiment": "axis2_v3_final_sealed_test" if test else "axis2_v3_final_dry_run_on_validation",
        "test_evaluated": test, "split": args.split, "evaluations_of_this_split_with_frozen_model": 1,
        "final_architecture": "50-tree Random Forest + natural-offset HGB, combined per example by the dynamic gate (w on RF)",
        "provenance": {"dataset": FE.rel(DATASET), "split_file": FE.rel(DATA_FILES["splits"]), "artifacts": {k: FE.rel(p) for k, p in ARTIFACTS.items()},
                       "configs": {k: FE.rel(p) for k, p in CONFIGS.items()}, "inference_code": [FE.rel(p) for p in CODE],
                       "git": before["git"], "environment": before["environment"], "rf_inference_n_jobs": 1,
                       "unlabeled_predictions_sha256_before_labels_loaded": unl_hash},
        "counts": {"songs": len(P["songs"]), "song_ids": P["songs"], "clean_notes": int(np.sum(var == vnames.index("clean"))), "examples": int(P["idx"].size)},
        "validation_reference": {"dynamic": VALIDATION_REFERENCE, **{k: {kk: val_ref[k][kk] for kk in ("mae", "rmse")} for k in MODELS}},
        "results": R, "validation_vs_test": val_vs,
        "generalization": {"dynamic_test_minus_validation_mae": dm - vm, "label": gen, "note": "descriptive only; no decision follows from it"},
        "validation_patterns_for_comparison": {"by_variant_mae": {k: val_ref[k]["by_variant_mae"] for k in ("rf", "natural", "dynamic")},
                                               "rf_weight": val_ref["dynamic_rf_weight"], "by_abs_rf_minus_natural": val_ref["by_abs_rf_minus_natural"],
                                               "oracle": val_ref["oracle"], "tails": {k: val_ref[k]["tail"] for k in ("rf", "natural", "dynamic")}},
        "dry_run": dry,
        "integrity": {"frozen_items_changed_after_evaluation": len(changed), "changed": changed, "models_refit": False,
                      "labels_loaded_after_unlabeled_predictions_saved": True, "runtime_s": time.time() - t0},
    }
    (out / "final_test_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (out / "final_test_summary.md").write_text(summary_md(results))
    print(json.dumps({"split": args.split, "dynamic": {k: R["overall"]["dynamic"][k] for k in ("mae", "rmse")},
                      "rf": R["overall"]["rf"]["mae"], "natural": R["overall"]["natural"]["mae"], "generalization": gen,
                      "changed_after": len(changed)}, indent=1))
    return 0


def summary_md(r) -> str:
    f = lambda x, k=3: "—" if x is None else f"{x:.{k}f}"  # noqa: E731
    R, vv = r["results"], r["validation_vs_test"]
    o = R["overall"]
    sp = "test" if r["test_evaluated"] else "validation (DRY RUN)"
    L = [f"# Axis 2 V3 final evaluation — {sp}", "",
         f"Frozen model: {r['final_architecture']}. Artifacts loaded, not refit; RF inference n_jobs=1. "
         f"{r['counts']['songs']} songs, {r['counts']['clean_notes']} clean notes, {r['counts']['examples']} examples. "
         f"test_evaluated: **{str(r['test_evaluated']).lower()}**.", "",
         f"**FINAL {sp.upper()} MAE {f(o['dynamic']['mae'], 4)} c, RMSE {f(o['dynamic']['rmse'], 4)} c** "
         f"(validation 12.9703 / 16.7909; Δ MAE {r['generalization']['dynamic_test_minus_validation_mae']:+.4f} c — {r['generalization']['label']}).", "",
         "## Headline", "", "| model | validation MAE | test MAE | Δ | test/val | validation RMSE | test RMSE | test median AE |", "|---|---|---|---|---|---|---|---|"]
    for k in MODELS:
        x = vv[k]
        L.append(f"| {LABEL[k]} | {f(x['validation_mae'], 4)} | {f(x['test_mae'], 4)} | {x['test_minus_validation_mae']:+.3f} | {x['test_over_validation_mae']:.3f} | "
                 f"{f(x['validation_rmse'], 4)} | {f(x['test_rmse'], 4)} | {f(o[k]['median_absolute_error'])} |")
    d = o["dynamic"]
    L += ["", f"Dynamic: mean signed {d['mean_signed_error']:+.3f}; P75 {f(d['quantiles']['p75'], 2)}, P90 {f(d['quantiles']['p90'], 2)}, "
              f"P95 {f(d['quantiles']['p95'], 2)}, P99 {f(d['quantiles']['p99'], 2)}, P99.5 {f(d['quantiles']['p99.5'], 2)}, "
              f"P99.9 {f(d['quantiles']['p99.9'], 2)}, max {f(d['max'], 2)}; > 25 c {d['exceeding']['gt_25_cents']['count']} "
              f"({d['exceeding']['gt_25_cents']['percent']:.2f}%), > 50 c {d['exceeding']['gt_50_cents']['count']} ({d['exceeding']['gt_50_cents']['percent']:.2f}%).", "",
          "## By variant (MAE; validation in parentheses)", "", "| model | clean | c1 | c2 | c3 | c4 |", "|---|---|---|---|---|---|"]
    vp = r["validation_patterns_for_comparison"]["by_variant_mae"]
    for k in ("rf", "natural", "dynamic"):
        L.append(f"| {LABEL[k]} | " + " | ".join(f"{f(R['by_variant'][n][k]['mae'], 2)} ({f(vp[k][n], 2)})" for n in R["by_variant"]) + " |")
    L += ["", "| dynamic | n | MAE | RMSE | median AE | mean signed |", "|---|---|---|---|---|---|"]
    for n, x in R["by_variant"].items():
        y = x["dynamic"]
        L.append(f"| {n} | {y['count']} | {f(y['mae'])} | {f(y['rmse'])} | {f(y['median_absolute_error'])} | {y['mean_signed_error']:+.3f} |")
    L += ["", "## Sign (corrupted)", "", "| sign | model | n | MAE | RMSE | mean signed |", "|---|---|---|---|---|---|"]
    for s, x in R["by_sign_corrupted"].items():
        for k in ("rf", "natural", "dynamic"):
            L.append(f"| {s} | {LABEL[k]} | {x[k]['count']} | {f(x[k]['mae'])} | {f(x[k]['rmse'])} | {x[k]['mean_signed_error']:+.3f} |")
    L += ["", "## Tails (test; validation in parentheses)", "", "| model | P90 | P95 | P99 | P99.5 | P99.9 | max | > 25 c | > 50 c |", "|---|---|---|---|---|---|---|---|---|"]
    vt = r["validation_patterns_for_comparison"]["tails"]
    for k in ("rf", "natural", "dynamic"):
        x, v = o[k], vt[k]
        L.append(f"| {LABEL[k]} | " + " | ".join(f"{f(x['quantiles'][q], 1)} ({f(v['quantiles'][q], 1)})" for q in ("p90", "p95", "p99", "p99.5", "p99.9"))
                 + f" | {f(x['max'], 1)} ({f(v['max'], 1)}) | {x['exceeding']['gt_25_cents']['count']} ({v['exceeding']['gt_25_cents']['count']}) | "
                   f"{x['exceeding']['gt_50_cents']['count']} ({v['exceeding']['gt_50_cents']['count']}) |")
    L += ["", "## Per song (dynamic; model-error diagnostic, not singer quality)", "", "| song | clean notes | examples | MAE | RMSE | median AE | signed | P95 | max | RF MAE | natural MAE |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(R["per_song"], key=lambda x: x["mae"]):
        L.append(f"| {x['song_id']} | {x['clean_notes']} | {x['examples']} | {f(x['mae'])} | {f(x['rmse'])} | {f(x['median_absolute_error'])} | "
                 f"{x['mean_signed_error']:+.2f} | {f(x['p95'], 1)} | {f(x['max'], 1)} | {f(x['rf_mae'])} | {f(x['natural_mae'])} |")
    nd = R["note_level_dynamic_mae_distribution"]
    L += ["", f"## Notes ({nd['notes']}; per note, mean |error| over its five variants)", "",
          f"Dynamic: mean {f(nd['mean'])}, median {f(nd['median'])}, P75 {f(nd['p75'])}, P90 {f(nd['p90'])}, P95 {f(nd['p95'])}, P99 {f(nd['p99'])}, max {f(nd['max'])}.", "",
          "| comparison | dynamic better | tie (0.5 c) | expert better | median gain | median loss | mean gain |", "|---|---|---|---|---|---|---|"]
    for k, lab, ek in (("dynamic_vs_rf", "Dynamic vs RF", "rf"), ("dynamic_vs_natural", "Dynamic vs Natural", "natural")):
        x = R["note_group_comparison"][k]
        L.append(f"| {lab} | {x['dynamic_better_percent']:.1f}% | {x['tie_percent']:.1f}% | {x[f'{ek}_better_percent']:.1f}% | "
                 f"{f(x['median_gain_when_dynamic_wins'])} | {f(x['median_loss_when_dynamic_loses'])} | {x['mean_gain']:+.3f} |")
    wd, vw = R["rf_weight_distribution"], r["validation_patterns_for_comparison"]["rf_weight"]
    L += ["", "## Gate (test; validation in parentheses)", "", "| mean | std | P10 | P25 | P50 | P75 | P90 | min | max |", "|---|---|---|---|---|---|---|---|---|",
          "| " + " | ".join(f"{f(wd[q])} ({f(vw[q])})" if q in vw else f(wd[q]) for q in ("mean", "std", "p10", "p25", "p50", "p75", "p90", "min", "max")) + " |", "",
          "| variant | " + " | ".join(R["rf_weight_by_variant"]) + " |", "|---|" + "---|" * 5,
          "| w̄_RF | " + " | ".join(f"{f(R['rf_weight_by_variant'][n])} ({f(vw['by_variant'][n])})" for n in R["rf_weight_by_variant"]) + " |", "",
          "| \\|RF − natural\\| | n | RF | Natural | Dynamic | w̄_RF | validation w̄_RF |", "|---|---|---|---|---|---|---|"]
    vd = r["validation_patterns_for_comparison"]["by_abs_rf_minus_natural"]
    for b, x in R["by_abs_rf_minus_natural"].items():
        L.append(f"| {b} | {x['count']} | {f(x['rf_mae'])} | {f(x['natural_mae'])} | {f(x['dynamic_mae'])} | {f(x['mean_rf_weight'])} | {f(vd[b]['mean_rf_weight'])} |")
    ro, orc = R["routing"], R["oracle"]
    L += ["", f"Routing (post hoc): RF lower error on {100 * ro['rf_lower_error_fraction']:.1f}%, natural on {100 * ro['natural_lower_error_fraction']:.1f}%; "
              f"w̄_RF {f(ro['mean_rf_weight_where_rf_better'])} where RF was better vs {f(ro['mean_rf_weight_where_natural_better'])} where natural was; "
              f"higher-weighted expert was right {100 * ro['higher_weighted_expert_is_lower_error_fraction']:.1f}%.",
          f"Oracle (not deployable): MAE {f(orc['mae'], 4)}, RMSE {f(orc['rmse'], 4)}; dynamic − oracle {orc['dynamic_minus_oracle_mae']:+.4f} c "
          f"(validation oracle {f(r['validation_patterns_for_comparison']['oracle']['mae'], 4)}).", "",
          "## Slope (corrupted; secondary diagnostic)", "", "| model | slope | intercept | R² |", "|---|---|---|---|"]
    for k, x in R["prediction_slope_corrupted"].items():
        L.append(f"| {LABEL[k]} | {f(x['slope'])} | {x['intercept']:+.3f} | {f(x['r2'])} |")
    bn, bs = R["bootstrap"]["note_group"]["statistics"], R["bootstrap"]["song_cluster"]["statistics"]
    L += ["", "## Bootstrap 95% CIs (10,000 resamples, seed 20260824)", "", f"| statistic | point | note-group CI | song-cluster CI (unstable, {R['bootstrap']['song_cluster']['songs']} songs) |", "|---|---|---|---|"]
    for k in bn:
        L.append(f"| {k} | {f(bn[k]['point'], 4)} | [{f(bn[k]['ci95'][0], 4)}, {f(bn[k]['ci95'][1], 4)}] | [{f(bs[k]['ci95'][0], 4)}, {f(bs[k]['ci95'][1], 4)}] |")
    ig = r["integrity"]
    L += ["", "## Integrity", "", f"- Frozen artifacts/configs/dataset/code changed after evaluation: {ig['frozen_items_changed_after_evaluation']}.",
          f"- Unlabeled predictions hashed before labels were loaded: `{r['provenance']['unlabeled_predictions_sha256_before_labels_loaded']}`.",
          f"- Git commit {r['provenance']['git']['commit'][:8]}; uncommitted changes in working tree: {r['provenance']['git']['uncommitted_changes']}.",
          f"- Python {r['provenance']['environment']['python']}, numpy {r['provenance']['environment']['numpy']}, scikit-learn "
          f"{r['provenance']['environment']['scikit_learn']}, torch {r['provenance']['environment']['torch']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
