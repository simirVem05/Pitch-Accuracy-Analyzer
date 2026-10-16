#!/usr/bin/env python3
"""
Axis 2 V2: first sequence model for a clean note's natural pitch offset.

Target, per CLEAN note:  N = median(_core_slice(clean cents))   (production core median)
Input, per example:      centered[t] = round(cents[t] - median(_core_slice(cents)), 9)
                         + CREPE confidence + normalized position, and scalar melodic
                         context (neighbours' target MIDI and timing only).
Shift estimate:          S_hat = observed core median - N_hat

Each example is centered on its own core median, so no input carries absolute pitch
center and a constant shift leaves every input unchanged; sibling equality of N_hat is
measured, never imposed.

Model: small dilated residual 1D CNN (TCN-style) over padded variable-length sequences,
masked mean+max pooling, MLP head with optional context. L1 loss, AdamW.

Protocol: 5-fold GroupKFold by song on train clean notes (the same fold construction as
the handcrafted experiment) selects the input configuration and epoch count; per-fold
normalization is fit on fold-train frames only. The selected model is then trained on
all train clean notes and evaluated once on validation. Test is never read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "backend"))
from note_segmentation import _core_slice  # noqa: E402

DATASET = REPO / "data" / "ml_dataset" / "axis2_v2"
MAIN = REPO / "data" / "main_data"
HANDCRAFTED_DIR = REPO / "ml" / "experiments" / "axis2_v2_natural_offset"
RIDGE_RESULTS = REPO / "ml" / "results" / "axis2_v2" / "ridge_baseline.json"
VARIANTS = ("clean", "corruption_1", "corruption_2", "corruption_3", "corruption_4")
SEED = 20260824
THREADS = 4
CV_FOLDS = 5
CENTER_ROUND_DECIMALS = 9

TRAIN = dict(lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=60, patience=12, grad_clip=1.0, loss="L1Loss")
ARCH = dict(kernel_size=5, dilations=(1, 2, 4), dropout=0.1, head=(64, 32), activation="GELU")
CHANNELS_DEFAULT = 32
WIDTH_RULE_MARGIN = 0.10   # choose 64 channels only if its CV MAE is lower by more than this
TIE_TOLERANCE = 0.5        # |abs error difference| <= this counts as a tie in the note-by-note comparison

CONFIGS = {
    "A_contour_only": dict(channels=("centered", "position"), context=False),
    "B_contour_confidence": dict(channels=("centered", "confidence", "position"), context=False),
    "C_contour_context": dict(channels=("centered", "position"), context=True),
    "D_full": dict(channels=("centered", "confidence", "position"), context=True),
}
CONTEXT_NAMES = ("interval_in", "interval_out", "abs_interval_in", "abs_interval_out", "gap_before", "gap_after",
                 "prev_duration", "duration", "next_duration", "has_previous", "has_next", "target_midi",
                 "pitch_class_sin", "pitch_class_cos")
CHANNEL_INDEX = {"centered": 0, "confidence": 1, "position": 2}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data


def centered_contour(cents: np.ndarray) -> Tuple[np.ndarray, float]:
    core = float(np.median(_core_slice(cents)))
    return np.round(cents - core, CENTER_ROUND_DECIMALS), core


def song_context(song: str) -> np.ndarray:
    """Context rows for every note of one song, from its canonical notes.npz (target MIDI and timing only)."""
    with np.load(MAIN / song / "notes.npz", allow_pickle=False) as n:
        m = np.asarray(n["target_midi_note"], np.float64)
        st, en, du = (np.asarray(n[k], np.float64) for k in ("start_time_s", "end_time_s", "duration_s"))
    c = np.zeros((m.size, len(CONTEXT_NAMES)))
    c[1:, 0] = m[1:] - m[:-1]; c[:-1, 1] = m[1:] - m[:-1]
    c[:, 2], c[:, 3] = np.abs(c[:, 0]), np.abs(c[:, 1])
    c[1:, 4] = st[1:] - en[:-1]; c[:-1, 5] = st[1:] - en[:-1]
    c[1:, 6] = du[:-1]; c[:, 7] = du; c[:-1, 8] = du[1:]
    c[1:, 9] = 1; c[:-1, 10] = 1
    c[:, 11] = m
    pc = np.mod(m, 12)
    c[:, 12], c[:, 13] = np.sin(2 * np.pi * pc / 12), np.cos(2 * np.pi * pc / 12)
    return c


class Examples:
    """Per-example sequences (centered, confidence, position) + context + targets, for a fixed index set."""

    def __init__(self, idx: np.ndarray, D: Dict[str, np.ndarray], cents_flat, conf_flat, ctx_by_song):
        off = D["contour_offsets"]
        self.idx = idx
        self.seqs: List[np.ndarray] = []
        self.core = np.empty(idx.size)
        self.ctx = np.empty((idx.size, len(CONTEXT_NAMES)))
        for r, i in enumerate(idx):
            a, b = int(off[i]), int(off[i + 1])
            c, self.core[r] = centered_contour(cents_flat[a:b])
            n = c.size
            pos = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
            self.seqs.append(np.stack([c, conf_flat[a:b], pos], axis=1))
            self.ctx[r] = ctx_by_song[int(D["song_index"][i])][int(D["source_note_index"][i])]
        self.lengths = np.array([s.shape[0] for s in self.seqs])

    def subset(self, rows: np.ndarray) -> "Examples":
        e = object.__new__(Examples)
        e.idx, e.core, e.ctx, e.lengths = self.idx[rows], self.core[rows], self.ctx[rows], self.lengths[rows]
        e.seqs = [self.seqs[r] for r in rows]
        return e


def fit_norm(ex: Examples) -> Dict[str, Any]:
    frames = np.concatenate(ex.seqs, axis=0)
    mu, sd = ex.ctx.mean(0), ex.ctx.std(0)
    return {"centered_mean": float(frames[:, 0].mean()), "centered_std": float(frames[:, 0].std()),
            "confidence_mean": float(frames[:, 1].mean()), "confidence_std": float(frames[:, 1].std()),
            "context_mean": mu.tolist(), "context_std": np.where(sd > 0, sd, 1.0).tolist()}


def batches(ex: Examples, norm: Dict[str, Any], cfg: Dict[str, Any], batch_size: int, order: np.ndarray, dtype):
    """Yield (x [B,C,T], mask [B,T], ctx [B,K], rows). Rows are grouped by length to limit padding."""
    chans = [CHANNEL_INDEX[c] for c in cfg["channels"]]
    cmu, csd = np.asarray(norm["context_mean"]), np.asarray(norm["context_std"])
    for start in range(0, order.size, batch_size):
        rows = order[start:start + batch_size]
        T = int(ex.lengths[rows].max())
        x = np.zeros((rows.size, len(chans), T))
        mask = np.zeros((rows.size, T))
        for j, r in enumerate(rows):
            s = ex.seqs[r]
            n = s.shape[0]
            f = np.empty((n, 3))
            f[:, 0] = (s[:, 0] - norm["centered_mean"]) / norm["centered_std"]
            f[:, 1] = (s[:, 1] - norm["confidence_mean"]) / norm["confidence_std"]
            f[:, 2] = s[:, 2]
            x[j, :, :n] = f[:, chans].T
            mask[j, :n] = 1.0
        ctx = (ex.ctx[rows] - cmu) / csd
        yield (torch.tensor(x, dtype=dtype), torch.tensor(mask, dtype=dtype), torch.tensor(ctx, dtype=dtype), rows)


def length_sorted(ex: Examples) -> np.ndarray:
    return np.argsort(ex.lengths, kind="stable")


def train_order(ex: Examples, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Length-sorted, then length-bucketed batches in a shuffled order; ties broken randomly."""
    jitter = rng.random(ex.lengths.size)
    o = np.lexsort((jitter, ex.lengths))
    chunks = [o[i:i + batch_size] for i in range(0, o.size, batch_size)]
    rng.shuffle(chunks)
    return np.concatenate(chunks)


# ---------------------------------------------------------------- model


class Block(nn.Module):
    def __init__(self, ch: int, k: int, d: int, p: float):
        super().__init__()
        pad = d * (k - 1) // 2
        self.c1 = nn.Conv1d(ch, ch, k, dilation=d, padding=pad)
        self.c2 = nn.Conv1d(ch, ch, k, dilation=d, padding=pad)
        self.act, self.drop = nn.GELU(), nn.Dropout(p)

    def forward(self, x, m):
        h = self.drop(self.act(self.c1(x))) * m
        h = self.drop(self.act(self.c2(h))) * m
        return (x + h) * m          # padded frames stay exactly zero


class TCN(nn.Module):
    def __init__(self, in_ch: int, ctx_dim: int, ch: int):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, ch, 1)
        self.blocks = nn.ModuleList(Block(ch, ARCH["kernel_size"], d, ARCH["dropout"]) for d in ARCH["dilations"])
        h1, h2 = ARCH["head"]
        self.head = nn.Sequential(nn.Linear(2 * ch + ctx_dim, h1), nn.GELU(), nn.Dropout(ARCH["dropout"]),
                                  nn.Linear(h1, h2), nn.GELU(), nn.Linear(h2, 1))
        self.ctx_dim = ctx_dim

    def forward(self, x, mask, ctx):
        m = mask.unsqueeze(1)
        h = self.proj(x) * m
        for b in self.blocks:
            h = b(h, m)
        mean = h.sum(-1) / m.sum(-1)
        mx = h.masked_fill(m == 0, float("-inf")).amax(-1)
        z = torch.cat([mean, mx] + ([ctx] if self.ctx_dim else []), dim=1)
        return self.head(z).squeeze(1)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def predict(model: TCN, ex: Examples, norm, cfg, dtype=torch.float64) -> np.ndarray:
    m = TCN(len(cfg["channels"]), model.ctx_dim, model.proj.out_channels).to(dtype)
    m.load_state_dict({k: v.to(dtype) for k, v in model.state_dict().items()})
    m.eval()
    out = np.empty(ex.lengths.size)
    with torch.no_grad():
        for x, mask, ctx, rows in batches(ex, norm, cfg, 512, length_sorted(ex), dtype):
            out[rows] = m(x, mask, ctx).numpy()
    return out


def train(cfg, ch, fit: Examples, N_fit: np.ndarray, norm, epochs: int, seed: int,
          heldout: Examples | None = None, N_held: np.ndarray | None = None) -> Tuple[TCN, Dict[str, Any]]:
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = TCN(len(cfg["channels"]), len(CONTEXT_NAMES) if cfg["context"] else 0, ch)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    loss_fn = nn.L1Loss()
    y = torch.tensor(N_fit, dtype=torch.float32)
    hist, best, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, mask, ctx, rows in batches(fit, norm, cfg, TRAIN["batch_size"], train_order(fit, TRAIN["batch_size"], rng), torch.float32):
            opt.zero_grad()
            loss = loss_fn(model(x, mask, ctx), y[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            opt.step()
            tot += loss.item() * rows.size
        rec = {"epoch": epoch, "train_l1": tot / fit.lengths.size}
        if heldout is not None:
            p = predict(model, heldout, norm, cfg, torch.float32)
            rec["heldout_mae"] = float(np.mean(np.abs(p - N_held)))
            rec["heldout_rmse"] = float(np.sqrt(np.mean((p - N_held) ** 2)))
            if rec["heldout_mae"] < best:
                best, best_epoch, bad = rec["heldout_mae"], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_rmse = rec["heldout_rmse"]
            else:
                bad += 1
        hist.append(rec)
        if heldout is not None and bad >= TRAIN["patience"]:
            break
    if heldout is not None:
        model.load_state_dict(best_state)
        return model, {"best_epoch": best_epoch, "best_mae": best, "best_rmse": best_rmse, "epochs_run": len(hist), "history": hist}
    return model, {"epochs_run": len(hist), "history": hist}


# ---------------------------------------------------------------- metrics


def metrics(err: np.ndarray) -> Dict[str, Any]:
    a = np.abs(err)
    return {"count": int(err.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median_absolute_error": float(np.median(a)), "mean_signed_error": float(err.mean())}


def tail(err: np.ndarray) -> Dict[str, float]:
    a = np.abs(err)
    return {**{f"p{q}": float(np.percentile(a, q)) for q in (75, 90, 95, 99)}, "max": float(a.max())}


def slope_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    xc = x - x.mean()
    b = float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))
    a = float(y.mean() - b * x.mean())
    return {"slope": b, "intercept": a, "r2": 1.0 - float(np.sum((y - a - b * x) ** 2) / np.sum((y - y.mean()) ** 2)),
            "count": int(x.size)}


def hash_paths(paths: Sequence[Path]) -> Dict[str, str]:
    out = {}
    for root in paths:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        h = hashlib.sha256()
        for p in files:
            h.update(str(p.relative_to(REPO)).encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
        out[str(root.relative_to(REPO))] = h.hexdigest()
    return out


# ---------------------------------------------------------------- main


def main() -> int:
    t0 = time.time()
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    protected = [MAIN, REPO / "data/ml_dataset/axis2_v1", DATASET, REPO / "data/dataset_selections.json",
                 REPO / "ml/results/axis2_v1", REPO / "ml/results/axis2_v2", HANDCRAFTED_DIR]
    before = hash_paths(protected)
    for name in ("sequence_results.json", "cv_results.csv", "best_model.pt"):
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; refusing to overwrite")

    splits = json.loads((DATASET / "splits.json").read_text())
    with np.load(DATASET / "examples.npz", allow_pickle=False) as z:
        D = {k: np.asarray(z[k]) for k in z.files if k not in ("cents_flat", "confidence_flat")}
        cents_flat, conf_flat = np.asarray(z["cents_flat"]), np.asarray(z["confidence_flat"])
    split_names = [str(s) for s in D["split_names"]]
    assert split_names == ["train", "validation", "test"] and [str(v) for v in D["variant_names"]] == list(VARIANTS)
    TR, VA, TE = 0, 1, 2

    # Test indices never enter these sets; test songs' notes.npz is never opened.
    tr_clean = np.flatnonzero((D["split_index"] == TR) & (D["variant_index"] == 0))
    tr_all = np.flatnonzero(D["split_index"] == TR)
    va_all = np.flatnonzero(D["split_index"] == VA)
    assert tr_clean.size == splits["train_clean_note_count"] and va_all.size == splits["validation_example_count"]
    songs_needed = sorted(set(D["song_index"][np.r_[tr_all, va_all]].tolist()))
    assert all(D["song_split_index"][s] != TE for s in songs_needed)
    ctx_by_song = {s: song_context(str(D["song_ids"][s])) for s in songs_needed}

    ex_tr = Examples(tr_clean, D, cents_flat, conf_flat, ctx_by_song)
    N_tr = ex_tr.core
    core_check = float(np.max(np.abs(N_tr - D["core_median_cents"][tr_clean])))
    ex_va = Examples(va_all, D, cents_flat, conf_flat, ctx_by_song)
    core_check = max(core_check, float(np.max(np.abs(ex_va.core - D["core_median_cents"][va_all]))))
    if core_check > 1e-9:
        raise RuntimeError(f"recomputed core median differs from stored by {core_check}")
    groups = D["song_index"][tr_clean]
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(np.zeros(tr_clean.size), N_tr, groups))
    assert all(not (set(groups[a]) & set(groups[b])) for a, b in folds)

    # ---------------- grouped CV over the four input configurations (32 channels)
    cv: Dict[str, Any] = {}
    def run_cv(name, cfg, ch):
        res = []
        for f, (a, b) in enumerate(folds):
            fit, held = ex_tr.subset(a), ex_tr.subset(b)
            norm = fit_norm(fit)
            _, info = train(cfg, ch, fit, N_tr[a], norm, TRAIN["max_epochs"], SEED + f, held, N_tr[b])
            res.append(info)
            print(f"{name:24s} ch={ch} fold {f}: best epoch {info['best_epoch']:2d} MAE {info['best_mae']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        maes = [r["best_mae"] for r in res]
        return {"config": cfg, "channels": ch, "cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
                "cv_rmse": float(np.mean([r["best_rmse"] for r in res])), "fold_mae": maes,
                "best_epochs": [r["best_epoch"] for r in res], "epochs_run": [r["epochs_run"] for r in res],
                "histories": [r["history"] for r in res]}
    for name, cfg in CONFIGS.items():
        cv[f"{name}_ch{CHANNELS_DEFAULT}"] = run_cv(name, cfg, CHANNELS_DEFAULT)
    best_input = min(CONFIGS, key=lambda n: cv[f"{n}_ch{CHANNELS_DEFAULT}"]["cv_mae"])
    cv[f"{best_input}_ch64"] = run_cv(best_input, CONFIGS[best_input], 64)
    r32, r64 = cv[f"{best_input}_ch{CHANNELS_DEFAULT}"], cv[f"{best_input}_ch64"]
    width = 64 if r32["cv_mae"] - r64["cv_mae"] > WIDTH_RULE_MARGIN else CHANNELS_DEFAULT
    sel_key = f"{best_input}_ch{width}"
    sel = cv[sel_key]
    final_epochs = int(np.median(sel["best_epochs"]))

    hand_res = json.loads((HANDCRAFTED_DIR / "natural_offset_results.json").read_text())
    with (OUT / "cv_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "channels", "cv_mae", "cv_mae_std", "cv_rmse", "fold_mae", "best_epochs", "parameter_count"])
        w.writerow(["handcrafted_HGB_ALL", "", f"{hand_res['selected_model']['cv_mae']:.4f}",
                    f"{hand_res['selected_model']['cv_mae_std']:.4f}", f"{hand_res['selected_model']['cv_rmse']:.4f}", "", "", ""])
        for k, r in cv.items():
            pc = n_params(TCN(len(r["config"]["channels"]), len(CONTEXT_NAMES) if r["config"]["context"] else 0, r["channels"]))
            w.writerow([k, r["channels"], f"{r['cv_mae']:.4f}", f"{r['cv_mae_std']:.4f}", f"{r['cv_rmse']:.4f}",
                        ";".join(f"{x:.4f}" for x in r["fold_mae"]), ";".join(map(str, r["best_epochs"])), pc])

    # ---------------- final model on all train clean notes, fixed epoch count
    cfg = CONFIGS[best_input]
    norm = fit_norm(ex_tr)
    model, final_info = train(cfg, width, ex_tr, N_tr, norm, final_epochs, SEED + 100)
    params = n_params(model)
    torch.save(model.state_dict(), OUT / "best_model.pt")
    config_out = {"input_configuration": best_input, "config": cfg, "channels": width, "architecture": ARCH,
                  "training": TRAIN, "final_epochs": final_epochs, "seed": SEED + 100, "normalization": norm,
                  "context_feature_order": list(CONTEXT_NAMES), "centering": "round(cents - median(_core_slice(cents)), 9)",
                  "inference_dtype": "float64", "parameter_count": params, "torch_version": torch.__version__}
    (OUT / "model_config.json").write_text(json.dumps(config_out, indent=2) + "\n")

    # ---------------- single validation evaluation (every example independently)
    Nhat_va = predict(model, ex_va, norm, cfg)
    var_va = D["variant_index"][va_all]
    clean = var_va == 0
    N_va = ex_va.core[clean]
    eN = Nhat_va[clean] - N_va
    S_hat = ex_va.core - Nhat_va
    y_va = D["injected_shift_cents"][va_all]
    eS = S_hat - y_va
    order = np.lexsort((var_va, D["source_note_index"][va_all], D["song_index"][va_all]))
    g = Nhat_va[order].reshape(-1, 5)
    spread = g.max(1) - g.min(1)
    corrupted = ~clean
    shift = {
        "overall": metrics(eS) | tail(eS),
        "by_variant": {v: metrics(eS[var_va == i]) for i, v in enumerate(VARIANTS)},
        "by_sign": {"positive": metrics(eS[corrupted & (y_va > 0)]), "negative": metrics(eS[corrupted & (y_va < 0)])},
        "prediction_slope": {"corrupted_only": slope_stats(y_va[corrupted], S_hat[corrupted]),
                             "all_validation": slope_stats(y_va, S_hat)},
        "sibling_invariance": {"max_within_group_N_hat_spread": float(spread.max()), "median_spread": float(np.median(spread)),
                               "groups_exceeding_1e-6": int(np.count_nonzero(spread > 1e-6)), "groups": int(spread.size),
                               "note": "each sibling centered and predicted independently; equality is measured"},
    }
    # train-side shift estimate (in-sample, descriptive)
    ex_tr_all = Examples(tr_all, D, cents_flat, conf_flat, ctx_by_song)
    S_tr = ex_tr_all.core - predict(model, ex_tr_all, norm, cfg)
    shift_train = metrics(S_tr - D["injected_shift_cents"][tr_all])

    # ---------------- comparisons
    hand = {}
    with (HANDCRAFTED_DIR / "validation_predictions.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["variant"] == "clean":
                hand[(r["song_id"], int(r["source_note_index"]))] = float(r["predicted_natural_offset"])
    vci = va_all[clean]
    keys = [(str(D["song_ids"][D["song_index"][i]]), int(D["source_note_index"][i])) for i in vci]
    Nh_hand = np.array([hand[k] for k in keys])
    a_seq, a_hand = np.abs(eN), np.abs(Nh_hand - N_va)
    diff = a_seq - a_hand
    better, worse = diff < -TIE_TOLERANCE, diff > TIE_TOLERANCE
    note_cmp = {
        "tie_tolerance_cents": TIE_TOLERANCE,
        "percent_sequence_better": 100 * float(better.mean()), "percent_tied": 100 * float((~better & ~worse).mean()),
        "percent_handcrafted_better": 100 * float(worse.mean()),
        "median_improvement_when_sequence_wins": float(np.median(-diff[better])) if better.any() else None,
        "median_degradation_when_sequence_loses": float(np.median(diff[worse])) if worse.any() else None,
        "abs_error_correlation": float(np.corrcoef(a_seq, a_hand)[0, 1]),
        "prediction_correlation": float(np.corrcoef(Nhat_va[clean], Nh_hand)[0, 1]),
    }
    hand_nat = hand_res["natural_offset_validation"]["selected"]
    rid = json.loads(RIDGE_RESULTS.read_text())["validation"]
    ridge_slope = json.loads(RIDGE_RESULTS.read_text())["shrinkage_diagnostic"]["validation_slope_prediction_on_injected_shift_corrupted"]
    hand_shift = hand_res["synthetic_shift_validation"]
    bench = {
        "direct_ridge": {"overall_mae": rid["overall"]["mae"], "overall_rmse": rid["overall"]["rmse"],
                         **{f"{v}_mae": rid["by_variant"][v]["mae"] for v in VARIANTS}, "slope": ridge_slope},
        "handcrafted_natural_offset": {"overall_mae": hand_shift["overall"]["mae"], "overall_rmse": hand_shift["overall"]["rmse"],
                                       **{f"{v}_mae": hand_shift["by_variant"][v]["mae"] for v in VARIANTS},
                                       "slope": hand_shift["prediction_slope"]["corrupted_only"]["slope"]},
        "sequence_natural_offset": {"overall_mae": shift["overall"]["mae"], "overall_rmse": shift["overall"]["rmse"],
                                    **{f"{v}_mae": shift["by_variant"][v]["mae"] for v in VARIANTS},
                                    "slope": shift["prediction_slope"]["corrupted_only"]["slope"]},
    }

    ctxc = ex_va.ctx[clean]
    nf, du = ex_va.lengths[clean], ctxc[:, CONTEXT_NAMES.index("duration")]
    def buckets(values, edges, labels):
        out = {}
        for (lo, hi), lab in zip(edges, labels):
            m = np.ones(values.shape, bool)
            if lo is not None: m &= values >= lo
            if hi is not None: m &= values < hi
            out[lab] = {"notes": int(m.sum()), "sequence_mae": float(a_seq[m].mean()), "handcrafted_mae": float(a_hand[m].mean()),
                        "N0_mae": float(np.abs(N_va[m]).mean()), "sequence_minus_handcrafted": float(a_seq[m].mean() - a_hand[m].mean())} if m.any() else {"notes": 0}
        return out
    iv = [(None, 1), (1, 3), (3, 5), (5, 8), (8, None)]
    ivl = ["0 or none", "1-2", "3-4", "5-7", "8+"]
    by_type = {
        "frame_count": buckets(nf, [(3, 5), (5, 7), (7, 10), (10, 21), (21, 51), (51, None)], ["3-4", "5-6", "7-9", "10-20", "21-50", "51+"]),
        "duration_s": buckets(du, [(None, .1), (.1, .2), (.2, .4), (.4, .8), (.8, None)], ["<0.10", "0.10-0.20", "0.20-0.40", "0.40-0.80", ">=0.80"]),
        "abs_interval_in": buckets(ctxc[:, CONTEXT_NAMES.index("abs_interval_in")], iv, ivl),
        "abs_interval_out": buckets(ctxc[:, CONTEXT_NAMES.index("abs_interval_out")], iv, ivl),
    }
    def note_rows(sel_rows):
        return [{"song_id": keys[j][0], "source_note_index": keys[j][1], "frame_count": int(nf[j]), "duration_s": float(du[j]),
                 "target_midi": int(ctxc[j, CONTEXT_NAMES.index("target_midi")]),
                 "interval_in": float(ctxc[j, 0]) if ctxc[j, 9] else None, "interval_out": float(ctxc[j, 1]) if ctxc[j, 10] else None,
                 "true_N": float(N_va[j]), "handcrafted_N_hat": float(Nh_hand[j]), "sequence_N_hat": float(Nhat_va[clean][j]),
                 "handcrafted_abs_error": float(a_hand[j]), "sequence_abs_error": float(a_seq[j])} for j in sel_rows]
    o = np.argsort(diff)

    with (OUT / "validation_predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["song_id", "source_note_index", "variant", "frame_count", "target_midi", "observed_core_median",
                    "predicted_natural_offset", "predicted_shift", "injected_shift", "shift_error"])
        for j, i in enumerate(va_all):
            w.writerow([str(D["song_ids"][D["song_index"][i]]), int(D["source_note_index"][i]), VARIANTS[int(var_va[j])],
                        int(ex_va.lengths[j]), int(D["target_midi"][i]), f"{ex_va.core[j]:.6f}", f"{Nhat_va[j]:.6f}",
                        f"{S_hat[j]:.6f}", f"{y_va[j]:.6f}", f"{eS[j]:.6f}"])

    after = hash_paths(protected)
    results = {
        "experiment": "axis2_v2_sequence_natural_offset",
        "test_evaluated": False,
        "seed": SEED,
        "target": "clean_core_median_cents = median(_core_slice(clean cents)); recomputed vs stored max diff below",
        "absolute_pitch_center_removed": True,
        "input_representation": {"sequence_channels": ["centered = round(cents - own core median, 9)", "CREPE confidence",
                                                       "normalized position linspace(0,1,n)"],
                                 "context": list(CONTEXT_NAMES),
                                 "excluded": ["raw cents", "any absolute center", "neighbour cents/offsets", "variant", "shift", "song/artist"]},
        "input_configurations": CONFIGS,
        "selection_rules": {"input": "lowest mean CV MAE among A-D at 32 channels",
                            "width": f"64 channels only if CV MAE lower than 32 by > {WIDTH_RULE_MARGIN} c",
                            "epochs": "median of the selected configuration's per-fold best epochs"},
        "selected_model": {"input_configuration": best_input, "channels": width, "architecture": ARCH, "training": TRAIN,
                           "parameter_count": params, "epochs": final_epochs, "cv_mae": sel["cv_mae"], "cv_mae_std": sel["cv_mae_std"]},
        "cv_results": {k: {kk: vv for kk, vv in r.items() if kk != "histories"} for k, r in cv.items()},
        "cv_histories": {k: r["histories"] for k, r in cv.items()},
        "handcrafted_cv_reference": {"cv_mae": hand_res["selected_model"]["cv_mae"], "cv_mae_std": hand_res["selected_model"]["cv_mae_std"],
                                     "N0_cv_mae": hand_res["cv_reference"]["N0_cv_mae"], "same_fold_construction": True},
        "natural_offset_validation": {"sequence": metrics(eN) | tail(eN), "handcrafted": hand_nat,
                                      "N0": metrics(-N_va),
                                      "improvement_over_handcrafted": {"mae": hand_nat["mae"] - metrics(eN)["mae"],
                                                                       "percent": 100 * (hand_nat["mae"] - metrics(eN)["mae"]) / hand_nat["mae"],
                                                                       "rmse": hand_nat["rmse"] - metrics(eN)["rmse"]}},
        "synthetic_shift_validation": shift,
        "synthetic_shift_train_in_sample": shift_train,
        "comparison_to_handcrafted": note_cmp,
        "comparison_benchmarks_validation": bench,
        "analysis_by_note_type": by_type,
        "notes_sequence_much_better": note_rows(o[:20]),
        "notes_sequence_much_worse": note_rows(o[::-1][:20]),
        "integrity": {"test_examples_read": 0, "test_song_note_files_opened": 0, "validation_used_for_selection": False,
                      "validation_evaluations": 1, "normalization_fit_on": "fold-train (CV) / all train (final) only",
                      "recomputed_vs_stored_core_median_max_abs_diff": core_check,
                      "torch_version": torch.__version__, "device": "cpu", "threads": THREADS, "deterministic_algorithms": True,
                      "inference_dtype": "float64", "training_dtype": "float32",
                      "protected_hashes_before": before, "protected_hashes_after": after, "protected_unchanged": before == after,
                      "runtime_s": time.time() - t0},
    }
    (OUT / "sequence_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    (OUT / "sequence_summary.md").write_text(summary(results))
    print(json.dumps({"selected": results["selected_model"], "natural": results["natural_offset_validation"]["sequence"],
                      "shift": shift["overall"], "slope": shift["prediction_slope"]["corrupted_only"],
                      "invariance": shift["sibling_invariance"], "protected_unchanged": before == after}, indent=1, default=float))
    return 0


def summary(r: Dict[str, Any]) -> str:
    f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
    s = r["selected_model"]
    L = ["# Axis 2 V2: sequence natural-offset model", "",
         "Generated by `sequence_natural_offset.py`. Target N = clean core median; inputs centered on each example's own core "
         "median (no absolute pitch). Selection by 5-fold GroupKFold (song) on train; validation evaluated once; **test not evaluated**.", "",
         "| model | CV MAE | ± | CV RMSE | best epochs |", "|---|---|---|---|---|",
         f"| handcrafted HGB (ALL) | {f(r['handcrafted_cv_reference']['cv_mae'])} | {f(r['handcrafted_cv_reference']['cv_mae_std'], 3)} | — | — |"]
    for k, x in r["cv_results"].items():
        L.append(f"| TCN {k} | {f(x['cv_mae'])} | {f(x['cv_mae_std'], 3)} | {f(x['cv_rmse'])} | {x['best_epochs']} |")
    L += ["", f"**Selected:** {s['input_configuration']}, {s['channels']} channels, {s['parameter_count']:,} parameters, "
              f"{s['epochs']} epochs.", ""]
    nv = r["natural_offset_validation"]
    L += ["| natural offset (validation clean) | MAE | RMSE | median abs | mean signed |", "|---|---|---|---|---|"]
    for k, lab in (("N0", "N = 0"), ("handcrafted", "handcrafted HGB"), ("sequence", "sequence TCN")):
        x = nv[k]
        L.append(f"| {lab} | {f(x['mae'])} | {f(x['rmse'])} | {f(x['median_absolute_error'])} | {x['mean_signed_error']:+.3f} |")
    b = r["comparison_benchmarks_validation"]
    L += ["", "| shift estimate (validation) | Ridge | handcrafted N | sequence N |", "|---|---|---|---|"]
    for k in ["overall_mae", "overall_rmse"] + [f"{v}_mae" for v in VARIANTS] + ["slope"]:
        L.append(f"| {k} | {f(b['direct_ridge'][k], 3)} | {f(b['handcrafted_natural_offset'][k], 3)} | {f(b['sequence_natural_offset'][k], 3)} |")
    c = r["comparison_to_handcrafted"]
    L += ["", f"Note by note (tie = within {c['tie_tolerance_cents']} c): sequence better {c['percent_sequence_better']:.1f}%, "
              f"tied {c['percent_tied']:.1f}%, handcrafted better {c['percent_handcrafted_better']:.1f}%; abs-error correlation "
              f"{c['abs_error_correlation']:.3f}.", ""]
    for name, d in r["analysis_by_note_type"].items():
        L += [f"**By {name}**", "", "| bucket | notes | sequence | handcrafted | N=0 |", "|---|---|---|---|---|"]
        for k, x in d.items():
            if x["notes"]:
                L.append(f"| {k} | {x['notes']} | {f(x['sequence_mae'], 2)} | {f(x['handcrafted_mae'], 2)} | {f(x['N0_mae'], 2)} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
