"""
Axis 1 (harmonic fit) — label-free evaluation of chroma scoring rules on the saved feature store.

Implements protocol.md exactly; read that first. Everything is computed from
data/main_data/<song>/{notes.npz, harmony.npz}, which are opened read-only with
allow_pickle=False. No audio is touched, no chromagram is recomputed, and
production code is imported only to verify that `prod_geo_w25` reproduces
backend/harmony.py:score_key_compliance on every note.

Each method gives, for every note, a support score for each of the 12 candidate
pitch classes over the note's (fixed) time window. Experiments B (per-note
alternatives), C (seeded perturbations) and D (song-level transposition) are all
read off that (n_notes, 12) matrix.

Usage (from the repo root):
    venv/bin/python ml/experiments/axis1_chroma_evaluation/axis1_chroma_evaluation.py [--overwrite]

Outputs go to this directory, except the note-level table, which goes to
data/experiment_archives/axis1_chroma_evaluation/ (Git-ignored).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = REPO / "data" / "experiment_archives" / "axis1_chroma_evaluation"
MAIN = REPO / "data" / "main_data"
sys.path.insert(0, str(REPO / "backend"))

import harmony as H  # noqa: E402  (production Axis 1, used only for the reproduction check)

# ---------------------------------------------------------------- fixed settings (protocol.md)

TOL = 1e-9                     # score equality tolerance; ties count half
PERTURB_FRACTION = 0.20
PERTURB_DRAWS = 5
PATTERNS = {                   # name -> candidate signed pitch-class shifts (one drawn uniformly per note)
    "semitone": (-1, 1),
    "whole_tone": (-2, 2),
    "fourth_fifth": (5, 7),
    "tritone": (6,),
    "random_nonzero": tuple(range(1, 12)),
}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261126
DUR_BUCKETS = (0.0, 0.1, 0.2, 0.4, 0.8, np.inf)
CONF_BUCKETS = (0.0, 0.75, 0.85, 0.95, 1.0 + 1e-9)
SD_BUCKETS = (0.0, 15.0, 35.0, np.inf)
HC = {"min_mean_conf": 0.85, "min_duration_s": 0.10, "max_cents_sd": 35.0, "max_abs_core_cents": 40.0}
BLEED_CONTEXT_S = 1.0
INTERVAL_NAMES = {1: "+1 m2 up", 2: "+2 M2 up", 3: "+3 m3 up", 4: "+4 M3 up", 5: "+5 P4 up", 6: "+6 tritone",
                  7: "+7 P5 up", 8: "+8 m6 up", 9: "+9 M6 up", 10: "+10 (-2) M2 down", 11: "+11 (-1) m2 down"}

METHODS = ["raw_local", "raw_local_slack", "rank_local", "rank_local_slack", "rank_avgties_local_slack",
           "global", "global_raw", "prod_geo_w25", "geo_w25_noslack", "geo_w25_avgties", "arith_w25",
           "geo_w10", "geo_w50", "arith_w10", "arith_w50", "chord_local_slack", "chord_geo_w25"]
EXPLORATORY = {"chord_local_slack", "chord_geo_w25"}
PRODUCTION = "prod_geo_w25"
GROUPS = ("dev", "holdout", "robustness", "axis2_50", "all_63")

TRIADS = [(r, (r + 4) % 12, (r + 7) % 12) for r in range(12)] + [(r, (r + 3) % 12, (r + 7) % 12) for r in range(12)]
TEMPLATES = np.zeros((24, 12))
for _i, _t in enumerate(TRIADS):
    TEMPLATES[_i, list(_t)] = 1.0 / np.sqrt(3.0)
CONTAINS = np.array([[c in t for c in range(12)] for t in TRIADS])     # (24, 12)


# ---------------------------------------------------------------- helpers


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seeded_rng(*parts: Any) -> np.random.Generator:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def bucket(values: np.ndarray, edges: Tuple[float, ...]) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(edges), values, side="right") - 1, 0, len(edges) - 2)


def bucket_labels(edges: Tuple[float, ...]) -> List[str]:
    return [f"[{a:g},{b:g})" if np.isfinite(b) else f">={a:g}" for a, b in zip(edges[:-1], edges[1:])]


def window_means(cum: np.ndarray, start: np.ndarray, width: np.ndarray) -> np.ndarray:
    """cum is (12, n+1) cumulative sum along frames; returns (n_notes, 12) window means."""
    return ((cum[:, start + width] - cum[:, start]) / width).T


def norm_entropy_clarity(v: np.ndarray) -> np.ndarray:
    p = v / np.clip(v.sum(axis=1, keepdims=True), 1e-300, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    return 1.0 - ent / np.log(12.0)


def chord_support(win_mean: np.ndarray) -> np.ndarray:
    """(n, 12) window-mean CENS -> (n, 12): max cosine over triads containing each pitch class."""
    v = win_mean / np.clip(np.linalg.norm(win_mean, axis=1, keepdims=True), 1e-300, None)
    cos = v @ TEMPLATES.T                                                   # (n, 24)
    masked = np.where(CONTAINS[None, :, :], cos[:, :, None], -np.inf)       # (n, 24, 12)
    return masked.max(axis=1)


# ---------------------------------------------------------------- per-song computation


def load_song(song: str) -> Dict[str, Any]:
    with np.load(MAIN / song / "harmony.npz", allow_pickle=False) as z:
        h = {k: z[k] for k in z.files}
    with np.load(MAIN / song / "notes.npz", allow_pickle=False) as z:
        n = {k: z[k] for k in z.files}
    meta = json.loads((MAIN / song / "metadata.json").read_text(encoding="utf-8"))
    return {"h": h, "n": n, "meta": meta}


def production_windows(t: np.ndarray, start: np.ndarray, end: np.ndarray, slack_frames: int):
    """Exact replica of score_key_compliance's indexing; returns i0, width, and validity of +-slack windows."""
    nf = t.size
    i0 = np.searchsorted(t, start, side="left")
    i1 = np.searchsorted(t, end, side="right")
    i0 = np.minimum(np.maximum(i0, 0), max(nf - 1, 0))
    i1 = np.maximum(i1, i0 + 1)
    width = np.minimum(i1, nf) - i0
    lo, hi = i0 - slack_frames, i0 + slack_frames
    ok_lo = (lo >= 0) & (lo + width <= nf) & (slack_frames != 0)
    ok_hi = (hi >= 0) & (hi + width <= nf) & (slack_frames != 0)
    return i0, width, lo, hi, ok_lo, ok_hi


def slack_max(span: np.ndarray, m_lo: np.ndarray, m_hi: np.ndarray, ok_lo: np.ndarray, ok_hi: np.ndarray) -> np.ndarray:
    out = span.copy()
    out = np.where(ok_lo[:, None], np.maximum(out, m_lo), out)
    return np.where(ok_hi[:, None], np.maximum(out, m_hi), out)


def compute_song(song: str, group: str, axis2_info: Dict[str, Any]) -> Dict[str, Any]:
    d = load_song(song)
    h, n, meta = d["h"], d["n"], d["meta"]
    cens = h["other_chroma_cens"].astype(np.float64)
    rank = np.round(h["other_chroma_rank"].astype(np.float64) * 11.0) / 11.0   # exact k/11, as production holds it
    t = h["chroma_frame_times_s"].astype(np.float64)
    nf = t.size
    hop_s = H.CHROMA_HOP / H.CHROMA_SR
    tempo = float(h["estimated_tempo_bpm"])
    slack_s = 60.0 / tempo if tempo > 0 else H.DEFAULT_SLACK_S
    slack_frames = int(round(slack_s / hop_s))
    g_rank = h["global_chroma_rank"].astype(np.float64)
    g_raw = h["global_chroma_profile"].astype(np.float64)

    # ---- structural audit
    ordinal = np.argsort(np.argsort(cens, axis=0), axis=0) / 11.0
    zero_frames = np.all(cens == 0, axis=0)
    srt = np.sort(cens, axis=0)
    tie_frames = np.any(np.diff(srt, axis=0) == 0, axis=0) & ~zero_frames
    avg_rank = (rankdata(cens, axis=0, method="average") - 1.0) / 11.0

    start, end = n["start_time_s"].astype(np.float64), n["end_time_s"].astype(np.float64)
    pc = n["pitch_class"].astype(int)
    dur = n["duration_s"].astype(np.float64)
    offs = n["contour_offsets"]
    conf_flat, cents_flat, ct_flat = n["contour_crepe_confidence"], n["contour_cents_deviation"], n["contour_time_s"]
    nn = pc.size
    mean_conf = np.array([conf_flat[offs[i]:offs[i + 1]].mean() for i in range(nn)])
    min_conf = np.array([conf_flat[offs[i]:offs[i + 1]].min() for i in range(nn)])
    cents_sd = np.array([cents_flat[offs[i]:offs[i + 1]].std() for i in range(nn)])
    contour_in_note = np.array([bool(np.all((ct_flat[offs[i]:offs[i + 1]] >= start[i] - 1e-9)
                                            & (ct_flat[offs[i]:offs[i + 1]] <= end[i] + 1e-9))) for i in range(nn)])
    core_abs = np.abs(n["core_median_cents_deviation"].astype(np.float64))

    i0, width, lo, hi, ok_lo, ok_hi = production_windows(t, start, end, slack_frames)
    reasons = {
        "nonfinite_time": ~(np.isfinite(start) & np.isfinite(end)),
        "end_not_after_start": ~(end > start),
        "starts_after_chroma": start > t[-1],
        "window_empty": width < 1,
        "pc_mismatch_target_midi": pc != (n["target_midi_note"].astype(int) % 12),
    }
    zero_in_window = np.array([bool(zero_frames[a:a + max(w, 1)].any()) for a, w in zip(i0, width)])
    reasons["all_zero_chroma_in_window"] = zero_in_window
    eligible = ~np.any(np.vstack(list(reasons.values())), axis=0)
    hc = eligible & (mean_conf >= HC["min_mean_conf"]) & (dur >= HC["min_duration_s"]) \
        & (cents_sd < HC["max_cents_sd"]) & (core_abs <= HC["max_abs_core_cents"])
    overlaps = int(np.sum(start[1:] < end[:-1] - 1e-9))
    unsorted = int(np.sum(np.diff(start) < 0))

    # ---- window means for each base array
    def means(base: np.ndarray):
        cum = np.concatenate([np.zeros((12, 1)), np.cumsum(base, axis=1)], axis=1)
        span = window_means(cum, i0, width)
        m_lo = window_means(cum, np.clip(lo, 0, nf - 1), np.where(ok_lo, width, 1))
        m_hi = window_means(cum, np.clip(hi, 0, nf - 1), np.where(ok_hi, width, 1))
        return span, m_lo, m_hi

    raw_span, raw_lo, raw_hi = means(cens)
    rk_span, rk_lo, rk_hi = means(rank)
    av_span, av_lo, av_hi = means(avg_rank)
    S: Dict[str, np.ndarray] = {}
    S["raw_local"] = raw_span
    S["raw_local_slack"] = slack_max(raw_span, raw_lo, raw_hi, ok_lo, ok_hi)
    S["rank_local"] = rk_span
    S["rank_local_slack"] = slack_max(rk_span, rk_lo, rk_hi, ok_lo, ok_hi)
    S["rank_avgties_local_slack"] = slack_max(av_span, av_lo, av_hi, ok_lo, ok_hi)
    S["global"] = np.broadcast_to(g_rank, (nn, 12)).copy()
    S["global_raw"] = np.broadcast_to(g_raw, (nn, 12)).copy()
    gfloor = np.maximum(g_rank, 1e-6)[None, :]

    def geo(local: np.ndarray, w: float) -> np.ndarray:
        return (local ** (1.0 - w)) * (gfloor ** w)

    S["prod_geo_w25"] = geo(S["rank_local_slack"], 0.25)
    S["geo_w25_noslack"] = geo(S["rank_local"], 0.25)
    S["geo_w25_avgties"] = geo(S["rank_avgties_local_slack"], 0.25)
    S["arith_w25"] = 0.75 * S["rank_local_slack"] + 0.25 * g_rank[None, :]
    S["geo_w10"] = geo(S["rank_local_slack"], 0.10)
    S["geo_w50"] = geo(S["rank_local_slack"], 0.50)
    S["arith_w10"] = 0.90 * S["rank_local_slack"] + 0.10 * g_rank[None, :]
    S["arith_w50"] = 0.50 * S["rank_local_slack"] + 0.50 * g_rank[None, :]
    ch = slack_max(chord_support(raw_span), chord_support(raw_lo), chord_support(raw_hi), ok_lo, ok_hi)
    S["chord_local_slack"] = ch
    S["chord_geo_w25"] = (np.clip(ch, 1e-6, None) ** 0.75) * (gfloor ** 0.25)
    assert set(S) == set(METHODS)

    # ---- reproduction check against production code (every note)
    ctx = H.HarmonicContext(rank=rank, global_profile=g_raw, global_rank=g_rank, salience=cens, frame_times=t,
                            hop_s=hop_s, tuning_semitones=float(h["tuning_offset_semitones"]), beat_times=h["beat_times_s"],
                            tempo_bpm=tempo, slack_s=slack_s, has_accompaniment=True)
    segs = H.score_key_compliance([{"start": float(a), "end": float(b), "pitch_class": int(c)}
                                   for a, b, c in zip(start, end, pc)], ctx)
    prod_ref = np.array([s["harmonic_rank"] for s in segs], dtype=float)
    local_ref = np.array([s["local_support"] for s in segs], dtype=float)
    idx = np.arange(nn)
    repro = {"max_abs_diff_harmonic_rank": float(np.max(np.abs(prod_ref - S["prod_geo_w25"][idx, pc]))),
             "max_abs_diff_local_support": float(np.max(np.abs(local_ref - S["rank_local_slack"][idx, pc]))),
             "calibrated_key_compliance_mean_dw": float(np.average([s["key_compliance"] for s in segs], weights=dur))}

    # ---- method-independent diagnostics
    clarity = norm_entropy_clarity(raw_span)
    tie_expo = np.empty(nn)
    for i in range(nn):
        blk = cens[:, i0[i]:i0[i] + width[i]]
        tie_expo[i] = float(np.mean(np.any(np.isclose(np.delete(blk, pc[i], axis=0), blk[pc[i]][None, :], rtol=0, atol=0), axis=0)))
    # bleed proxy: orig-pc stored rank during note minus over nearby unvoiced, non-note chroma frames
    vt, vm = h["vocal_frame_times_s"], h["vocal_voiced_mask"].astype(bool)
    vi = np.clip(np.searchsorted(vt, t), 0, vt.size - 1)
    frame_voiced = vm[vi]
    in_note = np.zeros(nf, dtype=bool)
    for a, w in zip(i0, width):
        in_note[a:a + w] = True
    quiet = ~frame_voiced & ~in_note          # note frames are never "quiet", so [a, b) needs no exclusion
    cq = np.concatenate([np.zeros((12, 1)), np.cumsum(rank * quiet[None, :], axis=1)], axis=1)
    cn = np.concatenate([[0], np.cumsum(quiet)])
    a = np.searchsorted(t, start - BLEED_CONTEXT_S, side="left")
    b = np.searchsorted(t, end + BLEED_CONTEXT_S, side="right")
    nq = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        ctx_mean = (cq[pc, b] - cq[pc, a]) / nq
    bleed = np.where(nq > 0, rk_span[np.arange(nn), pc] - ctx_mean, np.nan)

    audit = {
        "song_id": song, "group": group, "axis2_ml": axis2_info["axis2_ml"], "axis2_split": axis2_info["axis2_split"],
        "axis2_exclusion_reason": axis2_info.get("axis2_exclusion_reason", ""),
        "n_notes": int(nn), "n_eligible": int(eligible.sum()), "n_high_conf": int(hc.sum()),
        **{f"invalid_{k}": int(v.sum()) for k, v in reasons.items()},
        "overlapping_notes": overlaps, "unsorted_note_starts": unsorted,
        "contour_outside_note_bounds": int((~contour_in_note).sum()),
        "frames_below_crepe_threshold": int(np.sum(conf_flat < float(meta["crepe_confidence_threshold"]) - 1e-12)),
        "mean_note_conf": float(mean_conf.mean()), "median_duration_s": float(np.median(dur)),
        "notes_under_0p1s": int(np.sum(dur < 0.1)), "notes_one_chroma_frame": int(np.sum(width == 1)),
        "pc_ambiguous_notes": int(np.sum(core_abs > HC["max_abs_core_cents"])),
        "chroma_frames": int(nf), "chroma_end_s": float(t[-1]), "vocal_end_s": float(vt[-1]),
        "metadata_duration_s": float(meta["duration_s"]), "last_note_end_s": float(end.max()),
        "zero_chroma_frames": int(zero_frames.sum()), "tie_frame_fraction": float(tie_frames.mean()),
        "rank_reproduces_production": bool(np.allclose(ordinal, rank, atol=1e-9)),
        "rank_entries_changed_by_avg_ties": float(np.mean(~np.isclose(avg_rank, rank, atol=1e-9))),
        "global_profile_matches_cens_mean": bool(np.allclose(cens.mean(axis=1), g_raw, atol=1e-6)),
        "global_rank_matches_profile": bool(np.allclose(np.argsort(np.argsort(g_raw)) / 11.0, g_rank)),
        "tempo_bpm": tempo, "slack_frames": slack_frames, "n_beats": int(h["beat_times_s"].size),
        "tuning_cents": float(h["tuning_offset_cents"]), "vocal_coverage": float(meta["vocal_coverage"]),
        "median_note_clarity": float(np.median(clarity)), "global_profile_clarity": float(norm_entropy_clarity(g_raw[None, :])[0]),
        **{f"repro_{k}": v for k, v in repro.items()},
    }
    notes = {"pc": pc, "dur": dur, "start": start, "mean_conf": mean_conf, "min_conf": min_conf, "cents_sd": cents_sd,
             "core_abs": core_abs, "clarity": clarity, "tie_expo": tie_expo, "bleed": bleed, "eligible": eligible,
             "hc": hc, "width": width}
    return {"song": song, "group": group, "audit": audit, "S": S, "notes": notes}


# ---------------------------------------------------------------- metrics


def per_note_metrics(S: np.ndarray, pc: np.ndarray) -> Dict[str, np.ndarray]:
    idx = np.arange(pc.size)
    orig = S[idx, pc]
    others = np.ones_like(S, dtype=bool)
    others[idx, pc] = False
    diff = S - orig[:, None]
    greater = ((diff > TOL) & others).sum(axis=1)
    ties = ((np.abs(diff) <= TOL) & others).sum(axis=1)
    alt_max = np.where(others, S, -np.inf).max(axis=1)
    inter = np.empty((pc.size, 11))
    for k in range(1, 12):
        alt = S[idx, (pc + k) % 12]
        inter[:, k - 1] = np.where(orig > alt + TOL, 1.0, np.where(np.abs(orig - alt) <= TOL, 0.5, 0.0))
    return {"orig": orig, "win": (11 - greater - ties + 0.5 * ties) / 11.0, "place": 1 + greater + 0.5 * ties,
            "top1": ((greater == 0) & (ties == 0)).astype(float), "top3": ((1 + greater + 0.5 * ties) <= 3).astype(float),
            "margin": orig - alt_max, "ties": ties.astype(float), "inter": inter}


def calibrated(x: np.ndarray) -> np.ndarray:
    return np.clip((x - H.RANK_LOW) / (H.RANK_HIGH - H.RANK_LOW), 0.0, 1.0)


def transposition(S: np.ndarray, pc: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    idx = np.arange(pc.size)
    m = np.array([S[idx, (pc + k) % 12].mean() for k in range(12)])
    mw = np.array([np.average(S[idx, (pc + k) % 12], weights=w) for k in range(12)])

    def place(v):
        d = v[1:] - v[0]
        g, ti = int(np.sum(d > TOL)), int(np.sum(np.abs(d) <= TOL))
        best = int(np.argmax(v[1:]) + 1)
        return {"placement": 1 + g + 0.5 * ti, "margin_vs_best": float(v[0] - v[1:].max()),
                "best_competing_shift": best, "separation_vs_mean": float(v[0] - v[1:].mean())}

    return {"unweighted": {**place(m), "by_shift": m.tolist()}, "duration_weighted": {**place(mw), "by_shift": mw.tolist()}}


def perturb(song: str, S_all: Dict[str, np.ndarray], pc: np.ndarray, w: np.ndarray, notes: Dict[str, np.ndarray]):
    n = pc.size
    idx = np.arange(n)
    k = max(1, int(n * PERTURB_FRACTION))
    out: Dict[str, Dict[str, Any]] = {}
    affected_rows = []
    for pat, choices in PATTERNS.items():
        draws = []
        for dr in range(PERTURB_DRAWS):
            rng = seeded_rng("axis1", song, pat, dr)
            chosen = np.sort(rng.choice(n, size=k, replace=False))
            shift = rng.choice(np.asarray(choices), size=k)
            draws.append((chosen, shift))
        per_method = {}
        for mth, S in S_all.items():
            orig = S[idx, pc]
            drops, drops_w, note_dec, note_chg = [], [], [], []
            hl_drop = []
            for chosen, shift in draws:
                pert = orig.copy()
                pert[chosen] = S[chosen, (pc[chosen] + shift) % 12]
                drops.append(orig.mean() - pert.mean())
                drops_w.append(np.average(orig, weights=w) - np.average(pert, weights=w))
                chg = orig[chosen] - pert[chosen]
                note_dec.append(np.mean(chg > TOL))
                note_chg.append(chg)
                if mth == PRODUCTION:
                    hl_drop.append(100.0 * (np.average(calibrated(orig), weights=w) - np.average(calibrated(pert), weights=w)))
                    for c, s in zip(chosen, shift):
                        affected_rows.append((pat, int(c), int(s), float(orig[c] - S[c, (pc[c] + s) % 12])))
            allchg = np.concatenate(note_chg)
            per_method[mth] = {"mean_drop": float(np.mean(drops)), "min_drop": float(np.min(drops)),
                               "mean_drop_dw": float(np.mean(drops_w)), "draws_lowering": int(np.sum(np.array(drops) > TOL)),
                               "affected_note_fraction_decreased": float(np.mean(note_dec)),
                               "affected_note_fraction_unchanged_or_higher": float(1 - np.mean(note_dec)),
                               "affected_note_change_median": float(np.median(allchg))}
            if mth == PRODUCTION:
                per_method[mth]["headline_pp_drop_mean"] = float(np.mean(hl_drop))
                per_method[mth]["headline_pp_drop_min"] = float(np.min(hl_drop))
        out[pat] = per_method
    return out, affected_rows


def summarize(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "max": float(a.max())}


def bootstrap_mean_ci(x: np.ndarray, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, x.size, size=(BOOTSTRAP_RESAMPLES, x.size))].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs of this experiment")
    args = ap.parse_args()
    outputs = [OUT / f for f in ("results.json", "tables.md", "data_audit_songs.csv", "song_level_methods.csv",
                                 "method_comparison.csv", "interval_win_rates.csv", "perturbation_summary.csv",
                                 "transposition_songs.csv", "diagnostic_buckets.csv")] + [ARCHIVE / "note_level.csv.gz"]
    if not args.overwrite and any(p.exists() for p in outputs):
        print("outputs exist; pass --overwrite to replace them", file=sys.stderr)
        return 2

    splits = json.loads((OUT / "axis1_splits.json").read_text())
    sel = json.loads((REPO / "data" / "dataset_selections.json").read_text())["songs"]
    manifest = json.loads((MAIN / "manifest.json").read_text())
    group_of = {s: g for g in ("dev", "holdout", "robustness") for s in splits[g]}
    assert set(group_of) == set(sel) == {r["song_id"] for r in manifest["songs"]}, "split / selection / manifest disagree"
    src_hashes_before = {s: {f: sha256_file(MAIN / s / f) for f in ("notes.npz", "harmony.npz")} for s in sorted(sel)}

    songs = []
    for s in sorted(sel):
        songs.append(compute_song(s, group_of[s], sel[s]))
        print(f"  {s}", flush=True)

    in_group = {"dev": lambda r: r["group"] == "dev", "holdout": lambda r: r["group"] == "holdout",
                "robustness": lambda r: r["group"] == "robustness",
                "axis2_50": lambda r: r["group"] in ("dev", "holdout"), "all_63": lambda r: True}

    # per-song, per-method metrics on the primary (eligible) and high-confidence note sets
    song_rows, note_rows, trans_rows, pert_all, affected_all = [], [], [], {}, []
    for r in songs:
        nt = r["notes"]
        el = nt["eligible"]
        r["metrics"], r["metrics_hc"], r["trans"] = {}, {}, {}
        for m in METHODS:
            S = r["S"][m]
            pm = per_note_metrics(S[el], nt["pc"][el])
            r["metrics"][m] = pm
            hc_in_el = nt["hc"][el]
            r["metrics_hc"][m] = {k: v[hc_in_el] for k, v in pm.items()}
            r["trans"][m] = transposition(S[el], nt["pc"][el], nt["dur"][el])
        pr, aff = perturb(r["song"], {m: r["S"][m][el] for m in METHODS}, nt["pc"][el], nt["dur"][el], nt)
        r["pert"] = pr
        el_idx = np.flatnonzero(el)
        for pat, c, s, chg in aff:
            j = el_idx[c]
            affected_all.append((r["song"], r["group"], pat, s, chg, nt["dur"][j], nt["clarity"][j], nt["mean_conf"][j]))
        for m in METHODS:
            pm, pmh, tr = r["metrics"][m], r["metrics_hc"][m], r["trans"][m]["unweighted"]
            row = {"song_id": r["song"], "group": r["group"], "method": m, "n_notes": int(el.sum()),
                   "win_rate": float(pm["win"].mean()), "win_rate_dw": float(np.average(pm["win"], weights=nt["dur"][el])),
                   "top1": float(pm["top1"].mean()), "top3": float(pm["top3"].mean()), "mean_placement": float(pm["place"].mean()),
                   "median_margin": float(np.median(pm["margin"])), "tied_alternatives_mean": float(pm["ties"].mean()),
                   "win_rate_hc": float(pmh["win"].mean()) if pmh["win"].size else float("nan"), "n_hc": int(pmh["win"].size),
                   "transposition_placement": tr["placement"], "transposition_margin": tr["margin_vs_best"],
                   "transposition_best_shift": tr["best_competing_shift"],
                   "transposition_placement_dw": r["trans"][m]["duration_weighted"]["placement"]}
            for pat in PATTERNS:
                row[f"pert_{pat}_draws_lowering"] = r["pert"][pat][m]["draws_lowering"]
                row[f"pert_{pat}_mean_drop"] = r["pert"][pat][m]["mean_drop"]
            song_rows.append(row)
            trans_rows.append({"song_id": r["song"], "group": r["group"], "method": m, **{k: v for k, v in tr.items() if k != "by_shift"},
                               **{f"mean_shift_{k}": v for k, v in enumerate(tr["by_shift"])}})
        pm, pmr = r["metrics"][PRODUCTION], r["metrics"]["rank_local_slack"]
        pmg = r["metrics"]["global"]
        for j, i in enumerate(np.flatnonzero(el)):
            note_rows.append([r["song"], r["group"], int(i), int(nt["pc"][i]), round(float(nt["start"][i]), 4), round(float(nt["dur"][i]), 4),
                              round(float(nt["mean_conf"][i]), 4), round(float(nt["cents_sd"][i]), 3), round(float(nt["core_abs"][i]), 3),
                              round(float(nt["clarity"][i]), 4), round(float(nt["tie_expo"][i]), 4),
                              "" if not np.isfinite(nt["bleed"][i]) else round(float(nt["bleed"][i]), 4), bool(nt["hc"][i]),
                              round(float(pm["orig"][j]), 6), round(float(pm["win"][j]), 4), float(pm["place"][j]),
                              round(float(pm["margin"][j]), 6), float(pmr["place"][j]), float(pmg["place"][j]),
                              *[round(float(r["metrics"][m]["win"][j]), 4) for m in METHODS]])

    # ---- group-level method comparison (song-level means are primary)
    by_song = {(row["song_id"], row["method"]): row for row in song_rows}
    comparison, interval_tab, pert_tab, bucket_tab = [], [], [], []
    results_groups: Dict[str, Any] = {}
    for g in GROUPS:
        gs = [r for r in songs if in_group[g](r)]
        gres: Dict[str, Any] = {"songs": len(gs), "notes": int(sum(r["notes"]["eligible"].sum() for r in gs)), "methods": {}}
        for m in METHODS:
            sw = [by_song[(r["song"], m)]["win_rate"] for r in gs]
            pooled = {k: np.concatenate([r["metrics"][m][k] for r in gs]) for k in ("win", "top1", "top3", "place", "margin")}
            pooled_hc = np.concatenate([r["metrics_hc"][m]["win"] for r in gs])
            tp = [r["trans"][m]["unweighted"]["placement"] for r in gs]
            tbest = [r["trans"][m]["unweighted"]["best_competing_shift"] for r in gs]
            tfail = [b for p, b in zip(tp, tbest) if p > 1]
            inter = np.concatenate([r["metrics"][m]["inter"] for r in gs])
            entry = {
                "song_mean_win_rate": summarize(sw),
                "song_mean_top1": summarize([by_song[(r["song"], m)]["top1"] for r in gs]),
                "song_mean_top3": summarize([by_song[(r["song"], m)]["top3"] for r in gs]),
                "song_mean_placement": summarize([by_song[(r["song"], m)]["mean_placement"] for r in gs]),
                "pooled_win_rate": float(pooled["win"].mean()), "pooled_top1": float(pooled["top1"].mean()),
                "pooled_top3": float(pooled["top3"].mean()), "pooled_mean_placement": float(pooled["place"].mean()),
                "pooled_margin_median": float(np.median(pooled["margin"])), "pooled_margin_positive": float(np.mean(pooled["margin"] > TOL)),
                "pooled_win_rate_high_conf": float(pooled_hc.mean()) if pooled_hc.size else None, "high_conf_notes": int(pooled_hc.size),
                "placement_distribution": {str(p): int(np.sum(np.isclose(pooled["place"], p))) for p in np.arange(1, 12.5, 0.5)
                                           if np.any(np.isclose(pooled["place"], p))},
                "interval_win_rate": {INTERVAL_NAMES[k]: float(inter[:, k - 1].mean()) for k in range(1, 12)},
                "transposition": {"songs_rank_first": int(np.sum(np.isclose(tp, 1))), "songs": len(gs),
                                  "placement_distribution": {str(p): int(np.sum(np.isclose(tp, p))) for p in sorted(set(tp))},
                                  "margin": summarize([r["trans"][m]["unweighted"]["margin_vs_best"] for r in gs]),
                                  "failures_best_shift": {str(b): int(tfail.count(b)) for b in sorted(set(tfail))},
                                  "failures_to_p4_p5": int(sum(b in (5, 7) for b in tfail)), "failures": len(tfail),
                                  "best_competitor_is_p4_p5_all_songs": int(sum(b in (5, 7) for b in tbest)),
                                  "songs_rank_first_duration_weighted": int(sum(np.isclose(r["trans"][m]["duration_weighted"]["placement"], 1) for r in gs))},
                "perturbation": {pat: {"draws_lowering": int(sum(r["pert"][pat][m]["draws_lowering"] for r in gs)),
                                       "draws_total": PERTURB_DRAWS * len(gs),
                                       "songs_all_draws_lowering": int(sum(r["pert"][pat][m]["draws_lowering"] == PERTURB_DRAWS for r in gs)),
                                       "mean_drop": summarize([r["pert"][pat][m]["mean_drop"] for r in gs]),
                                       "affected_note_fraction_decreased": summarize([r["pert"][pat][m]["affected_note_fraction_decreased"] for r in gs]),
                                       **({"headline_pp_drop_mean": summarize([r["pert"][pat][m]["headline_pp_drop_mean"] for r in gs])}
                                          if m == PRODUCTION else {})} for pat in PATTERNS},
            }
            gres["methods"][m] = entry
            comparison.append({"group": g, "method": m, "songs": len(gs), "song_mean_win_rate": entry["song_mean_win_rate"]["mean"],
                               "song_median_win_rate": entry["song_mean_win_rate"]["median"], "worst_song_win_rate": entry["song_mean_win_rate"]["min"],
                               "pooled_win_rate": entry["pooled_win_rate"], "pooled_top1": entry["pooled_top1"], "pooled_top3": entry["pooled_top3"],
                               "pooled_mean_placement": entry["pooled_mean_placement"], "pooled_margin_median": entry["pooled_margin_median"],
                               "pooled_win_rate_high_conf": entry["pooled_win_rate_high_conf"],
                               "transposition_rank_first": entry["transposition"]["songs_rank_first"],
                               "transposition_failures_to_p4_p5": entry["transposition"]["failures_to_p4_p5"],
                               **{f"pert_{p}_draws_lowering_frac": entry["perturbation"][p]["draws_lowering"] / entry["perturbation"][p]["draws_total"]
                                  for p in PATTERNS},
                               **{f"pert_{p}_affected_decreased": entry["perturbation"][p]["affected_note_fraction_decreased"]["mean"] for p in PATTERNS}})
            for k in range(1, 12):
                interval_tab.append({"group": g, "method": m, "shift": k, "interval": INTERVAL_NAMES[k], "win_rate": float(inter[:, k - 1].mean())})
            for pat in PATTERNS:
                e = entry["perturbation"][pat]
                pert_tab.append({"group": g, "method": m, "pattern": pat, "draws_lowering": e["draws_lowering"], "draws_total": e["draws_total"],
                                 "songs_all_draws_lowering": e["songs_all_draws_lowering"], "mean_drop_song_mean": e["mean_drop"]["mean"],
                                 "min_drop_song": e["mean_drop"]["min"], "affected_decreased_mean": e["affected_note_fraction_decreased"]["mean"],
                                 "headline_pp_drop_mean": e.get("headline_pp_drop_mean", {}).get("mean", "")})
        results_groups[g] = gres

    # ---- diagnostic buckets (pooled notes; production, selected-candidate pool of key methods)
    dev_clar = np.concatenate([r["notes"]["clarity"][r["notes"]["eligible"]] for r in songs if r["group"] == "dev"])
    clar_edges = (-np.inf, float(np.percentile(dev_clar, 100 / 3)), float(np.percentile(dev_clar, 200 / 3)), np.inf)
    diag_methods = [PRODUCTION, "rank_local_slack", "rank_local", "raw_local", "global", "geo_w25_avgties", "geo_w25_noslack", "chord_geo_w25"]
    bucket_specs = {
        "duration_s": ("dur", DUR_BUCKETS, bucket_labels(DUR_BUCKETS)),
        "mean_crepe_conf": ("mean_conf", CONF_BUCKETS, ["[0.6,0.75)", "[0.75,0.85)", "[0.85,0.95)", ">=0.95"]),
        "within_note_cents_sd": ("cents_sd", SD_BUCKETS, bucket_labels(SD_BUCKETS)),
        "abs_core_cents": ("core_abs", (0.0, 40.0 + 1e-9, np.inf), ["<=40", ">40 (pc-ambiguous)"]),
        "chroma_clarity_tertile": ("clarity", clar_edges, ["low", "mid", "high"]),
        "tie_exposure": ("tie_expo", (0.0, 1e-9, 0.5, 1.0 + 1e-9), ["none", "(0,0.5)", ">=0.5"]),
    }
    results_buckets: Dict[str, Any] = {"clarity_tertile_edges_from_dev": [clar_edges[1], clar_edges[2]]}
    for g in ("dev", "axis2_50", "robustness", "all_63"):
        gs = [r for r in songs if in_group[g](r)]
        results_buckets[g] = {}
        for name, (key, edges, labels) in bucket_specs.items():
            vals = np.concatenate([r["notes"][key][r["notes"]["eligible"]] for r in gs])
            b = bucket(vals, edges)
            results_buckets[g][name] = {}
            for bi, lab in enumerate(labels):
                mask = b == bi
                ent = {"notes": int(mask.sum()), "share": float(mask.mean())}
                for m in diag_methods:
                    win = np.concatenate([r["metrics"][m]["win"] for r in gs])
                    t1 = np.concatenate([r["metrics"][m]["top1"] for r in gs])
                    ent[m] = {"win_rate": float(win[mask].mean()) if mask.any() else None,
                              "top1": float(t1[mask].mean()) if mask.any() else None}
                    bucket_tab.append({"group": g, "variable": name, "bucket": lab, "method": m, "notes": int(mask.sum()),
                                       "win_rate": ent[m]["win_rate"], "top1": ent[m]["top1"]})
                results_buckets[g][name][lab] = ent

    # ---- local x global 2x2 on production terms, calibration saturation, bleed proxy, song-level correlates
    extra: Dict[str, Any] = {}
    for g in ("dev", "axis2_50", "robustness", "all_63"):
        gs = [r for r in songs if in_group[g](r)]
        lp = np.concatenate([r["metrics"]["rank_local_slack"]["place"] for r in gs])
        gp = np.concatenate([r["metrics"]["global"]["place"] for r in gs])
        prod_orig = np.concatenate([r["metrics"][PRODUCTION]["orig"] for r in gs])
        prod_all = np.concatenate([r["S"][PRODUCTION][r["notes"]["eligible"]] for r in gs])
        pcs = np.concatenate([r["notes"]["pc"][r["notes"]["eligible"]] for r in gs])
        alt_mask = np.ones_like(prod_all, dtype=bool)
        alt_mask[np.arange(pcs.size), pcs] = False
        cal_o, cal_a = calibrated(prod_orig), calibrated(prod_all[alt_mask])
        bl = np.concatenate([r["notes"]["bleed"][r["notes"]["eligible"]] for r in gs])
        extra[g] = {
            "local_x_global": {"local_top3_and_in_key": float(np.mean((lp <= 3) & (gp <= 7))),
                               "local_weak_in_key (passing/non-chord-tone candidates)": float(np.mean((lp > 3) & (gp <= 7))),
                               "local_top3_out_of_key": float(np.mean((lp <= 3) & (gp > 7))),
                               "both_weak (chromatic / out-of-key / extraction-error candidates)": float(np.mean((lp > 3) & (gp > 7)))},
            "calibration_saturation": {"orig_at_100pct": float(np.mean(cal_o >= 1.0)), "orig_at_0pct": float(np.mean(cal_o <= 0.0)),
                                       "alternatives_at_100pct": float(np.mean(cal_a >= 1.0)), "alternatives_at_0pct": float(np.mean(cal_a <= 0.0)),
                                       "orig_mean_calibrated": float(cal_o.mean()), "alternatives_mean_calibrated": float(cal_a.mean())},
            "bleed_proxy_orig_rank_during_minus_adjacent_unvoiced": {**summarize(bl.tolist()), "notes_without_context": int(np.sum(~np.isfinite(bl))),
                                                                      "fraction_positive": float(np.mean(bl[np.isfinite(bl)] > 0))},
        }

    def corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])

    a63 = {r["song"]: r["audit"] for r in songs}
    wr = [by_song[(r["song"], PRODUCTION)]["win_rate"] for r in songs]
    extra["song_level_spearman_vs_production_win_rate_all_63"] = {
        "vocal_coverage": corr([a63[r["song"]]["vocal_coverage"] for r in songs], wr),
        "median_note_clarity": corr([a63[r["song"]]["median_note_clarity"] for r in songs], wr),
        "mean_note_conf": corr([a63[r["song"]]["mean_note_conf"] for r in songs], wr),
        "tie_frame_fraction": corr([a63[r["song"]]["tie_frame_fraction"] for r in songs], wr),
        "median_bleed_proxy": corr([np.nanmedian(r["notes"]["bleed"][r["notes"]["eligible"]]) for r in songs], wr),
        "n_notes": corr([a63[r["song"]]["n_eligible"] for r in songs], wr),
        "tempo_bpm": corr([a63[r["song"]]["tempo_bpm"] for r in songs], wr)}
    # perturbation failures (production, all 63): which affected notes did not decrease
    aff = np.array([(x[4], x[5], x[6], x[7]) for x in affected_all], dtype=float)
    pats = np.array([x[2] for x in affected_all])
    shifts = np.array([x[3] for x in affected_all])
    fail = aff[:, 0] <= TOL
    extra["production_perturbation_failures_all_63"] = {
        pat: {"affected_notes": int(np.sum(pats == pat)), "not_lowered_fraction": float(np.mean(fail[pats == pat])),
              "not_lowered_by_shift": {str(s): float(np.mean(fail[(pats == pat) & (shifts == s)])) for s in sorted(set(shifts[pats == pat].tolist()))},
              "not_lowered_by_duration": {lab: float(np.mean(fail[(pats == pat) & (bucket(aff[:, 1], DUR_BUCKETS) == i)]))
                                          for i, lab in enumerate(bucket_labels(DUR_BUCKETS)) if np.any((pats == pat) & (bucket(aff[:, 1], DUR_BUCKETS) == i))},
              "not_lowered_by_clarity": {lab: float(np.mean(fail[(pats == pat) & (bucket(aff[:, 2], clar_edges) == i)]))
                                         for i, lab in enumerate(["low", "mid", "high"])}}
        for pat in PATTERNS}
    # robustness group by exclusion category (keyword, fixed)
    cats = {"rap_or_speech_like": ("rap", "speech"), "multiple_leads": ("two ", "multiple", "multi-artist"),
            "weak_extraction": ("weak", "low vocal coverage", "weakest", "only 145"), "heavy_processing": ("processed", "effected", "stylized")}
    extra["robustness_by_exclusion_category"] = {}
    for cname, keys in cats.items():
        members = [r for r in songs if r["group"] == "robustness" and any(k in r["audit"]["axis2_exclusion_reason"].lower() for k in keys)]
        extra["robustness_by_exclusion_category"][cname] = {
            "songs": [r["song"] for r in members],
            "production_song_mean_win_rate": summarize([by_song[(r["song"], PRODUCTION)]["win_rate"] for r in members])}

    # ---- pre-registered selection (dev) and paired bootstrap vs production
    dev = [r for r in songs if r["group"] == "dev"]
    prod_dev = np.array([by_song[(r["song"], PRODUCTION)]["win_rate"] for r in dev])
    paired = {}
    for mi, m in enumerate(METHODS):
        x = np.array([by_song[(r["song"], m)]["win_rate"] for r in dev]) - prod_dev
        paired[m] = {"dev_mean_win_rate": float(np.mean(x + prod_dev)), "diff_vs_production": float(x.mean()),
                     "ci95": bootstrap_mean_ci(x, BOOTSTRAP_SEED + mi) if m != PRODUCTION else [0.0, 0.0],
                     "songs_better": int(np.sum(x > TOL)), "songs_worse": int(np.sum(x < -TOL)), "exploratory": m in EXPLORATORY}
    selected = max(METHODS, key=lambda m: paired[m]["dev_mean_win_rate"])
    hold = [r for r in songs if r["group"] == "holdout"]
    hx = np.array([by_song[(r["song"], selected)]["win_rate"] - by_song[(r["song"], PRODUCTION)]["win_rate"] for r in hold])
    holdout_eval = {"selected": selected, "selected_is_production": selected == PRODUCTION,
                    "holdout_diff_mean": float(hx.mean()), "holdout_songs_better": int(np.sum(hx > TOL)),
                    "holdout_songs_worse": int(np.sum(hx < -TOL)), "holdout_ci95": bootstrap_mean_ci(hx, BOOTSTRAP_SEED + 999) if hx.any() else [0.0, 0.0],
                    "rule": ">= 7 of 10 holdout songs better than production to call it better",
                    "verdict": ("better than production on holdout" if np.sum(hx > TOL) >= 7 else "not distinguishable from production on 10 holdout songs")
                    if selected != PRODUCTION else "production itself was selected",
                    "per_song": {r["song"]: {"selected": by_song[(r["song"], selected)]["win_rate"], "production": by_song[(r["song"], PRODUCTION)]["win_rate"]}
                                 for r in hold}}
    # runner-up among non-exploratory, for context
    non_exp = sorted((m for m in METHODS if m not in EXPLORATORY), key=lambda m: -paired[m]["dev_mean_win_rate"])

    # ---- verification
    src_hashes_after = {s: {f: sha256_file(MAIN / s / f) for f in ("notes.npz", "harmony.npz")} for s in sorted(sel)}
    audits = [r["audit"] for r in songs]
    verification = {
        "source_files_unchanged": src_hashes_before == src_hashes_after,
        "production_reproduction_max_abs_diff_harmonic_rank": max(a["repro_max_abs_diff_harmonic_rank"] for a in audits),
        "production_reproduction_max_abs_diff_local_support": max(a["repro_max_abs_diff_local_support"] for a in audits),
        "stored_rank_reproduces_production_to_rank_all_songs": all(a["rank_reproduces_production"] for a in audits),
        "global_profile_and_rank_consistent_all_songs": all(a["global_profile_matches_cens_mean"] and a["global_rank_matches_profile"] for a in audits),
        "metadata_vs_harmony_sha256": all(json.loads((MAIN / s / "metadata.json").read_text())["harmony_npz_sha256"] == src_hashes_before[s]["harmony.npz"] for s in sel),
        "win_rate_bounds_ok": all(0 <= row["win_rate"] <= 1 for row in song_rows),
        "global_method_win_rate_equals_mean_global_placement_identity": None,
    }
    # identity check: for 'global', per-note win rate = (12 - placement)/11 exactly (no ties in global profile)
    gw = np.concatenate([r["metrics"]["global"]["win"] for r in songs])
    gpl = np.concatenate([r["metrics"]["global"]["place"] for r in songs])
    verification["global_method_win_rate_equals_mean_global_placement_identity"] = bool(np.allclose(gw, (12 - gpl) / 11))

    # ---- data audit totals
    data_audit = {
        "songs_in_manifest": len(manifest["songs"]), "songs_processed": sum(r["processing_status"] == "processed" for r in manifest["songs"]),
        "songs_with_usable_axis1_data": int(sum(a["n_eligible"] > 0 for a in audits)),
        "notes_total": int(sum(a["n_notes"] for a in audits)), "notes_eligible": int(sum(a["n_eligible"] for a in audits)),
        "notes_high_confidence": int(sum(a["n_high_conf"] for a in audits)),
        "by_group": {g: {"songs": int(sum(1 for r in songs if in_group[g](r))),
                         "notes_total": int(sum(r["audit"]["n_notes"] for r in songs if in_group[g](r))),
                         "notes_eligible": int(sum(r["audit"]["n_eligible"] for r in songs if in_group[g](r))),
                         "notes_high_confidence": int(sum(r["audit"]["n_high_conf"] for r in songs if in_group[g](r)))} for g in GROUPS},
        "invalid_reason_totals": {k: int(sum(a[k] for a in audits)) for k in audits[0] if k.startswith("invalid_")},
        "flags_totals": {k: int(sum(a[k] for a in audits)) for k in ("overlapping_notes", "unsorted_note_starts", "contour_outside_note_bounds",
                                                                      "frames_below_crepe_threshold", "zero_chroma_frames", "notes_under_0p1s",
                                                                      "notes_one_chroma_frame", "pc_ambiguous_notes")},
        "tie_frame_fraction": summarize([a["tie_frame_fraction"] for a in audits]),
        "rank_entries_changed_by_avg_ties": summarize([a["rank_entries_changed_by_avg_ties"] for a in audits]),
        "timing": {"chroma_end_minus_vocal_end_s": summarize([a["chroma_end_s"] - a["vocal_end_s"] for a in audits]),
                   "chroma_end_minus_metadata_duration_s": summarize([a["chroma_end_s"] - a["metadata_duration_s"] for a in audits]),
                   "last_note_end_minus_chroma_end_s": summarize([a["last_note_end_s"] - a["chroma_end_s"] for a in audits])},
        "tempo_bpm": summarize([a["tempo_bpm"] for a in audits]), "slack_frames": summarize([a["slack_frames"] for a in audits]),
        "n_beats": summarize([a["n_beats"] for a in audits]), "tuning_cents": summarize([a["tuning_cents"] for a in audits]),
        "vocal_coverage": summarize([a["vocal_coverage"] for a in audits]),
        "not_measurable_from_saved_data": ["a cappella guard (stem RMS not saved)", "bleed test (no bass or other+bass chroma saved)",
                                           "percussion-heavy sections (percussive stem not saved; CENS is per-frame normalized so energy is lost)",
                                           "alternative chroma types (CQT, HPSS, log-compression)", "backing-vocal presence (no multi-f0)"],
    }

    # ---- write outputs
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, rows: List[Dict[str, Any]]):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(OUT / "data_audit_songs.csv", audits)
    write_csv(OUT / "song_level_methods.csv", song_rows)
    write_csv(OUT / "method_comparison.csv", comparison)
    write_csv(OUT / "interval_win_rates.csv", interval_tab)
    write_csv(OUT / "perturbation_summary.csv", pert_tab)
    write_csv(OUT / "transposition_songs.csv", trans_rows)
    write_csv(OUT / "diagnostic_buckets.csv", bucket_tab)
    note_path = ARCHIVE / "note_level.csv.gz"
    with gzip.open(note_path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["song_id", "group", "note_index", "pitch_class", "start_s", "duration_s", "mean_conf", "cents_sd", "abs_core_cents",
                    "chroma_clarity", "tie_exposure", "bleed_proxy", "high_conf", "prod_score", "prod_win_rate", "prod_placement",
                    "prod_margin", "local_placement", "global_placement", *[f"win_{m}" for m in METHODS]])
        w.writerows(note_rows)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    results = {
        "experiment": "axis1_chroma_evaluation", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "protocol.md", "splits": "axis1_splits.json",
        "provenance": {"git_commit": git, "uncommitted_changes": dirty, "script_sha256": sha256_file(Path(__file__)),
                       "protocol_sha256": sha256_file(OUT / "protocol.md"), "splits_sha256": sha256_file(OUT / "axis1_splits.json"),
                       "feature_store_schema": manifest["dataset_schema_version"], "segmentation_version": manifest["segmentation_version"],
                       "python": platform.python_version(), "numpy": np.__version__,
                       "note_level_table": {"path": str(note_path.relative_to(REPO)), "sha256": sha256_file(note_path), "rows": len(note_rows)}},
        "settings": {"tolerance": TOL, "perturb_fraction": PERTURB_FRACTION, "perturb_draws": PERTURB_DRAWS, "patterns": PATTERNS,
                     "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED}, "high_conf_rule": HC,
                     "production_constants": {"RANK_LOW": H.RANK_LOW, "RANK_HIGH": H.RANK_HIGH, "GLOBAL_WEIGHT": H.GLOBAL_WEIGHT,
                                              "DEFAULT_SLACK_S": H.DEFAULT_SLACK_S}, "methods": METHODS, "exploratory": sorted(EXPLORATORY)},
        "verification": verification, "data_audit": data_audit, "groups": results_groups,
        "selection": {"rule": "highest dev mean song-level pairwise win rate (protocol.md)", "selected": selected,
                      "ranking_dev": sorted(METHODS, key=lambda m: -paired[m]["dev_mean_win_rate"]),
                      "best_non_exploratory": non_exp[0], "paired_vs_production_dev": paired, "holdout": holdout_eval},
        "diagnostic_buckets": results_buckets, "diagnostics": extra,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float) + "\n", encoding="utf-8")
    write_tables(results, OUT / "tables.md")
    print(json.dumps(verification, indent=1))
    print("selected:", selected, "| holdout:", holdout_eval["verdict"])
    return 0 if verification["source_files_unchanged"] else 1


def write_tables(R: Dict[str, Any], path: Path) -> None:
    """Machine-generated tables (no interpretation) so summary.md never hand-copies numbers."""
    L = ["# Axis 1 chroma evaluation — generated tables", "",
         f"Generated by `axis1_chroma_evaluation.py` at {R['created_utc']}. Interpretation is in `summary.md`.", ""]
    f = lambda x, d=3: "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))  # noqa: E731
    for g in GROUPS:
        G = R["groups"][g]
        L += [f"## Method comparison — {g} ({G['songs']} songs, {G['notes']} notes)", "",
              "| method | song-mean win | song-median | worst song | pooled win | top-1 | top-3 | mean place | median margin | win (high-conf) | transp. #1 | transp. fails→P4/P5 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            e = G["methods"][m]
            t = e["transposition"]
            L.append(f"| {m}{' *' if m in EXPLORATORY else ''} | {f(e['song_mean_win_rate']['mean'])} | {f(e['song_mean_win_rate']['median'])} | "
                     f"{f(e['song_mean_win_rate']['min'])} | {f(e['pooled_win_rate'])} | {f(e['pooled_top1'])} | {f(e['pooled_top3'])} | "
                     f"{f(e['pooled_mean_placement'], 2)} | {f(e['pooled_margin_median'], 4)} | {f(e['pooled_win_rate_high_conf'])} | "
                     f"{t['songs_rank_first']}/{t['songs']} | {t['failures_to_p4_p5']}/{t['failures']} |")
        L += ["", "\\* exploratory", ""]
    sel = R["selection"]
    L += ["## Dev paired comparison vs production (song-level win rate)", "",
          "| method | dev mean | diff vs prod | 95% CI (song bootstrap) | songs better | songs worse |", "|---|---|---|---|---|---|"]
    for m in sel["ranking_dev"]:
        p = sel["paired_vs_production_dev"][m]
        L.append(f"| {m}{' *' if p['exploratory'] else ''} | {p['dev_mean_win_rate']:.4f} | {p['diff_vs_production']:+.4f} | "
                 f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | {p['songs_better']} | {p['songs_worse']} |")
    h = sel["holdout"]
    L += ["", f"Selected on dev: **{sel['selected']}**. Holdout: {h['verdict']} (diff {h['holdout_diff_mean']:+.4f}, "
              f"better on {h['holdout_songs_better']}/10, worse on {h['holdout_songs_worse']}/10).", ""]
    for g in ("dev", "axis2_50", "robustness", "all_63"):
        L += [f"## Win rate vs specific intervals — {g}", "", "| method | " + " | ".join(INTERVAL_NAMES[k] for k in range(1, 12)) + " |",
              "|---" * 12 + "|"]
        for m in [PRODUCTION, "rank_local_slack", "rank_local", "raw_local", "global", "geo_w25_avgties", "chord_geo_w25"]:
            iw = R["groups"][g]["methods"][m]["interval_win_rate"]
            L.append(f"| {m} | " + " | ".join(f"{iw[INTERVAL_NAMES[k]]:.3f}" for k in range(1, 12)) + " |")
        L.append("")
    for g in ("dev", "axis2_50", "robustness", "all_63"):
        L += [f"## Perturbation (20% of notes, 5 draws/song) — {g}", "",
              "| method | pattern | draws lowering | songs 5/5 | mean song drop | worst song drop | affected notes lowered | headline pp drop (prod) |",
              "|---|---|---|---|---|---|---|---|"]
        for m in [PRODUCTION, "rank_local", "raw_local", "global", "geo_w25_avgties", "chord_geo_w25"]:
            for pat in PATTERNS:
                e = R["groups"][g]["methods"][m]["perturbation"][pat]
                hl = e.get("headline_pp_drop_mean", {}).get("mean")
                L.append(f"| {m} | {pat} | {e['draws_lowering']}/{e['draws_total']} | {e['songs_all_draws_lowering']} | "
                         f"{e['mean_drop']['mean']:+.4f} | {e['mean_drop']['min']:+.4f} | {e['affected_note_fraction_decreased']['mean']:.3f} | "
                         f"{'' if hl is None else f'{hl:+.2f}'} |")
        L.append("")
    for g in ("dev", "holdout", "robustness", "all_63"):
        L += [f"## Transposition — {g}", "", "| method | #1 songs | placement distribution | margin median | failures | of which P4/P5 | best competitor P4/P5 (all songs) | #1 (dur-weighted) |",
              "|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            t = R["groups"][g]["methods"][m]["transposition"]
            L.append(f"| {m} | {t['songs_rank_first']}/{t['songs']} | {t['placement_distribution']} | {t['margin']['median']:+.4f} | "
                     f"{t['failures']} | {t['failures_to_p4_p5']} | {t['best_competitor_is_p4_p5_all_songs']} | {t['songs_rank_first_duration_weighted']} |")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
