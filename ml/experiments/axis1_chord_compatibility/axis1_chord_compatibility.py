"""
Axis 1 Experiment 2 — chord-aware harmonic compatibility (experimental; not production).

Stages (protocol.md):
  1. calibrate the chord estimator on synthetic audio sets A (fit) / B (evaluate)
  2. choose the temporal estimator and key temperature on synthetic progressions (set C)
  3. run the pre-specified synthetic musical tests (template level and audio level)
  4. evaluate all 63 real songs (read-only), with ablations, stability and failure analysis

No parameter is fitted to real songs. Usage, from the repo root:
    venv/bin/python ml/experiments/axis1_chord_compatibility/axis1_chord_compatibility.py [--overwrite]
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
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAIN = REPO / "data" / "main_data"
ARCHIVE = REPO / "data" / "experiment_archives" / "axis1_chord_compatibility"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "backend"))

import chord_model as M  # noqa: E402
import synthetic_audio as SA  # noqa: E402
import harmony as H  # noqa: E402

SEED_A, SEED_B, SEED_C, SEED_T = 101, 202, 303, 404
TAU_GRID = (0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12)
THETA_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TAU_KEY_GRID = (0.02, 0.05, 0.1, 0.2, 0.3)
TEMPORALS = ("frame", "frame_hmm", "beat", "beat_hmm")
NOISE_SIGMA = 0.10
SHIFT_S = 0.05
TOL = 1e-9


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ stage 1: chord-estimator calibration


def synth_frames(vocab: M.Vocabulary, seed: int, degraded: bool) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []

    def take(cens):
        t = SA.frame_times(cens.shape[1])
        sel = np.flatnonzero((t >= 0.6) & (t <= 1.4))[::3]
        return cens[:, sel].T

    for k, s in enumerate(vocab.pitch_sets):
        for timbre in SA.TIMBRES:
            for _ in range(2):
                a = SA.chord_audio(SA.voicing(sorted(s), rng), 2.0, timbre, rng)
                if degraded:
                    extra = int(rng.choice([p for p in range(12) if p not in s]))
                    x = SA.tone(60 + extra, 2.0, timbre)
                    # one non-chord tone at half the RMS of an average chord tone (bleed / passing instrument)
                    x *= 0.5 * (np.sqrt(np.mean(a ** 2)) / np.sqrt(len(s))) / np.sqrt(np.mean(x ** 2))
                    a = a + x
                    a = SA.add_noise(a, 15.0, rng)
                else:
                    a = SA.add_noise(a, 40.0, rng)
                f = take(SA.production_cens(a))
                X.append(f)
                y += [k] * len(f)
    for i in range(100):
        n = int(2.0 * SA.SR)
        if i < 30:
            a = SA.white(n, rng) * 0.1
        elif i < 60:
            a = SA.pink(n, rng) * 0.1
        else:
            pcs = rng.choice(12, size=6, replace=False)
            a = SA.chord_audio(SA.voicing(sorted(pcs), rng), 2.0, str(rng.choice(list(SA.TIMBRES))), rng)
        f = take(SA.production_cens(a))
        X.append(f)
        y += [vocab.K] * len(f)
    return np.vstack(X), np.array(y)


def calibrate(vocab: M.Vocabulary) -> Dict[str, Any]:
    XA, yA = synth_frames(vocab, SEED_A, degraded=False)
    XB, yB = synth_frames(vocab, SEED_B, degraded=False)
    XD, yD = synth_frames(vocab, SEED_B + 1, degraded=True)
    K = vocab.K
    out: Dict[str, Any] = {"frames": {"A": int(len(yA)), "B_clean": int(len(yB)), "B_degraded": int(len(yD))}, "candidates": {}}
    qual_of = [vocab.qualities[k][0] if len(vocab.qualities[k]) == 1 else "+".join(vocab.qualities[k]) for k in range(K)]
    for template in ("binary", "harmonic"):
        for simk in ("cosine", "pearson"):
            T = vocab.binary if template == "binary" else vocab.harmonic
            sA = M.similarity(XA, T, simk)
            best = None
            for tau in TAU_GRID:
                for theta in THETA_GRID:
                    P = M.posterior(sA, tau, theta)
                    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(yA)), yA], 1e-300, None))))
                    if best is None or ll < best[0]:
                        best = (ll, tau, theta)
            ll, tau, theta = best
            res = {"tau": tau, "theta": theta, "logloss_A": ll}
            for name, X, y in (("clean", XB, yB), ("degraded", XD, yD)):
                P = M.posterior(M.similarity(X, T, simk), tau, theta)
                pred = np.argmax(P, axis=1)
                chord = y < K
                top3 = np.argsort(-P[:, :K], axis=1)[:, :3]
                per_q = {}
                for q in sorted(set(qual_of)):
                    m = chord & np.array([qual_of[k] == q for k in np.minimum(y, K - 1)])
                    if m.any():
                        wrong = m & (pred != y)
                        conf = {}
                        for k in pred[wrong]:
                            lab = "N" if k == K else qual_of[k]
                            conf[lab] = conf.get(lab, 0) + 1
                        per_q[q] = {"top1": float(np.mean(pred[m] == y[m])), "top3": float(np.mean(np.any(top3[m] == y[m, None], axis=1))),
                                    "frames": int(m.sum()), "confused_with": dict(sorted(conf.items(), key=lambda kv: -kv[1])[:4])}
                res[name] = {"top1_chord": float(np.mean(pred[chord] == y[chord])),
                             "top3_chord": float(np.mean(np.any(top3[chord] == y[chord, None], axis=1))),
                             "N_recall": float(np.mean(pred[~chord] == K)), "false_N_on_chords": float(np.mean(pred[chord] == K)),
                             "logloss": float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None)))),
                             "per_quality": per_q}
            res["selection_score"] = 0.5 * (res["clean"]["top1_chord"] + res["degraded"]["top1_chord"])
            out["candidates"][f"{template}/{simk}"] = res
    sel = max(out["candidates"], key=lambda k: out["candidates"][k]["selection_score"])
    t, s = sel.split("/")
    out["selected"] = {"template": t, "sim": s, "tau": out["candidates"][sel]["tau"], "theta": out["candidates"][sel]["theta"]}
    return out


# ================================================================ stage 2: temporal estimator and key temperature


def make_progressions(vocab: M.Vocabulary, seed: int):
    rng = np.random.default_rng(seed)
    set_index = {s: k for k, s in enumerate(vocab.pitch_sets)}
    major = [(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min")]
    minor = [(0, "min"), (3, "maj"), (5, "min"), (7, "maj"), (8, "maj"), (10, "maj")]
    variants = {"maj": [("dom7", 0.25), ("maj7", 0.15), ("sus4", 0.08)], "min": [("min7", 0.2)]}
    progs = []
    for i in range(48):
        d = (1, 2, 4)[i % 3]
        mode = "major" if rng.random() < 0.6 else "minor"
        tonic = int(rng.integers(12))
        degs = major if mode == "major" else minor
        n_ch = 32 // d
        seq, last = [], None
        for j in range(n_ch):
            while True:
                deg = degs[0] if j == 0 else degs[int(rng.integers(len(degs)))]
                if deg != last:
                    break
            last = deg
            q = deg[1]
            for vq, p in variants.get(q, []):
                if rng.random() < p:
                    q = vq
                    break
            root = (tonic + deg[0]) % 12
            seq.append(frozenset((root + iv) % 12 for iv in M.QUALITIES[q]))
        tempo = float(rng.uniform(80, 130))
        timbre = str(rng.choice(list(SA.TIMBRES)))
        y, beats, segs = SA.progression(seq, [d] * n_ch, tempo, timbre, rng)
        cens = SA.production_cens(y)
        t = SA.frame_times(cens.shape[1])
        bounds = np.array([s[0] for s in segs] + [segs[-1][1]])
        truth = np.array([set_index[segs[min(np.searchsorted(bounds, x, side="right") - 1, len(segs) - 1)][2]] for x in t])
        dist = np.min(np.abs(t[:, None] - bounds[None, 1:-1]), axis=1) if len(bounds) > 2 else np.full(t.size, np.inf)
        progs.append({"d": d, "mode": mode, "tonic": tonic, "tempo": tempo, "cens": cens, "t": t, "beats": beats,
                      "truth": truth, "dist": dist, "bounds": bounds, "key": tonic if mode == "major" else 12 + tonic})
    return progs


def temporal_eval(vocab: M.Vocabulary, sel: Dict[str, Any], progs) -> Dict[str, Any]:
    out: Dict[str, Any] = {"progressions": len(progs), "estimators": {}}
    K = vocab.K
    for temporal in TEMPORALS:
        cfg = M.EstimatorConfig(template=sel["template"], sim=sel["sim"], tau=sel["tau"], theta=sel["theta"], temporal=temporal)
        by_d: Dict[int, Dict[str, List[float]]] = {}
        dist_bins = {"<0.1s": [], "0.1-0.25s": [], "0.25-0.5s": [], ">0.5s": []}
        for p in progs:
            P, _ = M.frame_posteriors(p["cens"], p["t"], p["beats"], p["tempo"], vocab, cfg, M.HOP_S)
            pred = np.argmax(P, axis=1)
            ok = pred == p["truth"]
            beat = 60.0 / p["tempo"]
            centers = p["beats"][:-1] + beat / 2
            bi = np.clip(np.searchsorted(p["t"], centers), 0, p["t"].size - 1)
            seq = pred[bi]
            truth_seq = p["truth"][bi]
            flick = float(np.mean((seq[2:] == seq[:-2]) & (seq[1:-1] != seq[:-2]))) if seq.size > 2 else 0.0
            changes = np.flatnonzero(np.diff(pred) != 0)
            ctimes = p["t"][changes]
            tb = p["bounds"][1:-1]
            recall = float(np.mean([np.any(np.abs(ctimes - b) <= 0.5 * beat) for b in tb])) if tb.size else float("nan")
            r = by_d.setdefault(p["d"], {k: [] for k in ("acc", "acc_interior", "switch_per_beat", "true_switch_per_beat", "flicker", "boundary_recall", "N_share")})
            r["acc"].append(float(ok.mean()))
            r["acc_interior"].append(float(ok[p["dist"] > 0.3].mean()))
            r["switch_per_beat"].append(float(np.sum(np.diff(seq) != 0) / max(seq.size - 1, 1)))
            r["true_switch_per_beat"].append(float(np.sum(np.diff(truth_seq) != 0) / max(seq.size - 1, 1)))
            r["flicker"].append(flick)
            r["boundary_recall"].append(recall)
            r["N_share"].append(float(np.mean(pred == K)))
            for lab, (lo, hi) in {"<0.1s": (0, 0.1), "0.1-0.25s": (0.1, 0.25), "0.25-0.5s": (0.25, 0.5), ">0.5s": (0.5, np.inf)}.items():
                m = (p["dist"] >= lo) & (p["dist"] < hi)
                if m.any():
                    dist_bins[lab].append(float(ok[m].mean()))
        summ = {f"{d}_beat_chords": {k: float(np.nanmean(v)) for k, v in r.items()} for d, r in sorted(by_d.items())}
        summ["mean_acc_over_durations"] = float(np.mean([summ[f"{d}_beat_chords"]["acc"] for d in (1, 2, 4)]))
        summ["acc_by_distance_to_true_boundary"] = {k: float(np.mean(v)) for k, v in dist_bins.items()}
        out["estimators"][temporal] = summ
    out["selected"] = max(TEMPORALS, key=lambda k: out["estimators"][k]["mean_acc_over_durations"])
    # key temperature
    best = None
    for tk in TAU_KEY_GRID:
        ll = float(-np.mean([np.log(max(M.key_posterior(p["cens"].mean(axis=1), tk)[p["key"]], 1e-300)) for p in progs]))
        acc = float(np.mean([np.argmax(M.key_posterior(p["cens"].mean(axis=1), tk)) == p["key"] for p in progs]))
        if best is None or ll < best[0]:
            best = (ll, tk, acc)
    kp = [np.argmax(M.key_posterior(p["cens"].mean(axis=1), best[1])) for p in progs]
    rel = sum(1 for p, k in zip(progs, kp) if k != p["key"] and (k % 12, k // 12) in
              {((p["tonic"] + 9) % 12, 1) if p["mode"] == "major" else ((p["tonic"] + 3) % 12, 0)})
    fifth = sum(1 for p, k in zip(progs, kp) if k != p["key"] and k // 12 == p["key"] // 12 and (k - p["key"]) % 12 in (5, 7))
    out["key"] = {"tau_key": best[1], "logloss": best[0], "map_accuracy": best[2], "errors_relative_major_minor": rel, "errors_fourth_fifth": fifth,
                  "errors_total": int(sum(1 for p, k in zip(progs, kp) if k != p["key"]))}
    return out


# ================================================================ production reference (12-vector)


def production_vectors(cens, rank, t, tempo, g_rank, start, end):
    nf = t.size
    slack_s = 60.0 / tempo if tempo > 0 else H.DEFAULT_SLACK_S
    sf = int(round(slack_s / M.HOP_S))
    i0, w = M.production_windows(t, start, end)
    cum = np.concatenate([np.zeros((12, 1)), np.cumsum(rank, axis=1)], axis=1)
    mean = lambda a, ww: ((cum[:, a + ww] - cum[:, a]) / ww).T  # noqa: E731
    span = mean(i0, w)
    lo, hi = i0 - sf, i0 + sf
    ok_lo, ok_hi = (lo >= 0) & (lo + w <= nf) & (sf != 0), (hi >= 0) & (hi + w <= nf) & (sf != 0)
    loc = span.copy()
    loc = np.where(ok_lo[:, None], np.maximum(loc, mean(np.clip(lo, 0, nf - 1), np.where(ok_lo, w, 1))), loc)
    loc = np.where(ok_hi[:, None], np.maximum(loc, mean(np.clip(hi, 0, nf - 1), np.where(ok_hi, w, 1))), loc)
    comb = loc ** 0.75 * np.maximum(g_rank, 1e-6)[None, :] ** 0.25
    craw = np.concatenate([np.zeros((12, 1)), np.cumsum(cens, axis=1)], axis=1)
    raw = ((craw[:, i0 + w] - craw[:, i0]) / w).T
    return {"prod_calibrated": np.clip((comb - H.RANK_LOW) / (H.RANK_HIGH - H.RANK_LOW), 0, 1), "prod_harmonic_rank": comb,
            "raw_chroma": raw / np.clip(raw.max(axis=1, keepdims=True), 1e-12, None), "rank_chroma": span}


# ================================================================ stage 3: synthetic musical tests


def run_case_chroma(chroma, t, beats, tempo, notes, vocab, cfg, tau_key):
    start = np.array([n[0] for n in notes], float)
    end = np.array([n[1] for n in notes], float)
    midi = np.array([n[2] for n in notes], int)
    A = M.analyze(chroma, t, beats, tempo, chroma.mean(axis=1), start, end, midi, vocab, cfg, tau_key)
    rank = H._to_rank(chroma)
    g = chroma.mean(axis=1)
    g_rank = np.argsort(np.argsort(g)) / 11.0
    prod = production_vectors(chroma, rank, t, tempo, g_rank, start, end)["prod_calibrated"]
    return A, prod


def synthetic_tests(vocab, cfg, tau_key) -> Dict[str, Any]:
    tempo = 100.0
    beat = 0.6
    C, Am, G7, Cmaj7, Csus4, F, G = ({0, 4, 7}, {9, 0, 4}, {7, 11, 2, 5}, {0, 4, 7, 11}, {0, 5, 7}, {5, 9, 0}, {7, 11, 2})

    def accomp(segs, layer, seed, total=None, extra=None):
        """segs: list of (start, end, pcs | 'noise' | ('voicing', base)). Returns chroma (12, nf), t, beats."""
        total = total or segs[-1][1]
        beats = np.arange(0, total + 1e-9, beat)
        if layer == "T":
            nf = int(np.ceil(total / M.HOP_S)) + 1
            t = SA.frame_times(nf)
            X = np.zeros((12, nf))
            for a, b, s in segs:
                m = (t >= a) & (t < b + 1e-9)
                v = np.ones(12) if s == "noise" else M.harmonic_chroma(sorted(s[1] if isinstance(s, tuple) else s))
                X[:, m] = (v / np.linalg.norm(v))[:, None]
            return X, t, beats
        rng = np.random.default_rng(seed)
        parts = []
        for a, b, s in segs:
            n = int(round((b - a) * SA.SR))
            if s == "noise":
                parts.append(SA.white(n, rng) * 0.1)
            elif isinstance(s, tuple):
                parts.append(SA.chord_audio(s[2], b - a, "soft", rng))
            else:
                mids = [48 + p for p in sorted(s)] if len(s) <= 2 else SA.voicing(sorted(s), rng)
                if len(s) == 1:
                    mids = [48 + list(s)[0], 60 + list(s)[0]]
                parts.append(SA.chord_audio(mids, b - a, "soft", rng))
        y = np.concatenate(parts)
        y = SA.add_noise(y, extra if extra is not None else 40.0, rng)
        X = SA.production_cens(y)
        return X, SA.frame_times(X.shape[1]), beats

    cases = []
    probe = (1.5, 2.5, 64)
    for layer in ("T", "S"):
        res: Dict[str, Any] = {}

        def single(name, pcs, notes=(probe,), seed=1, extra=None):
            X, t, b = accomp([(0, 4.0, pcs)], layer, seed, extra=extra)
            A, prod = run_case_chroma(X, t, b, tempo, list(notes), vocab, cfg, tau_key)
            res[name] = (A, prod, X, t, b)
            return A, prod

        A1, p1 = single("1", C)
        A2, p2 = single("2", Am)
        A3, p3 = single("3", G7)
        A4, p4 = single("4", Cmaj7)
        A5, p5 = single("5", Csus4)
        A16, p16 = single("16", C, extra=10.0) if layer == "S" else single("16", C)
        A13a, p13a = single("13a", "noise")
        A13b, p13b = single("13b", {0})
        A13c, p13c = single("13c", {0, 7})
        h = lambda A: A.span.H_inst[0]  # noqa: E731
        ct = lambda A, pcs: [float(h(A)[p]) for p in pcs]  # noqa: E731
        checks = []

        def chk(cid, desc, passed, detail, prod_ref=None):
            checks.append({"layer": layer, "case": cid, "check": desc, "pass": bool(passed), "detail": detail,
                           "production_reference": prod_ref})

        def fmt(v):
            return " ".join(f"{M.PC_NAMES[i]}={v[i]:.2f}" for i in range(12))

        v1 = h(A1)
        t3t4 = [p for p in range(12) if p not in (0, 4, 7, 2, 9, 11)]
        chk("1", "C,E,G >= 0.75", min(ct(A1, (0, 4, 7))) >= 0.75, fmt(v1), fmt(p1[0]))
        chk("1", "min(chord tones) > max(T3,T4)", min(ct(A1, (0, 4, 7))) > max(ct(A1, t3t4)), fmt(v1), fmt(p1[0]))
        chk("1", "D,A,B in [0.4,0.95) and each > C#,F#,G#",
            all(0.4 <= v1[p] < 0.95 for p in (2, 9, 11)) and min(v1[[2, 9, 11]]) > max(v1[[1, 6, 8]]), fmt(v1), fmt(p1[0]))
        v2 = h(A2)
        chk("2", "A,C,E >= 0.75", min(ct(A2, (9, 0, 4))) >= 0.75, fmt(v2), fmt(p2[0]))
        chk("2", "min(chord tones) > max(T4: A#, C#)", min(ct(A2, (9, 0, 4))) > max(ct(A2, (10, 1))), fmt(v2), fmt(p2[0]))
        chk("3", "G,B,D,F >= 0.75", min(ct(A3, (7, 11, 2, 5))) >= 0.75, fmt(h(A3)), fmt(p3[0]))
        v4 = h(A4)
        chk("4", "C,E,G,B >= 0.75", min(ct(A4, (0, 4, 7, 11))) >= 0.75, fmt(v4), fmt(p4[0]))
        chk("4", "F (avoid 11) < D (9)", v4[5] < v4[2], fmt(v4), fmt(p4[0]))
        v5 = h(A5)
        chk("5", "C,F,G >= 0.75", min(ct(A5, (0, 5, 7))) >= 0.75, fmt(v5), fmt(p5[0]))
        chk("5", "E (3rd) >= 0.25", v5[4] >= 0.25, fmt(v5), fmt(p5[0]))
        # 6 progression
        segs = [(0, 2.4, C), (2.4, 4.8, Am), (4.8, 7.2, F), (7.2, 9.6, G)]
        X, t, b = accomp(segs, layer, 6)
        probes = [(a + 0.9, a + 1.5, 64) for a, _, _ in segs]
        A6, p6 = run_case_chroma(X, t, b, tempo, probes, vocab, cfg, tau_key)
        set_idx = {s: k for k, s in enumerate(vocab.pitch_sets)}
        truth = np.array([set_idx[frozenset(segs[min(int(x // 2.4), 3)][2])] for x in t])
        bounds = np.array([2.4, 4.8, 7.2])
        interior = np.min(np.abs(t[:, None] - bounds[None, :]), axis=1) > 0.3
        acc6 = float(np.mean(A6.frame_top[interior] == truth[interior]))
        chk("6", "interior-frame top-chord accuracy >= 0.8", acc6 >= 0.8, f"accuracy={acc6:.3f}")
        seg_ok = [min(float(A6.span.H_inst[j][p]) for p in s) for j, (_, _, s) in enumerate(segs)]
        chk("6", "each segment's chord tones >= 0.75 at its centre", min(seg_ok) >= 0.75, "min chord-tone H per segment: " + ", ".join(f"{x:.2f}" for x in seg_ok),
            ", ".join(f"{min(float(p6[j][p]) for p in s):.2f}" for j, (_, _, s) in enumerate(segs)))
        # 7 held through change
        X, t, b = accomp([(0, 2.4, C), (2.4, 4.8, F)], layer, 7)
        A7, p7 = run_case_chroma(X, t, b, tempo, [(1.6, 3.2, 64)], vocab, cfg, tau_key)
        hc = float(A7.H_ctx[0, 4])
        mh = max(float(A7.H_half1[0, 4]), float(A7.H_half2[0, 4]))
        chk("7", "CROSSES_CHANGE set and H_ctx >= max(halves)", bool(A7.crosses[0]) and hc >= mh - 1e-9,
            f"crosses={bool(A7.crosses[0])} H_inst={A7.span.H_inst[0, 4]:.2f} half1={A7.H_half1[0, 4]:.2f} half2={A7.H_half2[0, 4]:.2f} H_ctx={hc:.2f} code={A7.code[0, 4]!r}",
            f"{p7[0, 4]:.2f}")
        # 8 octave / voicing
        X, t, b = accomp([(0, 4.0, C)], layer, 8)
        A8, _ = run_case_chroma(X, t, b, tempo, [(1.5, 2.5, 52), (1.5, 2.5, 64), (1.5, 2.5, 76)], vocab, cfg, tau_key)
        same = bool(np.array_equal(A8.span.H_inst[0], A8.span.H_inst[1]) and np.array_equal(A8.span.H_inst[1], A8.span.H_inst[2])
                    and np.array_equal(A8.H_ctx[0], A8.H_ctx[1]))
        chk("8", "same pitch class at MIDI 52/64/76 -> identical H", same, "exact equality of the 12-vectors")
        vv = []
        for base, mids in ((48, [48, 52, 55]), (55, [55, 60, 64]), (64, [64, 67, 72])):
            X, t, b = accomp([(0, 4.0, ("voicing", sorted(C), mids))], layer, 80 + base)
            A, _ = run_case_chroma(X, t, b, tempo, [probe], vocab, cfg, tau_key)
            vv.append(A.span.H_inst[0])
        vv = np.array(vv)
        chk("8", "3 voicings of C: chord tones >= 0.75 in each, max |dH| <= 0.15",
            float(vv[:, [0, 4, 7]].min()) >= 0.75 and float(np.max(vv.max(0) - vv.min(0))) <= 0.15,
            f"min chord tone={vv[:, [0, 4, 7]].min():.2f}, max |dH| across voicings={np.max(vv.max(0) - vv.min(0)):.3f}")
        # 9 extensions
        chk("9", "over C: D, A >= 0.4", v1[2] >= 0.4 and v1[9] >= 0.4, f"D={v1[2]:.2f} A={v1[9]:.2f}", f"D={p1[0][2]:.2f} A={p1[0][9]:.2f}")
        chk("9", "over G7: A (9) >= 0.4", h(A3)[9] >= 0.4, f"A={h(A3)[9]:.2f}", f"A={p3[0][9]:.2f}")
        chk("9", "over C: F (11) < D (9)", v1[5] < v1[2], f"F={v1[5]:.2f} D={v1[2]:.2f}", f"F={p1[0][5]:.2f} D={p1[0][2]:.2f}")
        # 10 passing tone
        X, t, b = accomp([(0, 4.0, C)], layer, 10)
        A10, p10 = run_case_chroma(X, t, b, tempo, [(1.0, 1.4, 64), (1.4, 1.55, 65), (1.55, 1.95, 67)], vocab, cfg, tau_key)
        chk("10", "F: H_inst < 0.5, code PASSING, H_ctx >= 2/3",
            A10.span.H_inst[1, 5] < 0.5 and A10.code[1, 5] == "PASSING" and A10.H_ctx[1, 5] >= 2 / 3 - 1e-9,
            f"H_inst={A10.span.H_inst[1, 5]:.2f} code={A10.code[1, 5]!r} H_ctx={A10.H_ctx[1, 5]:.2f}", f"{p10[1, 5]:.2f}")
        # 11 suspension, held
        X, t, b = accomp([(0, 2.4, F), (2.4, 4.8, C)], layer, 11)
        A11, p11 = run_case_chroma(X, t, b, tempo, [(1.2, 1.8, 69), (1.8, 3.0, 65), (3.0, 3.8, 64)], vocab, cfg, tau_key)
        chk("11", "held F: code SUSPENSION and H_ctx >= 2/3", "SUSPENSION" in A11.code[1, 5] and A11.H_ctx[1, 5] >= 2 / 3 - 1e-9,
            f"H_inst={A11.span.H_inst[1, 5]:.2f} code={A11.code[1, 5]!r} H_ctx={A11.H_ctx[1, 5]:.2f} crosses={bool(A11.crosses[1])}", f"{p11[1, 5]:.2f}")
        A11b, p11b = run_case_chroma(X, t, b, tempo, [(1.2, 1.8, 69), (1.8, 2.4, 65), (2.4, 3.0, 65), (3.0, 3.8, 64)], vocab, cfg, tau_key)
        hi, cd, hcx = A11b.span.H_inst[2, 5], A11b.code[2, 5], A11b.H_ctx[2, 5]
        chk("11b", "re-articulated F over C: SUSPENSION or H_inst >= 2/3; H_ctx >= 2/3",
            ("SUSPENSION" in cd or hi >= 2 / 3 - 1e-9) and hcx >= 2 / 3 - 1e-9, f"H_inst={hi:.2f} code={cd!r} H_ctx={hcx:.2f}", f"{p11b[2, 5]:.2f}")
        # 12 unresolved dissonance
        X, t, b = accomp([(0, 4.0, C)], layer, 12)
        A12, p12 = run_case_chroma(X, t, b, tempo, [(1.0, 2.5, 61), (2.6, 3.2, 69)], vocab, cfg, tau_key)
        chk("12", "C# over C: H_inst <= 1/3 and H_ctx == H_inst", A12.span.H_inst[0, 1] <= 1 / 3 + 1e-9 and A12.H_ctx[0, 1] == A12.span.H_inst[0, 1],
            f"H_inst={A12.span.H_inst[0, 1]:.2f} H_ctx={A12.H_ctx[0, 1]:.2f} code={A12.code[0, 1]!r}", f"{p12[0, 1]:.2f}")
        # 13 ambiguity
        v = h(A13a)
        chk("13a", "noise: LOW_EVIDENCE (e<0.5) and no pc < 0.25", A13a.span.e[0] < 0.5 and v.min() >= 0.25,
            f"e={A13a.span.e[0]:.2f} min={v.min():.2f} max={v.max():.2f}", f"min={p13a[0].min():.2f} max={p13a[0].max():.2f}")
        v = h(A13b)
        chk("13b", "single C: C >= 0.75; E, Eb >= 0.4; AMBIGUOUS_CHORD (top-1 < 0.35)",
            v[0] >= 0.75 and v[4] >= 0.4 and v[3] >= 0.4 and A13b.span.top_p[0, 0] < 0.35,
            f"C={v[0]:.2f} E={v[4]:.2f} Eb={v[3]:.2f} top1={vocab.names[A13b.span.top[0, 0]]}@{A13b.span.top_p[0, 0]:.2f}", f"C={p13b[0][0]:.2f} E={p13b[0][4]:.2f} Eb={p13b[0][3]:.2f}")
        v = h(A13c)
        chk("13c", "C5: C, G >= 0.75; E, Eb >= 0.5", v[0] >= 0.75 and v[7] >= 0.75 and v[4] >= 0.5 and v[3] >= 0.5,
            f"C={v[0]:.2f} G={v[7]:.2f} E={v[4]:.2f} Eb={v[3]:.2f} top1={vocab.names[A13c.span.top[0, 0]]}@{A13c.span.top_p[0, 0]:.2f}",
            f"C={p13c[0][0]:.2f} G={p13c[0][7]:.2f} E={p13c[0][4]:.2f} Eb={p13c[0][3]:.2f}")
        # 14 specificity over 1-5
        spec = [(int(np.sum(h(A) >= 0.5)), int(np.sum(h(A) <= 1 / 3 + 1e-9))) for A in (A1, A2, A3, A4, A5)]
        chk("14", "cases 1-5: <= 7 pcs >= 0.5 and >= 2 pcs <= 1/3", all(a <= 7 and b >= 2 for a, b in spec), f"(#>=0.5, #<=1/3) = {spec}",
            str([(int(np.sum(p[0] >= 0.5)), int(np.sum(p[0] <= 1 / 3))) for p in (p1, p2, p3, p4, p5)]))
        # 15 timing
        X, t, b = res["1"][2:5]
        As, _ = run_case_chroma(X, t, b, tempo, [probe, (probe[0] - 0.05, probe[1] - 0.05, 64), (probe[0] + 0.05, probe[1] + 0.05, 64)], vocab, cfg, tau_key)
        dmax = float(np.max(np.abs(As.span.H_inst[1:] - As.span.H_inst[0])))
        chk("15", "+-50 ms shift: |dH| <= 0.05", dmax <= 0.05, f"max |dH|={dmax:.4f}")
        v = h(A16)
        chk("16", "C major + noise (10 dB SNR at audio level): C,E,G >= 0.7", min(v[[0, 4, 7]]) >= 0.7, fmt(v), fmt(p16[0]))
        cases += checks
    return {"checks": cases, "passed": {l: f"{sum(c['pass'] for c in cases if c['layer'] == l)}/{sum(1 for c in cases if c['layer'] == l)}" for l in ("T", "S")}}


# ================================================================ stage 4: real songs


def load_song(song: str):
    with np.load(MAIN / song / "harmony.npz", allow_pickle=False) as z:
        h = {k: z[k] for k in ("other_chroma_cens", "other_chroma_rank", "chroma_frame_times_s", "global_chroma_profile",
                               "global_chroma_rank", "beat_times_s", "estimated_tempo_bpm")}
    with np.load(MAIN / song / "notes.npz", allow_pickle=False) as z:
        n = {k: z[k] for k in z.files}
    return h, n


def vec_metrics(V: np.ndarray, pc: np.ndarray) -> Dict[str, float]:
    idx = np.arange(pc.size)
    orig = V[idx, pc]
    mask = np.ones_like(V, dtype=bool)
    mask[idx, pc] = False
    alt = V[mask].reshape(-1, 11)
    win = (np.sum(alt < orig[:, None] - TOL, axis=1) + 0.5 * np.sum(np.abs(alt - orig[:, None]) <= TOL, axis=1)) / 11
    return {"breadth_ge_0.75": float(np.mean(np.sum(V >= 0.75, axis=1))), "breadth_ge_0.5": float(np.mean(np.sum(V >= 0.5, axis=1))),
            "share_notes_ge10_pcs_ge_0.5": float(np.mean(np.sum(V >= 0.5, axis=1) >= 10)),
            "share_notes_no_pc_ge_0.75": float(np.mean(np.sum(V >= 0.75, axis=1) == 0)),
            "spread_max_minus_min": float(np.mean(V.max(1) - V.min(1))), "graded_share_in_0.25_0.75": float(np.mean((V > 0.25) & (V < 0.75))),
            "mean_H_sung": float(orig.mean()), "median_H_sung": float(np.median(orig)), "mean_H_alternatives": float(alt.mean()),
            "sung_minus_alt": float(orig.mean() - alt.mean()), "win_rate_sung_vs_alt": float(win.mean()),
            "share_sung_ge_0.75": float(np.mean(orig >= 0.75)), "share_sung_le_1/3": float(np.mean(orig <= 1 / 3 + 1e-9)),
            "P4_up_gap": float(np.mean(orig - V[idx, (pc + 5) % 12])), "P5_up_gap": float(np.mean(orig - V[idx, (pc + 7) % 12])),
            "semitone_up_gap": float(np.mean(orig - V[idx, (pc + 1) % 12]))}


def chord_seq_diag(frame_top, t, beats, K, kp, vocab):
    if beats.size < 3:
        return {}
    centers = 0.5 * (beats[:-1] + beats[1:])
    bi = np.clip(np.searchsorted(t, centers), 0, t.size - 1)
    seq = frame_top[bi]
    key = int(np.argmax(kp))
    tonic, minor = key % 12, key >= 12
    scale = {(tonic + i) % 12 for i in ((0, 2, 3, 5, 7, 8, 10, 11) if minor else (0, 2, 4, 5, 7, 9, 11))}
    chords = seq[seq < K]
    qual = {}
    for k in chords:
        q = "+".join(vocab.qualities[k])
        qual[q] = qual.get(q, 0) + 1
    tot = max(chords.size, 1)
    return {"switch_per_beat": float(np.mean(np.diff(seq) != 0)), "flicker_per_beat": float(np.mean((seq[2:] == seq[:-2]) & (seq[1:-1] != seq[:-2]))),
            "N_share": float(np.mean(seq == K)), "diatonic_share": float(np.mean([vocab.pitch_sets[k] <= scale for k in chords])) if chords.size else float("nan"),
            "quality_share": {q: c / tot for q, c in sorted(qual.items(), key=lambda kv: -kv[1])}, "key": M.key_name(key), "key_p": float(kp.max()),
            "beat_seq": seq}


def evaluate_songs(vocab, sel_cfg: M.EstimatorConfig, tau_key: float, songs: List[str], group_of: Dict[str, str]):
    rng_master = np.random.default_rng(20261127)
    ablations = {f"temporal_{tp}": dict(cfg=replace(sel_cfg, temporal=tp)) for tp in TEMPORALS}
    ablations["template_" + ("harmonic" if sel_cfg.template == "binary" else "binary")] = dict(cfg=replace(sel_cfg, template="harmonic" if sel_cfg.template == "binary" else "binary"))
    vocab_tri = M.build_vocabulary(("maj", "min"))
    vocab_7 = M.build_vocabulary(("maj", "min", "dom7", "maj7", "min7"))
    ablations["vocab_triads"] = dict(cfg=sel_cfg, vocab=vocab_tri)
    ablations["vocab_triads_7ths"] = dict(cfg=sel_cfg, vocab=vocab_7)
    per_song, note_rows, all_V = [], [], {}
    pooled: Dict[str, List[np.ndarray]] = {}
    pooled_pc: List[np.ndarray] = []
    stab = {"noise": {}, "shift": {}}
    ctx_stats = {"sung_lifted": 0, "alt_lifted": 0, "alt_total": 0, "sung_total": 0, "by_code_sung": {}, "by_code_alt": {}}
    failures: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ("chord_tone_low", "breadth_ge10", "extension_low", "noise_unstable", "crosses_change", "short_low", "low_evidence")}
    timings, chord_diag = [], {tp: [] for tp in TEMPORALS}
    K = vocab.K
    for si, song in enumerate(songs):
        h, n = load_song(song)
        cens = h["other_chroma_cens"].astype(np.float64)
        rank = np.round(h["other_chroma_rank"].astype(np.float64) * 11) / 11
        t = h["chroma_frame_times_s"].astype(np.float64)
        beats = h["beat_times_s"].astype(np.float64)
        tempo = float(h["estimated_tempo_bpm"])
        g = h["global_chroma_profile"].astype(np.float64)
        start, end = n["start_time_s"].astype(np.float64), n["end_time_s"].astype(np.float64)
        midi = n["target_midi_note"].astype(int)
        pc = n["pitch_class"].astype(int)
        dur = n["duration_s"].astype(np.float64)
        core_abs = np.abs(n["core_median_cents_deviation"].astype(np.float64))
        offs = n["contour_offsets"]
        conf = np.array([n["contour_crepe_confidence"][offs[i]:offs[i + 1]].mean() for i in range(pc.size)])
        t0 = time.perf_counter()
        A = M.analyze(cens, t, beats, tempo, g, start, end, midi, vocab, sel_cfg, tau_key, with_context=True)
        timings.append({"song": song, "seconds": time.perf_counter() - t0, "notes": int(pc.size), "frames": int(t.size)})
        prod = production_vectors(cens, rank, t, tempo, h["global_chroma_rank"].astype(np.float64), start, end)
        V = {**prod, "chord_hard": A.span.H_hard, "chord_dist": A.span.H_chord, "chord_dist_key": A.span.H_inst,
             "chord_dist_fixed_key": A.span.H_fixed, "chord_dist_key_ctx": A.H_ctx,
             "chord_dist_key_alt_tiers": None}
        # alt tier spacing (reuse posterior)
        Hk_alt = M.key_posterior(g, tau_key) @ M.key_tier_matrix(M.TIER_VALUES_ALT)
        V["chord_dist_key_alt_tiers"] = M.compatibility(A.span.P, vocab.tier_values_alt, Hk_alt).H_inst
        for name, ab in ablations.items():
            v = ab.get("vocab", vocab)
            Aa = M.analyze(cens, t, beats, tempo, g, start, end, midi, v, ab["cfg"], tau_key, with_context=False)
            V[f"abl_{name}"] = Aa.span.H_inst
            if name.startswith("temporal_"):
                d = chord_seq_diag(Aa.frame_top, t, beats, v.K, Aa.key_post, v)
                # beat-grid shift agreement (beat estimators only)
                if ab["cfg"].temporal.startswith("beat") and "beat_seq" in d:
                    As = M.analyze(cens, t, beats + SHIFT_S, tempo, g, start, end, midi, v, ab["cfg"], tau_key, with_context=False)
                    ds = chord_seq_diag(As.frame_top, t, beats, v.K, As.key_post, v)
                    d["beat_shift_50ms_agreement"] = float(np.mean(ds["beat_seq"] == d["beat_seq"]))
                d.pop("beat_seq", None)
                chord_diag[ab["cfg"].temporal].append(d)
        for k, v in V.items():
            pooled.setdefault(k, []).append(v)
        pooled_pc.append(pc)
        # stability: CENS noise and +-50 ms note shift (selected pipeline and production)
        noisy = cens * np.exp(rng_master.normal(0, NOISE_SIGMA, cens.shape))
        noisy /= np.clip(np.linalg.norm(noisy, axis=0, keepdims=True), 1e-12, None)
        An = M.analyze(noisy, t, beats, tempo, noisy.mean(axis=1), start, end, midi, vocab, sel_cfg, tau_key, with_context=False)
        pn = production_vectors(noisy, H._to_rank(noisy), t, tempo, np.argsort(np.argsort(noisy.mean(axis=1))) / 11.0, start, end)
        for k, (a, b) in {"chord_dist_key": (A.span.H_inst, An.span.H_inst), "prod_calibrated": (prod["prod_calibrated"], pn["prod_calibrated"]),
                          "raw_chroma": (prod["raw_chroma"], pn["raw_chroma"]), "rank_chroma": (prod["rank_chroma"], pn["rank_chroma"])}.items():
            stab["noise"].setdefault(k, []).append(np.abs(a - b))
        topn_agree = float(np.mean(An.frame_top == A.frame_top))
        for sh in (-SHIFT_S, SHIFT_S):
            Ash = M.analyze(cens, t, beats, tempo, g, start + sh, end + sh, midi, vocab, sel_cfg, tau_key, with_context=False)
            psh = production_vectors(cens, rank, t, tempo, h["global_chroma_rank"].astype(np.float64), start + sh, end + sh)
            for k, (a, b) in {"chord_dist_key": (A.span.H_inst, Ash.span.H_inst), "prod_calibrated": (prod["prod_calibrated"], psh["prod_calibrated"]),
                              "raw_chroma": (prod["raw_chroma"], psh["raw_chroma"]), "rank_chroma": (prod["rank_chroma"], psh["rank_chroma"])}.items():
                stab["shift"].setdefault(k, []).append(np.abs(a - b))
        # context specificity
        idx = np.arange(pc.size)
        lifted = (A.H_ctx > A.span.H_inst + 1e-12)
        ctx_stats["sung_lifted"] += int(lifted[idx, pc].sum())
        ctx_stats["sung_total"] += int(pc.size)
        alt_mask = np.ones_like(lifted)
        alt_mask[idx, pc] = False
        ctx_stats["alt_lifted"] += int(lifted[alt_mask].sum())
        ctx_stats["alt_total"] += int(alt_mask.sum())
        for i in range(pc.size):
            for c in range(12):
                cd = A.code[i, c]
                if cd:
                    for part in cd.split("+"):
                        key = "by_code_sung" if c == pc[i] else "by_code_alt"
                        ctx_stats[key][part] = ctx_stats[key].get(part, 0) + 1
        # reason codes and failure categories
        top = A.span.top[:, 0]
        tier_top = vocab.tier_index_first[top, :]           # tier index (0..3) of each pc under the top chord
        sung_tier = tier_top[idx, pc]
        H_s, Hc_s = A.span.H_inst[idx, pc], A.H_ctx[idx, pc]
        dH_noise = np.abs(A.span.H_inst[idx, pc] - An.span.H_inst[idx, pc])
        for i in range(pc.size):
            ex = {"song": song, "group": group_of[song], "note_index": int(n["note_index"][i]), "start_s": round(float(start[i]), 3),
                  "end_s": round(float(end[i]), 3), "duration_s": round(float(dur[i]), 3), "pc": M.PC_NAMES[pc[i]],
                  "top_chords": ", ".join(f"{vocab.names[A.span.top[i, j]]}:{A.span.top_p[i, j]:.2f}" for j in range(3)),
                  "evidence": round(float(A.span.e[i]), 3), "H_inst": round(float(H_s[i]), 3), "H_ctx": round(float(Hc_s[i]), 3),
                  "code": A.code[i, pc[i]], "prod_calibrated": round(float(prod["prod_calibrated"][i, pc[i]]), 3)}
            if sung_tier[i] == 0 and H_s[i] < 0.5:
                failures["chord_tone_low"].append(ex)
            if np.sum(A.span.H_inst[i] >= 0.5) >= 10:
                failures["breadth_ge10"].append(ex)
            if sung_tier[i] == 1 and H_s[i] < 1 / 3:
                failures["extension_low"].append(ex)
            if dH_noise[i] >= 0.25:
                failures["noise_unstable"].append({**ex, "dH_noise": round(float(dH_noise[i]), 3)})
            if A.crosses[i]:
                failures["crosses_change"].append(ex)
            if dur[i] < 0.15 and Hc_s[i] < 1 / 3:
                failures["short_low"].append(ex)
            if A.span.e[i] < 0.5:
                failures["low_evidence"].append(ex)
            rc = [("CT", "EXT", "TENSION", "OUT")[sung_tier[i]]]
            if A.span.e[i] < 0.5:
                rc.append("LOW_EVIDENCE")
            if A.span.top_p[i, 0] < 0.35:
                rc.append("AMBIGUOUS_CHORD")
            if 1 - A.span.e[i] > 0.2:
                rc.append("KEY_FALLBACK")
            if A.code[i, pc[i]]:
                rc.append(A.code[i, pc[i]])
            if dur[i] < M.CTX["short_s"]:
                rc.append("SHORT")
            if core_abs[i] > 40:
                rc.append("PC_AMBIGUOUS")
            note_rows.append([song, group_of[song], int(n["note_index"][i]), round(float(start[i]), 4), round(float(end[i]), 4),
                              round(float(dur[i]), 4), int(pc[i]), int(midi[i]), round(float(conf[i]), 4), round(float(core_abs[i]), 2),
                              *[vocab.names[A.span.top[i, j]] for j in range(3)], *[round(float(A.span.top_p[i, j]), 4) for j in range(3)],
                              round(float(A.span.e[i]), 4), round(float(A.span.entropy[i]), 3),
                              round(float(A.span.H_chord[i, pc[i]]), 4), round(float(A.span.H_key[i, pc[i]]), 4),
                              round(float(H_s[i]), 4), round(float(Hc_s[i]), 4), bool(A.crosses[i]), int(Hc_s[i] > H_s[i] + 1e-12),
                              ";".join(rc), round(float(prod["prod_calibrated"][i, pc[i]]), 4), round(float(prod["prod_harmonic_rank"][i, pc[i]]), 5),
                              *[round(float(x), 4) for x in A.span.H_inst[i]]])
        sm = vec_metrics(A.span.H_inst, pc)
        per_song.append({"song_id": song, "group": group_of[song], "notes": int(pc.size), "key": M.key_name(int(np.argmax(A.key_post))),
                         "key_p": round(float(A.key_post.max()), 3), "mean_H_sung": sm["mean_H_sung"], "mean_H_alt": sm["mean_H_alternatives"],
                         "win_rate": sm["win_rate_sung_vs_alt"], "breadth_ge_0.75": sm["breadth_ge_0.75"], "breadth_ge_0.5": sm["breadth_ge_0.5"],
                         "low_evidence_share": float(np.mean(A.span.e < 0.5)), "ambiguous_share": float(np.mean(A.span.top_p[:, 0] < 0.35)),
                         "median_top1_mass": float(np.median(A.span.top_p[:, 0])), "crosses_share": float(np.mean(A.crosses)),
                         "ctx_lift_share_sung": float(np.mean(lifted[idx, pc])), "prod_mean_sung": float(prod["prod_calibrated"][idx, pc].mean()),
                         "noise_top_chord_frame_agreement": topn_agree, "share_sung_CT": float(np.mean(sung_tier == 0)),
                         "share_sung_OUT": float(np.mean(sung_tier == 3))})
        print(f"  [{si + 1}/{len(songs)}] {song}", flush=True)
    pc_all = np.concatenate(pooled_pc)
    methods = {k: vec_metrics(np.vstack(v), pc_all) for k, v in pooled.items()}
    stability = {kind: {k: {"mean_abs_dH_all12": float(np.mean(np.vstack(v))), "p95_abs_dH": float(np.percentile(np.vstack(v), 95))}
                        for k, v in d.items()} for kind, d in stab.items()}
    return {"methods": methods, "per_song": per_song, "stability": stability, "context": ctx_stats, "failures": failures,
            "timings": timings, "chord_diag": chord_diag, "note_rows": note_rows, "pooled": pooled, "pc_all": pc_all}


# ================================================================ main and report writers


def summarize_diag(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for k in ("switch_per_beat", "flicker_per_beat", "N_share", "diatonic_share", "beat_shift_50ms_agreement"):
        vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    q: Dict[str, float] = {}
    for r in rows:
        for kk, vv in r.get("quality_share", {}).items():
            q[kk] = q.get(kk, 0.0) + vv / len(rows)
    out["quality_share_mean_over_songs"] = dict(sorted(q.items(), key=lambda kv: -kv[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    outs = [HERE / f for f in ("results.json", "method_comparison.md", "synthetic_tests.md", "failure_analysis.md", "song_level.csv")]
    if not args.overwrite and any(p.exists() for p in outs):
        print("outputs exist; pass --overwrite", file=sys.stderr)
        return 2
    sel = json.loads((REPO / "data" / "dataset_selections.json").read_text())["songs"]
    manifest = json.loads((MAIN / "manifest.json").read_text())
    songs = sorted(sel)
    assert set(songs) == {r["song_id"] for r in manifest["songs"]} and len(songs) == 63
    group_of = {s: ("axis2_50" if sel[s]["axis2_ml"] else "excluded_13") for s in songs}
    src_before = {s: [sha256_file(MAIN / s / f) for f in ("notes.npz", "harmony.npz")] for s in songs}
    # production-chroma path check on one synthetic clip
    rng = np.random.default_rng(1)
    clip = SA.chord_audio([60, 64, 67], 3.0, "soft", rng)
    ctx = H.build_harmonic_context(clip, clip, SA.SR)
    path_ok = bool(np.allclose(ctx.salience, SA.production_cens(clip)))

    vocab = M.build_vocabulary()
    t0 = time.perf_counter()
    print("stage 1: calibration", flush=True)
    cal = calibrate(vocab)
    s = cal["selected"]
    print("  selected", s, flush=True)
    print("stage 2: temporal + key", flush=True)
    progs = make_progressions(vocab, SEED_C)
    temp = temporal_eval(vocab, s, progs)
    cfg = M.EstimatorConfig(template=s["template"], sim=s["sim"], tau=s["tau"], theta=s["theta"], temporal=temp["selected"])
    tau_key = temp["key"]["tau_key"]
    print("  temporal", temp["selected"], "tau_key", tau_key, flush=True)
    print("stage 3: synthetic tests", flush=True)
    syn = synthetic_tests(vocab, cfg, tau_key)
    print("  passed", syn["passed"], flush=True)
    t_syn = time.perf_counter() - t0
    print("stage 4: real songs", flush=True)
    ev = evaluate_songs(vocab, cfg, tau_key, songs, group_of)
    src_after = {s_: [sha256_file(MAIN / s_ / f) for f in ("notes.npz", "harmony.npz")] for s_ in songs}

    # group splits of key metrics
    by_group = {}
    for g in ("axis2_50", "excluded_13"):
        idxs = [i for i, s_ in enumerate(songs) if group_of[s_] == g]
        pcs = np.concatenate([load_pc(songs[i]) for i in idxs])
        by_group[g] = {m: vec_metrics(np.vstack([ev["pooled"][m][i] for i in idxs]), pcs)
                       for m in ("prod_calibrated", "raw_chroma", "rank_chroma", "chord_dist", "chord_dist_key", "chord_dist_key_ctx")}
    diag = {tp: summarize_diag(rows) for tp, rows in ev["chord_diag"].items()}
    # duration / quality buckets for the selected method
    rows = ev["note_rows"]
    dur = np.array([r[5] for r in rows])
    Hs = np.array([r[20] for r in rows])
    Hc = np.array([r[21] for r in rows])
    ps = np.array([r[25] for r in rows])
    ev_e = np.array([r[16] for r in rows])
    conf = np.array([r[8] for r in rows])
    core = np.array([r[9] for r in rows])
    top1 = np.array([r[13] for r in rows])
    buckets = {}
    for name, vals, edges in (("duration_s", dur, (0, 0.1, 0.2, 0.4, 0.8, np.inf)), ("mean_crepe_conf", conf, (0, 0.75, 0.85, 0.95, 1.01)),
                              ("abs_core_cents", core, (0, 40.0001, np.inf)), ("evidence", ev_e, (0, 0.5, 0.8, 0.95, 1.0001)),
                              ("top1_chord_mass", top1, (0, 0.2, 0.35, 0.5, 0.75, 1.0001))):
        b = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, len(edges) - 2)
        buckets[name] = {f"[{edges[i]:g},{edges[i + 1]:g})": {"notes": int(np.sum(b == i)), "mean_H_inst_sung": float(Hs[b == i].mean()) if np.any(b == i) else None,
                                                              "mean_H_ctx_sung": float(Hc[b == i].mean()) if np.any(b == i) else None,
                                                              "mean_prod_sung": float(ps[b == i].mean()) if np.any(b == i) else None}
                         for i in range(len(edges) - 1)}
    reason_counts: Dict[str, int] = {}
    for r in rows:
        for c in r[24].split(";"):
            for part in c.split("+"):
                reason_counts[part] = reason_counts.get(part, 0) + 1
    tm = np.array([x["seconds"] for x in ev["timings"]])
    durs = np.array([float(json.loads((MAIN / x["song"] / "metadata.json").read_text())["duration_s"]) for x in ev["timings"]])

    # archive note table
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    note_path = ARCHIVE / "note_level.csv.gz"
    header = ["song_id", "group", "note_index", "start_s", "end_s", "duration_s", "pitch_class", "target_midi", "mean_conf", "abs_core_cents",
              "top1_chord", "top2_chord", "top3_chord", "top1_p", "top2_p", "top3_p", "evidence", "chord_entropy_bits",
              "H_chord_sung", "H_key_sung", "H_inst_sung", "H_ctx_sung", "crosses_change", "context_lifted", "reason_codes",
              "prod_calibrated_sung", "prod_harmonic_rank_sung", *[f"H_inst_{p}" for p in M.PC_NAMES]]
    with gzip.open(note_path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    with (HERE / "song_level.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev["per_song"][0].keys()))
        w.writeheader()
        w.writerows(ev["per_song"])

    fail_counts = {k: len(v) for k, v in ev["failures"].items()}
    examples = {}
    for k, lst in ev["failures"].items():
        keyf = {"chord_tone_low": lambda e: e["H_inst"], "breadth_ge10": lambda e: -e["duration_s"], "extension_low": lambda e: e["H_inst"],
                "noise_unstable": lambda e: -e["dH_noise"], "crosses_change": lambda e: -e["duration_s"], "short_low": lambda e: e["H_ctx"],
                "low_evidence": lambda e: e["evidence"]}[k]
        seen, ex = set(), []
        for e in sorted(lst, key=keyf):
            if e["song"] not in seen:
                ex.append(e)
                seen.add(e["song"])
            if len(ex) == 6:
                break
        examples[k] = ex
    ctx = ev["context"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    results = {
        "experiment": "axis1_chord_compatibility", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "method development on all 63 songs; no held-out split; in-corpus comparisons are not evidence of generalization",
        "provenance": {"git_commit": git, "python": platform.python_version(), "numpy": np.__version__,
                       "code_sha256": {f: sha256_file(HERE / f) for f in ("axis1_chord_compatibility.py", "chord_model.py", "synthetic_audio.py", "protocol.md")},
                       "note_level_table": {"path": str(note_path.relative_to(REPO)), "sha256": sha256_file(note_path), "rows": len(rows)}},
        "verification": {"source_files_unchanged": src_before == src_after, "synthetic_cens_equals_production_build_harmonic_context": path_ok,
                         "songs": len(songs), "notes": len(rows)},
        "vocabulary": {"distinct_pitch_sets": vocab.K, "merged": [n_ for n_ in vocab.names if "/" in n_]},
        "stage1_calibration": cal, "stage2_temporal": temp,
        "selected_config": {**asdict(cfg), "tau_key": tau_key},
        "stage3_synthetic": syn,
        "stage4_real": {"methods_all_63": ev["methods"], "by_group": by_group, "stability": ev["stability"],
                        "chord_sequence_diagnostics": diag, "buckets_selected_method": buckets, "reason_code_counts": reason_counts,
                        "context_specificity": {"sung_lift_rate": ctx["sung_lifted"] / ctx["sung_total"], "alternative_lift_rate": ctx["alt_lifted"] / ctx["alt_total"],
                                                "by_code_sung": ctx["by_code_sung"], "by_code_alternatives_per_alt": {k: v / 11 for k, v in ctx["by_code_alt"].items()}},
                        "failure_counts": fail_counts, "failure_examples": examples,
                        "abstention": {"low_evidence_share": fail_counts["low_evidence"] / len(rows),
                                       "ambiguous_chord_share": reason_counts.get("AMBIGUOUS_CHORD", 0) / len(rows)},
                        "runtime": {"selected_pipeline_seconds_per_song": {"mean": float(tm.mean()), "max": float(tm.max())},
                                    "seconds_per_minute_of_audio": float(np.sum(tm) / (np.sum(durs) / 60)), "synthetic_stages_seconds": t_syn}},
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    from report import write_reports  # noqa: E402
    write_reports(results, ev["per_song"], HERE)
    print(json.dumps(results["verification"]), flush=True)
    return 0 if results["verification"]["source_files_unchanged"] else 1


def load_pc(song: str) -> np.ndarray:
    with np.load(MAIN / song / "notes.npz", allow_pickle=False) as z:
        return z["pitch_class"].astype(int)


if __name__ == "__main__":
    raise SystemExit(main())
