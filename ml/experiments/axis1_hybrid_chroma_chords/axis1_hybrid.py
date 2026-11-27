"""
Axis 1 Experiment 3 — chroma-only vs chord-only vs top-5 hybrid vs confidence-weighted hybrid.

Stages (protocol.md, including Revision 1):
  1. synthetic threshold study -> kappa and D's (gamma, delta) by the pre-registered rule
  2. synthetic musical tests: every requirement scored for every method, layers T and S
  3. all 63 real songs (read-only): comparison, stability, confidence diagnostics, examples, timing

Usage, from the repo root:
    venv/bin/python ml/experiments/axis1_hybrid_chroma_chords/axis1_hybrid.py [--overwrite]
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAIN = REPO / "data" / "main_data"
ARCHIVE = REPO / "data" / "experiment_archives" / "axis1_hybrid_chroma_chords"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "axis1_chord_compatibility"))
sys.path.insert(0, str(REPO / "backend"))

import hybrid_model as HM  # noqa: E402
import chord_model as M  # noqa: E402
import synthetic_audio as SA  # noqa: E402
import axis1_chord_compatibility as X2  # noqa: E402
import harmony as H  # noqa: E402

VOCAB, K, TIERS = HM.VOCAB, HM.K, HM.TIERS
GAMMAS, DELTAS, EPSS, THETAS = (0.05, 0.10, 0.15), (0.03, 0.05, 0.10), (0.90, 0.95, 0.99), (0.70, 0.80, 0.90)
NOISE_SIGMA, SHIFT_S = 0.10, 0.05
TOL = 1e-9
MAIN_METHODS = ["A", "A_rank", "B", "C5u_a0.5", "C5w_a0.5", "D_a0.5"]
ALL_METHODS = (["A", "A_rank", "B"] + [f"C5u_a{a}" for a in HM.ALPHAS] + [f"C5w_a{a}" for a in HM.ALPHAS] + [f"D_a{a}" for a in HM.ALPHAS]
               + ["D1_weight_nothreshold", "D2_threshold_uniform", "D4_adaptive_alpha"])


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ stage 1: synthetic threshold study


def frames_of(audio):
    c = SA.production_cens(audio)
    t = SA.frame_times(c.shape[1])
    return c[:, np.flatnonzero((t >= 0.6) & (t <= 1.4))[::3]].T


def synth_extra(seed: int):
    """Unlabelled-harmony frames: noise, clusters, single pitches, two-chord mixtures (with acceptable sets)."""
    rng = np.random.default_rng(seed)
    n = int(2.0 * SA.SR)
    out = {"white": [], "pink": [], "cluster": [], "single": [], "mixture": []}
    acc_mix = []
    for _ in range(40):
        out["white"].append(frames_of(SA.white(n, rng) * 0.1))
        out["pink"].append(frames_of(SA.pink(n, rng) * 0.1))
        pcs = sorted(rng.choice(12, size=6, replace=False))
        out["cluster"].append(frames_of(SA.chord_audio(SA.voicing(pcs, rng), 2.0, str(rng.choice(list(SA.TIMBRES))), rng)))
    for pc in range(12):
        for tb in SA.TIMBRES:
            out["single"].append(frames_of(SA.chord_audio([48 + pc, 60 + pc], 2.0, tb, rng)))
    set_idx = {s: k for k, s in enumerate(VOCAB.pitch_sets)}
    pairs = [((0, "maj"), (9, "min")), ((0, "maj"), (5, "maj")), ((5, "maj"), (2, "min")), ((7, "maj"), (4, "min")), ((0, "maj"), (4, "min"))]
    for i in range(40):
        tonic = int(rng.integers(12))
        (a, qa), (b, qb) = pairs[i % len(pairs)]
        s1 = frozenset((tonic + a + iv) % 12 for iv in M.QUALITIES[qa])
        s2 = frozenset((tonic + b + iv) % 12 for iv in M.QUALITIES[qb])
        tb = str(rng.choice(list(SA.TIMBRES)))
        y = SA.chord_audio(SA.voicing(sorted(s1), rng), 2.0, tb, rng) + SA.chord_audio(SA.voicing(sorted(s2), rng, 55, 67), 2.0, tb, rng)
        f = frames_of(y)
        out["mixture"].append(f)
        union = s1 | s2
        ok = {set_idx[s1], set_idx[s2]} | ({set_idx[union]} if union in set_idx else set())
        acc_mix += [(set_idx[s1], set_idx[s2], ok)] * len(f)
    return {k: np.vstack(v) for k, v in out.items()}, acc_mix


def threshold_study() -> Dict[str, Any]:
    XB, yB = X2.synth_frames(VOCAB, X2.SEED_B, degraded=False)
    XD, yD = X2.synth_frames(VOCAB, X2.SEED_B + 1, degraded=True)
    XB, yB, XD, yD = XB[yB < K], yB[yB < K], XD[yD < K], yD[yD < K]
    extra, acc_mix = synth_extra(505)
    ev = {"clean": HM.evidence(XB), "degraded": HM.evidence(XD), **{k: HM.evidence(v) for k, v in extra.items()}}
    noise_ic = np.concatenate([ev["white"].Ic, ev["pink"].Ic])
    kappa = float(np.percentile(noise_ic, 99))
    comp_desc = {}
    for name, e in ev.items():
        c = e.r - e.m[:, None]
        comp_desc[name] = {"r_best_median": float(np.median(e.r.max(1))), "excess_best_median": float(np.median(c.max(1))),
                           "E_best_median": float(np.median(e.E.max(1))), "Ic_median": float(np.median(e.Ic)),
                           "posterior_top1_median": float(np.median(e.post[:, :K].max(1) / np.clip(e.post[:, :K].sum(1), 1e-300, None)))}
    strategies = [("ABS", dict(theta=t)) for t in THETAS] + [("REL", dict(delta=d)) for d in DELTAS] + \
                 [("NULL", dict(eps=e)) for e in EPSS] + [("NULL+REL", dict(eps=e, delta=d)) for e in EPSS for d in DELTAS] + \
                 [("EXCESS", dict(gamma=g)) for g in GAMMAS] + [("EXCESS+REL", dict(gamma=g, delta=d)) for g in GAMMAS for d in DELTAS]
    rows = []
    for strat, kw in strategies:
        r: Dict[str, Any] = {"strategy": strat, **kw}
        for name in ev:
            S = HM.eligible(ev[name], strat, **kw)
            cnt = S.sum(1)
            r[f"{name}_accept"] = float(np.mean(cnt > 0))
            r[f"{name}_mean_n"] = float(cnt.mean())
            if name in ("clean", "degraded"):
                y = yB if name == "clean" else yD
                r[f"{name}_recall"] = float(np.mean(S[np.arange(len(y)), y]))
                r[f"{name}_only_true"] = float(np.mean(S[np.arange(len(y)), y] & (cnt == 1)))
            if name == "mixture":
                r["mixture_any_acceptable"] = float(np.mean([S[i, list(ok)].any() for i, (_, _, ok) in enumerate(acc_mix)]))
                r["mixture_both_chords"] = float(np.mean([S[i, a] and S[i, b] for i, (a, b, _) in enumerate(acc_mix)]))
        r["noise_false_accept"] = 0.5 * (r["white_accept"] + r["pink_accept"])
        rows.append(r)
    cands = [r for r in rows if r["strategy"] in ("EXCESS", "EXCESS+REL")]
    ok = [r for r in cands if r["noise_false_accept"] <= 0.05 and r["clean_mean_n"] <= 3]
    if ok:
        best = sorted(ok, key=lambda r: (-r["degraded_recall"], r["clean_mean_n"]))[0]
        rule = "max degraded recall s.t. noise FA <= 0.05 and clean mean |S| <= 3"
    else:
        best = sorted(cands, key=lambda r: (r["noise_false_accept"], -r["degraded_recall"]))[0]
        rule = "fallback: lowest noise FA (no candidate met both constraints)"
    sel = {"gamma": best["gamma"], "delta": best.get("delta", np.inf), "strategy": best["strategy"], "rule_applied": rule}
    if sel["delta"] == np.inf:
        sel["delta"] = 10.0
    return {"kappa": kappa, "noise_Ic_p99": kappa, "frames": {k: int(len(v.r)) for k, v in ev.items()}, "components": comp_desc,
            "strategies": rows, "selected": sel}


# ================================================================ shared pipeline


def run_pipeline(cens, t, beats, start, end, kappa, gamma, delta):
    edges = M.beat_segments(t, beats)
    seg = M.segment_mean_chroma(cens, t, edges)
    ev = HM.evidence(seg)
    segm, diag = HM.segment_methods(seg, ev, kappa, gamma, delta)
    coo = HM.note_weights(edges, start, end)
    n = len(start)
    notes = {k: HM.aggregate(v, coo, n) for k, v in segm.items()}
    return {"edges": edges, "seg": seg, "ev": ev, "segm": segm, "diag": diag, "coo": coo, "notes": notes, "dom": HM.dominant_segment(coo, n)}


# ================================================================ stage 2: synthetic musical tests


def synthetic_tests(kappa, gamma, delta) -> Dict[str, Any]:
    beat = 0.6
    C, Am, G7, Csus4, C5 = (0, 4, 7), (9, 0, 4), (7, 11, 2, 5), (0, 5, 7), (0, 7)

    def tones_chroma(weights: Dict[int, float]) -> np.ndarray:
        v = np.zeros(12)
        for pc, w in weights.items():
            v += w * M.harmonic_chroma([pc])
        return v / np.linalg.norm(v)

    def build(layer, segs, seed, total=None, snr=None):
        """segs: (start, end, spec) with spec = pitch classes, {pc: amplitude}, 'noise'."""
        total = total or segs[-1][1]
        beats = np.arange(0, total + 1e-9, beat)
        rng = np.random.default_rng(seed)
        if layer == "T":
            nf = int(np.ceil(total / M.HOP_S)) + 1
            t = SA.frame_times(nf)
            X = np.zeros((12, nf))
            for a, b, spec in segs:
                mk = (t >= a) & (t < b + 1e-9)
                if spec == "noise":
                    X[:, mk] = rng.random((12, mk.sum())) + 1.0
                else:
                    w = spec if isinstance(spec, dict) else {p: 1.0 for p in spec}
                    X[:, mk] = tones_chroma(w)[:, None]
                    if snr is not None:
                        X[:, mk] += (1.0 / snr) * rng.random((12, mk.sum()))
            X /= np.clip(np.linalg.norm(X, axis=0, keepdims=True), 1e-12, None)
            return X, t, beats
        parts = []
        for a, b, spec in segs:
            d = b - a
            if spec == "noise":
                parts.append(SA.white(int(round(d * SA.SR)), rng) * 0.1)
                continue
            w = spec if isinstance(spec, dict) else {p: 1.0 for p in spec}
            y = sum(amp * SA.tone(52 + ((pc - 52) % 12), d, "soft") for pc, amp in w.items())
            if len(w) <= 2:
                y = y + sum(amp * SA.tone(64 + ((pc - 64) % 12), d, "soft") for pc, amp in w.items())
            parts.append(y / max(np.max(np.abs(y)), 1e-9) * 0.5)
        y = np.concatenate(parts)
        y = SA.add_noise(y, snr if snr is not None else 40.0, rng)
        X = SA.production_cens(y)
        return X, SA.frame_times(X.shape[1]), beats

    def run(layer, segs, notes, seed, snr=None, noise_sigma=None):
        X, t, b = build(layer, segs, seed, snr=snr)
        if noise_sigma:
            X = X * np.exp(np.random.default_rng(seed + 1).normal(0, noise_sigma, X.shape))
            X /= np.clip(np.linalg.norm(X, axis=0, keepdims=True), 1e-12, None)
        st = np.array([n[0] for n in notes])
        en = np.array([n[1] for n in notes])
        P = run_pipeline(X, t, b, st, en, kappa, gamma, delta)
        return P

    probe = [(1.5, 2.1)]
    checks: List[Dict[str, Any]] = []
    demo: Dict[str, Any] = {}
    for layer in ("T", "S"):
        snr8 = 1.5 if layer == "T" else 5.0
        R = {}
        R["1"] = run(layer, [(0, 3.6, C)], probe, 1)
        R["2"] = run(layer, [(0, 3.6, Am)], probe, 2)
        R["3"] = run(layer, [(0, 3.6, G7)], probe, 3)
        R["4"] = run(layer, [(0, 3.6, Csus4)], probe, 4)
        R["5"] = run(layer, [(0, 3.6, C5)], probe, 5)
        R["6"] = run(layer, [(0, 3.6, (0, 4, 7, 9))], probe, 6)
        R["7"] = run(layer, [(0, 3.6, (0, 4))], probe, 7)
        R["8"] = run(layer, [(0, 3.6, C)], probe, 8, snr=snr8)
        R["9"] = run(layer, [(0, 3.6, (0, 1, 2, 6, 7, 8))], probe, 9)
        R["10"] = run(layer, [(0, 3.6, "noise")], probe, 10)
        R["11"] = run(layer, [(0, 3.6, (0,))], probe, 11)
        prog = [(0, 1.2, C), (1.2, 2.4, (5, 9, 0)), (2.4, 3.6, (7, 11, 2)), (3.6, 4.8, C)]
        R["12"] = run(layer, prog, [(a + 0.3, a + 0.9) for a, _, _ in prog], 12)
        R["13"] = run(layer, [(0, 3.6, {0: 1.0, 4: 0.15, 7: 1.0})], probe, 13)
        R["14"] = run(layer, [(0, 3.6, {0: 1.0, 4: 1.0, 7: 1.0, 6: 1.0})], probe, 14)
        sh = run(layer, [(0, 3.6, C)], [(1.5, 2.1), (1.45, 2.05), (1.55, 2.15)], 1)
        R18n = run(layer, [(0, 3.6, C)], probe, 1, noise_sigma=0.1)

        def vec(case, m, i=0):
            return R[case]["notes"][m][i]

        def add(case, truth, desc, fn):
            for m in ALL_METHODS:
                try:
                    ok, detail = fn(m)
                except FloatingPointError:
                    ok, detail = False, "error"
                checks.append({"layer": layer, "case": case, "truth": truth, "check": desc, "method": m, "pass": bool(ok), "detail": detail})

        def fmt(v, pcs):
            return "abstain" if np.isnan(v[0]) else " ".join(f"{M.PC_NAMES[p]}={v[p]:.2f}" for p in pcs)

        def req(case, hi=(), hi_t=0.75, mid=(), mid_t=1 / 3, lo=(), lo_t=0.25, extra=None, pcs_show=None):
            def fn(m):
                v = vec(case, m)
                if np.isnan(v[0]):
                    return False, "abstain"
                ok = all(v[p] >= hi_t - TOL for p in hi) and all(v[p] >= mid_t - TOL for p in mid) and all(v[p] <= lo_t + TOL for p in lo)
                if extra:
                    ok = ok and extra(v)
                return ok, fmt(v, pcs_show or sorted(set(hi) | set(mid) | set(lo)))
            return fn

        add("1", "exact", "C,E,G>=.75; D,A>=1/3; C#,F#,G#<=.25; min(CEG)>max(DAB)",
            req("1", (0, 4, 7), mid=(2, 9), lo=(1, 6, 8), extra=lambda v: min(v[[0, 4, 7]]) > max(v[[2, 9, 11]]), pcs_show=(0, 4, 7, 2, 9, 11, 1, 6, 8)))
        add("2", "exact", "A,C,E>=.75; A#,C#<=.25", req("2", (9, 0, 4), lo=(10, 1)))
        add("3", "exact", "G,B,D,F>=.75", req("3", (7, 11, 2, 5)))
        add("4", "exact", "C,F,G>=.75; E>=.25", req("4", (0, 5, 7), mid=(4,), mid_t=0.25))
        add("5", "exact (thirds undetermined)", "C,G>=.75; E>=1/3 and Eb>=1/3", req("5", (0, 7), mid=(4, 3)))
        add("6", "ambiguous", "C,E,G,A>=.6; C#,F#<=.4", req("6", (0, 4, 7, 9), hi_t=0.6, lo=(1, 6), lo_t=0.4))
        add("7", "ambiguous", "C,E>=.75; G>=1/3 and A>=1/3", req("7", (0, 4), mid=(7, 9)))
        add("8", "exact (noisy)", "C,E,G>=.6; C#,F#<=.4", req("8", (0, 4, 7), hi_t=0.6, lo=(1, 6), lo_t=0.4))

        def c9(m):
            v = vec("9", m)
            if np.isnan(v[0]):
                return True, "abstain"
            absent = [p for p in range(12) if p not in (0, 1, 2, 6, 7, 8)]
            return float(np.max(v[absent])) <= 0.5 + TOL, f"max over absent pcs={np.max(v[absent]):.2f}"
        add("9", "ambiguous (no credible chord)", "abstain OR max over the 6 absent pcs <= .5", c9)
        add("10", "no harmony", "abstain / insufficient", lambda m: (bool(np.isnan(vec("10", m)[0])), "abstain" if np.isnan(vec("10", m)[0]) else f"scored: max={np.nanmax(vec('10', m)):.2f} min={np.nanmin(vec('10', m)):.2f}"))

        def c11(m):
            v = vec("11", m)
            if np.isnan(v[0]):
                return False, "abstain"
            return v[0] >= 0.75 - TOL and abs(v[4] - v[3]) <= 0.2 + TOL, fmt(v, (0, 4, 3))
        add("11", "identity undetermined", "C>=.75; |E-Eb|<=.2", c11)

        def c12(m):
            sets = [C, (5, 9, 0), (7, 11, 2), C]
            vals = []
            for i, s in enumerate(sets):
                v = vec("12", m, i)
                if np.isnan(v[0]):
                    return False, "abstain"
                vals.append(min(v[list(s)]))
            return min(vals) >= 0.75 - TOL, "min chord-tone H per chord: " + ", ".join(f"{x:.2f}" for x in vals)
        add("12", "exact", "each chord's tones >= .75 at its centre", c12)
        add("13", "exact", "quiet third E >= .6", req("13", (4,), hi_t=0.6, pcs_show=(0, 4, 7)))

        def c14(m):
            v = vec("14", m)
            if np.isnan(v[0]):
                return False, "abstain"
            return v[6] < min(v[[0, 4, 7]]) and v[6] <= 0.6 + TOL, fmt(v, (0, 4, 7, 6))
        add("14", "exact chord + foreign F#", "F# < min(C,E,G) and F# <= .6", c14)

        true_sets = {"1": ("maj", 0), "2": ("min", 9), "3": ("dom7", 7), "4": ("sus4", 0)}

        def c15(m):
            out = []
            for cse in ("1", "2", "3", "4"):
                v = vec(cse, m)
                if np.isnan(v[0]):
                    return False, "abstain"
                out.append((int(np.sum(v >= 0.75 - TOL)), int(np.sum(v >= 0.5 - TOL))))
            return all(a >= 3 and b <= 8 for a, b in out), f"(#>=.75, #>=.5) = {out}"
        add("15", "exact", "each of cases 1-4: >=3 pcs >=.75 and <=8 pcs >=.5", c15)

        def c17(m):
            worst = []
            for cse, (q, root) in true_sets.items():
                v = vec(cse, m)
                if np.isnan(v[0]):
                    return False, "abstain"
                t4 = [(root + i) % 12 for i in M.TIERS[q][3]]
                worst.append(float(np.max(v[t4])))
            return max(worst) <= 0.25 + TOL, "max T4 per case: " + ", ".join(f"{x:.2f}" for x in worst)
        add("17", "exact", "T4 pcs of the true chord <= .25 (cases 1-4)", c17)

        def c18(m):
            v0, v1, v2 = sh["notes"][m][0], sh["notes"][m][1], sh["notes"][m][2]
            vn = R18n["notes"][m][0]
            if np.isnan(v0[0]) or np.isnan(vn[0]):
                return False, "abstain"
            ds = float(max(np.max(np.abs(v1 - v0)), np.max(np.abs(v2 - v0))))
            dn = float(np.max(np.abs(vn - v0)))
            return ds <= 0.05 + TOL and dn <= 0.10 + TOL, f"shift max|dH|={ds:.3f}, noise max|dH|={dn:.3f}"
        add("18", "exact", "+-50 ms |dH|<=.05; CENS noise |dH|<=.10", c18)
        # 16 demonstration
        d16 = {}
        for m in ALL_METHODS:
            n50, t4hi = [], 0
            for cse, (q, root) in true_sets.items():
                v = vec(cse, m)
                if np.isnan(v[0]):
                    n50.append(None)
                    continue
                n50.append(int(np.sum(v >= 0.5 - TOL)))
                t4 = [(root + i) % 12 for i in M.TIERS[q][3]]
                t4hi += int(np.max(v[t4]) >= 0.5 - TOL)
            d16[m] = {"n_pcs_ge_0.5_cases_1_4": n50, "cases_with_T4_ge_0.5": t4hi}
        demo[layer] = d16
        # eligible sets for transparency
        demo[f"{layer}_eligible_sets"] = {cse: [[f"{VOCAB.names[k]}:{R[cse]['diag']['wD'][R[cse]['dom'][0], k] / max(R[cse]['diag']['wD'][R[cse]['dom'][0]].sum(), 1e-12):.2f}"
                                                for k in np.flatnonzero(R[cse]["diag"]["S"][R[cse]["dom"][0]])], str(R[cse]["diag"]["status_D"][R[cse]["dom"][0]]),
                                               f"r_best={R[cse]['diag']['r_best'][R[cse]['dom'][0]]:.3f}", f"top_r={VOCAB.names[R[cse]['diag']['top_r'][R[cse]['dom'][0]]]}"]
                                         for cse in R}
    passed = {}
    for layer in ("T", "S"):
        for m in ALL_METHODS:
            cs = [c for c in checks if c["layer"] == layer and c["method"] == m]
            passed.setdefault(m, {})[layer] = f"{sum(c['pass'] for c in cs)}/{len(cs)}"
    return {"checks": checks, "passed": passed, "demo16": demo}


# ================================================================ stage 3: real songs


def tier_index_matrix():
    return VOCAB.tier_index_first       # (K, 12) tier 0..3 of each pc under chord k (first interpretation)


def metrics(V: np.ndarray, pc: np.ndarray, ref_tier: np.ndarray) -> Dict[str, Any]:
    ok = ~np.isnan(V[:, 0])
    out = {"coverage": float(ok.mean()), "notes_scored": int(ok.sum())}
    if not ok.any():
        return out
    out.update(X2.vec_metrics(V[ok], pc[ok]))
    rt = ref_tier[ok]
    have = rt[:, 0] >= 0
    for t, name in enumerate(("T1_chord_tone", "T2_extension", "T3_tension", "T4_clash")):
        m_ = have[:, None] & (rt == t)
        out[f"mean_H_{name}"] = float(V[ok][m_].mean()) if m_.any() else None
    return out


def evaluate_real(kappa, gamma, delta, songs, group_of):
    rng = np.random.default_rng(20261128)
    pooled: Dict[str, List[np.ndarray]] = {m: [] for m in ALL_METHODS + ["prod_calibrated", "D_ctx"]}
    pcs, ref_tiers, groups, durs = [], [], [], []
    stab = {k: {m: [] for m in MAIN_METHODS + ["D2_threshold_uniform", "D1_weight_nothreshold", "D4_adaptive_alpha", "C5u_a0.25", "C5u_a0.75", "D_a0.25", "D_a0.75"]} for k in ("noise", "note_shift", "beat_shift")}
    flips = {k: {m: [] for m in stab["noise"]} for k in stab}
    jacc, status_agree = [], []
    seg_stats = {"n_eligible": [], "status": [], "Ic": [], "r_best": [], "post_top1": [], "diatonic_best": [], "accepted": []}
    strat_real: Dict[str, List[float]] = {}
    timings = []
    examples_pool = []
    note_rows = []
    key_tiers_scale = None
    for si, song in enumerate(songs):
        h, n = X2.load_song(song)
        cens = h["other_chroma_cens"].astype(float)
        t = h["chroma_frame_times_s"].astype(float)
        beats = h["beat_times_s"].astype(float)
        st, en = n["start_time_s"].astype(float), n["end_time_s"].astype(float)
        pc = n["pitch_class"].astype(int)
        midi = n["target_midi_note"].astype(int)
        dur = n["duration_s"].astype(float)
        # timing by stage
        t0 = time.perf_counter()
        edges = M.beat_segments(t, beats)
        seg = M.segment_mean_chroma(cens, t, edges)
        t1 = time.perf_counter()
        C_only = seg / np.clip(seg.max(1, keepdims=True), 1e-12, None)
        t2 = time.perf_counter()
        r = M.similarity(seg, VOCAB.binary, "pearson")
        post = M.posterior(r, HM.POST_TAU, HM.POST_THETA)
        t3 = time.perf_counter()
        ev = HM.evidence(seg)
        t4 = time.perf_counter()
        segm, diag = HM.segment_methods(seg, ev, kappa, gamma, delta)
        coo = HM.note_weights(edges, st, en)
        notes = {k: HM.aggregate(v, coo, len(pc)) for k, v in segm.items()}
        t5 = time.perf_counter()
        timings.append({"song": song, "beat_segments": t1 - t0, "A_chroma": t2 - t1, "B_C_template_fit_posterior": t3 - t2,
                        "D_null_evidence": t4 - t3, "assemble_all_methods_and_notes": t5 - t4, "segments": int(len(seg)), "notes": int(len(pc))})
        dom = HM.dominant_segment(coo, len(pc))
        be = diag["best_eligible"][dom]
        rt = np.where(be[:, None] >= 0, tier_index_matrix()[np.maximum(be, 0)], -1)
        prod = X2.production_vectors(cens, np.round(h["other_chroma_rank"].astype(float) * 11) / 11, t, float(h["estimated_tempo_bpm"]),
                                     h["global_chroma_rank"].astype(float), st, en)["prod_calibrated"]
        # context on D (PASSING / NEIGHBOR only)
        Hd = notes[f"D_a{HM.ALPHA_PRIMARY}"]
        Hfill = np.where(np.isnan(Hd), 0.0, Hd)
        Hc, code = M.context_adjust(st, en, midi, Hfill, Hfill, Hfill, Hfill, Hfill, np.zeros(len(pc), dtype=bool))
        keep = np.vectorize(lambda c: c.split("+")[0] in ("PASSING", "NEIGHBOR"))(code)
        Dctx = np.where(keep & ~np.isnan(Hd), np.maximum(Hd, M.CTX["lift_to"]), Hd)
        for m in ALL_METHODS:
            pooled[m].append(notes[m])
        pooled["prod_calibrated"].append(prod)
        pooled["D_ctx"].append(Dctx)
        pcs.append(pc)
        ref_tiers.append(rt)
        groups += [group_of[song]] * len(pc)
        durs.append(dur)
        # segment-level stats
        kp = M.key_posterior(h["global_chroma_profile"].astype(float), 0.1)
        key = int(np.argmax(kp))
        scale = {(key % 12 + i) % 12 for i in ((0, 2, 3, 5, 7, 8, 10, 11) if key >= 12 else (0, 2, 4, 5, 7, 9, 11))}
        top = diag["top_r"]
        seg_stats["n_eligible"].append(diag["n_eligible"])
        seg_stats["status"].append(diag["status_D"])
        seg_stats["Ic"].append(diag["Ic"])
        seg_stats["r_best"].append(diag["r_best"])
        seg_stats["post_top1"].append(diag["post_top1"])
        seg_stats["diatonic_best"].append(np.array([VOCAB.pitch_sets[k] <= scale for k in top]))
        seg_stats["accepted"].append(diag["has_chord"])
        for strat, kw in ([("ABS", dict(theta=th)) for th in THETAS] + [("REL", dict(delta=d)) for d in DELTAS] + [("NULL", dict(eps=e)) for e in EPSS]
                          + [("NULL+REL", dict(eps=e, delta=d)) for e in EPSS for d in DELTAS] + [("EXCESS", dict(gamma=g)) for g in GAMMAS]
                          + [("EXCESS+REL", dict(gamma=g, delta=d)) for g in GAMMAS for d in DELTAS]):
            S = HM.eligible(ev, strat, **kw)
            key_ = strat + "|" + ",".join(f"{a}={b}" for a, b in kw.items())
            strat_real.setdefault(key_ + "|accept", []).append(S.any(1))
            strat_real.setdefault(key_ + "|n", []).append(S.sum(1))
        # stability: CENS noise (same segment edges), note shift, beat-grid shift
        noisy = cens * np.exp(rng.normal(0, NOISE_SIGMA, cens.shape))
        noisy /= np.clip(np.linalg.norm(noisy, axis=0, keepdims=True), 1e-12, None)
        segN = M.segment_mean_chroma(noisy, t, edges)
        evN = HM.evidence(segN)
        segmN, diagN = HM.segment_methods(segN, evN, kappa, gamma, delta)
        notesN = {k: HM.aggregate(v, coo, len(pc)) for k, v in segmN.items()}
        Sa, Sb = diag["S"], diagN["S"]
        inter, union = (Sa & Sb).sum(1), (Sa | Sb).sum(1)
        jacc.append(np.where(union > 0, inter / np.maximum(union, 1), 1.0))
        status_agree.append(diag["status_D"] == diagN["status_D"])
        shifted = []
        for shv in (-SHIFT_S, SHIFT_S):
            cooS = HM.note_weights(edges, st + shv, en + shv)
            shifted.append({k: HM.aggregate(v, cooS, len(pc)) for k, v in segm.items()})
        edgesB = M.beat_segments(t, beats + SHIFT_S)
        segB = M.segment_mean_chroma(cens, t, edgesB)
        evB = HM.evidence(segB)
        segmB, _ = HM.segment_methods(segB, evB, kappa, gamma, delta)
        cooB = HM.note_weights(edgesB, st, en)
        notesB = {k: HM.aggregate(v, cooB, len(pc)) for k, v in segmB.items()}
        for m in stab["noise"]:
            for kind, others in (("noise", [notesN]), ("note_shift", shifted), ("beat_shift", [notesB])):
                for o in others:
                    a, b = notes[m], o[m]
                    both = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
                    stab[kind][m].append(np.abs(a[both] - b[both]))
                    flips[kind][m].append(np.isnan(a[:, 0]) != np.isnan(b[:, 0]))
        # example pool and note rows
        for i in range(len(pc)):
            j = dom[i]
            Sset = np.flatnonzero(diag["S"][j])
            w = diag["wD"][j]
            chords = ", ".join(f"{VOCAB.names[k]}:{w[k] / max(w.sum(), 1e-12):.2f}" for k in Sset[np.argsort(-w[Sset])][:4])
            nseg = int(np.sum(coo[0] == i))
            tops = {diag["best_eligible"][s_] for s_ in coo[1][coo[0] == i]}
            row = {"song": song, "group": group_of[song], "note_index": int(n["note_index"][i]), "start_s": round(float(st[i]), 3), "end_s": round(float(en[i]), 3),
                   "dur": round(float(dur[i]), 3), "pc": M.PC_NAMES[pc[i]], "status_D": str(diag["status_D"][j]), "n_eligible": int(len(Sset)),
                   "eligible_weights": chords, "top_r_chord": VOCAB.names[diag["top_r"][j]], "r_best": round(float(diag["r_best"][j]), 3),
                   "excess_best": round(float(np.max(ev.r[j] - ev.m[j])), 3), "Ic": round(float(diag["Ic"][j]), 3),
                   "post_top1": round(float(diag["post_top1"][j]), 3), "C_sung": round(float(diag["C"][j, pc[i]]), 3),
                   "ref_tier": int(rt[i, pc[i]]), "segments": nseg, "crosses_chord": int(len(tops - {-1}) > 1)}
            for m in MAIN_METHODS + ["D2_threshold_uniform", "D1_weight_nothreshold", "D4_adaptive_alpha"]:
                v = notes[m][i, pc[i]]
                row[m] = None if np.isnan(v) else round(float(v), 3)
            row["prod_calibrated"] = round(float(prod[i, pc[i]]), 3)
            row["D_ctx"] = None if np.isnan(Dctx[i, pc[i]]) else round(float(Dctx[i, pc[i]]), 3)
            row["ctx_code"] = code[i, pc[i]] if keep[i, pc[i]] else ""
            examples_pool.append(row)
            note_rows.append(row)
        print(f"  [{si + 1}/{len(songs)}] {song}", flush=True)
    pc_all = np.concatenate(pcs)
    rt_all = np.vstack(ref_tiers)
    gr = np.array(groups)
    dur_all = np.concatenate(durs)
    res: Dict[str, Any] = {"methods": {}, "methods_common_subset": {}, "by_group": {}, "by_duration": {}}
    V = {m: np.vstack(v) for m, v in pooled.items()}
    common = np.all([~np.isnan(V[m][:, 0]) for m in V], axis=0)
    for m, arr in V.items():
        res["methods"][m] = metrics(arr, pc_all, rt_all)
        res["methods_common_subset"][m] = metrics(arr[common], pc_all[common], rt_all[common])
    res["common_subset_notes"] = int(common.sum())
    for g in ("axis2_50", "excluded_13"):
        mk = gr == g
        res["by_group"][g] = {m: metrics(V[m][mk], pc_all[mk], rt_all[mk]) for m in MAIN_METHODS + ["prod_calibrated"]}
    edges_d = (0, 0.1, 0.2, 0.4, 0.8, np.inf)
    b = np.clip(np.searchsorted(edges_d, dur_all, side="right") - 1, 0, 4)
    for i in range(5):
        mk = b == i
        lab = f"[{edges_d[i]:g},{edges_d[i + 1]:g})"
        res["by_duration"][lab] = {m: {"notes": int(mk.sum()), "coverage": float(np.mean(~np.isnan(V[m][mk, 0]))),
                                       "mean_H_sung": float(np.nanmean(V[m][mk][np.arange(mk.sum()), pc_all[mk]]))}
                                   for m in MAIN_METHODS + ["prod_calibrated"]}
    res["stability"] = {kind: {m: {"mean_abs_dH": float(np.mean(np.vstack(v))), "p95_abs_dH": float(np.percentile(np.vstack(v), 95)),
                                   "abstention_flip_rate": float(np.mean(np.concatenate(flips[kind][m])))}
                               for m, v in d.items()} for kind, d in stab.items()}
    J = np.concatenate(jacc)
    res["eligible_set_stability_noise"] = {"mean_jaccard": float(J.mean()), "share_identical": float(np.mean(J == 1.0)),
                                          "status_agreement": float(np.mean(np.concatenate(status_agree)))}
    ne = np.concatenate(seg_stats["n_eligible"])
    stat = np.concatenate(seg_stats["status"])
    acc = np.concatenate(seg_stats["accepted"])
    Ic = np.concatenate(seg_stats["Ic"])
    rb = np.concatenate(seg_stats["r_best"])
    pt = np.concatenate(seg_stats["post_top1"])
    dia = np.concatenate(seg_stats["diatonic_best"])
    res["chord_selection"] = {"segments": int(ne.size), "n_eligible_distribution": {str(k): float(np.mean(ne == k)) for k in range(6)} | {"6+": float(np.mean(ne >= 6))},
                              "mean_n_eligible_when_any": float(ne[ne > 0].mean()) if np.any(ne > 0) else 0.0,
                              "status_share_segments": {s_: float(np.mean(stat == s_)) for s_ in ("CHORD", "CHROMA_ONLY", "INSUFFICIENT")}}
    res["confidence_quality"] = {
        "accepted_vs_rejected": {"Ic_median": [float(np.median(Ic[acc])), float(np.median(Ic[~acc]))],
                                 "r_best_median": [float(np.median(rb[acc])), float(np.median(rb[~acc]))],
                                 "diatonic_share_of_top_chord": [float(np.mean(dia[acc])), float(np.mean(dia[~acc]))],
                                 "posterior_top1_median": [float(np.median(pt[acc])), float(np.median(pt[~acc]))],
                                 "posterior_top1_ge_0.9_share": [float(np.mean(pt[acc] >= 0.9)), float(np.mean(pt[~acc] >= 0.9))],
                                 "order": "[accepted, rejected]"},
        "real_acceptance_by_strategy": {k.rsplit("|", 1)[0]: (float(np.mean(np.concatenate(v))) if k.endswith("|accept") else None)
                                        for k, v in strat_real.items() if k.endswith("|accept")},
        "real_mean_n_by_strategy": {k.rsplit("|", 1)[0]: float(np.mean(np.concatenate(v))) for k, v in strat_real.items() if k.endswith("|n")}}
    res["note_status_D"] = {}
    stD = np.array([r_["status_D"] for r_ in note_rows])
    for s_ in ("CHORD", "CHROMA_ONLY", "INSUFFICIENT"):
        res["note_status_D"][s_] = float(np.mean(stD == s_))
    ctxk = np.array([r_["ctx_code"] for r_ in note_rows])
    res["context"] = {k: int(np.sum(ctxk == k)) for k in ("PASSING", "NEIGHBOR")}
    tm = {k: float(np.mean([x[k] for x in timings])) for k in timings[0] if k not in ("song",)}
    tmax = {k: float(np.max([x[k] for x in timings])) for k in timings[0] if k not in ("song", "segments", "notes")}
    res["timing_mean_per_song"] = tm
    res["timing_max_per_song"] = tmax
    return res, note_rows, V, pc_all


def pick_examples(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    def diff(r, a, b):
        return None if r[a] is None or r[b] is None else abs(r[a] - r[b])

    def pick(cands, key, k=5):
        seen, out = set(), []
        for r in sorted(cands, key=key):
            if r["song"] in seen:
                continue
            seen.add(r["song"])
            out.append(r)
            if len(out) == k:
                break
        return out

    ex = {}
    ex["largest_D_vs_A"] = pick([r for r in rows if diff(r, "D_a0.5", "A") is not None and r["dur"] >= 0.2], lambda r: -diff(r, "D_a0.5", "A"))
    ex["largest_D_vs_C5u"] = pick([r for r in rows if diff(r, "D_a0.5", "C5u_a0.5") is not None and r["dur"] >= 0.2], lambda r: -diff(r, "D_a0.5", "C5u_a0.5"))
    ex["largest_D_vs_B"] = pick([r for r in rows if diff(r, "D_a0.5", "B") is not None and r["dur"] >= 0.2], lambda r: -diff(r, "D_a0.5", "B"))
    ex["chord_tone_low_chroma"] = pick([r for r in rows if r["ref_tier"] == 0 and r["A"] is not None and r["dur"] >= 0.3], lambda r: r["A"])
    ex["extension"] = pick([r for r in rows if r["ref_tier"] == 1 and r["dur"] >= 0.3 and r["D_a0.5"] is not None], lambda r: -r["dur"])
    ex["short_note"] = pick([r for r in rows if r["dur"] < 0.1 and r["D_a0.5"] is not None], lambda r: r["D_a0.5"])
    ex["chord_transition"] = pick([r for r in rows if r["crosses_chord"] and r["D_a0.5"] is not None], lambda r: -r["dur"])
    ex["ambiguous_many_chords"] = pick([r for r in rows if r["n_eligible"] >= 3 and r["dur"] >= 0.2], lambda r: -r["n_eligible"])
    ex["chroma_only_fallback"] = pick([r for r in rows if r["status_D"] == "CHROMA_ONLY" and r["dur"] >= 0.3], lambda r: -r["dur"])
    ex["insufficient"] = pick([r for r in rows if r["status_D"] == "INSUFFICIENT"], lambda r: -r["dur"])
    return ex


# ================================================================ reports


def f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def write_reports(R: Dict[str, Any]) -> None:
    real, ts, syn = R["real"], R["threshold_study"], R["synthetic"]
    cols = ["coverage", "breadth_ge_0.75", "breadth_ge_0.5", "share_notes_ge10_pcs_ge_0.5", "spread_max_minus_min", "graded_share_in_0.25_0.75",
            "mean_H_T1_chord_tone", "mean_H_T2_extension", "mean_H_T3_tension", "mean_H_T4_clash", "mean_H_sung", "mean_H_alternatives", "win_rate_sung_vs_alt", "P4_up_gap"]
    head = "| method | cover | pcs≥.75 | pcs≥.5 | ≥10 pcs≥.5 | max−min | graded | H@T1 | H@T2 | H@T3 | H@T4 | H sung | H alt | win | P4↑ gap |"

    def table(d, order):
        L = [head, "|---" * 15 + "|"]
        for m in order:
            if m in d:
                e = d[m]
                L.append(f"| {m} | " + " | ".join(f(e.get(c), 2 if c.startswith("breadth") else 3) for c in cols) + " |")
        return L

    order = ["A", "A_rank", "B", "C5u_a0.5", "C5w_a0.5", "D_a0.5", "D_ctx", "prod_calibrated"]
    L = ["# Method comparison — all 63 songs (exploratory, in-corpus)", "",
         "Generated from `results.json`. Label-free proxies only. H@Tn is the mean H of pitch classes at tier n of a **reference chord**: D's top eligible chord in the note's dominant beat (notes with no eligible chord are excluded). This is circular for chord-based methods and informative for A.", "",
         f"## All notes (each method scored where it does not abstain; coverage = scored share of {real['methods']['A']['notes_scored'] / max(real['methods']['A']['coverage'], 1e-9):.0f} notes)", ""]
    L += table(real["methods"], order)
    L += ["", f"## Common subset ({real['common_subset_notes']} notes where every method scores)", ""] + table(real["methods_common_subset"], order)
    L += ["", "## Alpha sensitivity (C5u, C5w, D)", ""] + table(real["methods"], [f"{p}_a{a}" for p in ("C5u", "C5w", "D") for a in (0.25, 0.5, 0.75)])
    L += ["", "## Stability", "", "|ΔH| over the 12 pitch classes of notes scored in both runs. *flip* is the share of notes that change between scored and abstained.", "",
          "| method | noise mean | noise p95 | noise flip | note ±50 ms mean | p95 | beat grid +50 ms mean | p95 | beat flip |", "|---|---|---|---|---|---|---|---|---|"]
    for m in real["stability"]["noise"]:
        a, b_, c = real["stability"]["noise"][m], real["stability"]["note_shift"][m], real["stability"]["beat_shift"][m]
        L.append(f"| {m} | {f(a['mean_abs_dH'], 4)} | {f(a['p95_abs_dH'], 4)} | {f(a['abstention_flip_rate'], 4)} | {f(b_['mean_abs_dH'], 4)} | {f(b_['p95_abs_dH'], 4)} | "
                 f"{f(c['mean_abs_dH'], 4)} | {f(c['p95_abs_dH'], 4)} | {f(c['abstention_flip_rate'], 4)} |")
    L += ["", "Reference from experiment 2 (note-window method): production noise 0.0145 / shift 0.0139.", "",
          "## By duration (mean H of the sung pitch class; coverage)", "", "| duration | " + " | ".join(MAIN_METHODS + ["prod_calibrated"]) + " |", "|---" * 8 + "|"]
    for lab, d in real["by_duration"].items():
        L.append(f"| {lab} (n={d['A']['notes']}) | " + " | ".join(f"{d[m]['mean_H_sung']:.3f} ({d[m]['coverage']:.2f})" for m in MAIN_METHODS + ["prod_calibrated"]) + " |")
    L += ["", "## By group", "", "| group | method | cover | H sung | H alt | pcs≥.75 | pcs≥.5 |", "|---|---|---|---|---|---|---|"]
    for g, d in real["by_group"].items():
        for m, e in d.items():
            L.append(f"| {g} | {m} | {f(e['coverage'])} | {f(e.get('mean_H_sung'))} | {f(e.get('mean_H_alternatives'))} | {f(e.get('breadth_ge_0.75'), 2)} | {f(e.get('breadth_ge_0.5'), 2)} |")
    tm = real["timing_mean_per_song"]
    L += ["", "## Computational cost (mean per song, seconds, excluding Demucs and CREPE)", "",
          f"- beat-segment chroma (shared by all): {tm['beat_segments']:.4f}",
          f"- A (chroma ratio): +{tm['A_chroma']:.4f}",
          f"- B / C5 (template fit + posterior): +{tm['B_C_template_fit_posterior']:.4f}",
          f"- D (permutation-null evidence, 200 permutations × 112 templates per beat): +{tm['D_null_evidence']:.4f}",
          f"- assembling all 15 method variants + note aggregation: +{tm['assemble_all_methods_and_notes']:.4f}",
          f"- mean {tm['segments']:.0f} beat segments and {tm['notes']:.0f} notes per song. Max per song for D's null step: {real['timing_max_per_song']['D_null_evidence']:.3f} s.", ""]
    (HERE / "method_comparison.md").write_text("\n".join(L) + "\n")

    # confidence_analysis.md
    L = ["# Confidence analysis", "", "Generated from `results.json`.", "",
         "Evidence quantities per beat segment:",
         "- `r`: Pearson fit (absolute)",
         "- `G`: gap to the best template (relative)",
         "- `E`: permutation-null significance",
         "- `c = r − m`: **excess fit** over the chance-level best fit of the same values scrambled",
         "- `I_c`: chroma informativeness", "",
         "Posterior numbers are **uncalibrated weights**, not probabilities of correctness.", "",
         f"κ (chroma informativeness floor) = 99th percentile of I_c on synthetic noise = **{ts['kappa']:.4f}**. "
         f"Selected for D: **{ts['selected']['strategy']} γ={ts['selected']['gamma']} δ={ts['selected']['delta']}** ({ts['selected']['rule_applied']}).", "",
         "## Component medians by synthetic condition", "", "| condition | frames | r_best | excess c_best | E_best | I_c | posterior top-1 |", "|---|---|---|---|---|---|---|"]
    for k, d in ts["components"].items():
        L.append(f"| {k} | {ts['frames'][k]} | {f(d['r_best_median'])} | {f(d['excess_best_median'])} | {f(d['E_best_median'])} | {f(d['Ic_median'])} | {f(d['posterior_top1_median'])} |")
    L += ["", "## Strategy sweep (synthetic)", "",
          "*accept* is the share of frames with at least one eligible chord, and *n* is the mean eligible count. Noise FA is the mean of white and pink acceptance. Real acceptance (last columns) is descriptive only: no real labels exist.", "",
          "| strategy | params | clean recall | clean n | degraded recall | degraded n | noise FA | cluster accept | single-pitch accept (n) | mixture any-acceptable | mixture both | **real accept** | real n |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ra, rn = real["confidence_quality"]["real_acceptance_by_strategy"], real["confidence_quality"]["real_mean_n_by_strategy"]
    for r in ts["strategies"]:
        params = ",".join(f"{k}={r[k]}" for k in ("theta", "eps", "gamma", "delta") if k in r)
        key = r["strategy"] + "|" + ",".join(f"{k}={r[k]}" for k in ("theta", "delta", "eps", "gamma") if k in r)
        alt = [k for k in ra if k.startswith(r["strategy"] + "|") and all(f"{p}={r[p]}" in k for p in ("theta", "eps", "gamma", "delta") if p in r)]
        rk = alt[0] if alt else key
        sel = " **(D)**" if r["strategy"] == ts["selected"]["strategy"] and r.get("gamma") == ts["selected"]["gamma"] and r.get("delta", 10.0) == ts["selected"]["delta"] else ""
        L.append(f"| {r['strategy']}{sel} | {params} | {f(r['clean_recall'])} | {f(r['clean_mean_n'], 2)} | {f(r['degraded_recall'])} | {f(r['degraded_mean_n'], 2)} | "
                 f"{f(r['noise_false_accept'])} | {f(r['cluster_accept'])} | {f(r['single_accept'])} ({f(r['single_mean_n'], 1)}) | {f(r['mixture_any_acceptable'])} | "
                 f"{f(r['mixture_both_chords'])} | {f(ra.get(rk))} | {f(rn.get(rk), 2)} |")
    cq = real["confidence_quality"]["accepted_vs_rejected"]
    cs = real["chord_selection"]
    L += ["", "## Real songs: D's chord selection", "", f"Beat segments: {cs['segments']}. Distribution of eligible-chord count per beat: "
          + ", ".join(f"{k}: {v:.1%}" for k, v in cs["n_eligible_distribution"].items()) + f". Mean count when any qualify: {cs['mean_n_eligible_when_any']:.2f}.", "",
          "Segment status under D: " + ", ".join(f"{k} {v:.1%}" for k, v in cs["status_share_segments"].items()) + ". Note status: "
          + ", ".join(f"{k} {v:.1%}" for k, v in real["note_status_D"].items()) + ".", "",
          "| real beat segments | accepted | rejected |", "|---|---|---|"]
    for k in ("Ic_median", "r_best_median", "diatonic_share_of_top_chord", "posterior_top1_median", "posterior_top1_ge_0.9_share"):
        L.append(f"| {k} | {f(cq[k][0])} | {f(cq[k][1])} |")
    es = real["eligible_set_stability_noise"]
    L += ["", f"Eligible-set stability under CENS noise σ = 0.1: mean Jaccard {es['mean_jaccard']:.3f}, identical sets {es['share_identical']:.1%}, status agreement {es['status_agreement']:.1%}.", ""]
    (HERE / "confidence_analysis.md").write_text("\n".join(L) + "\n")

    # ablation_results.md
    abl = ["C5u_a0.5", "C5w_a0.5", "D1_weight_nothreshold", "D2_threshold_uniform", "D_a0.5", "D4_adaptive_alpha", "B", "A"]
    L = ["# Ablations", "", "Generated from `results.json`. All at α = 0.5 except D4 (adaptive α).", "",
         "| method | what changes |", "|---|---|",
         "| C5u | top-5 by fit, equal weights (no rejection) |",
         "| C5w | top-5, posterior weights |",
         "| D1 | weights ∝ excess fit over all chords with c > 0 — weighting, no threshold |",
         "| D2 | EXCESS+REL threshold, uniform weights |",
         "| **D (= D3)** | threshold + excess-fit weights |",
         "| D4 | D3 + confidence-aware α |",
         "| B | chord posterior only (α = 0) |",
         "| A | chroma only (α = 1) |", "",
         "## Real songs (all notes)", ""] + table(real["methods"], abl)
    L += ["", "## Real songs (common subset)", ""] + table(real["methods_common_subset"], abl)
    L += ["", "## Synthetic pass counts", "", "| method | T | S |", "|---|---|---|"] + [f"| {m} | {syn['passed'][m]['T']} | {syn['passed'][m]['S']} |" for m in abl]
    L += ["", "## Stability", "", "| method | noise mean | note shift mean | beat shift mean |", "|---|---|---|---|"]
    for m in abl:
        if m in real["stability"]["noise"]:
            L.append(f"| {m} | {f(real['stability']['noise'][m]['mean_abs_dH'], 4)} | {f(real['stability']['note_shift'][m]['mean_abs_dH'], 4)} | {f(real['stability']['beat_shift'][m]['mean_abs_dH'], 4)} |")
    (HERE / "ablation_results.md").write_text("\n".join(L) + "\n")

    # synthetic_tests.md
    L = ["# Synthetic tests", "",
         "Generated from `results.json`. Each requirement was fixed in `protocol.md` before any run and is applied to **every** method.",
         "- **Layer T:** idealized harmonic-partial chroma; tests the scoring rules.",
         "- **Layer S:** synthetic audio → the production CENS call; also tests estimation.",
         "Passing shows rule behavior on synthetic material only, not perceptual validity.", "",
         "## Pass counts", "", "| method | T | S |", "|---|---|---|"]
    L += [f"| {m} | {v['T']} | {v['S']} |" for m, v in syn["passed"].items()]
    L += ["", "## Case × method (✓ pass, ✗ fail)", ""]
    cases = list(dict.fromkeys((c["case"], c["truth"], c["check"]) for c in syn["checks"]))
    for layer in ("T", "S"):
        L += [f"### Layer {layer}", "", "| case | truth | requirement | " + " | ".join(ALL_METHODS) + " |", "|---" * (3 + len(ALL_METHODS)) + "|"]
        for cid, truth, desc in cases:
            cells = []
            for m in ALL_METHODS:
                c = next(x for x in syn["checks"] if x["layer"] == layer and x["case"] == cid and x["method"] == m)
                cells.append("✓" if c["pass"] else "✗")
            L.append(f"| {cid} | {truth} | {desc} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Details for the main methods", "", "| layer | case | method | pass | detail |", "|---|---|---|---|---|"]
    for c in syn["checks"]:
        if c["method"] in ("A", "B", "C5u_a0.5", "D_a0.5"):
            L.append(f"| {c['layer']} | {c['case']} | {c['method']} | {'✓' if c['pass'] else '✗'} | {c['detail']} |")
    L += ["", "## Case 16 — top-5 permissiveness (demonstration)", "", "| layer | method | pcs ≥ 0.5 in cases 1–4 | cases with a T4 pc ≥ 0.5 |", "|---|---|---|---|"]
    for layer in ("T", "S"):
        for m, d in syn["demo16"][layer].items():
            L.append(f"| {layer} | {m} | {d['n_pcs_ge_0.5_cases_1_4']} | {d['cases_with_T4_ge_0.5']} |")
    L += ["", "## D's eligible chords per case (dominant beat)", "", "| layer | case | eligible (weight) | status | fit | top-r chord |", "|---|---|---|---|---|---|"]
    for layer in ("T", "S"):
        for cse, (ch, st_, rb, tr) in syn["demo16"][f"{layer}_eligible_sets"].items():
            L.append(f"| {layer} | {cse} | {', '.join(ch) or '—'} | {st_} | {rb} | {tr} |")
    (HERE / "synthetic_tests.md").write_text("\n".join(L) + "\n")

    # failure_analysis.md
    L = ["# Failure analysis and real-song examples", "", "Generated from `results.json`. One example per song per category, chosen by the most extreme value. "
         "*elig* lists D's eligible chords with normalized excess-fit weights. *C* is the direct chroma support of the sung pitch class, and *ref tier* is its tier under D's top eligible chord (−1 = none).", ""]
    for cat, ex in R["examples"].items():
        L += [f"## {cat}", "", "| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |", "|---" * 18 + "|"]
        for e in ex:
            L.append(f"| {e['song']} | {e['start_s']}–{e['end_s']} | {e['dur']} | {e['pc']} | {e['status_D']} | {e['eligible_weights'] or '—'} | {e['r_best']} | "
                     f"{e['excess_best']} | {e['Ic']} | {e['post_top1']} | {e['C_sung']} | {e['ref_tier']} | {f(e['A'])} | {f(e['B'])} | {f(e['C5u_a0.5'])} | "
                     f"{f(e['D_a0.5'])} | {f(e['D2_threshold_uniform'])} | {f(e['prod_calibrated'])} |")
        L.append("")
    (HERE / "failure_analysis.md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if (HERE / "results.json").exists() and not args.overwrite:
        print("outputs exist; pass --overwrite", file=sys.stderr)
        return 2
    sel = json.loads((REPO / "data" / "dataset_selections.json").read_text())["songs"]
    songs = sorted(sel)
    assert len(songs) == 63 and set(songs) == {r["song_id"] for r in json.loads((MAIN / "manifest.json").read_text())["songs"]}
    group_of = {s: ("axis2_50" if sel[s]["axis2_ml"] else "excluded_13") for s in songs}
    before = {s: [sha256_file(MAIN / s / f_) for f_ in ("notes.npz", "harmony.npz")] for s in songs}
    t0 = time.perf_counter()
    print("stage 1: threshold study", flush=True)
    ts = threshold_study()
    kappa, g, d = ts["kappa"], ts["selected"]["gamma"], ts["selected"]["delta"]
    print("  kappa", kappa, "selected", ts["selected"], flush=True)
    print("stage 2: synthetic tests", flush=True)
    syn = synthetic_tests(kappa, g, d)
    print("  passed", {m: syn["passed"][m] for m in MAIN_METHODS}, flush=True)
    t_syn = time.perf_counter() - t0
    print("stage 3: real songs", flush=True)
    real, rows, V, pc_all = evaluate_real(kappa, g, d, songs, group_of)
    after = {s: [sha256_file(MAIN / s / f_) for f_ in ("notes.npz", "harmony.npz")] for s in songs}
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    note_path = ARCHIVE / "note_level.csv.gz"
    with gzip.open(note_path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    vec_path = ARCHIVE / "note_vectors.npz"
    np.savez_compressed(vec_path, pitch_class=pc_all, **{m.replace(".", "_"): v.astype(np.float32) for m, v in V.items()})
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    R = {"experiment": "axis1_hybrid_chroma_chords", "created_utc": datetime.now(timezone.utc).isoformat(),
         "scope": "exploratory, all 63 songs; in-corpus proxies; not perceptual validation",
         "provenance": {"git_commit": git, "python": platform.python_version(), "numpy": np.__version__,
                        "code_sha256": {f_: sha256_file(HERE / f_) for f_ in ("axis1_hybrid.py", "hybrid_model.py", "protocol.md")},
                        "reused_experiment2_sha256": {f_: sha256_file(HERE.parent / "axis1_chord_compatibility" / f_) for f_ in ("chord_model.py", "synthetic_audio.py", "axis1_chord_compatibility.py")},
                        "archive": {str(note_path.relative_to(REPO)): sha256_file(note_path), str(vec_path.relative_to(REPO)): sha256_file(vec_path)}},
         "verification": {"source_files_unchanged": before == after, "songs": len(songs), "notes": len(rows)},
         "settings": {"n_permutations": HM.N_PERM, "alphas": HM.ALPHAS, "alpha_primary": HM.ALPHA_PRIMARY, "noise_sigma": NOISE_SIGMA, "shift_s": SHIFT_S,
                      "posterior_tau_theta": [HM.POST_TAU, HM.POST_THETA]},
         "threshold_study": ts, "synthetic": syn, "real": real, "examples": pick_examples(rows), "synthetic_seconds": t_syn}
    (HERE / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    write_reports(R)
    print(json.dumps(R["verification"]), flush=True)
    return 0 if R["verification"]["source_files_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
