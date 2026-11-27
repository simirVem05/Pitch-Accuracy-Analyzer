"""
Chroma + confidence-weighted chord hybrid (experimental; not production).

Reuses experiment 2's vocabulary, templates, tiers and beat segmentation by import
(that directory is not modified). Everything here is per beat segment; notes are
an overlap-weighted mean of their segments. See protocol.md.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "axis1_chord_compatibility"))
import chord_model as M  # noqa: E402  (experiment 2, read-only reuse)

VOCAB = M.build_vocabulary()
K = VOCAB.K
TIERS = VOCAB.tier_values                       # (K, 12): T1 1, T2 2/3, T3 1/3, T4 0
N_PERM = 200
PERMS = np.array([np.random.default_rng(20261128 + i).permutation(12) for i in range(N_PERM)])
POST_TAU, POST_THETA = 0.02, 0.8                # experiment 2's fitted posterior (Method B only)
ALPHAS = (0.25, 0.5, 0.75)
ALPHA_PRIMARY = 0.5

_Tc = VOCAB.binary - VOCAB.binary.mean(axis=1, keepdims=True)
_Tn = np.linalg.norm(_Tc, axis=1)
# r(pi(v), T_k) == r(v, pi^-1(T_k)): permute the templates once instead of every frame
_inv = np.argsort(PERMS, axis=1)
_TcP = _Tc[:, _inv].transpose(1, 0, 2)          # (P, K, 12): template k with bins permuted by pi^-1


@dataclass
class Evidence:
    r: np.ndarray        # (n, K) Pearson fit
    E: np.ndarray        # (n, K) null-calibrated evidence in [0, 1]
    m: np.ndarray        # (n,) median of null best-fit
    q: Dict[float, np.ndarray]  # eps -> (n,) eps-quantile of null best-fit
    Ic: np.ndarray       # (n,) chroma informativeness
    post: np.ndarray     # (n, K+1) experiment-2 posterior (last = N)


def informativeness(v: np.ndarray) -> np.ndarray:
    p = v / np.clip(v.sum(axis=1, keepdims=True), 1e-300, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    return 1.0 - h / np.log(12.0)


def evidence(v: np.ndarray, eps_list=(0.90, 0.95, 0.99), chunk: int = 400) -> Evidence:
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    vc = v - v.mean(axis=1, keepdims=True)
    vn = np.linalg.norm(vc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nan_to_num((vc @ _Tc.T) / (vn[:, None] * _Tn[None, :]))
    E = np.empty((n, K))
    m = np.empty(n)
    q = {e: np.empty(n) for e in eps_list}
    flat = _TcP.reshape(-1, 12).T                  # (12, P*K)
    denomT = np.tile(_Tn, N_PERM)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        with np.errstate(invalid="ignore", divide="ignore"):
            rn = np.nan_to_num((vc[a:b] @ flat) / (vn[a:b, None] * denomT[None, :]))
        rmax = rn.reshape(b - a, N_PERM, K).max(axis=2)       # (chunk, P)
        rmax.sort(axis=1)
        E[a:b] = (rmax[:, :, None] < r[a:b, None, :]).mean(axis=1)
        m[a:b] = np.median(rmax, axis=1)
        for e in eps_list:
            q[e][a:b] = np.quantile(rmax, e, axis=1)
    return Evidence(r, E, m, q, informativeness(v), M.posterior(r, POST_TAU, POST_THETA))


def eligible(ev: Evidence, strategy: str, eps: float = 0.95, delta: float = 0.05, theta: float = 0.8,
             gamma: float = 0.10) -> np.ndarray:
    G = ev.r.max(axis=1, keepdims=True) - ev.r
    c = ev.r - ev.m[:, None]
    if strategy == "EXCESS":
        return c >= gamma
    if strategy == "EXCESS+REL":
        return (c >= gamma) & (G <= delta)
    if strategy == "ABS":
        return (ev.r >= theta) & (G <= 0.05)
    if strategy == "REL":
        return G <= delta
    if strategy == "NULL":
        return ev.E >= eps
    if strategy == "NULL+REL":
        return (ev.E >= eps) & (G <= delta)
    raise ValueError(strategy)


def _mix(F: np.ndarray, valid: np.ndarray, C: np.ndarray, Ic: np.ndarray, kappa: float, alpha) -> Tuple[np.ndarray, np.ndarray]:
    """alpha*C + (1-alpha)*F where a chord set exists; chroma only if informative; else NaN. Returns (H, status)."""
    a = np.asarray(alpha, dtype=float)
    a = a[:, None] if a.ndim == 1 else a
    H = np.where(valid[:, None], a * C + (1 - a) * F, np.where((Ic >= kappa)[:, None], C, np.nan))
    status = np.where(valid, "CHORD", np.where(Ic >= kappa, "CHROMA_ONLY", "INSUFFICIENT"))
    return H, status


def segment_methods(v: np.ndarray, ev: Evidence, kappa: float, gamma: float, delta: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-segment (n, 12) vectors for every method; NaN rows = abstain. Also per-segment diagnostics."""
    n = v.shape[0]
    mx = np.clip(v.max(axis=1, keepdims=True), 1e-12, None)
    C = v / mx
    Crank = np.argsort(np.argsort(v, axis=1), axis=1) / 11.0
    informative = ev.Ic >= kappa
    out: Dict[str, np.ndarray] = {}
    out["A"] = np.where(informative[:, None], C, np.nan)
    out["A_rank"] = np.where(informative[:, None], Crank, np.nan)
    pc_ = ev.post[:, :K] / np.clip(ev.post[:, :K].sum(axis=1, keepdims=True), 1e-300, None)
    out["B"] = pc_ @ TIERS
    top5 = np.argsort(-ev.r, axis=1)[:, :5]
    F5u = TIERS[top5].mean(axis=1)
    w5 = np.take_along_axis(pc_, top5, axis=1)
    w5 = w5 / np.clip(w5.sum(axis=1, keepdims=True), 1e-300, None)
    F5w = np.einsum("nk,nkj->nj", w5, TIERS[top5])
    S = eligible(ev, "EXCESS+REL", gamma=gamma, delta=delta)
    x = ev.r - ev.m[:, None]
    wD = np.where(S, np.maximum(x, 0), 0.0)
    has = wD.sum(axis=1) > 0
    FD = (wD / np.clip(wD.sum(axis=1, keepdims=True), 1e-300, None)) @ TIERS
    Su = S.astype(float)
    FD2 = (Su / np.clip(Su.sum(axis=1, keepdims=True), 1e-300, None)) @ TIERS
    w1 = np.maximum(x, 0)
    has1 = w1.sum(axis=1) > 0
    FD1 = (w1 / np.clip(w1.sum(axis=1, keepdims=True), 1e-300, None)) @ TIERS
    cbest = np.where(has, np.max(np.where(S, x, -np.inf), axis=1), 0.0)
    s = np.clip((cbest - gamma) / np.clip(1 - ev.m - gamma, 1e-9, None), 0, 1)
    status = {}
    for a in ALPHAS:
        out[f"C5u_a{a}"] = a * C + (1 - a) * F5u
        out[f"C5w_a{a}"] = a * C + (1 - a) * F5w
        out[f"D_a{a}"], status[f"D_a{a}"] = _mix(FD, has, C, ev.Ic, kappa, a)
    out["D1_weight_nothreshold"], status["D1"] = _mix(FD1, has1, C, ev.Ic, kappa, ALPHA_PRIMARY)
    out["D2_threshold_uniform"], status["D2"] = _mix(FD2, S.any(axis=1), C, ev.Ic, kappa, ALPHA_PRIMARY)
    out["D4_adaptive_alpha"], status["D4"] = _mix(FD, has, C, ev.Ic, kappa, 0.25 + 0.5 * (1 - s))
    out["Fchord_D"] = np.where(has[:, None], FD, np.nan)   # chord component of D alone (diagnostic)
    diag = {"n_eligible": S.sum(axis=1), "has_chord": has, "Ic": ev.Ic, "informative": informative,
            "status_D": status[f"D_a{ALPHA_PRIMARY}"], "top_r": np.argmax(ev.r, axis=1), "r_best": ev.r.max(axis=1),
            "E_best": ev.E.max(axis=1), "post_top1": pc_.max(axis=1), "post_N": ev.post[:, K],
            "best_eligible": np.where(has, np.argmax(np.where(S, x, -np.inf), axis=1), -1), "S": S, "wD": wD, "C": C}
    return out, diag


def note_weights(edges: np.ndarray, start: np.ndarray, end: np.ndarray):
    """COO lists (note, segment, overlap seconds) for every note/segment overlap."""
    ni, si, ov = [], [], []
    nseg = len(edges) - 1
    for i, (a, b) in enumerate(zip(start, end)):
        j0 = max(int(np.searchsorted(edges, a, side="right")) - 1, 0)
        j1 = min(int(np.searchsorted(edges, b, side="left")) - 1, nseg - 1)
        for j in range(j0, max(j1, j0) + 1):
            o = min(b, edges[j + 1]) - max(a, edges[j])
            if o > 0:
                ni.append(i)
                si.append(j)
                ov.append(o)
        if not ni or ni[-1] != i:          # zero-length or numerical edge: nearest segment
            ni.append(i)
            si.append(min(max(j0, 0), nseg - 1))
            ov.append(1e-6)
    return np.array(ni), np.array(si), np.array(ov)


def aggregate(Hseg: np.ndarray, coo, n_notes: int) -> np.ndarray:
    ni, si, ov = coo
    valid = ~np.isnan(Hseg[si, 0])
    num = np.zeros((n_notes, 12))
    den = np.zeros(n_notes)
    np.add.at(num, ni[valid], ov[valid, None] * Hseg[si[valid]])
    np.add.at(den, ni[valid], ov[valid])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den[:, None]
    out[den == 0] = np.nan
    return out


def dominant_segment(coo, n_notes: int) -> np.ndarray:
    ni, si, ov = coo
    best = np.full(n_notes, -1)
    bo = np.zeros(n_notes)
    for a, b, o in zip(ni, si, ov):
        if o > bo[a]:
            bo[a], best[a] = o, b
    return best
