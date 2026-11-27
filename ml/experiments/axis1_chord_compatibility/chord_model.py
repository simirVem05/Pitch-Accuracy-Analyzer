"""
Chord-aware harmonic compatibility — the model (experimental; not production).

Pure numpy. Everything here operates on 12-bin chroma already computed by the
production chroma path, so the same functions run on saved real-song CENS and on
CENS rendered from synthetic audio. See protocol.md for the rationale of every
constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

QUALITIES: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "power": (0, 7),
    "dom7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "hdim7": (0, 3, 6, 10),
}
QUALITY_SUFFIX = {"maj": "", "min": "m", "dim": "dim", "aug": "+", "sus2": "sus2", "sus4": "sus4", "power": "5",
                  "dom7": "7", "maj7": "maj7", "min7": "m7", "hdim7": "m7b5"}

# Ordinal tiers by interval above the root (protocol.md, Step B). T1 chord tone, T2 stable
# extension/colour, T3 tension usable with context, T4 clash without stylistic licence.
TIERS: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {
    "maj":   ((0, 4, 7), (2, 9, 11), (3, 5, 10), (1, 6, 8)),
    "min":   ((0, 3, 7), (2, 5, 10), (6, 8, 9, 11), (1, 4)),
    "dim":   ((0, 3, 6), (9, 10), (2, 5, 8), (1, 4, 7, 11)),
    "aug":   ((0, 4, 8), (2, 6, 10), (11,), (1, 3, 5, 7, 9)),
    "sus2":  ((0, 2, 7), (4, 5, 9, 10), (3, 11), (1, 6, 8)),
    "sus4":  ((0, 5, 7), (2, 9, 10), (3, 4, 11), (1, 6, 8)),
    "power": ((0, 7), (2, 3, 4, 5, 10), (8, 9, 11), (1, 6)),
    "dom7":  ((0, 4, 7, 10), (2, 9), (1, 3, 5, 6, 8), (11,)),
    "maj7":  ((0, 4, 7, 11), (2, 6, 9), (3, 5), (1, 8, 10)),
    "min7":  ((0, 3, 7, 10), (2, 5, 9), (6, 8), (1, 4, 11)),
    "hdim7": ((0, 3, 6, 10), (2, 5, 8), (1,), (4, 7, 9, 11)),
}
KEY_TIERS = {"major": ((0, 4, 7), (2, 5, 9, 11), (3, 6, 10), (1, 8)),
             "minor": ((0, 3, 7), (2, 5, 8, 10, 11), (6, 9), (1, 4))}
# Krumhansl–Kessler probe-tone profiles.
KK = {"major": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
      "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])}

TIER_VALUES_LINEAR = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)
TIER_VALUES_ALT = (1.0, 0.75, 0.4, 0.1)

for _q, _t in list(TIERS.items()) + list(KEY_TIERS.items()):
    _all = sorted(i for tier in _t for i in tier)
    assert _all == list(range(12)), f"tier table for {_q} does not partition the 12 intervals"
for _q, _iv in QUALITIES.items():
    assert set(_iv) == set(TIERS[_q][0]), f"T1 of {_q} must equal its chord tones"


def tier_vector(intervals_tiers, root: int, values=TIER_VALUES_LINEAR) -> np.ndarray:
    v = np.zeros(12)
    for t, members in enumerate(intervals_tiers):
        for iv in members:
            v[(root + iv) % 12] = values[t]
    return v


def tier_index_vector(intervals_tiers, root: int) -> np.ndarray:
    v = np.zeros(12, dtype=int)
    for t, members in enumerate(intervals_tiers):
        for iv in members:
            v[(root + iv) % 12] = t
    return v


@dataclass
class Vocabulary:
    names: List[str]                    # one per distinct pitch set ("Csus4/Fsus2")
    pitch_sets: List[frozenset]
    qualities: List[Tuple[str, ...]]    # interpretations' qualities
    members: List[List[Tuple[str, int]]]  # (quality, root) interpretations
    binary: np.ndarray                  # (K, 12) unit-norm
    harmonic: np.ndarray                # (K, 12) unit-norm
    tier_values: np.ndarray             # (K, 12) mean of interpretations' tier values
    tier_values_alt: np.ndarray
    tier_index_first: np.ndarray        # (K, 12) tier index of the first interpretation (for reason codes)

    @property
    def K(self) -> int:
        return len(self.names)


def harmonic_chroma(pcs: Sequence[int], partials: int = 6, decay: float = 0.6) -> np.ndarray:
    v = np.zeros(12)
    for pc in pcs:
        for k in range(1, partials + 1):
            v[(pc + int(round(12 * np.log2(k)))) % 12] += decay ** (k - 1)
    return v


def build_vocabulary(qualities: Sequence[str] = tuple(QUALITIES)) -> Vocabulary:
    by_set: Dict[frozenset, List[Tuple[str, int]]] = {}
    order: List[frozenset] = []
    for q in qualities:
        for r in range(12):
            s = frozenset((r + i) % 12 for i in QUALITIES[q])
            if s not in by_set:
                by_set[s] = []
                order.append(s)
            by_set[s].append((q, r))
    names, sets, quals, mem, B, Hm, T, Ta, Ti = [], [], [], [], [], [], [], [], []
    for s in order:
        m = by_set[s]
        names.append("/".join(f"{PC_NAMES[r]}{QUALITY_SUFFIX[q]}" for q, r in m))
        sets.append(s)
        quals.append(tuple(sorted({q for q, _ in m})))
        mem.append(m)
        b = np.zeros(12)
        b[list(s)] = 1.0
        B.append(b / np.linalg.norm(b))
        h = harmonic_chroma(sorted(s))
        Hm.append(h / np.linalg.norm(h))
        T.append(np.mean([tier_vector(TIERS[q], r) for q, r in m], axis=0))
        Ta.append(np.mean([tier_vector(TIERS[q], r, TIER_VALUES_ALT) for q, r in m], axis=0))
        Ti.append(tier_index_vector(TIERS[m[0][0]], m[0][1]))
    return Vocabulary(names, sets, quals, mem, np.array(B), np.array(Hm), np.array(T), np.array(Ta), np.array(Ti))


# ---------------------------------------------------------------- chord scoring and posterior


def similarity(chroma: np.ndarray, templates: np.ndarray, kind: str) -> np.ndarray:
    """chroma (n, 12) -> (n, K) similarity in [-1, 1]. 'cosine' or 'pearson'."""
    x = np.asarray(chroma, dtype=float)
    T = templates
    if kind == "pearson":
        x = x - x.mean(axis=1, keepdims=True)
        T = T - T.mean(axis=1, keepdims=True)
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    Tn = np.linalg.norm(T, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (x @ T.T) / (xn * Tn.T)
    return np.nan_to_num(s, nan=0.0)


def posterior(sim: np.ndarray, tau: float, theta: float) -> np.ndarray:
    """(n, K) similarities -> (n, K+1) posterior; last column is N (no chord) with fixed score theta."""
    z = np.concatenate([sim, np.full((sim.shape[0], 1), theta)], axis=1) / tau
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def forward_backward(emis: np.ndarray, p_stay: float) -> np.ndarray:
    """
    Posterior marginals of an HMM with uniform switching. emis: (T, S) emission
    likelihoods (any positive scale). Transition: stay p_stay, else uniform over the
    other S-1 states. O(T*S).
    """
    T_, S = emis.shape
    if T_ == 0:
        return emis
    sw = (1.0 - p_stay) / (S - 1)
    a = np.empty_like(emis)
    prev = np.full(S, 1.0 / S) * emis[0]
    prev /= prev.sum()
    a[0] = prev
    for t in range(1, T_):
        cur = (prev * (p_stay - sw) + sw) * emis[t]   # prev @ A, using sum(prev) == 1
        cur /= cur.sum()
        a[t] = cur
        prev = cur
    b = np.ones(S)
    out = np.empty_like(emis)
    out[-1] = a[-1]
    for t in range(T_ - 2, -1, -1):
        m = b * emis[t + 1]
        b = (m * (p_stay - sw) + sw * m.sum())
        b /= b.sum()
        g = a[t] * b
        out[t] = g / g.sum()
    return out


def beat_segments(frame_times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Segment edges (seconds) covering the whole chromagram: beats, padded with median-beat segments."""
    t0, t1 = float(frame_times[0]), float(frame_times[-1]) + 1e-6
    b = np.asarray(beat_times, dtype=float)
    b = b[(b > t0) & (b < t1)]
    if b.size < 2:
        step = 0.5
        return np.arange(t0, t1 + step, step)
    med = float(np.median(np.diff(b)))
    pre = np.arange(b[0] - med, t0 - med, -med)[::-1]
    post = np.arange(b[-1] + med, t1 + med, med)
    edges = np.concatenate([pre, b, post])
    edges = np.concatenate([[t0], edges[(edges > t0) & (edges < t1)], [t1]])
    return np.unique(edges)


def segment_mean_chroma(chroma: np.ndarray, frame_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """chroma (12, nf) -> (n_seg, 12) mean over frames whose time lies in each segment."""
    idx = np.searchsorted(edges, frame_times, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    n = len(edges) - 1
    sums = np.zeros((n, 12))
    np.add.at(sums, idx, chroma.T)
    cnt = np.bincount(idx, minlength=n).astype(float)
    out = np.zeros((n, 12))
    ok = cnt > 0
    out[ok] = sums[ok] / cnt[ok, None]
    # empty segments (shorter than a hop): copy nearest frame
    for j in np.flatnonzero(~ok):
        c = 0.5 * (edges[j] + edges[j + 1])
        out[j] = chroma[:, int(np.clip(np.searchsorted(frame_times, c), 0, chroma.shape[1] - 1))]
    return out


@dataclass
class EstimatorConfig:
    template: str = "binary"        # binary | harmonic
    sim: str = "cosine"             # cosine | pearson
    tau: float = 0.05
    theta: float = 0.5
    temporal: str = "beat_hmm"      # frame | frame_hmm | beat | beat_hmm
    bar_beats: float = 4.0          # expected chord duration (beats) for the HMM prior


def frame_posteriors(cens: np.ndarray, frame_times: np.ndarray, beat_times: np.ndarray, tempo: float,
                     vocab: Vocabulary, cfg: EstimatorConfig, hop_s: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (per-frame posterior (nf, K+1), segment edges or None). For beat estimators
    the per-frame posterior is the posterior of the beat segment containing the frame.
    """
    T = vocab.binary if cfg.template == "binary" else vocab.harmonic
    if cfg.temporal in ("frame", "frame_hmm"):
        P = posterior(similarity(cens.T, T, cfg.sim), cfg.tau, cfg.theta)
        if cfg.temporal == "frame_hmm":
            beat_s = 60.0 / tempo if tempo > 0 else 0.5
            d = max(cfg.bar_beats * beat_s / hop_s, 2.0)
            P = forward_backward(P, 1.0 - 1.0 / d)
        return P, None
    edges = beat_segments(frame_times, beat_times)
    seg = segment_mean_chroma(cens, frame_times, edges)
    P = posterior(similarity(seg, T, cfg.sim), cfg.tau, cfg.theta)
    if cfg.temporal == "beat_hmm":
        P = forward_backward(P, 1.0 - 1.0 / max(cfg.bar_beats, 1.0 + 1e-9))
    idx = np.clip(np.searchsorted(edges, frame_times, side="right") - 1, 0, len(edges) - 2)
    return P[idx], edges


def window_mean(P: np.ndarray, i0: np.ndarray, width: np.ndarray) -> np.ndarray:
    cum = np.concatenate([np.zeros((1, P.shape[1])), np.cumsum(P, axis=0)], axis=0)
    return (cum[i0 + width] - cum[i0]) / width[:, None]


# ---------------------------------------------------------------- key context


def key_posterior(global_profile: np.ndarray, tau_key: float) -> np.ndarray:
    """(24,) posterior over keys: 0-11 major (tonic pc), 12-23 minor."""
    g = np.asarray(global_profile, dtype=float)
    r = []
    for mode in ("major", "minor"):
        for t in range(12):
            prof = np.roll(KK[mode], t)
            r.append(np.corrcoef(g, prof)[0, 1] if g.std() > 0 else 0.0)
    z = np.array(r) / tau_key
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def key_tier_matrix(values=TIER_VALUES_LINEAR) -> np.ndarray:
    return np.array([tier_vector(KEY_TIERS[m], t, values) for m in ("major", "minor") for t in range(12)])


def key_name(k: int) -> str:
    return f"{PC_NAMES[k % 12]} {'major' if k < 12 else 'minor'}"


# ---------------------------------------------------------------- compatibility


@dataclass
class NoteHarmony:
    P: np.ndarray               # (n, K+1) note-window chord posterior
    e: np.ndarray               # evidence = 1 - p(N)
    H_chord: np.ndarray         # (n, 12) expected chord tier (renormalised without N)
    H_hard: np.ndarray          # (n, 12) tier under argmax chord
    H_key: np.ndarray           # (n, 12)
    H_inst: np.ndarray          # (n, 12) evidence-gated
    H_fixed: np.ndarray         # (n, 12) fixed 0.75/0.25 blend
    top: np.ndarray             # (n, 3) chord indices
    top_p: np.ndarray           # (n, 3) renormalised chord masses
    entropy: np.ndarray         # (n,) bits, over chords only


def compatibility(Pn: np.ndarray, tiers: np.ndarray, Hk: np.ndarray) -> NoteHarmony:
    K = tiers.shape[0]
    chord = Pn[:, :K]
    e = 1.0 - Pn[:, K]
    mass = np.clip(chord.sum(axis=1, keepdims=True), 1e-300, None)
    pc = chord / mass
    H_chord = pc @ tiers
    best = np.argmax(pc, axis=1)
    H_hard = tiers[best]
    H_key = np.broadcast_to(Hk, H_chord.shape)
    H_inst = e[:, None] * H_chord + (1.0 - e[:, None]) * H_key
    H_fixed = 0.75 * H_chord + 0.25 * H_key
    order = np.argsort(-pc, axis=1)[:, :3]
    top_p = np.take_along_axis(pc, order, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pc > 0, pc * np.log2(pc), 0.0), axis=1)
    return NoteHarmony(Pn, e, H_chord, H_hard, H_key, H_inst, H_fixed, order, top_p, ent)


# ---------------------------------------------------------------- melodic / temporal context

CTX = {"neighbor_gap_s": 0.30, "short_embellish_s": 0.35, "resolution_max_s": 1.0, "anticipation_max_s": 0.6,
       "supported": 2.0 / 3.0, "candidate_below": 0.5, "lift_to": 2.0 / 3.0, "context_window_s": 0.5, "short_s": 0.10}


def context_adjust(start, end, midi, H_inst_cand, H_before_cand, H_after_cand, H_half1_cand, H_half2_cand,
                   crosses, cand_pc_matrix=None):
    """
    Deterministic phrase-context rules (protocol.md Step C), evaluated for every
    candidate pitch class in each note's place.

    H_*_cand: (n, 12) compatibility of each pitch class over the note / before / after /
    first half / second half windows. The sung neighbours are fixed; only the note
    itself is substituted, so a candidate's motion to its neighbours follows its own
    pitch (nearest MIDI with that pitch class).
    Returns H_ctx (n, 12) and code (n, 12) of strings.
    """
    n = len(start)
    S = CTX["supported"]
    H_ctx = H_inst_cand.copy()
    code = np.full((n, 12), "", dtype=object)
    pcs = np.arange(12)
    orig_pc = (midi % 12).astype(int)
    idx = np.arange(n)
    sung_H = H_inst_cand[idx, orig_pc]      # neighbours keep their sung pitch and score
    for i in range(n):
        dur = end[i] - start[i]
        has_prev = i > 0 and start[i] - end[i - 1] <= CTX["neighbor_gap_s"]
        has_next = i < n - 1 and start[i + 1] - end[i] <= CTX["neighbor_gap_s"]
        # candidate MIDI: nearest to the sung MIDI with each pitch class (ties -> below)
        diff = (pcs - orig_pc[i] + 6) % 12 - 6
        cand_midi = midi[i] + diff
        for c in range(12):
            h = H_inst_cand[i, c]
            if h < CTX["candidate_below"]:
                m = cand_midi[c]
                d_prev = m - midi[i - 1] if has_prev else None
                d_next = midi[i + 1] - m if has_next else None
                prev_ok = has_prev and sung_H[i - 1] >= S
                next_ok = has_next and sung_H[i + 1] >= S
                step = lambda d: d is not None and abs(d) in (1, 2)  # noqa: E731
                tag = ""
                if dur <= CTX["short_embellish_s"] and prev_ok and next_ok and step(d_prev) and step(d_next) \
                        and np.sign(d_prev) == np.sign(d_next):
                    tag = "PASSING"
                elif dur <= CTX["short_embellish_s"] and prev_ok and next_ok and step(d_prev) \
                        and (midi[i + 1] % 12) == (midi[i - 1] % 12):
                    tag = "NEIGHBOR"
                elif dur <= CTX["resolution_max_s"] and next_ok and step(d_next) and d_next < 0 \
                        and H_before_cand[i, c] >= S:
                    tag = "SUSPENSION"
                elif dur <= CTX["resolution_max_s"] and next_ok and step(d_next):
                    tag = "STEP_RESOLUTION"
                elif dur <= CTX["anticipation_max_s"] and H_after_cand[i, c] >= S:
                    tag = "ANTICIPATION"
                if tag:
                    H_ctx[i, c] = max(h, CTX["lift_to"])
                    code[i, c] = tag
            if crosses[i]:
                best_half = max(H_half1_cand[i, c], H_half2_cand[i, c])
                if best_half > H_ctx[i, c]:
                    H_ctx[i, c] = best_half
                    code[i, c] = (code[i, c] + "+" if code[i, c] else "") + "CROSSES_CHANGE"
    return H_ctx, code


# ---------------------------------------------------------------- one entry point for synthetic and real songs

HOP_S = 512.0 / 22050.0


def production_windows(t: np.ndarray, start: np.ndarray, end: np.ndarray):
    """Exactly backend/harmony.py:score_key_compliance's note-span indexing."""
    nf = t.size
    i0 = np.searchsorted(t, start, side="left")
    i1 = np.searchsorted(t, end, side="right")
    i0 = np.minimum(np.maximum(i0, 0), max(nf - 1, 0))
    i1 = np.maximum(i1, i0 + 1)
    return i0, np.minimum(i1, nf) - i0


def _span(t: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Frames with a <= time < b, at least one frame (nearest to a)."""
    nf = t.size
    i0 = np.clip(np.searchsorted(t, a, side="left"), 0, nf - 1)
    i1 = np.clip(np.searchsorted(t, b, side="left"), 0, nf)
    return i0, np.maximum(np.minimum(i1, nf) - i0, 1)   # i0 <= nf-1, so i0 + width <= nf


@dataclass
class SongAnalysis:
    span: NoteHarmony
    H_before: np.ndarray
    H_after: np.ndarray
    H_half1: np.ndarray
    H_half2: np.ndarray
    crosses: np.ndarray
    H_ctx: Optional[np.ndarray]
    code: Optional[np.ndarray]
    frame_top: np.ndarray            # per-frame argmax chord (K = N)
    key_post: np.ndarray
    edges: Optional[np.ndarray]


def analyze(cens: np.ndarray, t: np.ndarray, beats: np.ndarray, tempo: float, global_profile: np.ndarray,
            start: np.ndarray, end: np.ndarray, midi: np.ndarray, vocab: Vocabulary, cfg: EstimatorConfig,
            tau_key: float, tiers: Optional[np.ndarray] = None, key_tiers: Optional[np.ndarray] = None,
            with_context: bool = True) -> SongAnalysis:
    tiers = vocab.tier_values if tiers is None else tiers
    key_tiers = key_tier_matrix() if key_tiers is None else key_tiers
    P, edges = frame_posteriors(cens, t, beats, tempo, vocab, cfg, HOP_S)
    kp = key_posterior(global_profile, tau_key)
    Hk = kp @ key_tiers
    i0, w = production_windows(t, start, end)
    span = compatibility(window_mean(P, i0, w), tiers, Hk)
    win = {}
    mid = 0.5 * (start + end)
    cw = CTX["context_window_s"]
    for name, (a, b) in {"before": (start - cw, start), "after": (end, end + cw), "half1": (start, mid), "half2": (mid, end)}.items():
        j0, jw = _span(t, a, b)
        win[name] = compatibility(window_mean(P, j0, jw), tiers, Hk)
    K = vocab.K
    crosses = (win["half1"].top[:, 0] != win["half2"].top[:, 0]) & (win["half1"].e >= 0.5) & (win["half2"].e >= 0.5)
    H_ctx = code = None
    if with_context:
        H_ctx, code = context_adjust(start, end, np.asarray(midi, dtype=int), span.H_inst, win["before"].H_inst,
                                     win["after"].H_inst, win["half1"].H_inst, win["half2"].H_inst, crosses)
    return SongAnalysis(span, win["before"].H_inst, win["after"].H_inst, win["half1"].H_inst, win["half2"].H_inst,
                        crosses, H_ctx, code, np.argmax(P, axis=1), kp, edges)
