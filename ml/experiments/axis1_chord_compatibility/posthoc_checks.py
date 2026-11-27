"""
Post-hoc checks for axis1_chord_compatibility — written AFTER seeing results.json.

Not pre-registered. Real-song data is used only to *describe* behavior; every
parameter variant here is still fitted on synthetic audio (degraded set) or is a
structural alternative, never tuned to real songs.

  1. case 8 (octave invariance) re-run correctly: the original harness placed three
     octave notes at the same time, so the context rules treated them as neighbours
  2. why 29% of real notes have low evidence: real vs synthetic template similarity
  3. theta/tau refitted on a *degraded synthetic* calibration set, then applied to real songs
  4. low-evidence fallback variants: key (pre-registered), chord-only, neutral 0.5, abstain
  5. context restricted to the rules that proved specific (PASSING, NEIGHBOR)
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import axis1_chord_compatibility as X  # noqa: E402
import chord_model as M  # noqa: E402
import synthetic_audio as SA  # noqa: E402


def main() -> int:
    R = json.loads((HERE / "results.json").read_text())
    sc = R["selected_config"]
    cfg = M.EstimatorConfig(template=sc["template"], sim=sc["sim"], tau=sc["tau"], theta=sc["theta"], temporal=sc["temporal"])
    tau_key = sc["tau_key"]
    vocab = M.build_vocabulary()
    out = {"note": "post hoc; not pre-registered; no parameter fitted on real songs"}

    # 1. octave invariance, done correctly (one note per analysis)
    res = {}
    for layer in ("T", "S"):
        if layer == "T":
            nf = int(np.ceil(4.0 / M.HOP_S)) + 1
            t = SA.frame_times(nf)
            v = M.harmonic_chroma([0, 4, 7])
            C = np.repeat((v / np.linalg.norm(v))[:, None], nf, axis=1)
        else:
            rng = np.random.default_rng(8)
            C = SA.production_cens(SA.add_noise(SA.chord_audio(SA.voicing([0, 4, 7], rng), 4.0, "soft", rng), 40.0, rng))
            t = SA.frame_times(C.shape[1])
        beats = np.arange(0, 4.0 + 1e-9, 0.6)
        vecs = []
        for m in (52, 64, 76):
            A = M.analyze(C, t, beats, 100.0, C.mean(axis=1), np.array([1.5]), np.array([2.5]), np.array([m]), vocab, cfg, tau_key)
            vecs.append((A.span.H_inst[0], A.H_ctx[0]))
        res[layer] = bool(all(np.array_equal(vecs[0][0], v[0]) and np.array_equal(vecs[0][1], v[1]) for v in vecs[1:]))
    out["case8_octave_invariance_fixed_harness"] = res

    # 2. similarity distributions: real beat segments vs synthetic frames
    T = vocab.binary if cfg.template == "binary" else vocab.harmonic
    real_max, real_top1 = [], []
    songs = sorted(json.loads((X.REPO / "data" / "dataset_selections.json").read_text())["songs"])
    for s in songs:
        h, _ = X.load_song(s)
        cens = h["other_chroma_cens"].astype(float)
        t = h["chroma_frame_times_s"].astype(float)
        edges = M.beat_segments(t, h["beat_times_s"].astype(float))
        seg = M.segment_mean_chroma(cens, t, edges)
        sim = M.similarity(seg, T, cfg.sim)
        real_max.append(sim.max(axis=1))
        P = M.posterior(sim, cfg.tau, cfg.theta)[:, :vocab.K]
        real_top1.append((P.max(axis=1) / np.clip(P.sum(axis=1), 1e-300, None)))
    real_max = np.concatenate(real_max)
    XB, yB = X.synth_frames(vocab, X.SEED_B, degraded=False)
    XD, yD = X.synth_frames(vocab, X.SEED_B + 1, degraded=True)
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)}  # noqa: E731
    sB = M.similarity(XB[yB < vocab.K], T, cfg.sim).max(axis=1)
    sD = M.similarity(XD[yD < vocab.K], T, cfg.sim).max(axis=1)
    out["max_template_similarity"] = {"real_beat_segments": {**q(real_max), "share_below_theta": float(np.mean(real_max < cfg.theta))},
                                      "synthetic_clean_chord_frames": {**q(sB), "share_below_theta": float(np.mean(sB < cfg.theta))},
                                      "synthetic_degraded_chord_frames": {**q(sD), "share_below_theta": float(np.mean(sD < cfg.theta))},
                                      "theta": cfg.theta}
    out["real_top1_chord_mass_beat_segments"] = q(np.concatenate(real_top1))

    # 3. tau/theta refitted on a degraded synthetic calibration set (seed A, degraded)
    XA2, yA2 = X.synth_frames(vocab, X.SEED_A, degraded=True)
    sA2 = M.similarity(XA2, T, cfg.sim)
    best = None
    for tau in X.TAU_GRID:
        for theta in X.THETA_GRID:
            P = M.posterior(sA2, tau, theta)
            ll = float(-np.mean(np.log(np.clip(P[np.arange(len(yA2)), yA2], 1e-300, None))))
            if best is None or ll < best[0]:
                best = (ll, tau, theta)
    cfg_d = replace(cfg, tau=best[1], theta=best[2])
    out["degraded_calibration"] = {"tau": best[1], "theta": best[2], "logloss": best[0]}

    # 4/5. real-song variants
    variants = {}
    lift = {"sung": 0, "alt": 0, "n": 0}
    for s in songs:
        h, n = X.load_song(s)
        cens = h["other_chroma_cens"].astype(float)
        t = h["chroma_frame_times_s"].astype(float)
        beats = h["beat_times_s"].astype(float)
        tempo = float(h["estimated_tempo_bpm"])
        g = h["global_chroma_profile"].astype(float)
        st, en = n["start_time_s"].astype(float), n["end_time_s"].astype(float)
        midi = n["target_midi_note"].astype(int)
        pc = n["pitch_class"].astype(int)
        for name, c in (("pre_registered", cfg), ("degraded_calibration", cfg_d)):
            A = M.analyze(cens, t, beats, tempo, g, st, en, midi, vocab, c, tau_key, with_context=(name == "pre_registered"))
            e = A.span.e[:, None]
            V = {f"{name}/key_fallback": A.span.H_inst, f"{name}/chord_only": A.span.H_chord,
                 f"{name}/neutral_fallback": e * A.span.H_chord + (1 - e) * 0.5}
            for k, v in V.items():
                variants.setdefault(k, {"V": [], "pc": [], "e": []})
                variants[k]["V"].append(v)
                variants[k]["pc"].append(pc)
                variants[k]["e"].append(A.span.e)
            if name == "pre_registered":
                keep = np.vectorize(lambda cd: cd.split("+")[0] in ("PASSING", "NEIGHBOR"))(A.code)
                Hr = np.where(keep, np.maximum(A.span.H_inst, M.CTX["lift_to"]), A.span.H_inst)
                variants.setdefault("pre_registered/ctx_passing_neighbor_only", {"V": [], "pc": [], "e": []})
                variants["pre_registered/ctx_passing_neighbor_only"]["V"].append(Hr)
                variants["pre_registered/ctx_passing_neighbor_only"]["pc"].append(pc)
                variants["pre_registered/ctx_passing_neighbor_only"]["e"].append(A.span.e)
                lifted = Hr > A.span.H_inst + 1e-12
                idx = np.arange(pc.size)
                lift["sung"] += int(lifted[idx, pc].sum())
                m = np.ones_like(lifted)
                m[idx, pc] = False
                lift["alt"] += int(lifted[m].sum())
                lift["n"] += int(pc.size)
    vo = {}
    for k, d in variants.items():
        V, pc, e = np.vstack(d["V"]), np.concatenate(d["pc"]), np.concatenate(d["e"])
        met = X.vec_metrics(V, pc)
        met["low_evidence_share"] = float(np.mean(e < 0.5))
        keep = e >= 0.5
        met["abstain_if_low_evidence"] = {"coverage": float(keep.mean()), **{kk: vv for kk, vv in X.vec_metrics(V[keep], pc[keep]).items()
                                                                              if kk in ("mean_H_sung", "mean_H_alternatives", "breadth_ge_0.75", "breadth_ge_0.5", "win_rate_sung_vs_alt")}}
        vo[k] = met
    out["real_variants"] = vo
    out["restricted_context_lift"] = {"sung_lift_rate": lift["sung"] / lift["n"], "alt_lift_rate": lift["alt"] / (11 * lift["n"])}
    # 6. stability of chord-only vs evidence-gated H (same noise model and seed as the main run's spirit)
    rng = np.random.default_rng(20261127)
    acc = {"chord_only": {"noise": [], "shift": []}, "key_fallback": {"noise": [], "shift": []}}
    for s in songs:
        h, n = X.load_song(s)
        c = h["other_chroma_cens"].astype(float)
        t, b, tp, g = h["chroma_frame_times_s"], h["beat_times_s"], float(h["estimated_tempo_bpm"]), h["global_chroma_profile"]
        st, en, mi = n["start_time_s"], n["end_time_s"], n["target_midi_note"].astype(int)
        A = M.analyze(c, t, b, tp, g, st, en, mi, vocab, cfg, tau_key, with_context=False)
        no = c * np.exp(rng.normal(0, X.NOISE_SIGMA, c.shape))
        no /= np.linalg.norm(no, axis=0, keepdims=True)
        An = M.analyze(no, t, b, tp, no.mean(1), st, en, mi, vocab, cfg, tau_key, with_context=False)
        acc["chord_only"]["noise"].append(np.abs(A.span.H_chord - An.span.H_chord))
        acc["key_fallback"]["noise"].append(np.abs(A.span.H_inst - An.span.H_inst))
        for sh in (-X.SHIFT_S, X.SHIFT_S):
            As = M.analyze(c, t, b, tp, g, st + sh, en + sh, mi, vocab, cfg, tau_key, with_context=False)
            acc["chord_only"]["shift"].append(np.abs(A.span.H_chord - As.span.H_chord))
            acc["key_fallback"]["shift"].append(np.abs(A.span.H_inst - As.span.H_inst))
    out["stability"] = {k: {kind: {"mean": float(np.mean(np.vstack(v))), "p95": float(np.percentile(np.vstack(v), 95))} for kind, v in d.items()}
                        for k, d in acc.items()}
    (HERE / "posthoc_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "real_variants"}, indent=1))
    for k, v in vo.items():
        print(f"{k:50} lowE={v['low_evidence_share']:.3f} pcs>=.75={v['breadth_ge_0.75']:.2f} pcs>=.5={v['breadth_ge_0.5']:.2f} "
              f">=10={v['share_notes_ge10_pcs_ge_0.5']:.3f} graded={v['graded_share_in_0.25_0.75']:.3f} sung={v['mean_H_sung']:.3f} alt={v['mean_H_alternatives']:.3f} "
              f"win={v['win_rate_sung_vs_alt']:.3f} | abstain cov={v['abstain_if_low_evidence']['coverage']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
