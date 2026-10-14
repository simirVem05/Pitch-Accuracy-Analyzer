#!/usr/bin/env python3
"""
Follow-up to the parameter sweep: the hybrid's short-gap rule with a local
"before" reference (SegmentationV2Config.gap_before = "local") versus the
center-window reference the sweep used. Same harness, same metrics, same v1
mapping; only the configuration list and output directory differ, so the
completed sweep in parameter_comparison.csv is not touched.

Writes followup/parameter_comparison_gap_reference.csv and
followup/_verification.json.
"""

from __future__ import annotations

import evaluate_segmentation_v2 as E

_SWEEP_CONFIGS = E.build_configs


def configs():
    base = [c for c in _SWEEP_CONFIGS() if c["name"] in ("v1", "v1+hardgap0.5")]
    out = list(base)
    for g in (0.25, 0.35, 0.50):
        for sp in (70, 100, 120):
            for ref in ("center_window", "local"):
                out.append(dict(name=f"gapv1{'local' if ref == 'local' else ''}_g{g}_s{sp}", kind="v2",
                                mode="gap_v1", hard_gap_s=g, separation_cents=sp, persistence_frames=4,
                                gap_before=ref))
    # sensitivity of the local variant to the gap-context length (both sides)
    for ctx in (4, 6):
        out.append(dict(name=f"gapv1local_g0.5_s100_ctx{ctx}", kind="v2", mode="gap_v1", hard_gap_s=0.5,
                        separation_cents=100, persistence_frames=4, gap_before="local",
                        gap_context_frames=ctx))
    return out


if __name__ == "__main__":
    E.build_configs = configs
    E.OUT = E.OUT / "followup"
    E.OUT.mkdir(exist_ok=True)
    import os
    E.main()
    os.replace(E.OUT / "parameter_comparison.csv", E.OUT / "parameter_comparison_gap_reference.csv")
