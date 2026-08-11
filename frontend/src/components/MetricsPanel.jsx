import React from "react";
import { clamp01, toPct } from "../lib/scoring";

function Card({ label, value, sub, emphasis }) {
  return (
    <div
      className={
        emphasis
          ? "rounded-2xl border border-white/20 bg-black/40 p-4"
          : "rounded-2xl border border-white/10 bg-black/30 p-4"
      }
    >
      <div className="text-xs uppercase tracking-widest text-white/50">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      {sub ? <div className="mt-1 text-xs text-white/45">{sub}</div> : null}
    </div>
  );
}

export default function MetricsPanel({ metrics }) {
  const hasKey = metrics?.key_compliance !== null && metrics?.key_compliance !== undefined;
  const keyCompliance = clamp01(metrics?.key_compliance ?? 0);
  const intonation = clamp01(metrics?.intonation_accuracy ?? 0);
  const coverage = clamp01(metrics?.voiced_coverage ?? 0);

  const total = Number(metrics?.total_notes_analyzed ?? 0) || 0;
  const scoredForKey = Number(metrics?.notes_scored_for_key ?? 0) || 0;
  const deviation = Number(metrics?.median_cents_deviation ?? 0);
  const tuning = Number(metrics?.tuning_offset_cents ?? 0);
  const tempo = Number(metrics?.tempo_bpm ?? 0);

  const lowCoverage = coverage < 0.7;

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm font-medium text-white/90">Results</div>
        <div className="text-xs text-white/50">{total} notes</div>
      </div>

      <div className="grid grid-cols-1 gap-3">
        <Card
          emphasis
          label="Key compliance"
          value={hasKey ? toPct(keyCompliance, 1) : "—"}
          sub={
            hasKey
              ? `Do the chosen notes fit the song's harmony · ${scoredForKey} notes scored`
              : "No instrumental found — upload the full song to measure note choice"
          }
        />
        <Card
          emphasis
          label="Intonation accuracy"
          value={toPct(intonation, 1)}
          sub={`Were those notes sung cleanly · median ${deviation.toFixed(1)}c off target`}
        />

        <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
          <div className="text-xs uppercase tracking-widest text-white/50 mb-2">
            Analysis detail
          </div>
          <dl className="space-y-1.5 text-xs">
            <div className="flex justify-between">
              <dt className="text-white/50">Vocal analyzed</dt>
              <dd className={lowCoverage ? "text-amber-300/90" : "text-white/75"}>
                {toPct(coverage, 0)}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-white/50">Tuning offset</dt>
              <dd className="text-white/75">
                {tuning >= 0 ? "+" : ""}
                {tuning.toFixed(1)}c
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-white/50">Tempo</dt>
              <dd className="text-white/75">{tempo > 0 ? `${tempo.toFixed(0)} BPM` : "—"}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="mt-3 text-xs leading-relaxed text-white/40">
        The two scores are independent and deliberately not combined — expressive
        note choices can score low on key compliance while being sung perfectly.
        {lowCoverage
          ? " Coverage is low, so parts of this vocal could not be confidently analyzed."
          : ""}
      </div>
    </div>
  );
}
