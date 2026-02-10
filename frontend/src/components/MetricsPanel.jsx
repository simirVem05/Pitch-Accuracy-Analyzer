import React, { useMemo } from "react";
import { clamp01, intonationScoreFromAbsCents, toPct } from "../lib/scoring";

function Card({ label, value, sub }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/30 p-4">
      <div className="text-xs uppercase tracking-widest text-white/50">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      {sub ? <div className="mt-1 text-xs text-white/45">{sub}</div> : null}
    </div>
  );
}

export default function MetricsPanel({ metrics }) {
  const avgOnKeyPct = useMemo(() => {
    const cents = Number(metrics?.median_cents_deviation ?? 0);
    const score = clamp01(intonationScoreFromAbsCents(Math.abs(cents)));
    return `${(score * 100).toFixed(1)}%`;
  }, [metrics]);

  const high = clamp01(metrics?.high_ratio ?? 0);
  const med = clamp01(metrics?.mediocre_ratio ?? 0);
  const low = clamp01(metrics?.low_ratio ?? 0);

  const total = Number(metrics?.total_notes_analyzed ?? 0) || 0;
  const vibCount = Number(metrics?.vibrato_detected_count ?? 0) || 0;
  const portCount = Number(metrics?.portamento_detected_count ?? 0) || 0;

  const vibPrev = total > 0 ? vibCount / total : 0;
  const portPrev = total > 0 ? portCount / total : 0;

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm font-medium text-white/90">Summary</div>
        <div className="text-xs text-white/50">{total} notes analyzed</div>
      </div>

      <div className="grid grid-cols-1 gap-3">
        <Card
          label="Average on-key score"
          value={avgOnKeyPct}
          sub={`From median cents deviation (${Number(metrics?.median_cents_deviation ?? 0).toFixed(2)}c)`}
        />

        <div className="grid grid-cols-3 gap-3">
          <Card label="High" value={toPct(high, 0)} sub="≥ 80%" />
          <Card label="Mediocre" value={toPct(med, 0)} sub="60–80%" />
          <Card label="Low" value={toPct(low, 0)} sub="< 60%" />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Card label="Vibrato prevalence" value={toPct(vibPrev, 0)} sub={`${vibCount} segments`} />
          <Card label="Portamento prevalence" value={toPct(portPrev, 0)} sub={`${portCount} transitions`} />
        </div>
      </div>
    </div>
  );
}
