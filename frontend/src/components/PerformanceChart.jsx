import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from "recharts";

function fmtTime(sec) {
  const s = Number(sec) || 0;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(0).padStart(2, "0")}`;
}

function fmtPct(v) {
  if (v === null || v === undefined) return "N/A";
  return `${(Number(v) || 0).toFixed(1)}%`;
}

export default function PerformanceChart({ graphPoints }) {
  const data = useMemo(
    () =>
      (graphPoints || []).map(([t, s]) => ({
        t: Number(t) || 0,
        // Preserve null so the line breaks across silences instead of
        // interpolating through a rest.
        score: s === null ? null : (Number(s) || 0) * 100,
      })),
    [graphPoints]
  );

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-medium text-white/90">Intonation Over Time</div>
        <div className="text-xs text-white/50">Score (0–100%)</div>
      </div>

      <div className="h-[320px] md:h-[420px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeOpacity={0.12} vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={fmtTime}
              stroke="rgba(255,255,255,0.35)"
              tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }}
              axisLine={{ strokeOpacity: 0.2 }}
              tickLine={{ strokeOpacity: 0.2 }}
              minTickGap={24}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
              stroke="rgba(255,255,255,0.35)"
              tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }}
              axisLine={{ strokeOpacity: 0.2 }}
              tickLine={{ strokeOpacity: 0.2 }}
              width={42}
            />

            <ReferenceLine y={60} stroke="rgba(255,255,255,0.25)" strokeDasharray="4 4" />
            <ReferenceLine y={80} stroke="rgba(255,255,255,0.25)" strokeDasharray="4 4" />

            <Tooltip
              contentStyle={{
                background: "rgba(0,0,0,0.85)",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 12,
              }}
              labelStyle={{ color: "rgba(255,255,255,0.7)" }}
              formatter={(value) => [fmtPct(value), "Intonation"]}
              labelFormatter={(label) => `Time: ${fmtTime(label)}`}
            />

            <Line
              type="linear"
              dataKey="score"
              stroke="rgba(255,255,255,0.9)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={true}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 text-xs text-white/50">
        Each sustained note is drawn at its own width; gaps are rests. This graph
        shows intonation only — key compliance is reported separately.
      </div>
    </div>
  );
}
