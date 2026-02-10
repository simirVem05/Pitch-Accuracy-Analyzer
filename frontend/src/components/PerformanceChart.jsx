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
  const x = Number(v) || 0;
  return `${x.toFixed(1)}%`;
}

export default function PerformanceChart({ graphTuples }) {
  const data = useMemo(() => {
    // tuples: [timeSeconds, scoreFraction]
    return (graphTuples || []).map(([t, s]) => ({
      t: Number(t) || 0,
      score: (Number(s) || 0) * 100,
    }));
  }, [graphTuples]);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-medium text-white/90">Performance Over Time</div>
        <div className="text-xs text-white/50">Score (0–100%)</div>
      </div>

      <div className="h-[320px] md:h-[420px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeOpacity={0.12} vertical={false} />
            <XAxis
              dataKey="t"
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

            {/* Threshold lines */}
            <ReferenceLine y={60} stroke="rgba(255,255,255,0.25)" strokeDasharray="4 4" />
            <ReferenceLine y={80} stroke="rgba(255,255,255,0.25)" strokeDasharray="4 4" />

            <Tooltip
              contentStyle={{
                background: "rgba(0,0,0,0.85)",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 12,
                boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
              }}
              labelStyle={{ color: "rgba(255,255,255,0.7)" }}
              formatter={(value) => [fmtPct(value), "On-key"]}
              labelFormatter={(label) => `Time: ${fmtTime(label)}`}
            />

            <Line
              type="monotone"
              dataKey="score"
              stroke="rgba(255,255,255,0.9)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={true}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex items-center justify-between text-xs text-white/50">
        <div>Thresholds: 80%+ high, 60–80% mediocre, &lt;60% low</div>
        <div>{data.length} points</div>
      </div>
    </div>
  );
}
