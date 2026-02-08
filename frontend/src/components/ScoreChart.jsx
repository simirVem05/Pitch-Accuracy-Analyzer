import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceArea,
} from "recharts";
import { useMemo, useState } from "react";

function formatNum(x, digits = 2) {
  if (x === null || x === undefined) return "—";
  if (Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

export default function ScoreChart({ data }) {
  const [showRaw, setShowRaw] = useState(false);
  const [showVibrato, setShowVibrato] = useState(true);

  // Downsample for performance (in case App passes full-res rows)
  const dsData = useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) return [];
    const n = data.length;
    const MAX_POINTS = 2000;
    const stride = Math.max(1, Math.floor(n / MAX_POINTS));
    const out = [];
    for (let i = 0; i < n; i += stride) out.push(data[i]);
    return out;
  }, [data]);

  // Build vibrato regions from dsData
  const vibRanges = useMemo(() => {
    if (!dsData.length) return [];
    const ranges = [];
    let start = null;

    for (let i = 0; i < dsData.length; i++) {
      const isV = dsData[i].vib && dsData[i].score !== null;
      if (isV && start === null) start = dsData[i].t;
      if (!isV && start !== null) {
        const end = dsData[i - 1]?.t ?? dsData[i].t;
        ranges.push([start, end]);
        start = null;
      }
    }
    if (start !== null) ranges.push([start, dsData[dsData.length - 1].t]);

    return ranges.slice(0, 200);
  }, [dsData]);

  if (!dsData.length) {
    return (
      <div style={styles.card}>
        <div style={styles.title}>On-Key Score Over Time</div>
        <div style={styles.footer}>No data yet — upload a file and click Analyze.</div>
      </div>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.title}>On-Key Score Over Time</div>
        <div style={styles.toggles}>
          <label style={styles.toggle}>
            <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
            Show raw (intonation-only)
          </label>
          <label style={styles.toggle}>
            <input type="checkbox" checked={showVibrato} onChange={(e) => setShowVibrato(e.target.checked)} />
            Highlight vibrato
          </label>
        </div>
      </div>

      <div style={{ height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={dsData} margin={{ top: 10, right: 18, left: 6, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="t"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v) => `${Number(v).toFixed(1)}s`}
            />
            <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              formatter={(value, name) => {
                if (name === "score") return [`${formatNum(value, 1)}%`, "On-key"];
                if (name === "scoreRaw") return [`${formatNum(value, 1)}%`, "Raw"];
                return [value, name];
              }}
              labelFormatter={(t) => `Time: ${formatNum(t, 2)}s`}
              contentStyle={{ borderRadius: 12 }}
            />
            <Legend />

            {showVibrato &&
              vibRanges.map(([x1, x2], idx) => (
                <ReferenceArea
                  key={idx}
                  x1={x1}
                  x2={x2}
                  y1={0}
                  y2={100}
                  ifOverflow="hidden"
                  fillOpacity={0.08}
                  strokeOpacity={0}
                />
              ))}

            <Line
              type="monotone"
              dataKey="score"
              name="On-key"
              dot={false}
              connectNulls={false}
              strokeWidth={2}
            />

            {showRaw && (
              <Line
                type="monotone"
                dataKey="scoreRaw"
                name="Raw"
                dot={false}
                connectNulls={false}
                strokeWidth={1.5}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={styles.footer}>
        Tip: hover to see scores, and use strictness to tune how “theory-strict” the grade feels.
      </div>
    </div>
  );
}

const styles = {
  card: {
    border: "1px solid #e5e7eb",
    borderRadius: 16,
    padding: 16,
    background: "white",
    boxShadow: "0 8px 30px rgba(0,0,0,0.06)",
  },
  header: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 },
  title: { fontSize: 16, fontWeight: 800 },
  toggles: { display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "flex-end" },
  toggle: { display: "flex", gap: 8, alignItems: "center", fontSize: 13, color: "#374151" },
  footer: { marginTop: 10, fontSize: 12, color: "#6b7280" },
};
