import { useState } from "react";
import UploadForm from "./components/UploadForm";
import ScoreChart from "./components/ScoreChart";
import ReportPanel from "./components/ReportPanel";
import { analyzeVocals } from "./api/analyze";

function toNumOrNull(v) {
  if (v === null || v === undefined) return null;
  if (v === "null" || v === "None" || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function normalizeGraph(rawGraph) {
  const t = Array.isArray(rawGraph?.time_s) ? rawGraph.time_s.map(toNumOrNull) : [];
  const s = Array.isArray(rawGraph?.score_pct) ? rawGraph.score_pct.map(toNumOrNull) : [];
  const sr = Array.isArray(rawGraph?.score_raw_pct) ? rawGraph.score_raw_pct.map(toNumOrNull) : [];
  const dev = Array.isArray(rawGraph?.deviation_cents) ? rawGraph.deviation_cents.map(toNumOrNull) : [];
  const conf = Array.isArray(rawGraph?.confidence) ? rawGraph.confidence.map(toNumOrNull) : [];
  const vib = Array.isArray(rawGraph?.vibrato_mask) ? rawGraph.vibrato_mask : [];

  const n = Math.min(t.length, s.length, sr.length);
  const rows = [];
  for (let i = 0; i < n; i++) {
    if (t[i] === null) continue;
    rows.push({
      t: t[i],
      score: s[i],
      scoreRaw: sr[i],
      dev: dev[i] ?? null,
      conf: conf[i] ?? null,
      vib: vib[i] === 1,
    });
  }
  return rows;
}

export default function App() {
  const [loading, setLoading] = useState(false);
  const [graphRows, setGraphRows] = useState([]);   // <- store rows, not raw dict
  const [summary, setSummary] = useState(null);
  const [geminiPrompt, setGeminiPrompt] = useState("");
  const [error, setError] = useState("");

  async function handleAnalyze(payload) {
    setError("");
    setLoading(true);
    setGraphRows([]);
    setSummary(null);
    setGeminiPrompt("");

    try {
      const data = await analyzeVocals(payload);

      console.log("Graph Keys:", Object.keys(data.graph || {}));
      console.log("time_s[0..5]:", data.graph?.time_s?.slice?.(0, 5));
      console.log("score_pct[0..5]:", data.graph?.score_pct?.slice?.(0, 5));

      setGraphRows(normalizeGraph(data.graph));
      setSummary(data.summary);
      setGeminiPrompt(data.gemini_prompt);
    } catch (e) {
      setError(e.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <UploadForm onSubmit={handleAnalyze} isLoading={loading} />

        {error && <div style={styles.error}>{error}</div>}

        <div style={styles.grid}>
          <ScoreChart data={graphRows} />
          <ReportPanel summary={summary} geminiPrompt={geminiPrompt} />
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: { minHeight: "100vh", background: "#f6f7fb", padding: 18 },
  container: { maxWidth: 1100, margin: "0 auto", display: "flex", flexDirection: "column", gap: 14 },
  grid: { display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 14, alignItems: "start" },
  error: {
    border: "1px solid #fecaca",
    background: "#fff1f2",
    color: "#991b1b",
    padding: 12,
    borderRadius: 12,
    fontWeight: 700,
  },
};
