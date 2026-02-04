import { useState } from "react";
import UploadForm from "./components/UploadForm";
import ScoreChart from "./components/ScoreChart";
import ReportPanel from "./components/ReportPanel";
import { analyzeVocals } from "./api/analyze";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [graph, setGraph] = useState(null);
  const [summary, setSummary] = useState(null);
  const [geminiPrompt, setGeminiPrompt] = useState("");
  const [error, setError] = useState("");

  async function handleAnalyze(payload) {
    setError("");
    setLoading(true);
    setGraph(null);
    setSummary(null);
    setGeminiPrompt("");

    try {
      const data = await analyzeVocals(payload);
      setGraph(data.graph);
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
          <ScoreChart graph={graph} />
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