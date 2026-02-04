import { useState } from "react";

const GENRES = [
  { value: "pop", label: "Pop" },
  { value: "rock", label: "Rock" },
  { value: "blues", label: "Blues" },
  { value: "jazz", label: "Jazz" },
  { value: "rnb_soul", label: "R&B / Soul" },
  { value: "hiphop_rap", label: "Hip-Hop / Rap" },
  { value: "classical", label: "Classical" },
];

export default function UploadForm({ onSubmit, isLoading }) {
  const [file, setFile] = useState(null);
  const [keySig, setKeySig] = useState("B minor");
  const [genre, setGenre] = useState("rnb_soul");

  // Speed settings (your current decision)
  const [modelCapacity, setModelCapacity] = useState("tiny");
  const [stepSizeMs, setStepSizeMs] = useState(20);

  // Quality knobs
  const [strictness, setStrictness] = useState(0.6);
  const [confidence, setConfidence] = useState(0.25);
  const [viterbi, setViterbi] = useState(true);
  const [a4, setA4] = useState(440);

  function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;

    onSubmit({
      file,
      key: keySig,
      genre,
      strictness,
      confidence_threshold: confidence,
      step_size_ms: stepSizeMs,
      model_capacity: modelCapacity,
      viterbi,
      a4_hz: a4,
    });
  }

  return (
    <form onSubmit={handleSubmit} style={styles.card}>
      <div style={styles.headerRow}>
        <div>
          <div style={styles.title}>Pitch Accuracy Analyzer</div>
          <div style={styles.subtitle}>Upload vocals → get an on-key score over time.</div>
        </div>
      </div>

      <div style={styles.grid}>
        <label style={styles.label}>
          Vocal file (mp3/wav)
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            disabled={isLoading}
            style={styles.input}
          />
        </label>

        <label style={styles.label}>
          Key
          <input
            value={keySig}
            onChange={(e) => setKeySig(e.target.value)}
            disabled={isLoading}
            placeholder='e.g., "B minor", "C major"'
            style={styles.input}
          />
        </label>

        <label style={styles.label}>
          Genre
          <select value={genre} onChange={(e) => setGenre(e.target.value)} disabled={isLoading} style={styles.input}>
            {GENRES.map((g) => (
              <option key={g.value} value={g.value}>
                {g.label}
              </option>
            ))}
          </select>
        </label>

        <label style={styles.label}>
          Strictness: {strictness.toFixed(2)}
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={strictness}
            onChange={(e) => setStrictness(parseFloat(e.target.value))}
            disabled={isLoading}
            style={styles.range}
          />
          <div style={styles.helpText}>
            Low = more style freedom (blue/passing tones). High = tighter key enforcement.
          </div>
        </label>

        <label style={styles.label}>
          Confidence threshold: {confidence.toFixed(2)}
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            disabled={isLoading}
            style={styles.range}
          />
          <div style={styles.helpText}>
            Lower = more frames counted (riskier). Higher = fewer frames but more reliable.
          </div>
        </label>

        <label style={styles.label}>
          Model capacity
          <select
            value={modelCapacity}
            onChange={(e) => setModelCapacity(e.target.value)}
            disabled={isLoading}
            style={styles.input}
          >
            {["tiny", "small", "medium", "large", "full"].map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        <label style={styles.label}>
          Step size (ms)
          <select
            value={stepSizeMs}
            onChange={(e) => setStepSizeMs(parseInt(e.target.value, 10))}
            disabled={isLoading}
            style={styles.input}
          >
            {[10, 20, 50].map((n) => (
              <option key={n} value={n}>
                {n} ms
              </option>
            ))}
          </select>
        </label>

        <label style={styles.label}>
          A4 tuning (Hz)
          <input
            type="number"
            value={a4}
            onChange={(e) => setA4(parseFloat(e.target.value))}
            disabled={isLoading}
            style={styles.input}
          />
        </label>

        <label style={{ ...styles.label, display: "flex", gap: 10, alignItems: "center" }}>
          <input type="checkbox" checked={viterbi} onChange={(e) => setViterbi(e.target.checked)} disabled={isLoading} />
          Viterbi smoothing
        </label>
      </div>

      <button type="submit" disabled={!file || isLoading} style={styles.button}>
        {isLoading ? "Analyzing..." : "Analyze"}
      </button>
    </form>
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
  headerRow: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  title: { fontSize: 22, fontWeight: 700 },
  subtitle: { fontSize: 13, color: "#4b5563", marginTop: 4 },
  grid: { display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 },
  label: { display: "flex", flexDirection: "column", gap: 6, fontSize: 13, fontWeight: 600 },
  input: { padding: 10, borderRadius: 10, border: "1px solid #e5e7eb", fontSize: 14 },
  range: { width: "100%" },
  helpText: { fontSize: 12, fontWeight: 400, color: "#6b7280" },
  button: {
    marginTop: 14,
    width: "100%",
    padding: 12,
    borderRadius: 12,
    border: "none",
    fontWeight: 700,
    cursor: "pointer",
  },
};