export default function ReportPanel({ summary, geminiPrompt }) {
  if (!summary) return null;

  return (
    <div style={styles.card}>
      <div style={styles.title}>Performance Report</div>

      <div style={styles.kpis}>
        <KPI label="Overall score" value={`${summary.overall_score?.toFixed?.(1)}%`} />
        <KPI label="Median score" value={`${summary.median_score?.toFixed?.(1)}%`} />
        <KPI label="Median error" value={`${summary.median_abs_dev_cents?.toFixed?.(1)} cents`} />
        <KPI label="95th % error" value={`${summary.p95_abs_dev_cents?.toFixed?.(1)} cents`} />
        <KPI label="Vibrato detected" value={`${summary.vibrato_pct?.toFixed?.(1)}%`} />
        <KPI label="Voiced frames" value={`${summary.voiced_pct?.toFixed?.(1)}%`} />
      </div>

      <div style={{ marginTop: 12, fontSize: 13, color: "#374151", fontWeight: 700 }}>
        Gemini prompt (placeholder until API response is wired)
      </div>
      <textarea style={styles.textarea} value={geminiPrompt || ""} readOnly />
    </div>
  );
}

function KPI({ label, value }) {
  return (
    <div style={styles.kpi}>
      <div style={styles.kpiLabel}>{label}</div>
      <div style={styles.kpiValue}>{value}</div>
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
  title: { fontSize: 16, fontWeight: 800 },
  kpis: {
    display: "grid",
    gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
    gap: 10,
    marginTop: 12,
  },
  kpi: { border: "1px solid #f1f5f9", borderRadius: 14, padding: 12, background: "#fafafa" },
  kpiLabel: { fontSize: 12, color: "#6b7280", fontWeight: 700 },
  kpiValue: { fontSize: 15, color: "#111827", fontWeight: 900, marginTop: 4 },
  textarea: {
    marginTop: 8,
    width: "100%",
    minHeight: 180,
    borderRadius: 12,
    border: "1px solid #e5e7eb",
    padding: 12,
    fontSize: 12,
    lineHeight: 1.4,
  },
};
