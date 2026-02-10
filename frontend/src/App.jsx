import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import Spinner from "./components/Spinner";
import PerformanceChart from "./components/PerformanceChart";
import MetricsPanel from "./components/MetricsPanel";
import ReportPanel from "./components/ReportPanel";
import { analyzeAudio } from "./lib/api";

export default function App() {
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [error, setError] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [graphTuples, setGraphTuples] = useState([]);
  const [report, setReport] = useState("");

  async function handleAnalyze({ file, key, genre }) {
    setStatus("loading");
    setError("");
    setMetrics(null);
    setGraphTuples([]);
    setReport("");

    try {
      const data = await analyzeAudio({ file, key, genre });
      setMetrics(data.metrics);
      setGraphTuples(data.graph_tuples);
      setReport(data.report);
      setStatus("done");
    } catch (e) {
      setError(e?.message || "Something went wrong.");
      setStatus("error");
    }
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* subtle glow */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute -top-40 left-1/2 h-[520px] w-[820px] -translate-x-1/2 rounded-full bg-white/10 blur-3xl" />
        <div className="absolute top-40 left-1/2 h-[280px] w-[620px] -translate-x-1/2 rounded-full bg-white/5 blur-3xl" />
      </div>

      {/* Make content wider so the right panel sits closer to the right side */}
      <div className="mx-auto w-full max-w-[1280px] px-5 md:px-8 py-10 md:py-14">
        <header className="mb-8 md:mb-10">
          <h1 className="text-3xl md:text-5xl font-semibold tracking-tight">
            Pitch Accuracy Analyzer
          </h1>
          <p className="mt-3 text-sm md:text-base text-white/60 max-w-2xl">
            Upload your acapella, choose the key and genre, and get a pitch accuracy graph with a concise vocal coaching report.
          </p>
        </header>

        {status === "idle" && <UploadForm onSubmit={handleAnalyze} disabled={false} />}

        {status === "loading" && (
          <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-6 md:p-10">
            <Spinner label="Analyzing" />
          </div>
        )}

        {status === "error" && (
          <div className="space-y-6">
            <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-6">
              <div className="text-sm font-medium text-white/90">Something went wrong</div>
              <div className="mt-2 text-sm text-white/70">{error}</div>
              <button
                onClick={() => {
                  setStatus("idle");
                  setError("");
                }}
                className="mt-4 rounded-2xl bg-white text-black font-medium px-4 py-2.5 hover:bg-white/90 transition"
              >
                Back
              </button>
            </div>
            <UploadForm onSubmit={handleAnalyze} disabled={false} />
          </div>
        )}

        {status === "done" && (
          <div className="space-y-6">
            {/* Top row: graph left, metrics right */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              <div className="lg:col-span-8">
                <PerformanceChart graphTuples={graphTuples} />
              </div>
              <div className="lg:col-span-4">
                <MetricsPanel metrics={metrics} />
                <button
                  onClick={() => setStatus("idle")}
                  className="mt-4 w-full rounded-2xl border border-white/15 bg-black/40 text-white py-3 text-sm hover:border-white/25 hover:bg-black/55 transition"
                >
                  Analyze Another File
                </button>
              </div>
            </div>

            {/* Report underneath BOTH, centered */}
            <div className="flex justify-center">
              <div className="w-full max-w-4xl">
                <ReportPanel report={report} />
              </div>
            </div>
          </div>
        )}

        <footer className="mt-10 text-xs text-white/40">
          Backend: FastAPI / CREPE • Frontend: React + Recharts • Style: black/white minimal
        </footer>
      </div>
    </div>
  );
}
