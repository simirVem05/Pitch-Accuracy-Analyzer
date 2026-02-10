import React from "react";

export default function ReportPanel({ report }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="text-sm font-medium text-white/90 mb-3">Vocal Coach Report</div>
      <div className="text-sm leading-relaxed text-white/80 whitespace-pre-wrap">
        {report || "No report returned."}
      </div>
    </div>
  );
}
