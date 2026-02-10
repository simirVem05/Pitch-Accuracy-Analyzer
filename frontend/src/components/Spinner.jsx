import React from "react";

export default function Spinner({ label = "Analyzing" }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-12">
      <div className="h-10 w-10 animate-spin rounded-full border-2 border-white/20 border-t-white" />
      <div className="text-white/80 text-sm tracking-wide">{label}</div>
    </div>
  );
}
