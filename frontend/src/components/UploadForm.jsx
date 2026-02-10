import React, { useMemo, useState } from "react";

const TONICS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const SCALES = ["major", "minor"];
const GENRES = [
  { label: "Hip Hop", value: "hip hop" },
  { label: "Pop", value: "pop" },
  { label: "Alternative Pop", value: "alternative pop" },
  { label: "R&B", value: "rnb" }, // backend normalizes rnb -> r&b
  { label: "Rock", value: "rock" },
  { label: "Country", value: "country" },
  { label: "Jazz", value: "jazz" },
];

export default function UploadForm({ onSubmit, disabled }) {
  const [file, setFile] = useState(null);
  const [tonic, setTonic] = useState("B");
  const [scale, setScale] = useState("minor");
  const [genre, setGenre] = useState("rnb");

  const keyString = useMemo(() => `${tonic} ${scale}`, [tonic, scale]);

  const canSubmit = !!file && !!tonic && !!scale && !!genre && !disabled;

  function handleSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit({ file, key: keyString, genre });
  }

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-6 md:p-8 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
          {/* File */}
          <div className="md:col-span-3">
            <label className="block text-xs uppercase tracking-widest text-white/60 mb-2">
              Upload Acapella
            </label>
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white file:mr-4 file:rounded-lg file:border-0 file:bg-white file:px-3 file:py-2 file:text-black hover:border-white/20"
            />
            {file?.name ? (
              <div className="mt-2 text-xs text-white/60 truncate">Selected: {file.name}</div>
            ) : (
              <div className="mt-2 text-xs text-white/40">Upload a vocal-only file (no instrumental).</div>
            )}
          </div>

          {/* Key */}
          <div>
            <label className="block text-xs uppercase tracking-widest text-white/60 mb-2">
              Tonic
            </label>
            <select
              value={tonic}
              onChange={(e) => setTonic(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white outline-none focus:border-white/30"
            >
              {TONICS.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs uppercase tracking-widest text-white/60 mb-2">
              Scale
            </label>
            <select
              value={scale}
              onChange={(e) => setScale(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white outline-none focus:border-white/30"
            >
              {SCALES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Genre */}
          <div>
            <label className="block text-xs uppercase tracking-widest text-white/60 mb-2">
              Genre
            </label>
            <select
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white outline-none focus:border-white/30"
            >
              {GENRES.map((g) => (
                <option key={g.value} value={g.value}>{g.label}</option>
              ))}
            </select>
          </div>

          {/* Key preview */}
          <div className="md:col-span-3 -mt-1 text-xs text-white/50">
            Key sent to backend: <span className="text-white/70">{keyString}</span>
          </div>

          {/* Button */}
          <div className="md:col-span-3 pt-2">
            <button
              type="submit"
              disabled={!canSubmit}
              className="w-full rounded-2xl bg-white text-black font-medium py-3.5 hover:bg-white/90 disabled:opacity-40 disabled:cursor-not-allowed transition"
            >
              Analyze
            </button>
            <div className="mt-3 text-xs text-white/40 text-center">
              Tip: If results look weird, double-check you selected the correct key.
            </div>
          </div>
        </div>
      </div>
    </form>
  );
}
