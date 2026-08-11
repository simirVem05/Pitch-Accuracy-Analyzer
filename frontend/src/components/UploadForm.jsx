import React, { useState } from "react";

export default function UploadForm({ onSubmit, disabled }) {
  const [file, setFile] = useState(null);

  const canSubmit = !!file && !disabled;

  function handleSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit({ file });
  }

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur p-6 md:p-8 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
        <label className="block text-xs uppercase tracking-widest text-white/60 mb-2">
          Upload Song
        </label>
        <input
          type="file"
          accept="audio/*,.mp3,.wav,.flac,.m4a"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white file:mr-4 file:rounded-lg file:border-0 file:bg-white file:px-3 file:py-2 file:text-black hover:border-white/20"
        />

        {file?.name ? (
          <div className="mt-2 text-xs text-white/60 truncate">
            Selected: {file.name}
          </div>
        ) : (
          <div className="mt-2 text-xs text-white/40">
            Upload the full song — instrumental and vocals together. Harmony is
            read from the instrumental, so no key or genre is needed.
          </div>
        )}

        <button
          type="submit"
          disabled={!canSubmit}
          className="mt-6 w-full rounded-2xl bg-white text-black font-medium py-3.5 hover:bg-white/90 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          Analyze
        </button>

        <div className="mt-3 text-xs text-white/40 text-center">
          The song is separated into stems first, which takes a few minutes.
        </div>
      </div>
    </form>
  );
}
