const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function analyzeAudio({ file, key, genre }) {
  const form = new FormData();
  form.append("file", file);
  form.append("key", key);
  form.append("genre", genre);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.detail) msg = data.detail;
    } catch {}
    throw new Error(msg);
  }

  return await res.json();
}
