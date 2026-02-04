export async function analyzeVocals({
    file,
    key,
    genre,
    strictness,
    confidence_threshold,
    step_size_ms,
    model_capacity,
    viterbi,
    a4_hz,
}) {
    const form = new FormData();
    form.append("file", file);
    form.append("key", key);
    form.append("genre", genre);
    
    form.append("strictness", String(strictness));
    form.append("confidence_threshold", String(confidence_threshold));
    form.append("step_size_ms", String(step_size_ms));
    form.append("model_capacity", String(model_capacity));
    form.append("viterbi", String(viterbi));
    form.append("a4_hz", String(a4_hz));

    const res = await fetch("/api/analyze", {
        method: "POST",
        body: form,
    });

    if (!res.ok){
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
    }

    return await res.json();
}