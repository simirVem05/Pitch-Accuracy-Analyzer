export function intonationScoreFromAbsCents(absCents) {
  const d = Math.max(0, Number(absCents) || 0);

  // 0–20c => 1.00 -> 0.80
  if (d <= 20) {
    return 1.0 - (0.20 / 20.0) * d;
  }

  // 20–30c => 0.80 -> 0.60
  if (d <= 30) {
    return 0.80 - (0.20 / 10.0) * (d - 20);
  }

  // 30+c => exponential decay below 0.60
  return 0.60 * Math.exp(-(d - 30.0) / 43.0);
}

export function toPct(x, decimals = 0) {
  const v = (Number(x) || 0) * 100;
  return `${v.toFixed(decimals)}%`;
}

export function clamp01(x) {
  const v = Number(x) || 0;
  return Math.max(0, Math.min(1, v));
}
