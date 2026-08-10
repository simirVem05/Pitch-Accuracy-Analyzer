export function toPct(x, decimals = 0) {
  const v = (Number(x) || 0) * 100;
  return `${v.toFixed(decimals)}%`;
}

export function clamp01(x) {
  const v = Number(x) || 0;
  return Math.max(0, Math.min(1, v));
}
