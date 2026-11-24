// Line charts (uPlot) that follow the theme. Colours: fixed series order; dashed = reference (persistence).
import { h, cssVar, on } from "./core.js";
const live = new Set();
// compact number: plain for ordinary magnitudes, exponent for very large or very small ones
const num = (v, sig = 2) => { const a = Math.abs(v); return a === 0 ? "0" : a >= 1e5 || a < 1e-2 ? v.toExponential(sig > 2 ? 2 : 1).replace("e+", "e") : String(+v.toPrecision(Math.max(sig, 3))); };
const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];

// spec: { title, sub, x: [...], xLabel, series: [{ label, data, dash, color (index) }], yRange: [lo, hi] | null, xFmt }
export function chart(parent, spec) {
  const box = h("div", { class: "chart" }, h("div", { class: "ct" }, spec.title), spec.sub && h("div", { class: "cs" }, spec.sub), h("div", { class: "plot" }));
  parent.append(box);
  const c = { box, spec, plot: null };
  live.add(c); queueMicrotask(() => build(c));      // measure after sibling charts are in the grid
  return c;
}
function build(c) {
  if (!c.box.isConnected) { live.delete(c); return; }
  const { spec } = c, host = c.box.querySelector(".plot");
  c.plot?.destroy(); host.replaceChildren();
  const ink = cssVar("--muted"), grid = cssVar("--line"), w = Math.max(240, Math.floor(c.box.clientWidth - 22));
  const axis = { stroke: ink, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, font: "11px system-ui", labelFont: "11px system-ui", labelSize: 18, size: 34 };
  c.plot = new uPlot({
    width: w, height: spec.height || 190, padding: [8, 10, 0, 0],
    scales: { x: { time: false }, y: spec.yRange ? { range: spec.yRange } : { auto: true } },
    axes: [{ ...axis, label: spec.xLabel, space: spec.xSpace || 50, splits: spec.xAtData ? () => { const n = Math.ceil(spec.x.length / 8); return spec.x.filter((_, i) => i % n === 0); } : undefined, values: spec.xFmt ? (u, v) => v.map(spec.xFmt) : undefined }, { ...axis, size: 52, values: (u, v) => v.map(num) }],
    cursor: { points: { size: 7 }, y: false },
    legend: { live: true },
    series: [{ label: spec.xLabel || "x", value: spec.xFmt ? (u, v) => (v == null ? "–" : spec.xFmt(v)) : undefined },
      ...spec.series.map((s, i) => ({ label: s.label, stroke: cssVar(SERIES[(s.color ?? i) % SERIES.length]), width: 2, dash: s.dash, spanGaps: true,
        points: { show: spec.x.length <= 40, size: 6 }, value: (u, v) => (v == null ? "–" : num(v, 3)) }))],
  }, [spec.x, ...spec.series.map((s) => s.data.map((v) => (Number.isFinite(v) ? v : null)))], host);
}
export function updateChart(c, spec) { c.spec = { ...c.spec, ...spec }; queueMicrotask(() => build(c)); }
let t = 0;
const rebuild = () => { clearTimeout(t); t = setTimeout(() => live.forEach(build), 120); };
on("theme", rebuild); addEventListener("resize", rebuild); on("layout", rebuild);
