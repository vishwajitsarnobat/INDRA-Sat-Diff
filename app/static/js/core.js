// Shared helpers and state. No build step: plain ES modules.
export const $ = (s, el = document) => el.querySelector(s);
export const $$ = (s, el = document) => [...el.querySelectorAll(s)];

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value" || k === "checked" || k === "disabled" || k === "hidden") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat(Infinity)) if (k !== null && k !== undefined && k !== false) el.append(k instanceof Node ? k : document.createTextNode(k));
  return el;
}
const ICONS = {
  play: "M5 3l9 5-9 5z", pause: "M4 3h3v10H4zM9 3h3v10H9z", prev: "M11 3L5 8l6 5zM3 3h2v10H3z", next: "M5 3l6 5-6 5zM11 3h2v10h-2z",
  first: "M13 3L7 8l6 5zM8 3L2 8l6 5z", last: "M3 3l6 5-6 5zM8 3l6 5-6 5z", left: "M10 3L5 8l5 5z", right: "M6 3l5 5-5 5z",
  sun: "M8 4.5a3.5 3.5 0 100 7 3.5 3.5 0 000-7zM7.4 1h1.2v2H7.4zM7.4 13h1.2v2H7.4zM1 7.4h2v1.2H1zM13 7.4h2v1.2h-2z",
  moon: "M9.5 1.5a6.5 6.5 0 105 10.2A5.5 5.5 0 019.5 1.5z",
};
export function icon(name, size = 14) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 16 16"); s.setAttribute("width", size); s.setAttribute("height", size); s.setAttribute("aria-hidden", "true");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", ICONS[name]); p.setAttribute("fill", "currentColor"); s.append(p);
  return s;
}

// replaceChildren that ignores null / false entries
export const put = (el, ...kids) => el.replaceChildren(...kids.flat(Infinity).filter((k) => k !== null && k !== undefined && k !== false));

export async function api(url, opts = {}) {
  const init = opts.body !== undefined ? { method: "POST", headers: { "Content-Type": "application/json" }, ...opts, body: JSON.stringify(opts.body) } : opts;
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = await r.text();
    try { msg = JSON.parse(msg).detail || msg; } catch (e) { /* plain text */ }
    throw new Error(msg || r.statusText);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
export function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), Math.min(12000, 4000 + msg.length * 40));
}
export const fmt = (v, d = 2) => (v === null || v === undefined || !Number.isFinite(+v)) ? "–" : Number(v).toFixed(d);
// UTC timestamp "2024-07-26 00:30" from unix seconds or an ISO string
export function utc(t, withZone = true) {
  const d = typeof t === "number" ? new Date(t * 1000) : new Date(t);
  return Number.isNaN(+d) ? "–" : d.toISOString().slice(0, 16).replace("T", " ") + (withZone ? " UTC" : "");
}
export const isoParam = (t) => new Date(t * 1000).toISOString().slice(0, 16) + "Z";
export const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export const store = {
  get(k, d) { try { const v = localStorage.getItem("indra." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("indra." + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
};

// ------------------------------------------------------------------ state
export const bus = new EventTarget();
export const on = (name, fn) => bus.addEventListener(name, (e) => fn(e.detail));
export const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }));
export const S = { datasets: [], models: [], allModels: [], ds: null, timelines: new Map() };

export async function loadLists(refresh = false) {
  const [ds, models] = await Promise.all([api("/api/datasets" + (refresh ? "?refresh=true" : "")), api("/api/models?all=true")]);
  S.datasets = ds; S.allModels = models; S.models = models.filter((m) => m.usable !== false);
  if (refresh) S.timelines.clear();
  const sel = $("#ds-select"), want = S.ds?.id || store.get("ds");
  const groups = [["evaluation", "Unseen evaluation data"], ["user", "Your data"], ["sample", "Samples"], ["test", "Test sequences (training years)"], ["training", "Training archives"]];
  const known = new Set(groups.map((g) => g[0]));
  sel.replaceChildren(...groups.map(([role, label]) => {
    const items = ds.filter((d) => (known.has(d.role) ? d.role : "user") === role);
    return items.length ? h("optgroup", { label }, items.map((d) => h("option", { value: d.id, title: d.source || "" }, d.name))) : null;
  }).filter(Boolean));
  S.ds = ds.find((d) => d.id === want) || ds.find((d) => d.role === "evaluation" && /south/i.test(d.name)) || ds.find((d) => d.role === "evaluation") || ds[0] || null;
  if (S.ds) sel.value = S.ds.id;
  emit("lists");
}
// Fill a model <select>, keeping the choice; otherwise prefer a model trained for the current dataset's region.
export function fillModels(sel, label = (m) => m.name) {
  const v = sel.value, same = (a, b) => a && b && Math.abs(a[0] - b[0]) < 0.06 && Math.abs(a[1] - b[1]) < 0.06;
  sel.replaceChildren(...S.models.map((m) => h("option", { value: m.id }, label(m))));
  const match = S.models.find((m) => S.ds && same(m.lat_range, S.ds.lat_range) && same(m.lon_range, S.ds.lon_range));
  if (S.models.some((m) => m.id === v)) sel.value = v; else if (match) sel.value = match.id;
}
export const dsById = (id) => S.datasets.find((d) => d.id === id);
export const modelById = (id) => S.allModels.find((m) => m.id === id);
export function timeline(id) {
  if (!S.timelines.has(id)) S.timelines.set(id, api(`/api/datasets/${id}/timeline`));
  return S.timelines.get(id);
}
// Indices i such that frames i-inLen+1 .. i+outLen exist and are evenly spaced.
export function validStarts(tl, inLen, outLen = 0) {
  const out = [], T = tl.times, dt = tl.interval;
  let run = 0;                       // consecutive evenly spaced frames ending at k
  const ok = new Uint8Array(T.length);
  for (let k = 0; k < T.length; k++) { run = k && Math.abs(T[k] - T[k - 1] - dt) <= 1 ? run + 1 : 1; ok[k] = run >= inLen + outLen ? 1 : 0; }
  for (let i = inLen - 1; i < T.length - outLen; i++) if (ok[i + outLen]) out.push(i);
  return out;
}

// ------------------------------------------------------------------ jobs
export function runJob(type, params, { onProgress, onLog, onStart } = {}) {
  return new Promise(async (resolve, reject) => {
    let job;
    try { job = await api("/api/jobs", { body: { type, params } }); } catch (e) { return reject(e); }
    onStart?.(job); emit("jobs");
    const es = new EventSource(`/api/jobs/${job.id}/events`);
    es.addEventListener("progress", (e) => onProgress?.(JSON.parse(e.data)));
    es.addEventListener("log", (e) => onLog?.(JSON.parse(e.data).line));
    es.addEventListener("status", (e) => {
      const d = JSON.parse(e.data);
      if (d.status === "done") { es.close(); emit("jobs"); resolve({ ...job, status: "done", result: d.result }); }
      else if (d.status === "failed" || d.status === "cancelled") { es.close(); emit("jobs"); reject(new Error(d.error || d.status)); }
    });
    es.onerror = () => { if (es.readyState === 2) reject(new Error("connection to the server was lost")); };
  });
}

// ------------------------------------------------------------------ small UI builders
export const field = (label, control, hint) => h("label", { class: "field" }, h("span", { class: "lab" }, label), control, hint && h("span", { class: "hint" }, hint));
export const check = (label, attrs = {}) => { const i = h("input", { type: "checkbox", ...attrs }); const l = h("label", { class: "check" }, i, label); l.input = i; return l; };
export const options = (list, value) => list.map((o) => { const [v, t] = Array.isArray(o) ? o : [o, o]; return h("option", { value: v, selected: String(v) === String(value) }, t); });
export const select = (list, attrs = {}) => h("select", attrs, options(list, attrs.value));
export const chip = (text, cls = "") => h("span", { class: `chip ${cls}` }, text);
export const kv = (pairs) => h("dl", { class: "kv" }, pairs.filter((p) => p && p[1] !== undefined && p[1] !== null).map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
export function segmented(list, value, onChange, label) {
  const el = h("div", { class: "seg", role: "group", "aria-label": label });
  el.value = value;
  for (const o of list) {
    const [v, t] = Array.isArray(o) ? o : [o, String(o)];
    el.append(h("button", { type: "button", "aria-pressed": String(v === value), onclick: () => { el.set(v); onChange?.(v); } }, t));
  }
  el.set = (v) => { el.value = v; [...el.children].forEach((b, i) => b.setAttribute("aria-pressed", String((Array.isArray(list[i]) ? list[i][0] : list[i]) === v))); };
  return el;
}
// Page skeleton: collapsible side panel + work area.
export function shell(root, { fill = false, side = true } = {}) {
  const aside = h("aside", { class: "side", "aria-label": "Controls" });
  const work = h("div", { class: "work " + (fill ? "fill" : "pad") });
  root.classList.add("page");
  if (!side) { root.classList.add("noside"); root.append(work); return { work }; }
  const toggle = h("button", { class: "side-toggle", title: "Show / hide panel (L)", "aria-label": "Show or hide side panel", "aria-expanded": "true" });
  const apply = (c) => { root.classList.toggle("collapsed", c); toggle.replaceChildren(icon(c ? "right" : "left", 10)); toggle.setAttribute("aria-expanded", String(!c)); };
  toggle.addEventListener("click", () => { const c = !root.classList.contains("collapsed"); store.set("collapsed", c); apply(c); });
  apply(store.get("collapsed", false));
  root.toggleSide = () => toggle.click();
  root.append(aside, toggle, work);
  return { side: aside, work };
}
export const lonLabel = (v, d = 2) => `${Math.abs(v).toFixed(d)}°${v < 0 ? "W" : "E"}`;
export const latLabel = (v, d = 2) => `${Math.abs(v).toFixed(d)}°${v < 0 ? "S" : "N"}`;
export const regionLabel = (o) => o?.lat_range && o?.lon_range ? `${latLabel(o.lat_range[0], 1)}–${latLabel(o.lat_range[1], 1)}, ${lonLabel(o.lon_range[0], 1)}–${lonLabel(o.lon_range[1], 1)}` : "–";
