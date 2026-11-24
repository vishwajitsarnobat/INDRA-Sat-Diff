// INDRA-Sat-Diff app — no build step. Frames travel as uint8 (normalised*255) and are colour-mapped here.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) el.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) el.append(k instanceof Node ? k : document.createTextNode(k));
  return el;
};
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts.body ? { method: "POST", headers: { "Content-Type": "application/json" }, ...opts, body: JSON.stringify(opts.body) } : opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
};
const toast = (msg) => { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 3500); };
const iso = (t) => new Date(t * 1000).toISOString().slice(0, 16).replace("T", " ") + "Z";
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);

// ------------------------------------------------------------------ colour map (mm/hr)
const BINS = [0.1, 0.5, 1, 2.5, 5, 7.6, 10, 16, 25, 50, 100];
const COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#31a354", "#74c476", "#fed976", "#feb24c", "#fd8d3c", "#e31a1c", "#99000d"];
const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const lutCache = new Map();
function lut(clip = 100) {
  if (lutCache.has(clip)) return lutCache.get(clip);
  const L = new Uint8ClampedArray(256 * 4), k = Math.log1p(clip);
  for (let u = 0; u < 256; u++) {
    const mm = Math.expm1((u / 255) * k);
    let b = -1; for (let i = 0; i < BINS.length; i++) if (mm >= BINS[i]) b = i;
    if (b >= 0) { const [r, g, bl] = hex(COLORS[Math.min(b, COLORS.length - 1)]); L.set([r, g, bl, 255], u * 4); }
  }
  lutCache.set(clip, L); return L;
}
const PROB_COLORS = ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"];
function probLut() {
  const L = new Uint8ClampedArray(256 * 4);
  for (let u = 0; u < 256; u++) { const p = u / 255; if (p < 0.05) continue; const [r, g, b] = hex(PROB_COLORS[Math.min(6, Math.floor(p * 7))]); L.set([r, g, b, 255], u * 4); }
  return L;
}
function legend(el, clip = 100) {
  el.replaceChildren(h("span", {}, "mm/hr "), ...BINS.slice(0, -1).map((b, i) => h("span", { class: "sw", style: `background:${COLORS[i]}`, title: `${b}–${BINS[i + 1]} mm/hr` })),
    h("br"), h("span", { class: "lab" }, ""), ...BINS.slice(0, -1).map((b) => h("span", { class: "lab" }, String(b))));
}

// ------------------------------------------------------------------ geography
let GEO = null;
async function geo() {
  if (!GEO) {
    const [c, b] = await Promise.all(["ne_50m_coastline", "ne_50m_admin_0_boundary_lines_land"].map((n) => fetch(`/static/vendor/${n}.geojson`).then((r) => r.json())));
    const lines = (fc) => fc.features.flatMap((f) => f.geometry.type === "LineString" ? [f.geometry.coordinates] : f.geometry.coordinates);
    GEO = { coast: lines(c), border: lines(b) };
  }
  return GEO;
}
geo();

// Draw one field. a: Uint8Array of shape [n0, n1]; order "lon_lat" means a[i_lon*n1 + j_lat], lat increasing.
function drawField(canvas, a, shape, ds, { lutTable, axes = true, title = null } = {}) {
  const [n0, n1] = shape, lonLat = (ds.array_order || "lon_lat") === "lon_lat";
  const W = lonLat ? n0 : n1, H = lonLat ? n1 : n0;
  const img = new ImageData(W, H), L = lutTable || lut(ds.clip_value);
  for (let r = 0; r < H; r++) {                 // r = 0 is north
    const jLat = H - 1 - r;
    for (let c = 0; c < W; c++) {
      const u = lonLat ? a[c * n1 + jLat] : a[jLat * n1 + c];
      const o = (r * W + c) * 4, s = u * 4;
      img.data[o] = L[s]; img.data[o + 1] = L[s + 1]; img.data[o + 2] = L[s + 2]; img.data[o + 3] = L[s + 3];
    }
  }
  const off = new OffscreenCanvas(W, H); off.getContext("2d").putImageData(img, 0, 0);
  const ctx = canvas.getContext("2d"), m = axes ? { l: 38, r: 6, t: title ? 18 : 6, b: 22 } : { l: 0, r: 0, t: title ? 14 : 0, b: 0 };
  const pw = canvas.width - m.l - m.r, ph = canvas.height - m.t - m.b;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--panel"); ctx.fillRect(m.l, m.t, pw, ph);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(off, m.l, m.t, pw, ph);
  const [la0, la1] = ds.lat_range, [lo0, lo1] = ds.lon_range;
  const X = (lon) => m.l + ((lon - lo0) / (lo1 - lo0)) * pw, Y = (lat) => m.t + ((la1 - lat) / (la1 - la0)) * ph;
  if (GEO) {
    ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
    for (const [set, col, w] of [[GEO.border, "rgba(90,90,90,.7)", 0.8], [GEO.coast, "rgba(20,20,20,.9)", 1.1]]) {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath();
      for (const line of set) {
        let inside = false;
        for (const [lon, lat] of line) { if (lon < lo0 - 2 || lon > lo1 + 2 || lat < la0 - 2 || lat > la1 + 2) { inside = false; continue; }
          if (!inside) { ctx.moveTo(X(lon), Y(lat)); inside = true; } else ctx.lineTo(X(lon), Y(lat)); }
      }
      ctx.stroke();
    }
    ctx.restore();
  }
  ctx.strokeStyle = css.getPropertyValue("--line"); ctx.strokeRect(m.l + .5, m.t + .5, pw - 1, ph - 1);
  ctx.fillStyle = css.getPropertyValue("--muted"); ctx.font = "11px system-ui";
  if (axes) {
    const step = (lo1 - lo0) > 8 ? 2 : 1;
    for (let lon = Math.ceil(lo0); lon <= lo1; lon += step) { ctx.fillText(`${lon}°E`, X(lon) - 12, canvas.height - 6); }
    for (let lat = Math.ceil(la0); lat <= la1; lat += step) { ctx.fillText(`${lat}°N`, 2, Y(lat) + 4); }
  }
  if (title) { ctx.fillStyle = css.getPropertyValue("--ink"); ctx.font = "600 11px system-ui"; ctx.fillText(title, m.l + 2, 11); }
  canvas._geo = { m, pw, ph, W, H, a, n1, lonLat, ds };
}
function valueAt(canvas, ev) {
  const g = canvas._geo; if (!g) return null;
  const rect = canvas.getBoundingClientRect(), sx = canvas.width / rect.width;
  const x = (ev.clientX - rect.left) * sx - g.m.l, y = (ev.clientY - rect.top) * sx - g.m.t;
  if (x < 0 || y < 0 || x >= g.pw || y >= g.ph) return null;
  const c = Math.floor((x / g.pw) * g.W), r = Math.floor((y / g.ph) * g.H), jLat = g.H - 1 - r;
  const u = g.lonLat ? g.a[c * g.n1 + jLat] : g.a[jLat * g.n1 + c];
  const [la0, la1] = g.ds.lat_range, [lo0, lo1] = g.ds.lon_range;
  return { lat: la1 - (y / g.ph) * (la1 - la0), lon: lo0 + (x / g.pw) * (lo1 - lo0), mm: Math.expm1((u / 255) * Math.log1p(g.ds.clip_value || 100)) };
}

// ------------------------------------------------------------------ shared state
const S = { datasets: [], models: [], timelines: {} };
async function refreshLists() {
  [S.datasets, S.models] = await Promise.all([api("/api/datasets"), api("/api/models")]);
  const dsOpts = S.datasets.map((d) => h("option", { value: d.id }, `${d.name} (${d.count} frames)`));
  const mOpts = S.models.map((m) => h("option", { value: m.id }, `${m.name} · ${m.grid.join("×")} · L${m.compute_level} · ${m.size_mb} MB`));
  for (const id of ["#data-ds", "#pred-ds", "#sim-ds", "#eval-ds", "#train-ds"]) $(id).replaceChildren(...dsOpts.map((o) => o.cloneNode(true)));
  for (const id of ["#pred-model", "#sim-model", "#eval-model"]) $(id).replaceChildren(...mOpts.map((o) => o.cloneNode(true)));
}
const dsById = (id) => S.datasets.find((d) => d.id === id);
const modelById = (id) => S.models.find((m) => m.id === id);
async function timeline(id) { return S.timelines[id] ||= await api(`/api/datasets/${id}/timeline`); }

// Valid forecast start indices: in_len consecutive frames ending there.
function validStarts(tl, inLen, outLen = 0) {
  const out = [], T = tl.times, dt = tl.interval;
  for (let i = inLen - 1; i < T.length - outLen; i++) {
    let ok = true; for (let k = i - inLen + 1; k < i + outLen; k++) if (Math.abs(T[k + 1] - T[k] - dt) > 1) { ok = false; break; }
    if (ok) out.push(i);
  }
  return out;
}

// ------------------------------------------------------------------ jobs + SSE
function runJob(type, params, { onProgress, onLog } = {}) {
  return new Promise(async (resolve, reject) => {
    let job; try { job = await api("/api/jobs", { body: { type, params } }); } catch (e) { return reject(e); }
    pollBadge();
    const es = new EventSource(`/api/jobs/${job.id}/events`);
    es.addEventListener("progress", (e) => onProgress?.(JSON.parse(e.data)));
    es.addEventListener("log", (e) => onLog?.(JSON.parse(e.data).line));
    es.addEventListener("status", (e) => {
      const d = JSON.parse(e.data);
      if (d.status === "done") { es.close(); resolve({ ...job, result: d.result }); }
      else if (d.status === "failed" || d.status === "cancelled") { es.close(); reject(new Error(d.error || d.status)); }
    });
    es.onerror = () => { if (es.readyState === 2) reject(new Error("event stream closed")); };
  });
}
async function jobArray(jobId, name, member = 0) {
  const r = await fetch(`/api/jobs/${jobId}/array/${name}?member=${member}`);
  if (!r.ok) throw new Error(await r.text());
  return { a: new Uint8Array(await r.arrayBuffer()), shape: r.headers.get("X-Shape").split(",").map(Number) };
}
async function pollBadge() {
  const js = await api("/api/jobs"); const n = js.filter((j) => j.status === "running" || j.status === "queued").length;
  const b = $("#jobs-badge"); b.hidden = !n; b.textContent = n;
}
setInterval(pollBadge, 5000);

// ------------------------------------------------------------------ DATA page
const D = { ds: null, tl: null, i: 0, cache: new Map(), playing: false };
async function loadFrame(i) {
  const key = `${D.ds.id}:${i}`; if (D.cache.has(key)) return D.cache.get(key);
  const p = fetch(`/api/datasets/${D.ds.id}/frame?i=${i}`).then(async (r) => ({ a: new Uint8Array(await r.arrayBuffer()), shape: r.headers.get("X-Shape").split(",").map(Number) }));
  D.cache.set(key, p); if (D.cache.size > 400) D.cache.delete(D.cache.keys().next().value);
  return p;
}
async function showFrame(i) {
  D.i = i; $("#data-slider").value = i;
  const f = await loadFrame(i); if (D.i !== i) return;
  drawField($("#data-map"), f.a, f.shape, D.ds);
  $("#data-time").textContent = `${iso(D.tl.times[i])}  ·  frame ${i + 1}/${D.tl.times.length}`;
  for (let k = 1; k <= 6; k++) if (i + k < D.tl.times.length) loadFrame(i + k);   // prefetch
}
async function selectDataset(id) {
  D.ds = dsById(id); if (!D.ds) return;
  D.tl = await timeline(id); legend($("#legend-data"), D.ds.clip_value);
  $("#data-slider").max = D.tl.times.length - 1;
  $("#data-info").replaceChildren(h("h2", {}, D.ds.name),
    h("div", { class: "mono" }, `lat ${D.ds.lat_range.join("–")}°N · lon ${D.ds.lon_range.join("–")}°E · grid ${D.ds.grid?.join("×")}`),
    h("div", { class: "mono" }, `${D.ds.start} → ${D.ds.end} · every ${D.ds.interval_minutes} min · ${D.ds.gaps} gap(s)`));
  const av = $("#data-avail"), ctx = av.getContext("2d"); av.width = av.clientWidth || 800;
  ctx.fillStyle = "#3182bd"; ctx.fillRect(0, 0, av.width, 10);
  const T = D.tl.times, span = T[T.length - 1] - T[0] || 1; ctx.fillStyle = "#e31a1c";
  for (const [a, b] of D.tl.gaps) { const x0 = ((Date.parse(a) / 1000 - T[0]) / span) * av.width, x1 = ((Date.parse(b) / 1000 - T[0]) / span) * av.width; ctx.fillRect(x0, 0, Math.max(2, x1 - x0), 10); }
  showFrame(0);
}
$("#data-ds").addEventListener("change", (e) => selectDataset(e.target.value));
$("#data-slider").addEventListener("input", (e) => showFrame(+e.target.value));
$("#data-play").addEventListener("click", async () => {
  D.playing = !D.playing; $("#data-play").textContent = D.playing ? "❚❚ Pause" : "▶ Play";
  while (D.playing) { await showFrame((D.i + 1) % D.tl.times.length); await new Promise((r) => setTimeout(r, +$("#data-speed").value)); }
});
$("#data-map").addEventListener("mousemove", (e) => { const v = valueAt(e.target, e); $("#data-hover").textContent = v ? `${v.lat.toFixed(2)}°N ${v.lon.toFixed(2)}°E  ·  ${v.mm.toFixed(2)} mm/hr` : "hover the map for values"; });
document.addEventListener("keydown", (e) => { if ($("#page-data").hidden || e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if (e.key === "ArrowRight") showFrame(Math.min(D.i + 1, D.tl.times.length - 1)); if (e.key === "ArrowLeft") showFrame(Math.max(D.i - 1, 0)); });
$("#dl-form").addEventListener("submit", async (e) => {
  e.preventDefault(); const f = Object.fromEntries(new FormData(e.target));
  const p = { name: f.name, lat_range: [+f.lat0, +f.lat1], lon_range: [+f.lon0, +f.lon1], start: f.start, end: f.end, product: f.product, downscale_factor: +f.downscale_factor };
  toast("Download queued — see Jobs");
  try { await runJob("download", p); toast("Dataset ready"); await refreshLists(); } catch (err) { toast("Download failed: " + err.message); }
});

// ------------------------------------------------------------------ strips of frames
function strip(container, rows, cols, ds, size = 150) {
  const grid = h("div", { class: "strip", style: `grid-template-columns: 90px repeat(${cols.length}, minmax(80px, ${size}px))` });
  grid.append(h("div"), ...cols.map((c) => h("div", { class: "collab" }, c)));
  for (const r of rows) {
    grid.append(h("div", { class: "rowlab" }, r.label));
    const [T, n0, n1] = r.shape;
    for (let k = 0; k < cols.length; k++) {
      const cv = h("canvas", { width: 220, height: 220, "aria-label": `${r.label} ${cols[k]}` });
      if (k < T) { drawField(cv, r.a.subarray(k * n0 * n1, (k + 1) * n0 * n1), [n0, n1], ds, { axes: false, lutTable: r.lut });
        if (!r.lut) cv.addEventListener("mousemove", (e) => { const v = valueAt(cv, e); cv.title = v ? `${v.mm.toFixed(1)} mm/hr` : ""; }); }
      grid.append(k < T ? cv : h("div"));
    }
  }
  container.append(grid);
}

// ------------------------------------------------------------------ PREDICT page
async function fillStarts(modelSel, dsSel, target, outLen = 0) {
  const m = modelById(modelSel.value), ds = dsById(dsSel.value); if (!m || !ds) return [];
  const tl = await timeline(ds.id), idx = validStarts(tl, m.in_len, outLen);
  target.replaceChildren(...idx.map((i) => h("option", { value: iso(tl.times[i]).replace(" ", "T") }, iso(tl.times[i]))));
  if (idx.length) target.selectedIndex = Math.floor(idx.length / 2);
  return idx;
}
for (const id of ["#pred-model", "#pred-ds"]) $(id).addEventListener("change", () => fillStarts($("#pred-model"), $("#pred-ds"), $("#pred-t0")));

function lineChart(el, title, xs, series, { yMax = 1, w = 380, ht = 220 } = {}) {
  const palette = ["#e8710a", "#1f5fb4", "#1a9e3a", "#9b59b6", "#c0392b"];
  const opts = { title, width: w, height: ht, scales: { x: { time: false }, y: { range: [0, yMax] } },
    axes: [{ label: "lead (min)" }, {}], series: [{}, ...series.map((s, i) => ({ label: s.label, stroke: palette[i % 5], width: 2, dash: s.dash, points: { show: true } }))] };
  const box = h("div", { class: "card" }); el.append(box);
  new uPlot(opts, [xs, ...series.map((s) => s.data.map((v) => v ?? null))], box);
}

$("#pred-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target), btn = $("button[type=submit]", e.target), prog = $("#pred-progress");
  const p = { model: f.get("model"), dataset: f.get("dataset"), t0: f.get("t0"),
    alignment: f.get("alignment") === "on", members: +f.get("members"), steps: +f.get("steps"), precision: f.get("precision"), sampler: "ddim" };
  btn.disabled = true; prog.hidden = false; prog.value = 0; $("#pred-status").textContent = "queued…";
  try {
    const job = await runJob("predict", p, { onProgress: (d) => { prog.value = d.value; $("#pred-status").textContent = `sampling ${(d.value * 100).toFixed(0)}%`; } });
    $("#pred-status").textContent = `done in ${job.result.seconds}s`;
    await renderPrediction(job, dsById(p.dataset), modelById(p.model));
  } catch (err) { $("#pred-status").textContent = "failed"; toast(err.message); }
  btn.disabled = false; prog.hidden = true;
});

async function renderPrediction(job, ds, m, out = $("#pred-out"), compact = false) {
  const r = job.result; out.replaceChildren();
  const ctx = await jobArray(job.id, "context"), fc = await jobArray(job.id, "forecast");
  const rows = [{ label: "Input", ...ctx }, { label: "Forecast", ...fc }];
  if (r.has_truth) rows.push({ label: "IMERG truth", ...(await jobArray(job.id, "truth")) });
  if (r.members > 1) { const ex = await jobArray(job.id, "exceed_2.5"); rows.push({ label: "P(≥2.5 mm/hr)", ...ex, lut: probLut() }); }
  const leads = r.lead_minutes.map((x) => `+${x} min`);
  const inputCols = r.times_context.map((t) => t.slice(11, 16) + "Z");
  if (!compact) { out.append(h("h2", {}, "Input frames")); strip(out, [rows[0]], inputCols, ds); }
  out.append(h("h2", {}, `Forecast from ${r.times_context.at(-1)}`)); strip(out, rows.slice(1), leads, ds);
  const lg = h("div", { class: "legend" }); legend(lg, ds.clip_value); out.append(lg);
  if (r.metrics) {
    const thr = "2.5", mm = r.metrics[thr], kpis = h("div", { class: "kpis" });
    for (const [k, name] of [["pod", "POD"], ["far", "FAR"], ["csi", "CSI"]]) {
      const avg = (arr) => { const v = arr.map((x) => x[k]).filter((x) => x !== null && !Number.isNaN(x)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
      kpis.append(h("div", { class: "kpi" }, h("small", {}, `${name} @${thr} mm/hr`), h("b", {}, fmt(avg(mm.model))), h("small", {}, `persistence ${fmt(avg(mm.persistence))}`)));
    }
    out.append(kpis);
    const charts = h("div", { class: "charts" }); out.append(charts);
    for (const [k, name] of [["csi", "CSI"], ["pod", "POD"], ["far", "FAR"]])
      lineChart(charts, `${name} vs lead (≥${thr} mm/hr, this forecast)`, r.lead_minutes, [{ label: "model", data: mm.model.map((x) => x[k]) }, { label: "persistence", data: mm.persistence.map((x) => x[k]), dash: [6, 4] }]);
  }
  out.append(h("p", { class: "hint" }, h("a", { href: `/api/jobs/${job.id}/file/result.npz` }, "Download arrays (npz)"), `  ·  sampled ${r.members} member(s) in ${r.seconds}s`));
}

// ------------------------------------------------------------------ SIMULATE page
for (const id of ["#sim-model", "#sim-ds"]) $(id).addEventListener("change", fillSim);
async function fillSim() {
  const m = modelById($("#sim-model").value), ds = dsById($("#sim-ds").value); if (!m || !ds) return;
  const tl = await timeline(ds.id), idx = validStarts(tl, m.in_len, m.out_len);
  const opts = idx.map((i) => h("option", { value: i }, iso(tl.times[i])));
  $("#sim-start").replaceChildren(...opts.map((o) => o.cloneNode(true))); $("#sim-end").replaceChildren(...opts);
  $("#sim-end").selectedIndex = Math.min(idx.length - 1, 12 * 4);
}
$("#sim-form").addEventListener("submit", async (e) => {
  e.preventDefault(); const f = new FormData(e.target), btn = $("button[type=submit]", e.target);
  const m = modelById(f.get("model")), ds = dsById(f.get("dataset")), tl = await timeline(ds.id);
  const i0 = +f.get("start"), i1 = +f.get("end"), every = +f.get("every");
  const starts = validStarts(tl, m.in_len, m.out_len).filter((i) => i >= i0 && i <= i1 && (i - i0) % every === 0);
  if (!starts.length) return toast("No complete windows in that range.");
  btn.disabled = true; const out = $("#sim-out"); out.replaceChildren();
  const chartBox = h("div", { class: "charts" }), latest = h("div"); out.append(chartBox, latest);
  const hist = { t: [], model: [], pers: [] };
  for (const [n, i] of starts.entries()) {
    $("#sim-status").textContent = `forecast ${n + 1}/${starts.length} (${iso(tl.times[i])})`;
    try {
      const job = await runJob("predict", { model: m.id, dataset: ds.id, t0: iso(tl.times[i]).replace(" ", "T"), alignment: f.get("alignment") === "on", steps: +f.get("steps"), members: 1, precision: "fp16" });
      const met = job.result.metrics["2.5"], mean = (a) => { const v = a.map((x) => x.csi).filter((x) => x !== null && !Number.isNaN(x)); return v.length ? v.reduce((p, q) => p + q, 0) / v.length : null; };
      hist.t.push(n + 1); hist.model.push(mean(met.model)); hist.pers.push(mean(met.persistence));
      chartBox.replaceChildren(); lineChart(chartBox, "Mean CSI (≥2.5 mm/hr) of each replayed forecast", hist.t, [{ label: "model", data: hist.model }, { label: "persistence", data: hist.pers, dash: [6, 4] }], { w: 560 });
      $(".u-legend", chartBox)?.insertAdjacentText("afterbegin", "x = forecast number  ");
      await renderPrediction(job, ds, m, latest, true);
    } catch (err) { toast(err.message); break; }
  }
  $("#sim-status").textContent = "replay finished"; btn.disabled = false;
});

// ------------------------------------------------------------------ EVALUATE page
$("#eval-form").addEventListener("submit", async (e) => {
  e.preventDefault(); const f = new FormData(e.target), btn = $("button[type=submit]", e.target);
  const p = { model: f.get("model"), dataset: f.get("dataset"), stride: +f.get("stride"), steps: +f.get("steps"), compare_ka: f.get("compare_ka") === "on" };
  btn.disabled = true; $("#eval-status").textContent = "queued…";
  try {
    const job = await runJob("evaluate", p, { onProgress: (d) => { $("#eval-status").textContent = d.detail || `${(d.value * 100).toFixed(0)}%`; } });
    renderEval(job.result);
    $("#eval-status").textContent = `${job.result.forecasts} start times · saved in ${job.result.dir}`;
  } catch (err) { $("#eval-status").textContent = "failed"; toast(err.message); }
  btn.disabled = false;
});
function renderEval(res, out = $("#eval-out")) {
  out.replaceChildren();
  const thrs = [...new Set(res.table.map((r) => r.threshold_mm))], variants = [...new Set(res.table.map((r) => r.variant))];
  const sel = h("select", {}, ...thrs.map((t) => h("option", { value: t, ...(t === 2.5 ? { selected: "" } : {}) }, `≥ ${t} mm/hr`)));
  const charts = h("div", { class: "charts" }), tableBox = h("div", { class: "card" });
  out.append(h("label", {}, "Threshold", sel), charts, tableBox);
  const draw = () => {
    const thr = +sel.value, rows = res.table.filter((r) => r.threshold_mm === thr), leads = [...new Set(rows.map((r) => r.lead_min))];
    charts.replaceChildren();
    for (const [k, name] of [["csi", "CSI"], ["pod", "POD"], ["far", "FAR"], ["bias", "Frequency bias"]])
      lineChart(charts, `${name} (≥${thr} mm/hr)`, leads, variants.map((v) => ({ label: v, data: leads.map((l) => rows.find((r) => r.variant === v && r.lead_min === l)?.[k]), dash: v === "persistence" ? [6, 4] : undefined })), { yMax: k === "bias" ? 2 : 1 });
    tableBox.replaceChildren(h("table", {}, h("thead", {}, h("tr", {}, ...["variant", "lead", "POD", "FAR", "CSI", "bias", "hits", "false alarms", "misses"].map((x) => h("th", {}, x)))),
      h("tbody", {}, ...rows.map((r) => h("tr", {}, h("td", {}, r.variant), h("td", {}, `+${r.lead_min}`), ...["pod", "far", "csi", "bias"].map((k) => h("td", {}, fmt(r[k]))), h("td", {}, r.hits), h("td", {}, r.false_alarms), h("td", {}, r.misses))))));
  };
  sel.addEventListener("change", draw); draw();
}

// ------------------------------------------------------------------ TRAIN page
$("#fit-btn").addEventListener("click", async () => {
  const ds = dsById($("#train-ds").value), box = $("#fit-out"), btn = $("#fit-btn");
  btn.disabled = true; box.replaceChildren(h("h2", {}, "Memory per level"), h("p", { class: "hint" }, "Measuring (a few training steps per level, ~2–4 min)…"));
  try {
    const job = await runJob("fitcheck", { grid: ds?.grid });
    const rows = job.result.rows;
    box.replaceChildren(h("h2", {}, `Memory per level · grid ${ds?.grid?.join("×")}`), h("table", {}, h("thead", {}, h("tr", {}, ...["level", "VAE", "alignment", "diffusion", "verdict"].map((x) => h("th", {}, x)))),
      h("tbody", {}, ...rows.map((r) => h("tr", {}, h("td", {}, r.level), h("td", {}, r.vae), h("td", {}, r.alignment), h("td", {}, r.diffusion), h("td", {}, r.verdict))))),
      h("p", {}, job.result.recommended ? `Recommended: level ${job.result.recommended}` : "Nothing fits: use a smaller region or downscale."));
    if (job.result.recommended) $("#train-level").value = job.result.recommended;
  } catch (err) { toast(err.message); }
  btn.disabled = false;
});
$("#train-form").addEventListener("submit", async (e) => {
  e.preventDefault(); const f = Object.fromEntries(new FormData(e.target)), log = $("#train-log");
  const years = (s) => s ? s.split(",").map((x) => x.trim()).filter(Boolean) : [];
  const p = { dataset: f.dataset, name: f.name || undefined, compute_level: +f.compute_level, max_epochs: +f.max_epochs, split: f.split, train_years: years(f.train_years), val_years: years(f.val_years) };
  log.textContent = ""; const btn = $("button[type=submit]", e.target); btn.disabled = true;
  try {
    const job = await runJob("train", p, { onLog: (l) => { log.textContent += l + "\n"; if (log.textContent.length > 200000) log.textContent = log.textContent.slice(-150000); log.scrollTop = log.scrollHeight; } });
    toast(`Model ${job.result.model} trained`); await refreshLists();
  } catch (err) { toast("Training: " + err.message); }
  btn.disabled = false;
});

// ------------------------------------------------------------------ MODELS page
async function renderModels() {
  const out = $("#models-out");
  out.replaceChildren(h("table", { class: "card" }, h("thead", {}, h("tr", {}, ...["model", "region (lat / lon)", "grid", "level", "latent ch", "diffusion steps", "size", "trained", ""].map((x) => h("th", {}, x)))),
    h("tbody", {}, ...S.models.map((m) => h("tr", {}, h("td", {}, m.name), h("td", {}, `${m.lat_range?.join("–")}°N / ${m.lon_range?.join("–")}°E`), h("td", {}, m.grid.join("×")), h("td", {}, m.compute_level), h("td", {}, m.latent_channels), h("td", {}, m.timesteps), h("td", {}, `${m.size_mb} MB`), h("td", {}, m.card?.trained || "–"),
      h("td", {}, h("button", { onclick: () => showCurves(m) }, "Training curves")))))));
}
async function showCurves(m) {
  const box = $("#model-curves"); box.replaceChildren(h("p", { class: "hint" }, "loading…"));
  const data = await api(`/api/models/${m.id}/metrics`); box.replaceChildren();
  if (!Object.keys(data).length) return box.append(h("p", { class: "hint" }, "No metrics files saved with this model (older runs did not keep metrics_history.json)."));
  const charts = h("div", { class: "charts" }); box.append(charts);
  for (const [path, d] of Object.entries(data)) {
    const hist = Array.isArray(d) ? d : (d.history || d.epochs || null);
    if (!Array.isArray(hist) || !hist.length) { box.append(h("details", { class: "card" }, h("summary", {}, path), h("pre", { class: "log" }, JSON.stringify(d, null, 1).slice(0, 20000)))); continue; }
    const xs = hist.map((r, i) => r.epoch ?? i), keys = Object.keys(hist[0]).filter((k) => k !== "epoch" && typeof hist[0][k] === "number");
    const losses = keys.filter((k) => /loss|mse|mae/i.test(k)), skills = keys.filter((k) => !losses.includes(k)).slice(0, 6);
    if (losses.length) lineChart(charts, `${path.split("/").slice(-3).join("/")} — losses`, xs, losses.map((k) => ({ label: k, data: hist.map((r) => r[k]) })), { yMax: Math.max(...losses.flatMap((k) => hist.map((r) => r[k] || 0))) * 1.05, w: 460 });
    if (skills.length) lineChart(charts, `${path.split("/").slice(-3).join("/")} — skill`, xs, skills.map((k) => ({ label: k, data: hist.map((r) => r[k]) })), { w: 460 });
  }
}

// ------------------------------------------------------------------ JOBS page
let logES = null;
async function renderJobs() {
  const js = await api("/api/jobs"), list = $("#jobs-list");
  list.replaceChildren(...js.map((j) => h("div", { class: "job", tabindex: 0, onclick: () => showLog(j.id) },
    h("div", {}, h("b", {}, j.type), " ", h("span", { class: `status-${j.status}` }, j.status), j.status === "running" ? ` ${(j.progress * 100).toFixed(0)}%` : ""),
    h("div", { class: "mono" }, new Date(j.created * 1000).toLocaleString()), j.error ? h("div", { class: "status-failed mono" }, j.error) : null,
    (j.status === "running" || j.status === "queued") ? h("button", { onclick: async (e) => { e.stopPropagation(); await api(`/api/jobs/${j.id}/cancel`, { method: "POST" }); renderJobs(); } }, "Cancel") : null)));
  if (!js.length) list.append(h("p", { class: "hint" }, "No jobs yet."));
}
function showLog(id) {
  logES?.close(); const pre = $("#job-log"); pre.textContent = "";
  logES = new EventSource(`/api/jobs/${id}/events`);
  logES.addEventListener("log", (e) => { pre.textContent += JSON.parse(e.data).line + "\n"; pre.scrollTop = pre.scrollHeight; });
  logES.addEventListener("status", (e) => { const d = JSON.parse(e.data); pre.textContent += `[${d.status}]${d.error ? " " + d.error : ""}\n`; });
}

// ------------------------------------------------------------------ tabs + init
function openPage(name) {
  for (const b of $$("nav button")) b.setAttribute("aria-selected", String(b.dataset.page === name));
  for (const s of $$(".page")) s.hidden = s.id !== `page-${name}`;
  if (name === "jobs") renderJobs();
  if (name === "models") renderModels();
  if (name === "predict") fillStarts($("#pred-model"), $("#pred-ds"), $("#pred-t0"));
  if (name === "simulate") fillSim();
  history.replaceState(null, "", `#${name}`);
}
$$("nav button").forEach((b) => b.addEventListener("click", () => openPage(b.dataset.page)));
(async () => {
  await refreshLists();
  if (S.datasets.length) await selectDataset(S.datasets[0].id);
  openPage(location.hash.slice(1) || "data");
  pollBadge();
})();
