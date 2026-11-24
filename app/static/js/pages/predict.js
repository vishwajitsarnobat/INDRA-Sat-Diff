// Predict: run one forecast and inspect it: input, forecast, observed, difference, ensemble probability and spread, skill vs persistence.
import { h, api, S, on, emit, fillModels as fill, timeline, validStarts, runJob, shell, field, check, select, segmented, chip, fmt, utc, isoParam, toast, dsById, modelById, latLabel, lonLabel, regionLabel } from "../core.js";
import { rainLut, probLut, spreadLut, diffLut, diffField, drawThumb, colorbar } from "../field.js";
import { geoLines } from "../geo.js";
import { chart } from "../charts.js";
import { openFullView } from "../fullview.js";

export async function loadArrays(jobId, names) {
  const r = await fetch(`/api/jobs/${jobId}/arrays?names=${names.map(encodeURIComponent).join(",")}`);
  if (!r.ok) throw new Error("The arrays of this forecast are no longer on disk.");
  const buf = new Uint8Array(await r.arrayBuffer()), out = {};
  let o = 0;
  for (const [name, shape] of JSON.parse(r.headers.get("X-Layout"))) { const n = shape.reduce((a, b) => a * b, 1); out[name] = { a: buf.subarray(o, o + n), shape }; o += n; }
  return out;
}
const mean = (rows, k) => { const v = rows.map((x) => x[k]).filter((x) => Number.isFinite(x)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const SCORES = [["csi", "CSI", "Critical success index: hits / (hits + misses + false alarms). 1 is perfect.", 1], ["pod", "POD", "Probability of detection: hits / (hits + misses). 1 is perfect.", 1],
  ["far", "FAR", "False alarm ratio: false alarms / (hits + false alarms). 0 is perfect.", -1], ["bias", "Bias", "Frequency bias: forecast area / observed area. 1 is unbiased.", 0]];

// Renders a finished predict job into `host`. compact: no input row, no table (used by Simulate).
export async function renderResult(host, job, { compact = false } = {}) {
  const r = job.result, p = job.params, ds = dsById(p.dataset), clip = ds?.clip_value || 100;
  host.replaceChildren(h("p", { class: "hint" }, "Loading forecast…"));
  const arr = await loadArrays(job.id, r.arrays.filter((n) => n !== "ensemble"));
  const lines = ds ? await geoLines(ds) : null, lonLat = (ds?.array_order || "lon_lat") === "lon_lat";
  const t0 = Date.parse(r.times_context.at(-1)) / 1000, hhmm = (t) => new Date(t * 1000).toISOString().slice(11, 16);
  const thresholds = r.arrays.filter((n) => n.startsWith("exceed_")).map((n) => n.slice(7)).sort((a, b) => a - b);
  if (arr.truth) arr.diff = { a: diffField(arr.forecast.a, arr.truth.a, clip), shape: arr.truth.shape };
  const st = { rows: { forecast: true, truth: !!arr.truth, diff: !!arr.truth && !compact, prob: thresholds.length > 0 && !compact, spread: false }, thr: thresholds.includes("2.5") ? "2.5" : thresholds[0], member: 0, skillThr: "2.5" };
  const luts = { rain: rainLut(clip), prob: probLut(), spread: spreadLut(), diff: diffLut(clip) };

  const probe = h("div", { class: "probe", "aria-live": "off" }, "Move over a panel for values");
  const grids = h("div"), legends = h("div");
  const nCols = Math.max(compact ? 0 : r.times_context.length, r.lead_minutes.length);
  function strip(g, rows, cols, head) {
    if (head) g.append(h("div", { class: "blockhead" }, head));
    g.append(h("div"), ...Array.from({ length: nCols }, (_, k) => (cols[k] ? h("div", { class: "cl" + (cols[k].t0 ? " t0" : "") }, cols[k].top, h("br"), cols[k].sub) : h("div"))));
    for (const row of rows) {
      g.append(h("div", { class: "rl" }, row.label, h("small", {}, row.sub)));
      const [T, n0, n1] = row.data.shape, nLon = lonLat ? n0 : n1, nLat = lonLat ? n1 : n0;
      for (let k = 0; k < nCols; k++) {
        if (k >= T || !cols[k]) { g.append(h("div")); continue; }
        const f = row.data.a.subarray(k * n0 * n1, (k + 1) * n0 * n1);
        const cv = h("canvas", { width: 256, height: 256, role: "img", "aria-label": `${row.label}, ${cols[k].top} ${cols[k].sub}` });
        drawThumb(cv, f, [n0, n1], row.lut, { lonLat, lines, ds });
        cv.title = "Click to open full size";
        cv.addEventListener("click", () => openFull(row.key, row.isInput ? k : nIn + k));
        cv.addEventListener("mousemove", (e) => {
          const b = cv.getBoundingClientRect(), i = Math.min(nLon - 1, Math.floor((e.clientX - b.left) / b.width * nLon)), j = Math.min(nLat - 1, Math.floor((1 - (e.clientY - b.top) / b.height) * nLat));
          const u = lonLat ? f[i * n1 + j] : f[j * n1 + i];
          const where = ds ? `${latLabel(ds.lat_range[0] + (j + .5) / nLat * (ds.lat_range[1] - ds.lat_range[0]))} ${lonLabel(ds.lon_range[0] + (i + .5) / nLon * (ds.lon_range[1] - ds.lon_range[0]))}` : `cell ${i}, ${j}`;
          probe.textContent = `${row.label} ${cols[k].top} (${cols[k].sub})  ·  ${where}  ·  ${row.lut.value(u)}`;
        });
        g.append(cv);
      }
    }
  }
  let memberData = null, current = null;
  const nIn = compact ? 0 : r.times_context.length;
  function openFull(key, index) {
    const sl = (d, k) => { if (!d) return null; const [, n0, n1] = d.shape; return { a: d.a.subarray(k * n0 * n1, (k + 1) * n0 * n1), shape: [n0, n1] }; };
    const fc = current?.fcst || arr.forecast, diffArr = st.member && memberData && arr.truth ? { a: diffField(fc.a, arr.truth.a, clip), shape: fc.shape } : arr.diff;
    const steps = [
      ...(compact ? [] : r.times_context.map((t, i, all) => ({ top: i === all.length - 1 ? "T0" : `−${(all.length - 1 - i) * (r.lead_minutes[0] || 30)} min`, sub: t.replace("T", " "), isInput: true,
        frames: { observed: sl(arr.context, i) } }))),
      ...r.lead_minutes.map((m, k) => ({ top: `+${m} min`, sub: new Date((t0 + m * 60) * 1000).toISOString().slice(0, 16).replace("T", " ") + "Z",
        frames: { forecast: sl(fc, k), observed: sl(arr.truth, k), diff: sl(diffArr, k), prob: sl(arr["exceed_" + st.thr], k), spread: sl(arr.spread, k) } })),
    ];
    const layers = [["forecast", "Forecast"], ...(arr.truth ? [["diff", "Difference (forecast − observed)"]] : []),
      ...(arr["exceed_" + st.thr] ? [["prob", `Probability ≥ ${st.thr} mm/h`]] : []), ...(arr.spread ? [["spread", "Ensemble spread"]] : [])];
    openFullView({ ds, title: `Forecast from ${utc(t0)}`, steps, layers, luts, index: Math.min(index, steps.length - 1),
      left: key === "observed" ? "forecast" : key });
  }
  function draw() {
    const fcst = st.member && memberData ? memberData : arr.forecast, used = new Set(["rain"]);
    const cols = r.lead_minutes.map((m) => ({ top: `+${m} min`, sub: hhmm(t0 + m * 60) })), rows = [];
    const ppName = { pmi: "probability + intensity matched", pm: "probability-matched", prob: "probability threshold", mean: "ensemble mean", member: "single sample" };
    const fsub = st.member ? `member ${st.member + 1} of ${r.members}` : (r.members > 1 ? `${ppName[r.settings?.postprocess] || "combined"}, ${r.members} members` : "single sample");
    if (st.rows.forecast) rows.push({ key: "forecast", label: "Forecast", sub: fsub, data: fcst, lut: luts.rain });
    if (st.rows.truth && arr.truth) rows.push({ key: "observed", label: "Observed", sub: "IMERG", data: arr.truth, lut: luts.rain });
    if (st.rows.diff && arr.diff) { rows.push({ key: "diff", label: "Difference", sub: "forecast − observed", data: st.member && memberData ? { a: diffField(fcst.a, arr.truth.a, clip), shape: fcst.shape } : arr.diff, lut: luts.diff }); used.add("diff"); }
    if (st.rows.prob && arr["exceed_" + st.thr]) { rows.push({ key: "prob", label: `P(≥ ${st.thr} mm/h)`, sub: `${r.members} members`, data: arr["exceed_" + st.thr], lut: luts.prob }); used.add("prob"); }
    if (st.rows.spread && arr.spread) { rows.push({ key: "spread", label: "Spread", sub: "σ between members", data: arr.spread, lut: luts.spread }); used.add("spread"); }
    const g = h("div", { class: "strip", style: `grid-template-columns: 104px repeat(${nCols}, minmax(0, 168px))` });
    const step = r.lead_minutes[0] || 30, last = r.lead_minutes.at(-1);
    if (!compact) strip(g, [{ key: "observed", isInput: true, label: "Input", sub: "observed (IMERG)", data: arr.context, lut: luts.rain }],
      r.times_context.map((t, i, all) => ({ top: i === all.length - 1 ? "T0" : `−${(all.length - 1 - i) * step} min`, sub: t.slice(11, 16), t0: i === all.length - 1 })),
      `Observed inputs · the ${r.times_context.length} frames up to T0`);
    strip(g, rows, cols, `Forecast · ${r.lead_minutes.length} steps of ${step} min (+${step} to +${last} min) · click any map to open it full size`);
    current = { fcst, rows };
    grids.replaceChildren(g);
    legends.replaceChildren(...[...used].map((k) => colorbar(luts[k], "inline")));
  }
  const toggle = (key, label, ok = true) => ok && check(label, { checked: st.rows[key], onchange: (e) => { st.rows[key] = e.target.checked; draw(); } });
  const toolbar = h("div", { class: "toolbar" }, h("span", { class: "hint", style: "margin:0" }, "Rows"), toggle("forecast", "Forecast"), toggle("truth", "Observed", !!arr.truth), toggle("diff", "Difference", !!arr.truth),
    toggle("prob", "Probability", thresholds.length > 0), toggle("spread", "Spread", !!arr.spread),
    thresholds.length > 0 && h("label", { class: "inline" }, h("span", { class: "lab" }, "Threshold"), select(thresholds.map((t) => [t, `≥ ${t} mm/h`]), { value: st.thr, "aria-label": "Exceedance threshold", onchange: (e) => { st.thr = e.target.value; draw(); } })),
    r.members > 1 && r.arrays.includes("ensemble") && h("label", { class: "inline" }, h("span", { class: "lab" }, "Member"), select([[0, "combined"], ...Array.from({ length: r.members - 1 }, (_, i) => [i + 1, `member ${i + 2}`])], { value: 0, "aria-label": "Ensemble member", onchange: async (e) => {
      st.member = +e.target.value;
      if (st.member) { const q = await fetch(`/api/jobs/${job.id}/array/ensemble?member=${st.member}`); memberData = { a: new Uint8Array(await q.arrayBuffer()), shape: q.headers.get("X-Shape").split(",").map(Number) }; }
      draw();
    } })));
  const m = modelById(p.model);
  host.replaceChildren(
    h("div", { class: "pagehead" }, h("h1", {}, `Forecast from ${utc(t0)}`),
      h("span", { class: "sub" }, [m?.name || p.model, ds?.name || p.dataset, `${r.members} member${r.members > 1 ? "s" : ""}`, p.steps && `${p.steps} ${(p.sampler || "ddim").toUpperCase()} steps`,
        `knowledge alignment ${p.alignment === false ? "off" : "on"}`, `sampled in ${r.seconds} s`].filter(Boolean).join("  ·  ")),
      h("a", { href: `/api/jobs/${job.id}/file/result.npz`, download: `${job.id}.npz` }, "Download arrays (.npz)")),
    h("div", { class: "card" }, toolbar, grids, legends, probe));
  draw();

  // ---- skill
  if (!r.metrics) { host.append(h("p", { class: "hint" }, "No observations exist after T0 in this dataset, so this forecast cannot be verified.")); return; }
  const skill = h("div"), thrs = Object.keys(r.metrics).sort((a, b) => a - b);
  if (!r.metrics[st.skillThr]) st.skillThr = thrs[0];
  const seg = segmented(thrs.map((t) => [t, `≥ ${t}`]), st.skillThr, (v) => { st.skillThr = v; drawSkill(); }, "Rain threshold in mm/h");
  function drawSkill() {
    const mm = r.metrics[st.skillThr], kpis = h("div", { class: "kpis" }), charts = h("div", { class: "charts" });
    for (const [k, name, tip, better] of SCORES) {
      const a = mean(mm.model, k), b = mean(mm.persistence, k), d = a !== null && b !== null ? a - b : null;
      const good = d === null || !better ? "" : (d * better > 0.005 ? "up" : d * better < -0.005 ? "down" : "");
      kpis.append(h("div", { class: "kpi", title: tip }, h("div", { class: "l" }, `${name}, mean over leads`), h("div", { class: "v" }, fmt(a)),
        h("div", { class: "d" }, `persistence ${fmt(b)}`, d !== null && better ? h("span", { class: good }, `  ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)}${good === "up" ? " better" : good === "down" ? " worse" : ""}`) : null)));
    }
    skill.replaceChildren(kpis, charts);
    for (const [k, name, tip] of SCORES.slice(0, compact ? 1 : 3))
      chart(charts, { title: `${name} by lead time`, sub: tip, x: r.lead_minutes, xLabel: "Lead time (min)", yRange: [0, 1],
        series: [{ label: "Model", data: mm.model.map((x) => x[k]) }, { label: "Persistence", data: mm.persistence.map((x) => x[k]), dash: [6, 4] }] });
    if (compact) return;
    const best = (a, b, better) => (!better || a === null || b === null ? [false, false] : [(a - b) * better > 0, (b - a) * better > 0]);
    skill.append(h("div", { class: "tablewrap" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", {}, "Lead"), SCORES.flatMap(([, n]) => [h("th", {}, `${n} model`), h("th", {}, `${n} persist.`)]))),
      h("tbody", {}, r.lead_minutes.map((lead, i) => h("tr", {}, h("td", {}, `+${lead} min`), SCORES.flatMap(([k, , , better]) => {
        const a = mm.model[i][k], b = mm.persistence[i][k], [x, y] = best(a, b, better);
        return [h("td", { class: x ? "best" : "" }, fmt(a)), h("td", { class: y ? "best" : "dim" }, fmt(b))];
      })))))));
  }
  host.append(h("div", { class: "card" }, h("h2", {}, "Skill of this forecast against persistence", h("span", { class: "spacer" }), h("span", { class: "hint", style: "margin:0;font-weight:400" }, "Threshold (mm/h)"), seg), skill,
    h("p", { class: "hint" }, "Scores of a single forecast are noisy. Use Evaluate for statistics over a period. Persistence repeats the last observed frame.")));
  drawSkill();
}

// ------------------------------------------------------------------ page
export async function mount(root) {
  const { side, work } = shell(root);
  const out = h("div", {}, h("div", { class: "empty" }, "Choose a model and a start time, then run a forecast. Finished forecasts are kept and listed on the left."));
  work.append(out);
  const model = h("select", { "aria-label": "Model" }), dsName = h("div", { class: "mono" });
  const t0 = h("input", { type: "datetime-local", step: "1800", required: true }), t0Hint = h("span", { class: "hint" });
  const mismatch = h("div", { class: "notice warn", role: "alert", hidden: true });
  // Warn when the model was trained for another region than the selected data.
  function checkMatch() {
    const m = modelById(model.value), d = S.ds;
    const centre = (o) => o?.lat_range && o?.lon_range ? [(o.lat_range[0] + o.lat_range[1]) / 2, (o.lon_range[0] + o.lon_range[1]) / 2] : null;
    const cm = centre(m), cd = centre(d);
    mismatch.hidden = true; mismatch.replaceChildren();
    if (!m || !d) return;
    if (!cm) { mismatch.hidden = false; mismatch.append("This model's training region is not recorded; check that it suits this data."); return; }
    const far = Math.hypot(cm[0] - cd[0], cm[1] - cd[1]) > 3;
    if (!far) return;
    const better = S.models.find((x) => { const c = centre(x); return c && Math.hypot(c[0] - cd[0], c[1] - cd[1]) <= 3; });
    mismatch.hidden = false;
    mismatch.append(`${m.name} was trained for another region (${regionLabel(m)}) than this data (${regionLabel(d)}); its forecasts will not be meaningful here.`,
      better ? h("button", { type: "button", style: "margin-top:6px", onclick: () => { model.value = better.id; modelDefaults(); checkMatch(); refreshStarts(true); } }, `Use ${better.name}`) : null);
  }
  const verify = check("Only times with observations to verify against", { checked: true });
  const align = check("Knowledge alignment (guided sampling)", { checked: true });
  const members = h("input", { type: "number", min: "1", max: "16", value: "4" }), steps = h("input", { type: "number", min: "5", max: "1000", value: "20" });
  const post = select([["pmi", "Probability matching + intensities (validated)"], ["pm", "Probability matching, area only"], ["prob", "Probability threshold"], ["mean", "Ensemble mean"], ["member", "Single sample"]]);
  const kaMult = select([["1", "1 × input mean (validated)"], ["1.5", "1.5 ×"], ["2", "2 × (SAC evaluation)"]]);
  const precision = select(["fp16", "bf16", "fp32"]), seed = h("input", { type: "number", min: "0", placeholder: "random" });
  const runBtn = h("button", { class: "primary", type: "submit" }, "Run forecast"), prog = h("progress", { max: "1", value: "0", hidden: true }), status = h("div", { class: "hint", role: "status" });
  const past = h("div", { class: "list" });
  let tl = null, starts = [], currentJob = null;

  const form = h("form", { onsubmit: submit },
    field("Model", model), mismatch, field("Dataset", dsName, "Change it in the top bar."),
    field("Forecast from (UTC, last observed frame)", t0, t0Hint), verify, align,
    h("div", { class: "row2" }, field("Ensemble members", members), field("Sampler steps", steps)),
    field("Combine members into one forecast", post, "How the ensemble becomes one map; scored against IMERG."),
    field("Knowledge-alignment target", kaMult),
    h("div", { class: "row2" }, field("Precision", precision), field("Seed", seed)),
    runBtn, h("div", { style: "height:8px" }), prog, status);
  side.append(h("section", {}, h("h3", {}, "New forecast"), form), h("section", {}, h("h3", {}, "Past forecasts"), past));

  const toLocal = (t) => new Date(t * 1000).toISOString().slice(0, 16), fromLocal = (s) => Date.parse(s + ":00Z") / 1000;
  async function refreshStarts(keep) {
    const m = modelById(model.value), ds = S.ds;
    dsName.textContent = ds ? ds.name : "none";
    if (!m || !ds) { t0Hint.textContent = "Needs a model and a dataset."; starts = []; return; }
    tl = await timeline(ds.id);
    starts = validStarts(tl, m.in_len, verify.input.checked ? m.out_len : 0);
    if (!starts.length) { t0Hint.textContent = `No run of ${m.in_len + (verify.input.checked ? m.out_len : 0)} consecutive frames in this dataset.`; return; }
    t0.min = toLocal(tl.times[starts[0]]); t0.max = toLocal(tl.times[starts.at(-1)]);
    if (!keep || !t0.value) t0.value = toLocal(tl.times[starts[Math.floor(starts.length / 2)]]);
    t0Hint.textContent = `${starts.length} valid start times, ${utc(tl.times[starts[0]], false)} to ${utc(tl.times[starts.at(-1)])}. Uses the ${m.in_len} frames up to this time; forecasts ${m.out_len} × ${tl.interval / 60} min.`;
  }
  const fillModels = () => fill(model, (m) => `${m.name} · ${m.grid.join("×")} · level ${m.compute_level}`);
  async function refreshPast() {
    const js = (await api("/api/jobs?type=predict&brief=true&limit=200")).filter((j) => j.status === "done");
    past.replaceChildren(...js.map((j) => h("button", { type: "button", "aria-current": String(j.id === currentJob), onclick: () => open(j.id) },
      h("span", {}, utc(j.params.t0)), h("small", {}, `${j.params.model} · ${j.params.members || 1} member(s) · ${j.params.alignment === false ? "no KA" : "KA"}`))));
    if (!js.length) past.append(h("p", { class: "hint" }, "None yet."));
    return js;
  }
  async function open(id) {
    try {
      const job = await api(`/api/jobs/${id}`);
      if (job.status !== "done") throw new Error(`Job ${id} is ${job.status}.`);
      currentJob = id; history.replaceState(null, "", `#predict?job=${id}`);
      await renderResult(out, job); work.scrollTop = 0;
      [...past.children].forEach((b, i) => b.setAttribute?.("aria-current", "false")); refreshPast();
    } catch (e) { toast(e.message); }
  }
  async function submit(e) {
    e.preventDefault();
    if (!starts.length) return toast("No valid start time in this dataset for this model.");
    const want = fromLocal(t0.value), i = starts.reduce((a, b) => (Math.abs(tl.times[b] - want) < Math.abs(tl.times[a] - want) ? b : a));
    if (Math.abs(tl.times[i] - want) > 1) { t0.value = toLocal(tl.times[i]); toast(`Start time moved to the nearest valid one: ${utc(tl.times[i])}`); }
    const p = { model: model.value, dataset: S.ds.id, t0: isoParam(tl.times[i]), alignment: align.input.checked, members: +members.value || 1, steps: +steps.value || 20, precision: precision.value, sampler: "ddim", postprocess: post.value, ka_multiplier: +kaMult.value };
    if (seed.value !== "") p.seed = +seed.value;
    runBtn.disabled = true; prog.hidden = false; prog.removeAttribute("value"); status.textContent = "Queued…";
    out.replaceChildren(h("div", { class: "empty" }, `Forecasting from ${utc(tl.times[i])} with ${modelById(p.model)?.name || p.model}… this takes about ${Math.max(10, (p.members || 1) * 8)} s.`));
    try {
      const job = await runJob("predict", p, { onProgress: (d) => { prog.value = d.value; status.textContent = `Sampling ${(d.value * 100).toFixed(0)} %`; }, onStart: () => { status.textContent = "Queued (the first run loads the model)…"; } });
      status.textContent = `Done in ${job.result.seconds} s`; await open(job.id);
    } catch (err) { status.textContent = "Failed: " + err.message; toast(err.message); }
    runBtn.disabled = false; prog.hidden = true;
  }
  function modelDefaults() {         // preset the form with the settings validated for this model
    const inf = modelById(model.value)?.inference || {};
    align.input.checked = inf.use_alignment ?? true;
    if (inf.postprocess && [...post.options].some((o) => o.value === inf.postprocess)) post.value = inf.postprocess;
    if (inf.ka_multiplier !== undefined && [...kaMult.options].some((o) => +o.value === +inf.ka_multiplier)) kaMult.value = String(+inf.ka_multiplier);
    if (inf.ensemble_size) members.value = inf.ensemble_size;
    if (inf.steps) steps.value = inf.steps;
  }
  model.addEventListener("change", () => { modelDefaults(); checkMatch(); refreshStarts(true); }); verify.input.addEventListener("change", () => refreshStarts(true));
  on("dataset", () => { checkMatch(); refreshStarts(false); }); on("lists", () => { fillModels(); refreshStarts(true); });
  on("theme", () => { if (currentJob && !root.hidden) open(currentJob); });
  fillModels(); modelDefaults(); checkMatch(); await refreshStarts(false);
  const js = await refreshPast();

  return {
    async show(params) {
      if (params.t0) { await refreshStarts(true); t0.value = params.t0.slice(0, 16); }
      if (params.job) open(params.job); else if (!currentJob && !params.t0 && js.length) open(js[0].id);
    },
  };
}
