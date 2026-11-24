// Evaluate: categorical scores of the model (with / without knowledge alignment) against persistence over a period.
import { h, S, on, fillModels, api, timeline, validStarts, runJob, shell, field, check, select, segmented, fmt, utc, isoParam, toast, modelById } from "../core.js";
import { chart } from "../charts.js";

const NAMES = { with_ka: "Model + KA", no_ka: "Model, no KA", persistence: "Persistence" }, COLOR = { with_ka: 0, no_ka: 2, persistence: 1 };
const SCORES = [["csi", "CSI", "Critical success index. 1 is perfect.", [0, 1]], ["pod", "POD", "Probability of detection. 1 is perfect.", [0, 1]],
  ["far", "FAR", "False alarm ratio. 0 is perfect.", [0, 1]], ["bias", "Frequency bias", "Forecast area / observed area. 1 is unbiased.", [0, 2]]];

export function renderEval(host, job) {
  const res = job.result, p = job.params || {};
  const thrs = [...new Set(res.table.map((r) => r.threshold_mm))].sort((a, b) => a - b), variants = [...new Set(res.table.map((r) => r.variant))];
  let thr = thrs.includes(2.5) ? 2.5 : thrs[0];
  const charts = h("div", { class: "charts" }), tableBox = h("div", { class: "tablewrap" });
  const csv = () => { const keys = Object.keys(res.table[0]); const blob = new Blob([[keys.join(","), ...res.table.map((r) => keys.map((k) => r[k] ?? "").join(","))].join("\n")], { type: "text/csv" });
    const a = h("a", { href: URL.createObjectURL(blob), download: `scores_${p.model || "model"}_${p.dataset || "data"}.csv` }); a.click(); URL.revokeObjectURL(a.href); };
  function draw() {
    const rows = res.table.filter((r) => r.threshold_mm === thr), leads = [...new Set(rows.map((r) => r.lead_min))].sort((a, b) => a - b);
    const cell = (v, lead, k) => rows.find((r) => r.variant === v && r.lead_min === lead)?.[k];
    charts.replaceChildren();
    for (const [k, name, tip, range] of SCORES)
      chart(charts, { title: `${name} by lead time, ≥ ${thr} mm/h`, sub: tip, x: leads, xLabel: "Lead time (min)", yRange: range,
        series: variants.map((v) => ({ label: NAMES[v] || v, color: COLOR[v], data: leads.map((l) => cell(v, l, k)), dash: v === "persistence" ? [6, 4] : undefined })) });
    tableBox.replaceChildren(h("table", { class: "data" }, h("thead", {}, h("tr", {}, ["Variant", "Lead", "POD", "FAR", "CSI", "Bias", "Hits", "False alarms", "Misses"].map((x) => h("th", {}, x)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, h("td", {}, NAMES[r.variant] || r.variant), h("td", {}, `+${r.lead_min} min`), ["pod", "far", "csi", "bias"].map((k) => h("td", {}, fmt(r[k]))),
        ["hits", "false_alarms", "misses"].map((k) => h("td", {}, Number(r[k]).toLocaleString())))))));
  }
  host.replaceChildren(
    h("div", { class: "pagehead" }, h("h1", {}, `Evaluation: ${modelById(p.model)?.name || p.model || "model"}`),
      h("span", { class: "sub" }, [`${res.forecasts} forecast start times`, p.dataset, p.steps && `${p.steps} DDIM steps`, res.dir && `saved in ${res.dir}`].filter(Boolean).join("  ·  "))),
    h("div", { class: "toolbar" }, h("span", { class: "hint", style: "margin:0" }, "Threshold (mm/h)"), segmented(thrs.map((t) => [t, `≥ ${t}`]), thr, (v) => { thr = v; draw(); }, "Rain threshold"),
      h("span", { class: "spacer" }), h("button", { onclick: csv }, "Download scores (.csv)")),
    charts, h("div", { class: "card" }, h("h2", {}, "Scores per lead time"), tableBox,
      h("p", { class: "hint" }, "Counts are grid cells summed over all forecasts. The knowledge-alignment target is computed from the input frames only (causal).")));
  draw();
}

export async function mount(root) {
  const { side, work } = shell(root);
  const out = h("div", {}, h("div", { class: "pagehead" }, h("h1", {}, "Evaluation")), h("div", { class: "empty" }, "Run an evaluation to compare the model with persistence over a period. Finished evaluations are kept and listed on the left."));
  work.append(out);
  const model = h("select", { "aria-label": "Model" }), dsName = h("div", { class: "mono" });
  const from = h("input", { type: "datetime-local", step: "1800" }), to = h("input", { type: "datetime-local", step: "1800" });
  const stride = select([[1, "30 min"], [2, "1 h"], [6, "3 h"], [12, "6 h"], [48, "24 h"]], { value: 6 }), steps = h("input", { type: "number", min: "5", max: "200", value: "20" });
  const members = h("input", { type: "number", min: "1", max: "16", value: "4" });
  const post = select([["pmi", "Probability matching + intensities (validated)"], ["pm", "Probability matching, area only"], ["prob", "Probability threshold"], ["mean", "Ensemble mean"], ["member", "Single sample"]]);
  const kaMult = select([["1", "1 × input mean (validated)"], ["2", "2 × (SAC evaluation)"]]);
  const cmp = check("Compare with and without knowledge alignment", { checked: true });
  const plan = h("p", { class: "hint" }), runBtn = h("button", { class: "primary", type: "submit" }, "Run evaluation"), prog = h("progress", { max: "1", value: "0", hidden: true }), status = h("div", { class: "hint", role: "status" });
  const past = h("div", { class: "list" });
  let tl = null, current = null;
  const toLocal = (t) => new Date(t * 1000).toISOString().slice(0, 16), fromLocal = (s) => Date.parse(s + ":00Z") / 1000;

  function showPlan() {
    const m = modelById(model.value); if (!m || !tl) return;
    const a = from.value ? fromLocal(from.value) : -Infinity, b = to.value ? fromLocal(to.value) : Infinity, st = +stride.value, ok = new Set(validStarts(tl, m.in_len, m.out_len));
    let n = 0; for (let i = m.in_len - 1; i < tl.times.length - m.out_len; i += st) if (ok.has(i) && tl.times[i] >= a && tl.times[i] <= b) n++;
    plan.textContent = `${n} start times × ${cmp.input.checked ? 2 : 1} variant(s) = ${n * (cmp.input.checked ? 2 : 1)} forecasts on the GPU.`;
  }
  async function refresh() {
    const ds = S.ds; dsName.textContent = ds ? ds.name : "none"; if (!ds) return;
    tl = await timeline(ds.id); from.min = to.min = toLocal(tl.times[0]); from.max = to.max = toLocal(tl.times.at(-1));
    from.value = toLocal(tl.times[0]); to.value = toLocal(tl.times.at(-1)); showPlan();
  }
  const fill = () => fillModels(model);
  async function refreshPast() {
    const js = (await api("/api/jobs?type=evaluate&brief=true&limit=200")).filter((j) => j.status === "done");
    past.replaceChildren(...js.map((j) => h("button", { type: "button", "aria-current": String(j.id === current), onclick: () => open(j.id) },
      h("span", {}, `${j.params.model} on ${j.params.dataset}`), h("small", {}, `${utc(j.created, false)} · every ${(j.params.stride || 6) * 30} min`))));
    if (!js.length) past.append(h("p", { class: "hint" }, "None yet."));
    return js;
  }
  async function open(id) {
    try { const job = await api(`/api/jobs/${id}`); current = id; renderEval(out, job); refreshPast(); } catch (e) { toast(e.message); }
  }
  async function submit(e) {
    e.preventDefault();
    const p = { model: model.value, dataset: S.ds.id, stride: +stride.value, steps: +steps.value || 20, compare_ka: cmp.input.checked, members: +members.value || 1, postprocess: post.value, ka_multiplier: +kaMult.value };
    if (from.value) p.start = isoParam(fromLocal(from.value)); if (to.value) p.end = isoParam(fromLocal(to.value));
    runBtn.disabled = true; prog.hidden = false; prog.value = 0; status.textContent = "Queued…";
    try {
      const job = await runJob("evaluate", p, { onProgress: (d) => { prog.value = d.value; status.textContent = d.detail || `${(d.value * 100).toFixed(0)} %`; } });
      status.textContent = `Done: ${job.result.forecasts} start times.`; await open(job.id);
    } catch (err) { status.textContent = "Failed: " + err.message; toast(err.message); }
    runBtn.disabled = false; prog.hidden = true;
  }
  for (const el of [model, from, to, stride, cmp.input]) el.addEventListener("change", showPlan);
  on("dataset", refresh); on("lists", () => { fill(); showPlan(); });
  side.append(h("section", {}, h("h3", {}, "New evaluation"), h("form", { onsubmit: submit }, field("Model", model), field("Dataset", dsName, "Change it in the top bar."),
    field("From (UTC)", from), field("To (UTC)", to), h("div", { class: "row2" }, field("Start every", stride), field("Sampler steps", steps)),
    h("div", { class: "row2" }, field("Ensemble members", members), field("KA target", kaMult)), field("Combine members", post), cmp, plan, runBtn, h("div", { style: "height:8px" }), prog, status)),
    h("section", {}, h("h3", {}, "Past evaluations"), past));
  fill(); await refresh();
  const js = await refreshPast();
  return { show(params) { if (params.job) open(params.job); else if (!current && js.length) open(js[0].id); } };
}
