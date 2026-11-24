// Simulate: replay an event. Forecasts are issued at regular times through the window and verified against what happened.
import { h, S, on, fillModels, timeline, validStarts, runJob, api, shell, field, check, select, fmt, utc, isoParam, toast, modelById } from "../core.js";
import { chart, updateChart } from "../charts.js";
import { renderResult } from "./predict.js";

export async function mount(root) {
  const { side, work } = shell(root);
  const model = h("select", { "aria-label": "Model" }), dsName = h("div", { class: "mono" });
  const from = h("input", { type: "datetime-local", step: "1800", required: true }), to = h("input", { type: "datetime-local", step: "1800", required: true });
  const every = select([[2, "1 h"], [4, "2 h"], [6, "3 h"], [12, "6 h"]], { value: 4 }), steps = h("input", { type: "number", min: "5", max: "200", value: "25" });
  const thr = select(["1.0", "2.5", "7.6", "16.0"].map((t) => [t, `≥ ${t} mm/h`]), { value: "2.5" });
  const align = check("Knowledge alignment", { checked: true });
  const runBtn = h("button", { class: "primary", type: "submit" }, "Run replay"), stopBtn = h("button", { type: "button", hidden: true, onclick: () => { stop = true; if (running) api(`/api/jobs/${running}/cancel`, { method: "POST" }).catch(() => {}); } }, "Stop");
  const status = h("div", { class: "hint", role: "status" }), prog = h("progress", { max: "1", value: "0", hidden: true }), plan = h("span", { class: "hint" });
  const head = h("div"), chartHost = h("div", { class: "charts" }), tableHost = h("div"), latest = h("div");
  work.append(h("div", { class: "pagehead" }, h("h1", {}, "Event replay"), h("span", { class: "sub" }, "Forecasts issued through an event, each verified against the observations that followed.")),
    head, chartHost, tableHost, latest);
  head.append(h("div", { class: "empty" }, "Choose a window and run the replay. Each forecast is a job on the GPU queue; results appear here as they finish."));
  let tl = null, starts = [], stop = false, running = null;
  const toLocal = (t) => new Date(t * 1000).toISOString().slice(0, 16), fromLocal = (s) => Date.parse(s + ":00Z") / 1000;

  function chosen() {
    const a = fromLocal(from.value), b = fromLocal(to.value), ev = +every.value;
    const inWin = starts.filter((i) => tl.times[i] >= a - 1 && tl.times[i] <= b + 1);
    return inWin.filter((i) => (i - inWin[0]) % ev === 0);
  }
  const showPlan = () => { if (tl && from.value && to.value) plan.textContent = `${chosen().length} forecasts will be issued.`; };
  async function refresh() {
    const m = modelById(model.value), ds = S.ds; dsName.textContent = ds ? ds.name : "none";
    if (!m || !ds) return;
    tl = await timeline(ds.id); starts = validStarts(tl, m.in_len, m.out_len);
    if (!starts.length) { plan.textContent = "No complete input + verification windows in this dataset."; return; }
    from.min = to.min = toLocal(tl.times[starts[0]]); from.max = to.max = toLocal(tl.times[starts.at(-1)]);
    from.value = toLocal(tl.times[starts[0]]); to.value = toLocal(tl.times[starts[Math.min(starts.length - 1, 24)]]);
    showPlan();
  }
  const fill = () => fillModels(model);
  for (const el of [from, to, every]) el.addEventListener("change", showPlan);
  model.addEventListener("change", refresh); on("dataset", refresh); on("lists", () => { fill(); refresh(); });

  async function submit(e) {
    e.preventDefault();
    const list = chosen(), m = modelById(model.value), ds = S.ds, T = thr.value;
    if (!list.length) return toast("No complete windows in that range.");
    stop = false; runBtn.disabled = true; stopBtn.hidden = false; prog.hidden = false; prog.value = 0;
    const hist = { t: [], model: [], pers: [], jobs: [] };
    head.replaceChildren(); chartHost.replaceChildren(); tableHost.replaceChildren(); latest.replaceChildren();
    const fmtT = (t) => utc(t, false).slice(8).replace(" ", "d ");
    const c = chart(chartHost, { title: `Mean CSI of each forecast (≥ ${T} mm/h, all lead times)`, sub: "One point per issue time. Higher is better.", x: [], xLabel: "Issue time (day, UTC)", xFmt: fmtT, xAtData: true, yRange: [0, 1], height: 220,
      series: [{ label: "Model", data: [] }, { label: "Persistence", data: [], dash: [6, 4] }] });
    const tbody = h("tbody");
    tableHost.append(h("div", { class: "card" }, h("h2", {}, "Issued forecasts"), h("div", { class: "tablewrap", style: "max-height:220px" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, ["Issued (UTC)", "CSI model", "CSI persistence", "POD model", "FAR model", "Sampling"].map((x) => h("th", {}, x)))), tbody))));
    const avg = (rows, k) => { const v = rows.map((x) => x[k]).filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
    for (const [n, i] of list.entries()) {
      if (stop) break;
      status.textContent = `Forecast ${n + 1} of ${list.length}: ${utc(tl.times[i])}`;
      try {
        const job = await runJob("predict", { model: m.id, dataset: ds.id, t0: isoParam(tl.times[i]), alignment: align.input.checked, steps: +steps.value || 25, members: 1, precision: "fp16", seed: 0 },
          { onStart: (j) => { running = j.id; }, onProgress: (d) => { prog.value = (n + d.value) / list.length; } });
        const met = job.result.metrics?.[T] || Object.values(job.result.metrics || {})[0];
        if (!met) throw new Error("No verification data for " + utc(tl.times[i]));
        hist.t.push(tl.times[i]); hist.model.push(avg(met.model, "csi")); hist.pers.push(avg(met.persistence, "csi"));
        updateChart(c, { x: hist.t, series: [{ label: "Model", data: hist.model }, { label: "Persistence", data: hist.pers, dash: [6, 4] }] });
        const row = h("tr", { tabindex: "0", title: "Show this forecast" }, h("td", {}, utc(tl.times[i], false)), h("td", {}, fmt(hist.model.at(-1))), h("td", { class: "dim" }, fmt(hist.pers.at(-1))),
          h("td", {}, fmt(avg(met.model, "pod"))), h("td", {}, fmt(avg(met.model, "far"))), h("td", {}, `${job.result.seconds} s`));
        const showIt = () => { [...tbody.children].forEach((r) => r.classList.toggle("sel", r === row)); renderResult(latest, job, { compact: true }); };
        row.addEventListener("click", showIt); row.addEventListener("keydown", (ev) => { if (ev.key === "Enter") showIt(); });
        tbody.prepend(row); showIt(); prog.value = (n + 1) / list.length;
      } catch (err) { if (!stop) toast(err.message); break; }
    }
    running = null; status.textContent = stop ? "Replay stopped." : `Replay finished: ${hist.t.length} forecasts.`;
    runBtn.disabled = false; stopBtn.hidden = true; prog.hidden = true;
  }
  side.append(h("section", {}, h("h3", {}, "Replay"), h("form", { onsubmit: submit },
    field("Model", model), field("Dataset", dsName, "Change it in the top bar."),
    field("From (UTC)", from), field("To (UTC)", to),
    h("div", { class: "row2" }, field("Issue every", every), field("Sampler steps", steps)),
    field("Score threshold", thr), align, plan, h("div", { style: "display:flex;gap:8px;margin:8px 0" }, runBtn, stopBtn), prog, status)));
  fill(); await refresh();
  return {};
}
