// Jobs: the single-GPU queue, with live logs. Finished jobs are kept on disk and listed after restarts.
import { h, api, on, emit, put, shell, kv, chip, utc, toast } from "../core.js";

const dur = (s) => (s == null || s < 0 ? "–" : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const what = (j) => { const p = j.params || {}; return [p.model, p.dataset, p.name, p.t0 && utc(p.t0, false)].filter(Boolean).join(" · "); };

export async function mount(root) {
  const { work } = shell(root, { side: false });
  const tbody = h("tbody"), detail = h("div", { class: "card" }, h("p", { class: "hint" }, "Select a job to see its parameters and log."));
  work.append(h("div", { class: "pagehead" }, h("h1", {}, "Jobs"), h("span", { class: "sub" }, "One job runs on the GPU at a time; the others wait in order. Finished jobs stay available after a restart.")),
    h("div", { class: "split" }, h("div", { class: "card" }, h("div", { class: "tablewrap", style: "max-height:calc(100vh - 190px)" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, ["Job", "Status", "Started (UTC)", "Duration"].map((x) => h("th", {}, x)), h("th", { class: "l" }, "Details"))), tbody))), detail));
  let current = null, es = null, timer = 0, jobs = [];

  async function refresh() {
    try { jobs = await api("/api/jobs?brief=true&limit=200"); } catch (e) { return; }
    tbody.replaceChildren(...jobs.map((j) => {
      const tr = h("tr", { tabindex: "0", class: j.id === current ? "sel" : "" }, h("td", {}, j.type), h("td", {}, chip(j.status + (j.status === "running" ? ` ${Math.round(j.progress * 100)} %` : ""), j.status)),
        h("td", {}, utc(j.created, false)), h("td", {}, dur((j.finished || (j.status === "running" ? Date.now() / 1000 : j.created)) - j.created)), h("td", { class: "l dim wrap" }, what(j)));
      tr.addEventListener("click", () => open(j.id)); tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(j.id); });
      return tr;
    }));
    if (!jobs.length) tbody.append(h("tr", {}, h("td", { colspan: "5", class: "l dim" }, "No jobs yet.")));
  }
  async function open(id) {
    current = id; es?.close(); refresh();
    let j; try { j = await api(`/api/jobs/${id}`); } catch (e) { return toast(e.message); }
    if (current !== id) return;           // a newer open() won the race; don't create a second stream
    const log = h("pre", { class: "log", tabindex: "0", "aria-label": "Job log", style: "max-height:calc(100vh - 430px)" });
    const active = j.status === "running" || j.status === "queued";
    put(detail, h("h2", {}, `${j.type}`, chip(j.status, j.status), h("span", { class: "spacer" }),
        j.type === "predict" && j.status === "done" && h("button", { onclick: () => emit("go", { page: "predict", params: { job: j.id } }) }, "Open forecast"),
        j.type === "evaluate" && j.status === "done" && h("button", { onclick: () => emit("go", { page: "evaluate", params: { job: j.id } }) }, "Open scores"),
        active && h("button", { class: "danger", onclick: async () => { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); toast("Cancel requested"); setTimeout(() => open(id), 800); } }, "Cancel job")),
      h("p", { class: "hint mono" }, j.id), j.error && h("p", { class: "err" }, j.error),
      kv(Object.entries(j.params || {}).map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)])), h("h3", {}, "Log"), log);
    let lines = 0, chars = 0;
    es?.close();
    es = new EventSource(`/api/jobs/${id}/events`);
    es.addEventListener("log", (e) => {
      const line = JSON.parse(e.data).line + "\n"; lines++; chars += line.length; log.append(line);
      if (chars > 400000) { log.textContent = log.textContent.slice(-300000); chars = 300000; }   // bounded, O(1) per line
      log.scrollTop = log.scrollHeight;
    });
    es.addEventListener("status", (e) => { const d = JSON.parse(e.data); if (["done", "failed", "cancelled"].includes(d.status)) { es.close(); if (!lines) log.textContent = "This job wrote no log lines."; if (active) { open(id); emit("jobs"); } } });
    es.onerror = () => es.close();
  }
  on("jobs", () => { if (!root.hidden) refresh(); });
  return {
    async show(params) { await refresh(); clearInterval(timer); timer = setInterval(refresh, 3000); if (params.job) open(params.job); else if (!current && jobs.length) open(jobs[0].id); },
    hide() { clearInterval(timer); es?.close(); },
  };
}
