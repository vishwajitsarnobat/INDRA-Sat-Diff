// App shell: navigation with lazily imported pages, theme, clock, job indicator, global shortcuts.
import { $, $$, h, api, icon, S, on, emit, loadLists, store, toast } from "./core.js";

const PAGES = [
  ["viewer", "Viewer", () => import("./pages/viewer.js")], ["predict", "Predict", () => import("./pages/predict.js")],
  ["simulate", "Simulate", () => import("./pages/simulate.js")], ["evaluate", "Evaluate", () => import("./pages/evaluate.js")],
  ["train", "Train", () => import("./pages/train.js")], ["download", "Download", () => import("./pages/download.js")],
  ["models", "Models", () => import("./pages/models.js")], ["jobs", "Jobs", () => import("./pages/jobs.js")],
];
const mounted = new Map();
let current = null;

export async function go(name, params = {}) {
  if (name === "data") name = "viewer";                              // old link
  const def = PAGES.find((p) => p[0] === name) || PAGES[0]; name = def[0];
  for (const b of $$("#nav button")) b.setAttribute("aria-selected", String(b.dataset.page === name));
  if (current && current !== name) { mounted.get(current)?.api.hide?.(); mounted.get(current).root.hidden = true; }
  current = name;
  const q = new URLSearchParams(params).toString();
  history.replaceState(null, "", `#${name}${q ? "?" + q : ""}`);
  if (!mounted.has(name)) {
    const root = h("section", { id: `page-${name}`, role: "tabpanel", "aria-label": def[1] });
    $("#app").append(root); mounted.set(name, { root, api: {} });
    try { mounted.get(name).api = (await (await def[2]()).mount(root)) || {}; }
    catch (e) { console.error(e); root.append(h("div", { class: "work pad" }, h("p", { class: "err" }, `Could not load this section: ${e.message}`))); }
  }
  const m = mounted.get(name);
  if (current !== name) return;
  m.root.hidden = false; m.api.show?.(params); emit("layout");
}
on("go", ({ page, params }) => go(page, params));

// ------------------------------------------------------------------ top bar
$("#nav").append(...PAGES.map(([id, label], i) => h("button", { role: "tab", "data-page": id, title: `${label} (${i + 1})`, onclick: () => go(id) }, label)));
$("#ds-select").addEventListener("change", (e) => { S.ds = S.datasets.find((d) => d.id === e.target.value); store.set("ds", S.ds.id); emit("dataset", S.ds); });
function setTheme(t) {
  document.documentElement.dataset.theme = t; store.set("theme", t);
  try { localStorage.setItem("indra.theme", t); } catch (e) { /* ignore */ }
  $("#theme-btn").replaceChildren(icon(t === "dark" ? "sun" : "moon")); emit("theme", t);
}
const toggleTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
$("#theme-btn").addEventListener("click", toggleTheme);
$("#theme-btn").replaceChildren(icon(document.documentElement.dataset.theme === "dark" ? "sun" : "moon"));
$("#help-btn").addEventListener("click", () => $("#help").showModal());
$("#jobs-ind").addEventListener("click", () => go("jobs"));
const tick = () => { $("#clock").textContent = new Date().toISOString().slice(0, 19).replace("T", " ") + " UTC"; };
tick(); setInterval(tick, 1000);

async function pollJobs() {
  if (document.hidden) return;
  try {
    const js = await api("/api/jobs?brief=true&limit=100"), run = js.filter((j) => j.status === "running"), q = js.filter((j) => j.status === "queued").length;
    $("#jobs-ind").classList.toggle("busy", run.length + q > 0);
    $("#jobs-ind-text").textContent = run.length ? `${run[0].type} ${Math.round(run[0].progress * 100)}%${q ? ` +${q}` : ""}` : q ? `${q} queued` : "Idle";
  } catch (e) { $("#jobs-ind-text").textContent = "Offline"; }
}
setInterval(pollJobs, 4000); on("jobs", pollJobs);

// ------------------------------------------------------------------ shortcuts
addEventListener("keydown", (e) => {
  const t = e.target, typing = t.matches?.("input:not([type=checkbox]):not([type=range]), select, textarea") || t.isContentEditable;
  if (typing || e.ctrlKey || e.metaKey || e.altKey || $("#help").open) return;
  if (mounted.get(current)?.api.key?.(e)) { e.preventDefault(); return; }
  if (e.key === "?") $("#help").showModal();
  else if (e.key === "t" || e.key === "T") toggleTheme();
  else if (e.key === "l" || e.key === "L") mounted.get(current)?.root.toggleSide?.();
  else if (/^[1-8]$/.test(e.key)) go(PAGES[+e.key - 1][0]);
});

// ------------------------------------------------------------------ start
(async () => {
  try { await loadLists(); } catch (e) { toast("Server not reachable: " + e.message); }
  const [name, q] = location.hash.slice(1).split("?");
  go(name || "viewer", Object.fromEntries(new URLSearchParams(q || "")));
  pollJobs();
})();
