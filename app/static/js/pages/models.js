// Models: registry with model cards and training curves.
import { h, S, api, on, shell, kv, chip, regionLabel, toast, emit } from "../core.js";
import { chart } from "../charts.js";

const ORDER = ["loss", "total_loss", "rec_loss", "nll_loss", "kl_loss", "mse_epoch", "mae_epoch", "mse", "mae", "csi", "pod", "far", "bias"];
const rank = (n) => { const i = ORDER.findIndex((o) => n === o || n.startsWith(o)); return i < 0 ? 99 : i; };

export async function mount(root) {
  const { work } = shell(root, { side: false });
  const cards = h("div", { class: "cards" }), detail = h("div");
  work.append(h("div", { class: "pagehead" }, h("h1", {}, "Models"), h("span", { class: "sub" }, "Trained models under artifacts/models. Select one for its card and training curves.")), cards, detail);
  let current = null;

  function draw() {
    cards.replaceChildren(...S.allModels.map((m) => h("button", { class: "mcard", "aria-current": String(m.id === current), onclick: () => open(m.id) },
      h("div", { class: "n" }, h("span", {}, m.name), m.usable === false ? chip("incomplete", "warn") : (m.card?.notes || "").includes("INCOMPLETE") ? chip("partial", "warn") : chip("ready", "ok")),
      kv([["Grid", m.grid ? m.grid.join(" × ") : m.card?.latent ? `latent ${m.card.latent}` : null], ["Sequence", m.in_len ? `${m.in_len} in → ${m.out_len} out` : null], ["Compute level", m.compute_level != null ? String(m.compute_level) : null],
        ["Weights", m.size_mb ? `${m.size_mb.toLocaleString()} MB` : null], ["Region", m.lat_range ? regionLabel(m) : null]]), m.card?.notes && h("p", {}, m.card.notes))));
    if (!S.allModels.length) cards.append(h("div", { class: "empty" }, "No models yet. Train one, or copy a model folder into artifacts/models."));
  }
  async function open(id) {
    current = id; draw();
    const m = S.allModels.find((x) => x.id === id); if (!m) return;
    detail.replaceChildren(h("p", { class: "hint" }, "Loading…"));
    let curves = {};
    try { curves = await api(`/api/models/${id}/curves`); } catch (e) { toast(e.message); }
    if (current !== id) return;
    const c = m.card || {};
    detail.replaceChildren(h("div", { class: "card" }, h("h2", {}, m.name, h("span", { class: "spacer" }), m.usable !== false && h("button", { onclick: () => emit("go", { page: "predict", params: {} }) }, "Use in Predict")),
      h("div", { class: "split" }, kv([["Folder", m.dir], ["Framework", c.framework], ["Trained", c.trained], ["Dataset", c.dataset], ["Data span", c.data_span?.join(" – ")], ["Source", c.source], ["Stages complete", c.stage_complete?.join(", ")],
        ["Latent", c.latent || (m.latent_channels ? `${m.latent_channels} channels` : null)], ["Diffusion steps", m.timesteps != null ? String(m.timesteps) : null], ["Sequence", m.in_len ? `${m.in_len} frames in, ${m.out_len} out` : null]]),
        h("div", {}, h("p", { style: "margin-top:0" }, c.notes || "No notes in model_card.yaml."), m.config && h("p", { class: "hint mono" }, m.config)))));
    const stages = Object.entries(curves);
    if (!stages.length) { detail.append(h("div", { class: "empty" }, "No training history was saved with this model (older runs did not keep metrics_history.json).")); return; }
    for (const [stage, d] of stages) {
      const bases = [...new Set(Object.keys(d.series).map((k) => k.replace(/^(train|val)\//, "")))].filter((b) => ["train/" + b, "val/" + b, b].some((k) => (d.series[k] || []).filter((v) => v !== null).length > 1)
        && !/logits|d_weight|disc_factor|logvar|g_loss|disc_loss/.test(b)).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
      const host = h("div", { class: "charts" }), more = bases.length > 6 ? h("button", { onclick: () => { more.remove(); plot(bases.slice(6)); } }, `Show ${bases.length - 6} more series`) : null;
      const plot = (list) => { for (const b of list) chart(host, { title: b, x: d.epochs, xLabel: "Epoch", height: 170,
        series: [["val/" + b, "Validation", 0], ["train/" + b, "Training", 1], [b, b, 2]].filter(([k]) => d.series[k]).map(([k, label, color]) => ({ label, color, data: d.series[k] })) }); };
      detail.append(h("div", { class: "card" }, h("h2", {}, `Training curves: ${stage}`, h("span", { class: "spacer" }), h("span", { class: "hint", style: "margin:0;font-weight:400" }, `${d.epochs.length} epochs · ${d.path}`)), host, more));
      plot(bases.slice(0, 6));
    }
  }
  on("lists", draw); draw();
  return { show() { if (!current && S.allModels.length) open((S.allModels.find((m) => m.metrics?.length) || S.allModels[0]).id); } };
}
