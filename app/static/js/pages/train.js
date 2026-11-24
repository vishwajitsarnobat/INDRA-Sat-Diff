// Train: every setting from the UI. The server writes the YAML (previewed live here) and runs the three stages.
import { h, S, on, api, runJob, shell, field, check, select, segmented, kv, chip, regionLabel, toast, loadLists, utc } from "../core.js";
import { MapView } from "../mapview.js";

const LEVELS = [[1, "Full model", "Reference configuration of the paper. About 32 GB of GPU memory at 128 × 128."], [2, "Full model, memory-saving", "Same networks with gradient checkpointing and micro-batches of one. Slower, about half the memory."],
  [4, "Half-width networks", "Narrower VAE, alignment and diffusion networks."], [8, "Quarter-width, shallow", "Small networks with a 16-channel latent."], [16, "Smallest", "Fits a 4 GB laptop GPU at 128 × 128; 500 diffusion steps."]];

export async function mount(root) {
  const { side, work } = shell(root);
  let level = 16, split = "random", fit = null, timer = 0;
  const name = h("input", { pattern: "[A-Za-z0-9_.\\-]+", placeholder: "dataset_L16" });
  const epochs = h("input", { type: "number", min: "1", max: "1000", value: "20" });
  const valR = h("input", { type: "number", min: "0.01", max: "0.4", step: "0.01", value: "0.05" }), testR = h("input", { type: "number", min: "0.01", max: "0.4", step: "0.01", value: "0.05" });
  const trainY = h("input", { placeholder: "2001-2020" }), valY = h("input", { placeholder: "2021" }), testY = h("input", { placeholder: "2022" });
  const lr = h("input", { type: "number", step: "any", min: "0", placeholder: "0.0001" }), batch = h("input", { type: "number", min: "1", placeholder: "16" });
  const regrid = check("Re-cut the raw files to another box or resolution"), rg = { lat0: 0, lat1: 0, lon0: 0, lon1: 0 };
  const num = (k) => (rg[k] = h("input", { type: "number", step: "0.1" }));
  const rgBox = h("div", { hidden: true }, h("div", { class: "row4" }, field("Lat S", num("lat0")), field("Lat N", num("lat1")), field("Lon W", num("lon0")), field("Lon E", num("lon1"))));
  const down = select([[1, "1 (0.1°, native)"], [2, "2 (0.2°)"], [4, "4 (0.4°)"]]); rgBox.append(field("Downscale factor", down), h("p", { class: "hint" }));
  const dataInfo = h("div"), levelInfo = h("p", { class: "hint" }), splitRandom = h("div", { class: "row2" }, field("Validation fraction", valR), field("Test fraction", testR));
  const splitYear = h("div", { hidden: true }, field("Train years", trainY, "Years or ranges, comma separated"), h("div", { class: "row2" }, field("Validation years", valY), field("Test years", testY)));
  const levelSeg = segmented(LEVELS.map((l) => l[0]), level, (v) => { level = v; update(); }, "Compute level");
  const splitSeg = segmented([["random", "Random"], ["year", "By year"]], split, (v) => { split = v; splitRandom.hidden = v !== "random"; splitYear.hidden = v !== "year"; update(); }, "Split");
  const fitBtn = h("button", { type: "button", onclick: runFit }, "Measure GPU memory"), startBtn = h("button", { class: "primary", type: "submit" }, "Start training");
  const prog = h("progress", { max: "1", value: "0", hidden: true }), status = h("div", { class: "hint", role: "status" });
  const yaml = h("pre", { class: "log yaml", tabindex: "0", "aria-label": "Configuration preview" }), log = h("pre", { class: "log", tabindex: "0", "aria-label": "Training log" }, "No training started from this page yet.");
  const memBox = h("div"), mapEl = h("div", { class: "minimap" });
  const map = new MapView(mapEl, { interactive: false, margin: 0.72 });

  function params() {
    const p = { dataset: S.ds?.id, name: name.value || undefined, compute_level: level, max_epochs: +epochs.value || 20, split };
    if (split === "year") Object.assign(p, { train_years: trainY.value, val_years: valY.value, test_years: testY.value });
    else Object.assign(p, { val_ratio: valR.value, test_ratio: testR.value });
    if (lr.value) p.lr = lr.value; if (batch.value) p.total_batch_size = batch.value;
    if (regrid.input.checked) p.regrid = { lat_range: [+rg.lat0.value, +rg.lat1.value], lon_range: [+rg.lon0.value, +rg.lon1.value], downscale_factor: +down.value };
    return p;
  }
  function grid() {
    const d = S.ds; if (!d) return null;
    if (!regrid.input.checked) return d.grid;
    const f = +down.value; return [Math.floor(Math.round((+rg.lat1.value - +rg.lat0.value) * 10) / f), Math.floor(Math.round((+rg.lon1.value - +rg.lon0.value) * 10) / f)];
  }
  function update() {
    const d = S.ds, L = LEVELS.find((l) => l[0] === level), g = grid();
    levelInfo.textContent = `${L[1]}. ${L[2]}`;
    name.placeholder = d ? `${d.id}_L${level}` : "";
    if (regrid.input.checked && g) {
      rgBox.lastChild.textContent = `Grid ${g[0]} × ${g[1]} cells.` + (g.some((n) => n % 16 || n < 32) ? " Sizes that are multiples of 16 (64, 96, 128) are the safest; the memory check verifies." : "");
      map.boxes = [{ lat_range: [+rg.lat0.value, +rg.lat1.value], lon_range: [+rg.lon0.value, +rg.lon1.value], kind: "sel", label: "training region" }]; map.invalidate();
    } else if (map.boxes.length) { map.boxes = []; map.invalidate(); }
    drawMem();
    clearTimeout(timer);
    timer = setTimeout(async () => {
      if (!d) { yaml.textContent = "No dataset selected."; return; }
      try { yaml.textContent = (await api("/api/train/preview", { body: params() })).yaml; yaml.classList.remove("err"); }
      catch (e) { yaml.textContent = "Cannot build the configuration: " + e.message; yaml.classList.add("err"); }
    }, 200);
  }
  function drawMem() {
    const rows = fit?.result?.rows || [], rec = fit?.result?.recommended, g = grid();
    const stale = fit && g && rows[0] && rows[0].grid.join() !== g.join();
    memBox.replaceChildren(h("table", { class: "data levels" },
      h("thead", {}, h("tr", {}, ["Level", "Networks", "VAE", "Alignment", "Diffusion", "Verdict"].map((x) => h("th", {}, x)))),
      h("tbody", {}, LEVELS.map(([lv, title]) => { const r = rows.find((x) => x.level === lv);
        return h("tr", { class: lv === level ? "sel" : "", tabindex: "0", onclick: () => { level = lv; levelSeg.set(lv); update(); } }, h("td", {}, String(lv), lv === rec ? " " : "", lv === rec ? chip("recommended", "ok") : null), h("td", { class: "l" }, title),
          h("td", {}, r?.vae || "–"), h("td", {}, r?.alignment || "–"), h("td", {}, r?.diffusion || "–"), h("td", {}, r ? chip(r.verdict, r.verdict.startsWith("fits") ? "ok" : "bad") : chip("not measured", "plain"))); }))),
      h("p", { class: "hint" }, fit ? `Measured ${utc(fit.finished || fit.created)} on grid ${rows[0]?.grid.join(" × ") || "?"}${stale ? " — the grid has changed since; measure again." : "."} Peak memory over a few real training steps per stage.`
        : "Not measured on this machine yet. “Measure GPU memory” runs a few training steps per level (2 to 4 minutes) and recommends the largest level that fits. Reference: on a 3.7 GB GPU at 128 × 128, level 16 peaks at 0.9 GB, level 8 at 1.4 GB, level 4 at 3.5 GB."));
  }
  async function runFit() {
    fitBtn.disabled = true; status.textContent = "Measuring memory per level…";
    try { fit = await runJob("fitcheck", { grid: grid() }); if (fit.result.recommended) { level = fit.result.recommended; levelSeg.set(level); } status.textContent = "Memory check finished."; }
    catch (e) { status.textContent = "Memory check failed: " + e.message; toast(e.message); }
    fitBtn.disabled = false; update();
  }
  async function submit(e) {
    e.preventDefault(); if (!S.ds) return toast("Select a dataset first.");
    startBtn.disabled = true; prog.hidden = false; prog.value = 0; log.textContent = ""; status.textContent = "Queued…";
    try {
      const job = await runJob("train", params(), { onProgress: (d) => { prog.value = d.value; status.textContent = `Training ${(d.value * 100).toFixed(0)} % (3 stages: VAE, alignment, diffusion)`; },
        onLog: (l) => { log.textContent += l + "\n"; if (log.textContent.length > 300000) log.textContent = log.textContent.slice(-200000); log.scrollTop = log.scrollHeight; } });
      status.textContent = `Model ${job.result.model} trained.`; toast(`Model ${job.result.model} is ready`); await loadLists(true);
    } catch (err) { status.textContent = "Training stopped: " + err.message; toast(err.message); }
    startBtn.disabled = false; prog.hidden = true;
  }
  function setDataset() {
    const d = S.ds;
    if (!d) { dataInfo.replaceChildren(h("p", { class: "hint" }, "No dataset. Use Download first.")); return; }
    dataInfo.replaceChildren(h("div", { class: "mono", style: "margin-bottom:6px" }, d.name), kv([["Region", regionLabel(d)], ["Grid", `${d.grid?.join(" × ")} cells`], ["Frames", `${d.count} (${d.start?.slice(0, 10)} to ${d.end?.slice(0, 10)})`]]));
    regrid.input.checked = false; regrid.input.disabled = !d.has_raw; rgBox.hidden = true;
    regrid.title = d.has_raw ? "" : "This dataset has no raw files, so its region and resolution are fixed. Use Download for another region.";
    [rg.lat0.value, rg.lat1.value] = d.lat_range; [rg.lon0.value, rg.lon1.value] = d.lon_range;
    map.setExtent(d, d); map.boxes = []; update();
  }
  regrid.input.addEventListener("change", () => { rgBox.hidden = !regrid.input.checked; update(); });
  const form = h("form", { onsubmit: submit, oninput: update },
    h("section", {}, h("h3", {}, "Data"), dataInfo, regrid, rgBox, h("p", { class: "hint" }, "Region and resolution belong to the dataset. Datasets downloaded here keep their raw files and can be re-cut; for a new region use Download.")),
    h("section", {}, h("h3", {}, "Model size"), field("Compute level", levelSeg), levelInfo, fitBtn),
    h("section", {}, h("h3", {}, "Schedule"), field("Epochs per stage", epochs), field("Train / validation / test split", splitSeg), splitRandom, splitYear,
      h("details", {}, h("summary", { class: "hint", style: "cursor:pointer" }, "Optimiser (defaults are fine)"), h("div", { class: "row2" }, field("Learning rate", lr), field("Total batch size", batch)))),
    h("section", {}, h("h3", {}, "Run"), field("Model name", name, "Saved under artifacts/models/"), startBtn, h("div", { style: "height:8px" }), prog, status));
  side.append(form);
  work.append(h("div", { class: "pagehead" }, h("h1", {}, "Train a model"), h("span", { class: "sub" }, "Three stages run in sequence: VAE, knowledge alignment, latent diffusion. Training uses the GPU queue.")),
    h("div", { class: "split wide" }, h("div", { class: "card" }, h("h2", {}, "GPU memory per compute level"), memBox), h("div", { class: "card" }, h("h2", {}, "Training region"), mapEl, h("p", { class: "hint" }, "Outline: the dataset. Shaded: a re-cut region, when enabled."))),
    h("div", { class: "split" }, h("div", { class: "card" }, h("h2", {}, "Configuration written by the server"), yaml), h("div", { class: "card" }, h("h2", {}, "Training log"), log)));
  on("dataset", setDataset); on("lists", setDataset);
  try { fit = (await api("/api/jobs?type=fitcheck&limit=20")).find((j) => j.status === "done" && j.result?.rows?.length) || null; } catch (e) { /* none */ }
  setDataset();
  return { show() { update(); } };
}
