// Download: choose a box on the map (or type it), a period and a product; the server downloads IMERG and preprocesses it.
import { h, S, on, runJob, shell, field, select, toast, loadLists, regionLabel } from "../core.js";
import { MapView } from "../mapview.js";

const INDIA = { lat_range: [4, 38], lon_range: [64, 100] };
const PRESETS = [["South India", [5.0, 17.8], [71.0, 83.8]], ["Northeast India", [19.1, 31.9], [86.1, 98.9]]];

export async function mount(root) {
  const { side, work } = shell(root, { fill: true });
  const mapEl = h("div"), readout = h("div", { class: "map-ui readout" }, "Drag on the map to draw the region"), logEl = h("pre", { class: "log", hidden: true, style: "max-height:180px;border-radius:0;border-width:1px 0 0" });
  work.append(mapEl, logEl);
  const inp = (attrs) => h("input", { type: "number", step: "0.1", required: true, ...attrs });
  const lat0 = inp({ value: "5.0", min: "-60", max: "60" }), lat1 = inp({ value: "17.8", min: "-60", max: "60" }), lon0 = inp({ value: "71.0", min: "-180", max: "180" }), lon1 = inp({ value: "83.8", min: "-180", max: "180" });
  const name = h("input", { required: true, pattern: "[A-Za-z0-9_.-]+", placeholder: "kerala_aug2024" }), title = h("input", { placeholder: "Kerala floods, Aug 2024" });
  const product = select([["IMERG_FINAL_HH_V07", "IMERG Final Run, half-hourly V07"], ["IMERG_LATE_HH_V07", "IMERG Late Run, half-hourly V07"], ["IMERG_EARLY_HH_V07", "IMERG Early Run, half-hourly V07"]]);
  const start = h("input", { type: "datetime-local", step: "1800", required: true }), end = h("input", { type: "datetime-local", step: "1800", required: true });
  const down = select([[1, "1 (0.1°, native)"], [2, "2 (0.2°)"], [4, "4 (0.4°)"]]), gridInfo = h("p", { class: "hint" });
  const drawBtn = h("button", { type: "button", "aria-pressed": "true", onclick: () => setDraw(!map.drawMode) }, "Draw box on map");
  const btn = h("button", { class: "primary", type: "submit" }, "Download and preprocess"), prog = h("progress", { max: "1", value: "0", hidden: true }), status = h("div", { class: "hint", role: "status" });
  const box = () => ({ lat_range: [+lat0.value, +lat1.value], lon_range: [+lon0.value, +lon1.value] });
  const map = new MapView(mapEl, { margin: 0.92, onBox: (b) => { [lat0.value, lat1.value] = b.lat_range.map((v) => v.toFixed(1)); [lon0.value, lon1.value] = b.lon_range.map((v) => v.toFixed(1)); update(); },
    onHover: (p) => { readout.textContent = p ? `${p.lat.toFixed(2)}°N  ${p.lon.toFixed(2)}°E   ${map.drawMode ? "drag to draw the region" : "drag to pan"}` : "Drag on the map to draw the region"; } });
  mapEl.append(readout, map.zoomControls());
  function setDraw(on_) { map.drawMode = on_; drawBtn.setAttribute("aria-pressed", String(on_)); drawBtn.classList.toggle("primary", on_); drawBtn.textContent = on_ ? "Drawing: drag on the map" : "Draw box on map"; }
  function update() {
    const b = box(), f = +down.value, ok = b.lat_range[1] > b.lat_range[0] && b.lon_range[1] > b.lon_range[0];
    const nLat = Math.floor(Math.round((b.lat_range[1] - b.lat_range[0]) * 10) / f), nLon = Math.floor(Math.round((b.lon_range[1] - b.lon_range[0]) * 10) / f);
    const frames = start.value && end.value ? Math.max(0, Math.round((Date.parse(end.value) - Date.parse(start.value)) / 1800000) + 1) : 0;
    gridInfo.textContent = ok ? `Grid ${nLat} × ${nLon} cells at ${(0.1 * f).toFixed(1)}°` + (nLat % 16 || nLon % 16 ? ". A 128 × 128 grid (12.8° box at 0.1°) matches the released models; multiples of 16 are the safest." : ".") + (frames ? ` ${frames.toLocaleString()} half-hourly files.` : "")
      : "South must be below north and west below east.";
    map.boxes = [...S.datasets.map((d) => ({ lat_range: d.lat_range, lon_range: d.lon_range, kind: "ref", label: d.id })).filter((d, i, all) => all.findIndex((o) => String(o.lat_range) + o.lon_range === String(d.lat_range) + d.lon_range) === i),
      ...(ok ? [{ ...b, kind: "sel", label: `${nLat} × ${nLon}` }] : [])];
    map.invalidate();
  }
  async function submit(e) {
    e.preventDefault();
    const b = box(), p = { name: name.value, title: title.value || name.value, ...b, start: start.value, end: end.value, product: product.value, downscale_factor: +down.value };
    btn.disabled = true; prog.hidden = false; prog.removeAttribute("value"); logEl.hidden = false; logEl.textContent = ""; status.textContent = "Queued…";
    try {
      await runJob("download", p, { onProgress: (d) => { prog.value = d.value; status.textContent = `Downloading ${(d.value / 0.8 * 100).toFixed(0)} %`; },
        onLog: (l) => { logEl.textContent += l + "\n"; if (logEl.textContent.length > 200000) logEl.textContent = logEl.textContent.slice(-150000); logEl.scrollTop = logEl.scrollHeight; } });
      status.textContent = "Dataset ready. Select it in the top bar."; toast(`Dataset ${p.name} is ready`); await loadLists(true);
    } catch (err) { status.textContent = "Failed: " + err.message; toast(err.message); }
    btn.disabled = false; prog.hidden = true;
  }
  const preset = (lab, la, lo) => h("button", { type: "button", onclick: () => { [lat0.value, lat1.value] = la; [lon0.value, lon1.value] = lo; update(); } }, lab);
  side.append(h("form", { onsubmit: submit, oninput: update },
    h("section", {}, h("h3", {}, "Region"), h("div", { style: "display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px" }, drawBtn, PRESETS.map((p) => preset(...p))),
      h("div", { class: "row2" }, field("Latitude south (°N)", lat0), field("Latitude north (°N)", lat1)), h("div", { class: "row2" }, field("Longitude west (°E)", lon0), field("Longitude east (°E)", lon1)),
      field("Downscale factor", down), gridInfo),
    h("section", {}, h("h3", {}, "Product and period"), field("Product", product), field("Start (UTC)", start), field("End (UTC)", end),
      h("p", { class: "hint" }, "Final Run is the research-quality product (about 3.5 months behind real time); Late and Early are near real time.")),
    h("section", {}, h("h3", {}, "Dataset"), field("Identifier", name, "Letters, digits, _ . - ; becomes artifacts/data/<identifier>"), field("Display name", title), btn, h("div", { style: "height:8px" }), prog, status,
      h("p", { class: "hint" }, "Needs a NASA Earthdata login in ~/.netrc on the server. Raw files are kept, so the region can be re-cut later in Train."))));
  map.setExtent(INDIA); setDraw(true); update();
  on("lists", update);
  return { show() { update(); }, key(e) { if (e.key === "+" || e.key === "=") map.zoom(1.5); else if (e.key === "-") map.zoom(1 / 1.5); else if (e.key === "0") map.fit(); else return false; return true; } };
}
