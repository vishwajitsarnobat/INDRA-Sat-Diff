// Add data: point at a folder of your own gridded rain files; the app inspects, previews and imports them.
// Only two inputs are required (folder, name); everything else is detected and can be changed under "Advanced".
import { h, S, api, emit, runJob, shell, field, select, toast, loadLists, kv, put, regionLabel } from "../core.js";
import { MapView } from "../mapview.js";
import { rainLut, colorbar, toMM } from "../field.js";

const UNITS = [
  ["1", "mm/h (rain rate)"], ["2", "mm per 30 minutes"], ["4", "mm per 15 minutes"],
  ["3600", "kg m⁻² s⁻¹ or mm/s"], ["0.041667", "mm per day"], ["1000", "m per hour"], ["custom", "Other (enter factor)"],
];

export async function mount(root) {
  const { side, work } = shell(root, { fill: true });
  const mapEl = h("div"), readout = h("div", { class: "map-ui readout" }, "The preview of your data appears here");
  const cbar = h("div"), logEl = h("pre", { class: "log", hidden: true, style: "max-height:160px;border-radius:0;border-width:1px 0 0" });
  work.append(mapEl, logEl);
  const map = new MapView(mapEl, { margin: 0.9, onHover: (p) => { readout.textContent = p ? `${p.lat.toFixed(2)}°N  ${p.lon.toFixed(2)}°E${p.u !== null && p.u !== undefined ? `   ${toMM(p.u).toFixed(2)} mm/h` : ""}` : "Move over the map for values"; } });
  mapEl.append(readout, cbar, map.zoomControls());
  map.setExtent({ lat_range: [4, 38], lon_range: [64, 100] });

  // ---- required
  const path = h("input", { required: true, placeholder: "/home/you/data/insat_hem_2024", spellcheck: "false", autocomplete: "off" });
  const name = h("input", { required: true, pattern: "[A-Za-z0-9_][A-Za-z0-9_.\\-]*", placeholder: "insat_hem_jul2024" });
  const inspectBtn = h("button", { type: "button", class: "primary", onclick: inspect }, "Check files");
  // ---- detected (editable)
  const variable = select([]), units = select(UNITS), factor = h("input", { type: "number", step: "any", value: "1", hidden: true });
  const order = select([["lon_lat", "[longitude, latitude]"], ["lat_lon", "[latitude, longitude]"]]);
  const latS = num("5"), latN = num("17.8"), lonW = num("71"), lonE = num("83.8");
  const boxRow = h("div", { hidden: true }, h("p", { class: "hint warn" }, "These files have no latitude/longitude inside. Enter the area they cover (outer edges):"),
    h("div", { class: "row2" }, field("South (°N)", latS), field("North (°N)", latN)), h("div", { class: "row2" }, field("West (°E)", lonW), field("East (°E)", lonE)));
  const title = h("input", { placeholder: "INSAT-3DS HEM, July 2024" });
  const crop = h("input", { type: "checkbox" }), cropLab = h("label", { class: "check" }, crop, "Keep only a region (draw it on the map)");
  const down = select([[1, "Keep native resolution"], [2, "Half resolution (2× coarser)"], [4, "Quarter resolution (4× coarser)"]]);
  const fill = h("input", { type: "number", step: "any", placeholder: "auto" });
  const questions = h("ul", { style: "margin:4px 0 0 16px;padding:0" }), questionsBox = h("div", { class: "notice warn", hidden: true }, h("strong", {}, "Needs your answer"), questions);
  const summary = h("div"), sanity = h("p", { class: "hint", role: "status" }), status = h("div", { class: "hint", role: "status" });
  const importBtn = h("button", { class: "primary", type: "submit", disabled: true }, "Import data"), prog = h("progress", { max: "1", value: "0", hidden: true });
  const openBtn = h("button", { type: "button", hidden: true }, "Open in Viewer");
  let info = null, cropBox = null;

  function num(v) { return h("input", { type: "number", step: "0.01", value: v }); }
  const unitsFactor = () => units.value === "custom" ? +factor.value || 1 : +units.value;
  const box = () => ({ lat_range: [+latS.value, +latN.value], lon_range: [+lonW.value, +lonE.value] });
  const varInfo = () => info?.variables.find((v) => v.name === variable.value);
  const hasCoords = () => !!(info?.lat_var && info?.lon_var);
  const params = () => ({ path: info?.path || path.value, variable: variable.value, lat_var: info?.lat_var, lon_var: info?.lon_var,
    time_var: info?.time_var, time_units: info?.time_units, order: order.value, factor: unitsFactor(),
    fill: fill.value === "" ? varInfo()?.fill : +fill.value, box: hasCoords() ? null : box() });

  async function inspect() {
    if (!path.value.trim()) { path.reportValidity(); return; }
    inspectBtn.disabled = true; status.textContent = "Reading your files…"; put(summary); importBtn.disabled = true;
    try {
      info = await api("/api/sources/inspect", { body: { path: path.value } });
      if (!info.variables.length) throw new Error("No 2-D numeric variable found in the files.");
      variable.replaceChildren(...info.variables.map((v) => h("option", { value: v.name, selected: v.name === info.suggested }, `${v.name}  ${v.shape.join("×")}${v.units ? "  [" + v.units + "]" : ""}`)));
      applyVariable();
      boxRow.hidden = hasCoords();
      if (!name.value) name.value = (info.path.split("/").filter(Boolean).pop() || "my_data").replace(/[^A-Za-z0-9_.-]/g, "_").replace(/^\.+/, "");
      put(summary, kv([
        ["Files", `${info.n_files.toLocaleString()} (e.g. ${info.examples[0]})`],
        ["Period (UTC)", info.start ? `${info.start.replace("T", " ")} → ${info.end.replace("T", " ")}` : "times not found; files are ordered by name"],
        ["Time step", info.interval_minutes ? `${info.interval_minutes} min${info.gaps ? `, ${info.gaps} gap(s)` : ", no gaps"}` : (info.note || "–")],
        ["Grid", hasCoords() ? `${info.nlat} × ${info.nlon} cells, ${fmtDeg(info.dlat)} × ${fmtDeg(info.dlon)}` : "no coordinates in files"],
        ["Area", hasCoords() ? regionLabel(info) : "enter below"],
        ["Times from", info.time_source === "file" ? "time variable inside the files" : info.time_source === "filename" ? "file names" : "not found"],
      ]));
      put(questions, ...(info.questions || []).map((q) => h("li", {}, q)));
      questionsBox.hidden = !(info.questions || []).length;
      status.textContent = "Check the preview on the map, then name the dataset and import.";
      await preview();
    } catch (e) { status.textContent = e.message; info = null; }
    inspectBtn.disabled = false;
  }
  const fmtDeg = (d) => d ? `${(+d).toFixed(3).replace(/0+$/, "").replace(/\.$/, "")}°` : "?";

  function applyVariable() {
    const v = varInfo(); if (!v) return;
    if (v.order) order.value = v.order;
    const f = v.units_factor; const match = UNITS.find(([x]) => x !== "custom" && Math.abs(+x - (f ?? 1)) < 1e-6);
    units.value = match ? match[0] : "1"; factor.hidden = units.value !== "custom";
  }

  async function preview() {
    if (!info) return;
    const p = params();
    try {
      const r = await fetch("/api/sources/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) });
      if (!r.ok) { const t = await r.text(); throw new Error((() => { try { return JSON.parse(t).detail; } catch (e) { return t; } })()); }
      const shape = r.headers.get("X-Shape").split(",").map(Number), ext = JSON.parse(r.headers.get("X-Extent")), st = JSON.parse(r.headers.get("X-Stats"));
      const data = new Uint8Array(await r.arrayBuffer()), lut = rainLut(100), ds = { ...ext, array_order: "lon_lat" };
      map.setExtent(ds, ds); map.setField(data, shape, lut); cbar.replaceChildren(colorbar(lut, "map-ui vert"));
      sanity.className = "hint"; sanity.textContent = `First file: max ${st.max_mm_h.toFixed(1)} mm/h, rain (≥ 0.1 mm/h) over ${(st.wet_fraction * 100).toFixed(0)} % of the area.`;
      if (st.max_mm_h > 0 && (st.max_mm_h < 0.05 || st.max_mm_h > 500)) {
        sanity.className = "hint warn"; sanity.textContent += " That looks unusual for rain; check the units under “Advanced”.";
      }
      sanity.textContent += " If the rain does not line up with the coastlines, use “Swap latitude/longitude”.";
      importBtn.disabled = false;
    } catch (e) { sanity.className = "hint err"; sanity.textContent = "Preview failed: " + e.message; importBtn.disabled = true; }
  }

  async function submit(e) {
    e.preventDefault();
    if (!info) return;
    const p = { ...params(), name: name.value, title: title.value || name.value, interval_minutes: info.interval_minutes || 30,
      crop: crop.checked && cropBox ? [cropBox.lat_range, cropBox.lon_range] : null, downscale: +down.value };
    importBtn.disabled = true; prog.hidden = false; prog.value = 0; logEl.hidden = false; logEl.textContent = ""; status.textContent = "Importing…"; openBtn.hidden = true;
    try {
      const job = await runJob("import", p, { onProgress: (d) => { prog.value = d.value; status.textContent = `Importing ${(d.value * 100).toFixed(0)} %`; },
        onLog: (l) => { logEl.textContent += l + "\n"; } });
      const r = job.result || {};
      status.textContent = `Done: ${r.frames?.toLocaleString?.() || "?"} frames, ${r.start || ""} → ${r.end || ""}. It is now in the dataset list under “Your data”.`
        + (r.skipped ? ` ${r.skipped} file(s) could not be read and were skipped (e.g. ${r.skipped_examples?.[0] || ""}).` : "");
      toast(`Dataset ${p.name} is ready`); await loadLists(true);
      openBtn.hidden = false; openBtn.onclick = () => { const d = S.datasets.find((x) => x.id === r.dataset); if (d) { S.ds = d; emit("dataset", d); } emit("go", { page: "viewer", params: {} }); };
    } catch (err) { status.textContent = "Import failed: " + err.message; }
    importBtn.disabled = false; prog.hidden = true;
  }

  map.onBox = (b) => { cropBox = b; crop.checked = true; map.boxes = [{ ...b, kind: "sel", label: "kept" }]; map.invalidate(); };
  crop.addEventListener("change", () => { map.drawMode = crop.checked; if (!crop.checked) { cropBox = null; map.boxes = []; map.invalidate(); } });
  variable.addEventListener("change", () => { applyVariable(); preview(); });
  for (const el of [units, factor, order, latS, latN, lonW, lonE, fill]) el.addEventListener("change", () => { factor.hidden = units.value !== "custom"; preview(); });
  path.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); inspect(); } });
  const swap = h("button", { type: "button", onclick: () => { order.value = order.value === "lon_lat" ? "lat_lon" : "lon_lat"; preview(); } }, "Swap latitude/longitude");

  side.append(h("form", { onsubmit: submit },
    h("section", {}, h("h3", {}, "1  Your files"),
      field("Folder with your data *", path, "One HDF5 or netCDF4 file per time step on a latitude/longitude grid (e.g. IMERG, INSAT-3DS HEM, GSMaP, ERA5)."),
      inspectBtn, h("div", { style: "height:8px" }), status),
    h("section", {}, h("h3", {}, "2  What we found"), summary, questionsBox,
      field("Rain variable", variable, "Picked automatically; change it if your file has several."), boxRow, sanity,
      h("div", { style: "display:flex;gap:6px;flex-wrap:wrap" }, swap)),
    h("section", {}, h("h3", {}, "3  Save as a dataset"),
      field("Dataset name *", name, "Short, no spaces. Used in the dataset list."),
      importBtn, openBtn, h("div", { style: "height:8px" }), prog,
      h("details", { style: "margin-top:12px" }, h("summary", { class: "hint", style: "cursor:pointer" }, "Advanced (detected automatically; change only if needed)"),
        field("Units of the values", units, "Converted to mm/h. Detected from the file when possible."), factor,
        field("Array order", order), cropLab, field("Resolution", down), field("Missing-value code", fill, "Values equal to this are treated as no data."),
        field("Display name", title))),
    h("p", { class: "hint" }, "* required. Your original files are only read, never changed.")));
  return { show() {}, key(e) { if (e.key === "+" || e.key === "=") map.zoom(1.5); else if (e.key === "-") map.zoom(1 / 1.5); else if (e.key === "0") map.fit(); else return false; return true; } };
}
