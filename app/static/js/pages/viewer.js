// Viewer: full map of the selected dataset with a time slider. Frames are fetched lazily in chunks.
import { h, S, on, emit, timeline, shell, kv, check, field, utc, isoParam, latLabel, lonLabel, regionLabel, toast } from "../core.js";
import { MapView } from "../mapview.js";
import { TimeBar } from "../timebar.js";
import { FrameStore } from "../frames.js";
import { rainLut, colorbar } from "../field.js";

export async function mount(root) {
  const { side, work } = shell(root, { fill: true });
  const mapEl = h("div"), barEl = h("div"), readout = h("div", { class: "map-ui readout", "aria-live": "off" });
  const busy = h("div", { class: "busy-ind", hidden: true }), cbarHost = h("div"), note = h("div", { class: "map-ui mapnote" }, "IMERG · coastlines: Natural Earth · India boundaries: states_IND");
  work.append(mapEl, barEl);
  let ds = null, tl = null, frames = null, cur = 0, lut = rainLut(), drawn = -1, lastHover = null;
  const perf = { first: null, t0: performance.now(), draws: [] };

  const map = new MapView(mapEl, { onHover: (p) => { lastHover = p; showReadout(); } });
  mapEl.append(readout, map.zoomControls(), cbarHost, note, busy);
  const bar = new TimeBar(barEl, {
    onSeek: (i, dir) => show(i, dir), ready: (i) => frames?.has(i), need: (i, dir) => { busy.hidden = false; frames?.request(i, { dir }); },
    loaded: () => frames?.loadedRanges() || [],
  });

  // ---- side panel
  const info = h("div"), stat = h("div", { class: "hint mono" });
  const layer = (key, label) => check(label, { checked: map.layers[key], onchange: (e) => { map.layers[key] = e.target.checked; map.invalidate(); } });
  const opacity = h("input", { type: "range", min: "0.2", max: "1", step: "0.05", value: "1", "aria-label": "Precipitation opacity", oninput: (e) => { map.opacity = +e.target.value; map.invalidate(); } });
  const smooth = check("Smooth (bilinear) instead of grid cells", { onchange: (e) => { map.smooth = e.target.checked; map.invalidate(); } });
  side.append(
    h("section", {}, h("h3", {}, "Dataset"), info),
    h("section", {}, h("h3", {}, "Layers"), layer("field", "Precipitation rate"), field("Opacity", opacity), layer("coast", "Coastlines"), layer("india", "India state boundaries"), layer("border", "Other country borders (Natural Earth)"),
      layer("grat", "Graticule and scale bar"), layer("box", "Data region outline"), smooth),
    h("section", {}, h("h3", {}, "Forecast"), h("p", { class: "hint" }, "Use the frame on screen as the last observation and forecast the next hours."),
      h("button", { class: "primary", onclick: () => emit("go", { page: "predict", params: { dataset: ds.id, t0: isoParam(tl.times[cur]) } }) }, "Forecast from this time")),
    h("section", {}, h("h3", {}, "Data loading"), stat));

  function showReadout() {
    const p = lastHover;
    if (!p) { readout.textContent = "Move over the map for values"; return; }
    readout.replaceChildren(`${latLabel(p.lat).padStart(8)}  ${lonLabel(p.lon).padStart(8)}   `, p.u === null || p.u === undefined ? "outside data" : h("b", {}, lut.value(p.u)));
  }
  function showStats() {
    if (!frames) return;
    const s = frames.stats, n = frames.chunks.size * frames.chunk, d = perf.draws;
    stat.textContent = `${Math.min(n, tl.times.length)} of ${tl.times.length} frames in memory\n${(s.bytes / 1048576).toFixed(2)} MB in ${s.requests} request(s), ${s.aborted} aborted\n` +
      `${frames.frameBytes.toLocaleString()} B per frame, chunk ${s.lastMs.toFixed(0)} ms\ndraw ${d.length ? (d.reduce((a, b) => a + b, 0) / d.length).toFixed(2) : "–"} ms` +
      (perf.first !== null ? ` · first frame ${perf.first.toFixed(0)} ms` : "");
    stat.style.whiteSpace = "pre-line";
  }
  function paint(i) {
    const f = frames.get(i); if (!f) return false;
    map.setField(f, frames.shape, lut); map.draw(); drawn = i;
    perf.draws.push(map.lastMs); if (perf.draws.length > 60) perf.draws.shift();
    if (perf.first === null) perf.first = performance.now() - perf.t0;
    window.__viewerPerf = { first: perf.first, draws: perf.draws, stats: frames.stats };
    if (lastHover) { lastHover.u = map.valueAt(lastHover.lon, lastHover.lat); showReadout(); }
    return true;
  }
  function show(i, dir = 1) {
    if (!frames) return;
    cur = i; bar.setIndex(i);
    const p = frames.request(i, { dir });              // also aborts requests that are no longer near the cursor
    if (paint(i)) { busy.hidden = true; showStats(); return; }
    busy.hidden = false;
    p.then(() => { if (cur === i && paint(i)) { busy.hidden = true; showStats(); } });
  }
  async function setDataset(d) {
    if (!d) { info.replaceChildren(h("p", { class: "hint" }, "No datasets yet. Use Download to fetch IMERG for a region.")); return; }
    ds = d; bar.pause(); perf.first = null; perf.t0 = performance.now(); perf.draws = [];
    try { tl = await timeline(d.id); } catch (e) { toast(e.message); return; }
    if (ds !== d) return;
    lut = rainLut(d.clip_value || 100); cbarHost.replaceChildren(colorbar(lut, "map-ui vert"));
    frames = new FrameStore(d, tl.times.length, { onLoad: () => { bar.draw(); showStats(); } });
    map.setField(null); map.setExtent(d, d); bar.setTimeline(tl);
    info.replaceChildren(h("h2", {}, d.name), kv([["Region", regionLabel(d)], ["Grid", `${d.grid?.join(" × ")} cells`], ["Resolution", `${((d.lat_range[1] - d.lat_range[0]) / (d.grid?.[1] || 128)).toFixed(2)}°`],
      ["Period", `${utc(d.start, false)} – ${utc(d.end)}`], ["Time step", `${d.interval_minutes} min`], ["Frames", String(d.count)], ["Gaps", d.gaps ? `${d.gaps} (marked on the time bar)` : "none"],
      ["Units", "mm/h (clipped at " + (d.clip_value || 100) + ")"]]), ...[d.source && h("p", { class: "hint" }, "Source: " + d.source), d.note && h("p", { class: "hint" }, "Note: " + d.note)].filter(Boolean));
    show(0);
  }
  on("dataset", (d) => { if (!root.hidden) setDataset(d); else ds = null; });
  on("lists", () => { if (!root.hidden && S.ds?.version !== ds?.version) setDataset(S.ds); });
  showReadout();

  return {
    show(params) { if (S.ds !== ds || !frames) setDataset(S.ds); },
    hide() { bar.pause(); },
    key(e) {
      const k = e.key;
      if (k === " ") bar.toggle(); else if (k === "ArrowRight") bar.step(1); else if (k === "ArrowLeft") bar.step(-1);
      else if (k === "]") bar.jump(3600); else if (k === "[") bar.jump(-3600); else if (k === "Home") bar.step(-1e9); else if (k === "End") bar.step(1e9);
      else if (k === "+" || k === "=") map.zoom(1.5); else if (k === "-") map.zoom(1 / 1.5); else if (k === "0") map.fit();
      else return false;
      return true;
    },
  };
}
