// Full-screen forecast viewer: forecast (or another layer) next to the observation, one large map each,
// stepping or playing through the inputs and lead times. Opened by clicking any forecast thumbnail.
import { h, latLabel, lonLabel } from "./core.js";
import { MapView } from "./mapview.js";
import { colorbar } from "./field.js";

// steps: [{ top, sub, isInput, frames: { layerKey: {a, shape} | null } }]
// layers: [[key, label]] choices for the left map; the right map always shows "observed"
export function openFullView({ ds, title, steps, layers, luts, index = 0, left = "forecast" }) {
  const dlg = h("dialog", { class: "fullview", "aria-label": "Forecast viewer" });
  const leftSel = h("select", { "aria-label": "Left map layer" }, layers.map(([k, l]) => h("option", { value: k, selected: k === left }, l)));
  const stepLab = h("strong", { class: "mono" }), slider = h("input", { type: "range", min: 0, max: steps.length - 1, value: index, "aria-label": "Time step" });
  const play = h("button", { type: "button", class: "primary", title: "Play / pause (Space)" }, "▶ Play");
  const prev = h("button", { type: "button", title: "Previous (←)", "aria-label": "Previous step" }, "◀"), next = h("button", { type: "button", title: "Next (→)", "aria-label": "Next step" }, "▶");
  const close = h("button", { type: "button", title: "Close (Esc)" }, "Close");
  const speed = h("select", { "aria-label": "Speed" }, [["900", "slow"], ["500", "normal"], ["250", "fast"]].map(([v, l]) => h("option", { value: v, selected: v === "500" }, l)));
  const pane = (name) => {
    const el = h("div", { class: "fv-map" }), lab = h("div", { class: "map-ui fv-lab" }), read = h("div", { class: "map-ui readout" }, "Move over the map"), cb = h("div");
    el.append(lab, read, cb);
    return { name, el, lab, read, cb };
  };
  const L = pane("left"), R = pane("right");
  dlg.append(h("div", { class: "fv-bar" }, h("span", { class: "fv-title" }, title), stepLab, prev, play, next, slider, h("label", { class: "inline" }, h("span", { class: "lab" }, "Left"), leftSel),
    h("label", { class: "inline" }, h("span", { class: "lab" }, "Speed"), speed), h("span", { class: "spacer" }), close),
    h("div", { class: "fv-body" }, L.el, R.el),
    h("p", { class: "hint fv-help" }, "← → step  ·  Space play/pause  ·  scroll to zoom, drag to pan (both maps move together)  ·  Esc close"));
  document.body.append(dlg); dlg.showModal();

  const mk = (P, other) => new MapView(P.el, {
    margin: 0.95,
    onHover: (p) => {
      for (const Q of [L, R]) {
        if (!p) { Q.read.textContent = "Move over the map"; continue; }
        const map = Q === L ? mapL : mapR, u = map.valueAt(p.lon, p.lat), lut = Q.lut;
        Q.read.textContent = `${latLabel(p.lat)} ${lonLabel(p.lon)}   ${u === null || u === undefined || !lut ? "–" : lut.value(u)}`;
      }
      const o = P === L ? mapR : mapL; o.cursor = p ? [p.lon, p.lat] : null; o.invalidate();
    } });
  const mapL = mk(L, R), mapR = mk(R, L);
  mapL.setExtent(ds, ds); mapR.setExtent(ds, ds);
  // keep both maps on the same view when one is zoomed or panned
  for (const [a, b] of [[mapL, mapR], [mapR, mapL]]) {
    const orig = a.invalidate.bind(a);
    a.invalidate = () => { if (a.view !== b.view && !a._syncing) { b._syncing = true; b.view = { ...a.view }; b.userMoved = a.userMoved; b.invalidate(); b._syncing = false; } orig(); };
  }

  let i = index, timer = null;
  function show(k) {
    i = Math.max(0, Math.min(steps.length - 1, k)); slider.value = i;
    const s = steps[i];
    stepLab.textContent = `${s.top}  ${s.sub}${s.isInput ? "  (observed input)" : ""}`;
    const lk = s.isInput ? "observed" : leftSel.value;
    for (const [P, map, key] of [[L, mapL, lk], [R, mapR, "observed"]]) {
      const f = s.frames[key], lut = luts[key === "observed" || key === "forecast" ? "rain" : key];
      P.lut = lut;
      P.lab.textContent = key === "observed" ? (s.isInput ? "Observed (input)" : "Observed (IMERG)") : layers.find(([k]) => k === key)?.[1] || key;
      if (f) map.setField(f.a, f.shape, lut); else map.setField(null);
      P.cb.replaceChildren(lut ? colorbar(lut, "map-ui vert") : "");
      if (!f && key === "observed") P.lab.textContent = "No observation for this time";
    }
  }
  function toggle() {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ Play"; return; }
    play.textContent = "❚❚ Pause";
    timer = setInterval(() => show(i + 1 >= steps.length ? 0 : i + 1), +speed.value);
  }
  speed.addEventListener("change", () => { if (timer) { toggle(); toggle(); } });
  prev.onclick = () => show(i - 1); next.onclick = () => show(i + 1); play.onclick = toggle;
  slider.addEventListener("input", () => show(+slider.value)); leftSel.addEventListener("change", () => show(i));
  const end = () => { clearInterval(timer); dlg.close(); dlg.remove(); };
  close.onclick = end; dlg.addEventListener("cancel", (e) => { e.preventDefault(); end(); });
  dlg.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" && e.target.type !== "range") return;
    if (e.key === "ArrowRight") { e.preventDefault(); show(i + 1); } else if (e.key === "ArrowLeft") { e.preventDefault(); show(i - 1); }
    else if (e.key === " ") { e.preventDefault(); toggle(); }
  });
  requestAnimationFrame(() => show(i));
}
