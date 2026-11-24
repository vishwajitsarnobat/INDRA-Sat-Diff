// Map workspace: WebGL field under a 2-D overlay (coastlines, borders, graticule, boxes, scale bar).
// Plate carrée with the east-west scale corrected by cos(latitude) at the region centre.
import { h, cssVar, on, latLabel, lonLabel } from "./core.js";
import { FieldRenderer } from "./field.js";
import { geoLines } from "./geo.js";

const NICE_DEG = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30], NICE_KM = [5, 10, 20, 50, 100, 200, 500, 1000, 2000];

export class MapView {
  constructor(host, { interactive = true, onHover, onBox, margin = 0.84 } = {}) {
    Object.assign(this, { host, interactive, onHover, onBox, margin });
    host.classList.add("map");
    this.glc = h("canvas", { "aria-hidden": "true" }); this.ov = h("canvas", { class: "ov", role: "img", "aria-label": "Map" });
    host.append(this.glc, this.ov);
    this.r = new FieldRenderer(this.glc); this.ctx = this.ov.getContext("2d");
    this.view = { lon: 78, lat: 15, ppd: 20 }; this.k = 1; this.w = this.h = 1; this.userMoved = false;
    this.layers = { field: true, coast: true, india: true, border: false, grat: true, box: true }; this.opacity = 1; this.smooth = false;
    this.boxes = []; this.field = null; this.ds = null; this.ext = null; this.lines = null; this.drawMode = false; this.lastMs = 0;
    new ResizeObserver(() => this._resize()).observe(host);
    on("theme", () => this.invalidate());
    if (interactive) this._bind();
  }
  // ext: { lat_range, lon_range } to frame; ds (optional): dataset whose grid the field uses
  setExtent(ext, ds = null) {
    this.ext = ext; this.ds = ds; this.k = Math.cos(((ext.lat_range[0] + ext.lat_range[1]) / 2) * Math.PI / 180);
    this.userMoved = false; this.fit();
    geoLines(ext).then((l) => { if (this.ext === ext) { this.lines = l; this.invalidate(); } });
  }
  setField(data, shape, lut) { this.field = data ? { data, shape, lut } : null; this.invalidate(); }
  fit() {
    if (!this.ext) return;
    const [la0, la1] = this.ext.lat_range, [lo0, lo1] = this.ext.lon_range;
    this.view = { lon: (lo0 + lo1) / 2, lat: (la0 + la1) / 2, ppd: Math.min(this.w / ((lo1 - lo0) * this.k), this.h / (la1 - la0)) * this.margin };
    this.userMoved = false; this.invalidate();
  }
  zoom(f, x = this.w / 2, y = this.h / 2) {
    const [lon, lat] = this.unproject(x, y), v = this.view;
    v.ppd = Math.min(2000, Math.max(3, v.ppd * f));
    v.lon = lon - (x - this.w / 2) / (v.ppd * this.k); v.lat = lat + (y - this.h / 2) / v.ppd;
    this.userMoved = true; this.invalidate();
  }
  project(lon, lat) { const v = this.view; return [this.w / 2 + (lon - v.lon) * v.ppd * this.k, this.h / 2 - (lat - v.lat) * v.ppd]; }
  unproject(x, y) { const v = this.view; return [v.lon + (x - this.w / 2) / (v.ppd * this.k), v.lat - (y - this.h / 2) / v.ppd]; }
  valueAt(lon, lat) {
    const f = this.field, d = this.ds; if (!f || !d) return null;
    const [la0, la1] = d.lat_range, [lo0, lo1] = d.lon_range, lonLat = (d.array_order || "lon_lat") === "lon_lat";
    const [n0, n1] = f.shape, nLon = lonLat ? n0 : n1, nLat = lonLat ? n1 : n0;
    const i = Math.floor((lon - lo0) / (lo1 - lo0) * nLon), j = Math.floor((lat - la0) / (la1 - la0) * nLat);
    if (i < 0 || j < 0 || i >= nLon || j >= nLat) return null;
    return lonLat ? f.data[i * n1 + j] : f.data[j * n1 + i];
  }
  invalidate() { if (!this._raf) this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); }); }
  _resize() {
    const w = this.host.clientWidth, hh = this.host.clientHeight; if (!w || !hh) return;
    const dpr = Math.min(2, devicePixelRatio || 1), ratio = this.w > 1 ? Math.min(w / this.w, hh / this.h) : 1;
    this.w = w; this.h = hh; this.dpr = dpr;
    for (const c of [this.glc, this.ov]) { c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); }
    if (this.userMoved) this.view.ppd *= ratio; else this.fit();
    this.draw();
  }
  draw() {
    const t0 = performance.now(), { w, h: hh, ctx, dpr } = this;
    if (!this.ext || w < 2) return;
    // field
    const f = this.field, d = this.ds;
    if (f && d && this.layers.field) {
      const [x0, y0] = this.project(d.lon_range[0], d.lat_range[0]), [x1, y1] = this.project(d.lon_range[1], d.lat_range[1]);
      this.r.draw(f.data, f.shape[0], f.shape[1], f.lut, { lonLat: (d.array_order || "lon_lat") === "lon_lat", smooth: this.smooth, opacity: this.opacity,
        rect: [x0 / w * 2 - 1, 1 - y0 / hh * 2, x1 / w * 2 - 1, 1 - y1 / hh * 2] });
    } else this.r.clear();
    // overlay
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, hh);
    const v = this.view, [lonW, latN] = this.unproject(0, 0), [lonE, latS] = this.unproject(w, hh);
    const step = NICE_DEG.find((s) => s * v.ppd >= 80) || 30, ink = cssVar("--ink-2"), halo = cssVar("--map");
    if (this.layers.grat) {
      ctx.strokeStyle = cssVar("--grat"); ctx.lineWidth = 1; ctx.beginPath();
      for (let lon = Math.ceil(lonW / step) * step; lon <= lonE; lon += step) { const x = Math.round(this.project(lon, 0)[0]) + .5; ctx.moveTo(x, 0); ctx.lineTo(x, hh); }
      for (let lat = Math.ceil(latS / step) * step; lat <= latN; lat += step) { const y = Math.round(this.project(0, lat)[1]) + .5; ctx.moveTo(0, y); ctx.lineTo(w, y); }
      ctx.stroke();
    }
    if (this.lines && (this.layers.coast || this.layers.border || this.layers.india)) {
      const m = new DOMMatrix([v.ppd * this.k, 0, 0, -v.ppd, w / 2 - v.lon * v.ppd * this.k, hh / 2 + v.lat * v.ppd]);
      ctx.lineJoin = "round";
      if (this.layers.border) { const p = new Path2D(); p.addPath(this.lines.border, m); ctx.strokeStyle = cssVar("--border"); ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.stroke(p); ctx.setLineDash([]); }
      if (this.layers.india && this.lines.india) { const p = new Path2D(); p.addPath(this.lines.india, m); ctx.strokeStyle = cssVar("--border"); ctx.lineWidth = 0.8; ctx.stroke(p); }
      if (this.layers.coast) { const p = new Path2D(); p.addPath(this.lines.coast, m); ctx.strokeStyle = cssVar("--coast"); ctx.lineWidth = 1.1; ctx.stroke(p); }
    }
    const boxes = [...(d && this.layers.box ? [{ ...d, kind: "data" }] : []), ...this.boxes];
    for (const b of boxes) {
      const [x0, y1] = this.project(b.lon_range[0], b.lat_range[0]), [x1, y0] = this.project(b.lon_range[1], b.lat_range[1]);
      ctx.lineWidth = b.kind === "sel" ? 2 : 1; ctx.strokeStyle = b.kind === "ref" ? cssVar("--muted") : cssVar("--accent");
      ctx.setLineDash(b.kind === "ref" ? [5, 4] : []);
      if (b.kind === "sel") { ctx.fillStyle = cssVar("--accent-soft"); ctx.fillRect(x0, y0, x1 - x0, y1 - y0); }
      ctx.strokeRect(Math.round(x0) + .5, Math.round(y0) + .5, Math.round(x1 - x0), Math.round(y1 - y0)); ctx.setLineDash([]);
      if (b.label) { ctx.font = "11px system-ui"; ctx.fillStyle = b.kind === "ref" ? cssVar("--muted") : cssVar("--accent"); ctx.textAlign = "left"; ctx.textBaseline = b.kind === "sel" ? "top" : "bottom"; ctx.fillText(b.label, x0 + 4, b.kind === "sel" ? y0 + 5 : y0 - 3); }
    }
    if (this.layers.grat) {
      ctx.font = "11px ui-monospace, Menlo, Consolas, monospace"; ctx.lineWidth = 3; ctx.strokeStyle = halo; ctx.fillStyle = ink; ctx.lineJoin = "round";
      const dec = step < 1 ? 1 : 0, text = (s, x, y) => { ctx.strokeText(s, x, y); ctx.fillText(s, x, y); };
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      for (let lon = Math.ceil(lonW / step) * step; lon <= lonE; lon += step) { const x = this.project(lon, 0)[0]; if (x > 50 && x < w - 170) text(lonLabel(lon, dec), x, hh - 4); }
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      for (let lat = Math.ceil(latS / step) * step; lat <= latN; lat += step) { const y = this.project(0, lat)[1]; if (y > 70 && y < hh - 22) text(latLabel(lat, dec), 5, y); }
      // scale bar
      const kmPerPx = 111.32 / v.ppd, km = [...NICE_KM].reverse().find((n) => n / kmPerPx <= 130) || 5, len = km / kmPerPx, x1 = w - 14, y = hh - 12;
      ctx.lineWidth = 3; ctx.strokeStyle = halo; ctx.beginPath(); ctx.moveTo(x1 - len, y - 5); ctx.lineTo(x1 - len, y); ctx.lineTo(x1, y); ctx.lineTo(x1, y - 5); ctx.stroke();
      ctx.lineWidth = 1.2; ctx.strokeStyle = ink; ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.lineWidth = 3; ctx.strokeStyle = halo; text(`${km} km`, x1 - len / 2, y - 3);
    }
    this.lastMs = performance.now() - t0;
  }
  _bind() {
    const ov = this.ov, pos = (e) => { const r = ov.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const snap = (v) => Math.round(v * 10) / 10;
    let drag = null;
    ov.addEventListener("wheel", (e) => { e.preventDefault(); const [x, y] = pos(e); this.zoom(Math.exp(-e.deltaY * 0.0015), x, y); }, { passive: false });
    ov.addEventListener("pointerdown", (e) => {
      if (e.button) return;
      const [x, y] = pos(e); ov.setPointerCapture(e.pointerId);
      drag = this.drawMode ? { box: this.unproject(x, y) } : { x, y, lon: this.view.lon, lat: this.view.lat };
      if (!this.drawMode) this.host.classList.add("panning");
    });
    ov.addEventListener("pointermove", (e) => {
      const [x, y] = pos(e), [lon, lat] = this.unproject(x, y);
      if (drag?.box) {
        const [a, b] = drag.box, sel = { lon_range: [snap(Math.min(a, lon)), snap(Math.max(a, lon))], lat_range: [snap(Math.min(b, lat)), snap(Math.max(b, lat))] };
        if (sel.lon_range[1] > sel.lon_range[0] && sel.lat_range[1] > sel.lat_range[0]) this.onBox?.(sel);
      } else if (drag) {
        this.view.lon = drag.lon - (x - drag.x) / (this.view.ppd * this.k); this.view.lat = drag.lat + (y - drag.y) / this.view.ppd; this.userMoved = true; this.invalidate();
      }
      this.onHover?.({ lon, lat, u: this.valueAt(lon, lat) });
    });
    const end = () => { drag = null; this.host.classList.remove("panning"); };
    ov.addEventListener("pointerup", end); ov.addEventListener("pointercancel", end);
    ov.addEventListener("pointerleave", () => this.onHover?.(null));
    ov.addEventListener("dblclick", () => this.fit());
  }
  // +, −, fit buttons
  zoomControls() {
    return h("div", { class: "map-ui zoom" },
      h("button", { title: "Zoom in (+)", "aria-label": "Zoom in", onclick: () => this.zoom(1.5) }, "+"),
      h("button", { title: "Zoom out (−)", "aria-label": "Zoom out", onclick: () => this.zoom(1 / 1.5) }, "−"),
      h("button", { title: "Fit region (0)", "aria-label": "Fit region", onclick: () => this.fit() }, "⤢"));
  }
}
