// Bottom time slider: transport controls, UTC readout, availability track (gaps marked, loaded frames shown).
import { h, icon, cssVar, on, utc } from "./core.js";

const STEPS = [1800, 3600, 3 * 3600, 6 * 3600, 12 * 3600, 86400, 2 * 86400, 7 * 86400, 30 * 86400, 91 * 86400, 365 * 86400];
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export class TimeBar {
  // opts: onSeek(i), ready(i) -> bool, need(i, dir), loaded() -> [[i0, i1), ...]
  constructor(host, opts) {
    this.o = opts; this.tl = null; this.i = 0; this.playing = false; this.fps = 4; this.hoverT = null;
    const b = (name, title, fn, cls = "") => h("button", { class: "icon ghost " + cls, title, "aria-label": title.replace(/ \(.*/, ""), onclick: fn }, icon(name));
    this.playBtn = b("play", "Play / pause (Space)", () => this.toggle(), "play");
    this.tEl = h("div", { class: "t", "aria-live": "off" }, "–"); this.sEl = h("div", { class: "s" }, "");
    this.cv = h("canvas", { role: "slider", tabindex: "0", "aria-label": "Time", "aria-valuemin": "0" });
    this.speed = h("select", { "aria-label": "Playback speed", onchange: (e) => { this.fps = +e.target.value; if (this.playing) { this.pause(); this.play(); } } },
      [1, 2, 4, 8, 16].map((f) => h("option", { value: f, selected: f === 4 }, `${f} fps`)));
    this.loop = h("input", { type: "checkbox", checked: true, "aria-label": "Loop playback" });
    host.classList.add("timebar");
    host.append(h("div", { class: "ctl", role: "group", "aria-label": "Playback" },
        b("first", "First frame (Home)", () => this.seek(0)), b("prev", "Previous frame (←)", () => this.step(-1)), this.playBtn,
        b("next", "Next frame (→)", () => this.step(1)), b("last", "Last frame (End)", () => this.seek(this.n - 1))),
      h("div", { class: "now" }, this.tEl, this.sEl), h("div", { class: "track" }, this.cv),
      h("div", { class: "opts" }, this.speed, h("label", { class: "check", style: "margin:0" }, this.loop, "Loop")));
    new ResizeObserver(() => this.draw()).observe(this.cv);
    on("theme", () => this.draw());
    const at = (e) => { const r = this.cv.getBoundingClientRect(); return this.t0 + ((e.clientX - r.left - 8) / (r.width - 16)) * this.span; };
    let down = false;
    this.cv.addEventListener("pointerdown", (e) => { if (!this.tl) return; down = true; this.cv.setPointerCapture(e.pointerId); this.pause(); this.seek(this.nearest(at(e))); });
    this.cv.addEventListener("pointermove", (e) => { if (!this.tl) return; this.hoverT = at(e); if (down) this.seek(this.nearest(this.hoverT)); else this.draw(); });
    this.cv.addEventListener("pointerup", () => { down = false; });
    this.cv.addEventListener("pointerleave", () => { this.hoverT = null; this.draw(); });
  }
  get n() { return this.tl ? this.tl.times.length : 0; }
  setTimeline(tl) {
    this.pause(); this.tl = tl; const T = tl.times;
    this.t0 = T[0]; this.span = Math.max(tl.interval, T[T.length - 1] - T[0]);
    this.gaps = tl.gaps.map(([a, b]) => [Date.parse(a.replace("Z", ":00Z")) / 1000, Date.parse(b.replace("Z", ":00Z")) / 1000]);
    this.cv.setAttribute("aria-valuemax", String(T.length - 1));
  }
  nearest(t) {
    const T = this.tl.times; let lo = 0, hi = T.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (T[m] < t) lo = m + 1; else hi = m; }
    return lo > 0 && t - T[lo - 1] < T[lo] - t ? lo - 1 : lo;
  }
  setIndex(i) {                         // display only
    this.i = i; const T = this.tl.times;
    this.tEl.textContent = utc(T[i]); this.sEl.textContent = `frame ${i + 1} / ${T.length}  ·  ${this.tl.interval / 60} min step`;
    this.cv.setAttribute("aria-valuenow", String(i)); this.cv.setAttribute("aria-valuetext", utc(T[i]));
    this.draw();
  }
  seek(i, dir = 0) { if (!this.tl) return; i = Math.max(0, Math.min(this.n - 1, i)); this.o.onSeek(i, dir || Math.sign(i - this.i) || 1); }
  step(d) { this.pause(); this.seek(this.i + d, d); }
  jump(seconds) { if (!this.tl) return; this.pause(); this.seek(this.nearest(this.tl.times[this.i] + seconds), Math.sign(seconds)); }
  toggle() { this.playing ? this.pause() : this.play(); }
  play() {
    if (!this.tl || this.playing) return;
    this.playing = true; this.playBtn.replaceChildren(icon("pause")); this.playBtn.setAttribute("aria-pressed", "true");
    this.timer = setInterval(() => {
      let nx = this.i + 1;
      if (nx >= this.n) { if (!this.loop.checked) return this.pause(); nx = 0; }
      if (this.o.ready(nx)) this.o.onSeek(nx, 1); else this.o.need(nx, 1);      // wait for the buffer rather than skip
    }, 1000 / this.fps);
  }
  pause() { if (!this.playing) return; this.playing = false; clearInterval(this.timer); this.playBtn.replaceChildren(icon("play")); this.playBtn.setAttribute("aria-pressed", "false"); }
  draw() {
    const cv = this.cv, w = cv.clientWidth, hh = cv.clientHeight; if (!w || !this.tl) return;
    const dpr = Math.min(2, devicePixelRatio || 1); if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr); }
    const c = cv.getContext("2d"), T = this.tl.times, dt = this.tl.interval, X = (t) => 8 + ((t - this.t0) / this.span) * (w - 16);
    c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, hh);
    const muted = cssVar("--muted"), line = cssVar("--line-2"), accent = cssVar("--accent"), y = 9, bh = 9;
    // availability
    c.fillStyle = line; c.fillRect(X(this.t0), y, X(this.t0 + this.span) - X(this.t0), bh);
    c.fillStyle = cssVar("--bad");
    for (const [a, b] of this.gaps) { const x0 = X(a + dt / 2), x1 = X(b - dt / 2); c.clearRect(x0, y, Math.max(2, x1 - x0), bh); c.globalAlpha = .55; for (let x = x0; x < Math.max(x0 + 2, x1); x += 4) c.fillRect(x, y, 1.5, bh); c.globalAlpha = 1; }
    // frames in memory
    c.fillStyle = accent; c.globalAlpha = .55;
    for (const [i0, i1] of this.o.loaded?.() || []) { const x0 = X(T[i0] - dt / 2), x1 = X(T[i1 - 1] + dt / 2); c.fillRect(Math.max(8, x0), y + bh - 3, Math.min(w - 8, x1) - Math.max(8, x0), 3); }
    c.globalAlpha = 1;
    // ticks
    const step = STEPS.find((s) => (s / this.span) * (w - 16) >= 74) || STEPS.at(-1);
    c.font = "10.5px ui-monospace, Menlo, Consolas, monospace"; c.textAlign = "center"; c.textBaseline = "top"; c.fillStyle = muted; c.strokeStyle = line; c.lineWidth = 1; c.beginPath();
    for (let t = Math.ceil(this.t0 / step) * step; t <= this.t0 + this.span; t += step) {
      const d = new Date(t * 1000), x = Math.round(X(t)) + .5, mid = d.getUTCHours() === 0 && d.getUTCMinutes() === 0;
      c.moveTo(x, y + bh); c.lineTo(x, y + bh + (mid ? 7 : 4));
      const lab = step >= 365 * 86400 ? String(d.getUTCFullYear()) : step >= 86400 || mid ? `${d.getUTCDate()} ${MON[d.getUTCMonth()]}${step >= 30 * 86400 ? " " + d.getUTCFullYear() : ""}` : `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`;
      if (x > 22 && x < w - 22) c.fillText(lab, x, y + bh + 9);
    }
    c.stroke();
    // hover time + cursor
    if (this.hoverT !== null && this.hoverT >= this.t0 && this.hoverT <= this.t0 + this.span) {
      const j = this.nearest(this.hoverT), x = Math.round(X(T[j])) + .5; c.strokeStyle = muted; c.beginPath(); c.moveTo(x, y - 3); c.lineTo(x, y + bh + 3); c.stroke();
      c.fillStyle = cssVar("--ink"); const s = utc(T[j], false).slice(5), tw = c.measureText(s).width / 2 + 4; c.textBaseline = "bottom"; c.fillText(s, Math.max(tw, Math.min(w - tw, x)), hh + 1);
    }
    const x = X(T[this.i]); c.fillStyle = accent; c.fillRect(Math.round(x) - 1, y - 6, 2, bh + 12);
    c.beginPath(); c.moveTo(x - 6, y - 8); c.lineTo(x + 6, y - 8); c.lineTo(x, y - 1); c.closePath(); c.fill();
  }
}
