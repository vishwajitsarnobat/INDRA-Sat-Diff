// Colour tables and the WebGL2 field renderer.
// A field is a Uint8Array [n0, n1] of value*255. Colouring is a 256-entry lookup done in the fragment shader.
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

// Precipitation rate, mm/h. Lower bin edges; the lightest classes are semi-transparent so drizzle stays quiet.
export const RAIN_EDGES = [0.1, 0.5, 1, 2.5, 5, 7.6, 10, 16, 25, 50];
const RAIN = [["#5b8fd0", 115], ["#4a90dc", 185], ["#2f7fe0", 255], ["#22b3c9", 255], ["#2fbf71", 255], ["#a5cf3a", 255],
  ["#f2d232", 255], ["#f59a23", 255], ["#e8492d", 255], ["#c2257f", 255]];
const css = ([hex, a = 255]) => { const [r, g, b] = rgb(hex); return `rgba(${r},${g},${b},${(a / 255).toFixed(2)})`; };

function table(fn) {            // fn(u) -> [hex, alpha] or null
  const t = new Uint8Array(1024);
  for (let u = 0; u < 256; u++) { const c = fn(u); if (c) t.set([...rgb(c[0]), c[1] ?? 255], u * 4); }
  return t;
}
const cache = new Map();
export function rainLut(clip = 100) {
  const key = "rain" + clip;
  if (!cache.has(key)) {
    const k = Math.log1p(clip);
    cache.set(key, { key, table: table((u) => { const mm = Math.expm1((u / 255) * k); let b = -1; for (let i = 0; i < RAIN_EDGES.length; i++) if (mm >= RAIN_EDGES[i]) b = i; return b < 0 ? null : RAIN[b]; }),
      legend: { title: "Precipitation rate (mm/h)", colors: RAIN.map(css), ticks: RAIN_EDGES.map(String) },
      value: (u) => `${Math.expm1((u / 255) * k).toFixed(2)} mm/h` });
  }
  return cache.get(key);
}
export const toMM = (u, clip = 100) => Math.expm1((u / 255) * Math.log1p(clip));

// Probability of exceedance, 0..1 (sequential, one direction: light -> dark red)
const PROB = ["#fff3b8", "#fedc7a", "#fdb94e", "#fb9038", "#f3642c", "#dc3a23", "#b5151e", "#7d0b1c"];
const PROB_EDGES = [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9];
export const probLut = () => cache.get("prob") || cache.set("prob", { key: "prob",
  table: table((u) => { const p = u / 255; let b = -1; for (let i = 0; i < PROB_EDGES.length; i++) if (p >= PROB_EDGES[i]) b = i; return b < 0 ? null : [PROB[b], b ? 255 : 170]; }),
  legend: { title: "Probability of exceedance (%)", colors: PROB.map((c, i) => css([c, i ? 255 : 170])), ticks: PROB_EDGES.map((e) => String(Math.round(e * 100))) },
  value: (u) => `${Math.round((u / 255) * 100)} %` }).get("prob");

// Ensemble spread: standard deviation between members in normalised log-rate units (0..1)
const SPREAD = ["#3b2f6b", "#5a3f9a", "#7c55c2", "#a273dc", "#c79bee", "#ead0fb"];
const SPREAD_EDGES = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3];
export const spreadLut = () => cache.get("spread") || cache.set("spread", { key: "spread",
  table: table((u) => { const s = u / 255; let b = -1; for (let i = 0; i < SPREAD_EDGES.length; i++) if (s >= SPREAD_EDGES[i]) b = i; return b < 0 ? null : [SPREAD[b], b ? 255 : 170]; }),
  legend: { title: "Ensemble spread (σ, normalised units)", colors: SPREAD.map((c, i) => css([c, i ? 255 : 170])), ticks: SPREAD_EDGES.map(String) },
  value: (u) => `σ ${(u / 255).toFixed(3)}` }).get("spread");

// Forecast minus observed, mm/h. Signed-log encoding around u = 128. Blue = forecast too dry, red = too wet.
const DIFF_EDGES = [0.5, 2.5, 7.6, 16, 50];
const DRY = ["#9cc3ec", "#5f9fe0", "#2f78cf", "#1c55a6", "#12397a"], WET = ["#f3b3a4", "#ec806a", "#de4f3f", "#b92a2a", "#851a22"];
export function diffLut(clip = 100) {
  const key = "diff" + clip;
  if (!cache.has(key)) {
    const k = Math.log1p(clip), dec = (u) => { const s = (u - 128) / 127; return Math.sign(s) * Math.expm1(Math.abs(s) * k); };
    cache.set(key, { key, table: table((u) => { const d = dec(u), a = Math.abs(d); let b = -1; for (let i = 0; i < DIFF_EDGES.length; i++) if (a >= DIFF_EDGES[i] - 1e-6) b = i; return b < 0 ? null : [(d < 0 ? DRY : WET)[b]]; }),
      legend: { title: "Forecast − observed (mm/h)", colors: [...DRY].reverse().map((c) => css([c])).concat(["transparent"], WET.map((c) => css([c]))),
        wide: true, ticks: [""].concat([...DIFF_EDGES].reverse().map((e) => "−" + e), DIFF_EDGES.map((e) => "+" + e)) },
      value: (u) => { const d = dec(u); return `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)} mm/h`; } });
  }
  return cache.get(key);
}
// a, b: uint8 normalised fields of equal length -> uint8 signed-log difference (a - b in mm/h)
export function diffField(a, b, clip = 100) {
  const k = Math.log1p(clip), mm = new Float32Array(256), out = new Uint8Array(a.length);
  for (let u = 0; u < 256; u++) mm[u] = Math.expm1((u / 255) * k);
  for (let i = 0; i < a.length; i++) { const d = mm[a[i]] - mm[b[i]]; out[i] = Math.round(128 + Math.sign(d) * (Math.log1p(Math.abs(d)) / k) * 127); }
  return out;
}
// Legend: one swatch per class with the class's lower boundary labelled at its lower (left / bottom) edge.
export function colorbar(lut, cls = "") {
  const L = lut.legend, el = document.createElement("div");
  el.className = "cbar " + cls + (L.wide ? " wide" : ""); el.setAttribute("role", "img");
  el.setAttribute("aria-label", `${L.title}: classes ${L.ticks.filter(Boolean).join(", ")}`);
  el.innerHTML = `<div class="t">${L.title}</div><div class="its">${L.colors.map((c, i) => `<div class="it"><i style="background:${c}"></i><span>${L.ticks[i] ?? ""}</span></div>`).join("")}</div>`;
  return el;
}

// ------------------------------------------------------------------ renderer
const VS = `#version 300 es
in vec2 p; uniform vec4 rect; out vec2 uv;
void main() { uv = p; gl_Position = vec4(mix(rect.xy, rect.zw, p), 0.0, 1.0); }`;
const FS = `#version 300 es
precision mediump float;
uniform sampler2D data, lut; uniform bool swap; uniform float opacity;
in vec2 uv; out vec4 o;
void main() {
  float v = texture(data, swap ? uv.yx : uv).r;
  vec4 c = texture(lut, vec2((floor(v * 255.0 + 0.5) + 0.5) / 256.0, 0.5));
  o = vec4(c.rgb * c.a, c.a) * opacity;
}`;

export class FieldRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = this.gl = canvas.getContext("webgl2", { antialias: false, alpha: true, premultipliedAlpha: true, depth: false, stencil: false, powerPreference: "low-power" });
    if (!gl) { this.ctx = canvas.getContext("2d"); return; }       // 2-D fallback below
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog); gl.useProgram(prog);
    this.u = Object.fromEntries(["rect", "data", "lut", "swap", "opacity"].map((n) => [n, gl.getUniformLocation(prog, n)]));
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.dataTex = this._tex(0); this.lutTex = this._tex(1); this.dim = ""; this.lutKey = null;
    gl.uniform1i(this.u.data, 0); gl.uniform1i(this.u.lut, 1); gl.clearColor(0, 0, 0, 0);
  }
  _tex(unit) {
    const gl = this.gl, t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  }
  clear() { if (this.gl) { this.gl.viewport(0, 0, this.canvas.width, this.canvas.height); this.gl.clear(this.gl.COLOR_BUFFER_BIT); } else this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); }
  // data [n0, n1]; lonLat: data[iLon * n1 + jLat] (else data[jLat * n1 + iLon]); rect = [x0, y0, x1, y1] in clip space (west-south, east-north)
  draw(data, n0, n1, lut, { lonLat = true, rect = [-1, -1, 1, 1], smooth = false, opacity = 1 } = {}) {
    const gl = this.gl;
    if (!gl) return this._draw2d(data, n0, n1, lut, lonLat, rect, opacity);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    if (this.dim !== n0 + "x" + n1) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, n1, n0, 0, gl.RED, gl.UNSIGNED_BYTE, data); this.dim = n0 + "x" + n1; }
    else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, n1, n0, gl.RED, gl.UNSIGNED_BYTE, data);
    const f = smooth ? gl.LINEAR : gl.NEAREST; gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    if (this.lutKey !== lut.key) { gl.activeTexture(gl.TEXTURE1); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut.table); this.lutKey = lut.key; }
    gl.uniform4f(this.u.rect, ...rect); gl.uniform1i(this.u.swap, lonLat ? 1 : 0); gl.uniform1f(this.u.opacity, opacity);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  _draw2d(a, n0, n1, lut, lonLat, rect, opacity) {
    const W = lonLat ? n0 : n1, H = lonLat ? n1 : n0, img = new ImageData(W, H), L = lut.table, d = img.data;
    for (let r = 0; r < H; r++) for (let c = 0, j = H - 1 - r; c < W; c++) { const s = (lonLat ? a[c * n1 + j] : a[j * n1 + c]) * 4, o = (r * W + c) * 4; d[o] = L[s]; d[o + 1] = L[s + 1]; d[o + 2] = L[s + 2]; d[o + 3] = L[s + 3]; }
    const off = this._off && this._off.width === W && this._off.height === H ? this._off : (this._off = Object.assign(document.createElement("canvas"), { width: W, height: H }));
    off.getContext("2d").putImageData(img, 0, 0);
    const cw = this.canvas.width, ch = this.canvas.height, x0 = (rect[0] + 1) / 2 * cw, x1 = (rect[2] + 1) / 2 * cw, y0 = (1 - rect[3]) / 2 * ch, y1 = (1 - rect[1]) / 2 * ch;
    this.ctx.clearRect(0, 0, cw, ch); this.ctx.imageSmoothingEnabled = false; this.ctx.globalAlpha = opacity; this.ctx.drawImage(off, x0, y0, x1 - x0, y1 - y0);
  }
}

// One shared renderer for thumbnails (browsers allow only a handful of WebGL contexts per page).
let shared = null;
export function drawThumb(canvas, data, shape, lut, { lonLat = true, lines = null, ds = null } = {}) {
  const [n0, n1] = shape, W = lonLat ? n0 : n1, H = lonLat ? n1 : n0;
  shared ||= new FieldRenderer(document.createElement("canvas"));
  if (shared.canvas.width !== W || shared.canvas.height !== H) { shared.canvas.width = W; shared.canvas.height = H; }
  shared.draw(data, n0, n1, lut, { lonLat });
  const ctx = canvas.getContext("2d"), cw = canvas.width, ch = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cw, ch); ctx.imageSmoothingEnabled = false; ctx.drawImage(shared.canvas, 0, 0, cw, ch);
  if (lines && ds) {
    const [la0, la1] = ds.lat_range, [lo0, lo1] = ds.lon_range, sx = cw / (lo1 - lo0), sy = ch / (la1 - la0);
    const st = getComputedStyle(document.documentElement);
    ctx.setTransform(sx, 0, 0, -sy, -lo0 * sx, la1 * sy); ctx.lineWidth = 1 / sx; ctx.lineJoin = "round";
    ctx.strokeStyle = st.getPropertyValue("--coast"); ctx.lineWidth = 1.2 / sx; ctx.stroke(lines.coast);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
}
