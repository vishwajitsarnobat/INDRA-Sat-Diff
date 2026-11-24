// Lazy frame store: frames arrive in fixed chunks (one request = `chunk` consecutive frames), are kept in an LRU,
// and requests that fall outside the window around the cursor are aborted while scrubbing.
export class FrameStore {
  constructor(ds, count, { chunk = 16, maxChunks = 64, onLoad } = {}) {
    Object.assign(this, { ds, count, chunk, maxChunks, onLoad });
    this.chunks = new Map();      // chunk index -> Uint8Array (insertion order = LRU order)
    this.pending = new Map();     // chunk index -> { ctrl, promise }
    this.shape = ds.grid || [128, 128];
    this.stats = { requests: 0, aborted: 0, bytes: 0, lastMs: 0 };
  }
  get frameBytes() { return this.shape[0] * this.shape[1]; }
  has(i) { return this.chunks.has(Math.floor(i / this.chunk)); }
  get(i) {
    const ci = Math.floor(i / this.chunk), c = this.chunks.get(ci);
    if (!c) return null;
    this.chunks.delete(ci); this.chunks.set(ci, c);                  // mark as recently used
    const o = (i - ci * this.chunk) * this.frameBytes;
    return c.subarray(o, o + this.frameBytes);
  }
  _load(ci) {
    if (this.chunks.has(ci)) return Promise.resolve();
    if (this.pending.has(ci)) return this.pending.get(ci).promise;
    const ctrl = new AbortController(), start = ci * this.chunk, t0 = performance.now();
    const promise = fetch(`/api/datasets/${this.ds.id}/frames?start=${start}&count=${Math.min(this.chunk, this.count - start)}&v=${this.ds.version}`, { signal: ctrl.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error(`frames ${start}: ${r.status}`);
        const [, n0, n1] = r.headers.get("X-Shape").split(",").map(Number); this.shape = [n0, n1];
        const buf = new Uint8Array(await r.arrayBuffer());
        this.chunks.set(ci, buf); this.stats.bytes += buf.length; this.stats.lastMs = performance.now() - t0;
        while (this.chunks.size > this.maxChunks) this.chunks.delete(this.chunks.keys().next().value);
        this.onLoad?.(ci);
      })
      .catch((e) => { if (e.name !== "AbortError") console.warn(e); })
      .finally(() => this.pending.delete(ci));
    this.pending.set(ci, { ctrl, promise }); this.stats.requests++;
    return promise;
  }
  // Make frame i available; prefetch `ahead` chunks in the direction of travel and one behind. Resolves when i is loaded.
  request(i, { dir = 1, ahead = 2 } = {}) {
    const ci = Math.floor(i / this.chunk), last = Math.floor((this.count - 1) / this.chunk);
    const want = new Set([ci]);
    for (let k = 1; k <= ahead; k++) want.add(ci + k * dir);
    want.add(ci - dir);
    for (const [k, p] of this.pending) if (!want.has(k)) { p.ctrl.abort(); this.stats.aborted++; }
    const main = this._load(ci);
    for (const k of want) if (k !== ci && k >= 0 && k <= last) this._load(k);
    return main;
  }
  loadedRanges() {                // [[i0, i1), ...] frame index ranges currently in memory
    return [...this.chunks.keys()].sort((a, b) => a - b).map((c) => [c * this.chunk, Math.min(this.count, (c + 1) * this.chunk)]);
  }
}
