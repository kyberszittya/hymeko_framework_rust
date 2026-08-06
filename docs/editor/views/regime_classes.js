// Regime analytics canvases, extracted verbatim (ESM) from demo_web/index.html:
// the animated αₖ Compass and the signed-topology Topo ring. Pure 2D-canvas, no
// three.js. The editor's kinematic tab uses these as side panels next to the
// real robot render; the demo's illustrative FK Scene3D skeleton is deliberately
// NOT reused here — the URDF-driven robot render is the honest geometric view.

// ── Regime compass (animated αₖ bars) ──────────────────────────────
export class Compass {
  constructor(canvas) {
    this.cv = canvas; this.ctx = canvas.getContext("2d");
    this.cur = {}; this.target = {}; this.ks = [3, 4, 5, 6]; this.raw = {};
    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
    this.resize();
  }
  // Only reallocates the backing store when the size actually changed — called
  // every animation frame, so an unconditional resize would churn memory.
  resize() {
    const r = this.cv.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    const w = Math.floor((r.width || 1) * dpr), h = Math.floor((r.height || 1) * dpr);
    this.W = r.width || 1; this.H = r.height || 1;
    if (this.cv.width === w && this.cv.height === h) return;
    this.cv.width = w; this.cv.height = h;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  dispose() { window.removeEventListener("resize", this._onResize); }
  set(cpa) {
    this.ks = [3, 4, 5, 6]; this.target = {};
    const max = Math.max(1, ...this.ks.map((k) => cpa[k] || 0));
    this.ks.forEach((k) => (this.target[k] = (cpa[k] || 0) / max));
    this.raw = cpa;
    this.dom = this.ks.reduce((b, k) => ((cpa[k] || 0) > (cpa[b] || 0) ? k : b), 3);
    this.flat = this.ks.every((k) => (cpa[k] || 0) === 0);
  }
  draw(t) {
    const ctx = this.ctx, W = this.W, H = this.H; ctx.clearRect(0, 0, W, H);
    const padL = 34, padB = 26, padT = 14, bw = (W - padL - 20) / this.ks.length, base = H - padB;
    ctx.strokeStyle = "rgba(120,150,190,.18)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, base); ctx.lineTo(W - 8, base); ctx.stroke();
    this.ks.forEach((k, i) => {
      const cv = this.cur[k] ?? 0, tv = this.target[k] ?? 0;
      this.cur[k] = cv + (tv - cv) * 0.12;
      const h = (base - padT) * this.cur[k];
      const x = padL + i * bw + bw * 0.18, w = bw * 0.64;
      const isDom = !this.flat && k === this.dom;
      const pulse = isDom ? 0.5 + 0.5 * Math.sin(t * 3) : 0;
      const grad = ctx.createLinearGradient(0, base - h, 0, base);
      if (isDom) { grad.addColorStop(0, "#fde68a"); grad.addColorStop(1, "#f59e0b"); }
      else { grad.addColorStop(0, "#38bdf8"); grad.addColorStop(1, "#0e3a52"); }
      ctx.fillStyle = grad;
      if (isDom) { ctx.shadowColor = "#fbbf24"; ctx.shadowBlur = 14 + 10 * pulse; }
      this._round(ctx, x, base - h, w, h, 4); ctx.fill(); ctx.shadowBlur = 0;
      const c = this.raw[k] || 0;
      ctx.fillStyle = isDom ? "#fde68a" : "#9fb4d0"; ctx.font = "600 12px Segoe UI"; ctx.textAlign = "center";
      if (c > 0) ctx.fillText(c, x + w / 2, base - h - 6);
      ctx.fillStyle = "#7d90ad"; ctx.font = "12px Segoe UI";
      ctx.fillText("k=" + k, x + w / 2, base + 16);
      if (isDom) { ctx.fillStyle = "#fbbf24"; ctx.font = "700 13px Segoe UI"; ctx.fillText("▲ αₖ", x + w / 2, base - h - 22); }
    });
  }
  _round(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, Math.abs(h) / 2); ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
}

// ── Topology graph (circular layout, signed edges, travelling pulse) ─
export class Topo {
  constructor(canvas) {
    this.cv = canvas; this.ctx = canvas.getContext("2d"); this.pos = [];
    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
    this.resize();
  }
  resize() {
    const r = this.cv.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    const w = Math.floor((r.width || 1) * dpr), h = Math.floor((r.height || 1) * dpr);
    this.W = r.width || 1; this.H = r.height || 1;
    if (this.cv.width === w && this.cv.height === h) { return; }
    this.cv.width = w; this.cv.height = h;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.mech) this.set(this.mech);
  }
  dispose() { window.removeEventListener("resize", this._onResize); }
  // mech = { n_links, edges:[[u,v]], joints:[{sign}] }
  set(mech) {
    this.mech = mech; const n = mech.n_links;
    const R = Math.min(this.W, this.H) * 0.36, cx = this.W / 2, cy = this.H / 2;
    this.pos = [];
    for (let i = 0; i < n; i++) {
      const a = -Math.PI / 2 + 2 * Math.PI * i / n;
      this.pos.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]);
    }
  }
  draw(t) {
    if (!this.mech) return; const ctx = this.ctx, W = this.W, H = this.H; ctx.clearRect(0, 0, W, H);
    const E = this.mech.edges, S = this.mech.joints;
    E.forEach((e, i) => {
      const a = this.pos[e[0]], b = this.pos[e[1]]; if (!a || !b) return;
      const sign = (S[i] && S[i].sign) || 1, col = sign < 0 ? "#f472b6" : "#38bdf8";
      ctx.strokeStyle = col; ctx.globalAlpha = 0.55; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.globalAlpha = 1;
      const u = (t * 0.25 + i * 0.13) % 1, px = a[0] + (b[0] - a[0]) * u, py = a[1] + (b[1] - a[1]) * u;
      ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 8;
      ctx.beginPath(); ctx.arc(px, py, 2.6, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
    });
    this.pos.forEach((p) => {
      ctx.fillStyle = "#cfe6ff"; ctx.strokeStyle = "#0a0e16"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 7); ctx.fill(); ctx.stroke();
    });
    ctx.fillStyle = "#7d90ad"; ctx.font = "11px Segoe UI"; ctx.textAlign = "center";
    ctx.fillText(this.mech.n_links + " links · " + this.mech.n_joints + " joints", W / 2, H - 6);
  }
}
