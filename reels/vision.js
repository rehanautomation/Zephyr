// Video scanning: cuts, shots, faces and zooms, flashes, on-screen text events and graphic-screen builds.
// Frames are analysed at 15 fps; none of those frames are exported.
import { readText, loadOcr, resetTextCache } from './ocr.js';

export const FPS = 15;
const OCR_EVERY = 4; // read text on every 4th analysis frame (~3.75 times per second), then refine timing per frame
const MP_VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

/* ---------- Helpers ---------- */
export const fmt = t => `${Math.floor(t / 60)}:${(Math.max(0, t) % 60).toFixed(2).padStart(5, '0')}`;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const median = arr => { if (!arr.length) return 0; const s = [...arr].sort((a, b) => a - b); return s[s.length >> 1]; };
const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
const r2 = t => Math.round(t * 100) / 100;
const pct = v => `${Math.round(v * 100)}%`;
export const rgbOf = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
export const colorDist = (a, b) => { const x = typeof a === 'string' ? rgbOf(a) : a, y = typeof b === 'string' ? rgbOf(b) : b; return Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]); };
const hex = c => '#' + c.map(v => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0')).join('').toUpperCase();

export function colorName(h) {
  const [r, g, b] = (typeof h === 'string' ? rgbOf(h) : h).map(v => v / 255);
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, d = mx - mn;
  const s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
  if (l > 0.95) return 'white';
  if (l < 0.13) {
    if (s < 0.5 || l < 0.04) return 'black';
    const hh = (d === 0 ? 0 : mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4) * 60;
    const hue0 = (hh + 360) % 360;
    return `very dark ${hue0 < 15 || hue0 >= 345 ? 'red' : hue0 < 40 ? 'brown' : hue0 < 160 ? 'green' : hue0 < 255 ? 'blue' : 'purple'}`;
  }
  if (s < 0.18) return l > 0.6 ? 'light gray' : l > 0.35 ? 'gray' : 'dark gray';
  let hue = d === 0 ? 0 : mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  hue = (hue * 60 + 360) % 360;
  const base = hue < 15 || hue >= 345 ? 'red' : hue < 40 ? 'orange' : hue < 65 ? 'yellow' : hue < 160 ? 'green' : hue < 195 ? 'teal' : hue < 255 ? 'blue' : hue < 290 ? 'purple' : 'pink';
  return l < 0.3 ? `dark ${base}` : l > 0.75 ? `light ${base}` : base;
}

function once(target, event) {
  return new Promise((resolve, reject) => {
    const ok = () => { target.removeEventListener('error', bad); resolve(); };
    const bad = () => { target.removeEventListener(event, ok); reject(new Error('This video format could not be read by your browser')); };
    target.addEventListener(event, ok, { once: true });
    target.addEventListener('error', bad, { once: true });
  });
}

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return [c, c.getContext('2d', { willReadFrequently: true })];
}

export async function openVideo(file) {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = URL.createObjectURL(file);
  await once(video, 'loadeddata');
  return video;
}

export async function seek(video, t) {
  if (video.readyState >= 2 && Math.abs(video.currentTime - t) < 1e-4) return;
  video.currentTime = t;
  await once(video, 'seeked');
}

// Steps through the video: plays it at 2x and grabs frames as they are shown (fast), or seeks frame by frame
// when the tab is in the background (browsers stop painting video frames there)
const RATE = 2;
function makeStepper(video, dur) {
  const rvfc = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  // Brightness of every frame the browser shows (cheap), so 2-3 frame flashes aren't missed between analysis frames
  const dense = [];
  const [, dctx] = makeCanvas(16, 28);
  let sampling = rvfc;
  const sample = (_, meta) => {
    if (!sampling) return;
    try {
      dctx.drawImage(video, 0, 0, 16, 28);
      const d = dctx.getImageData(0, 0, 16, 28).data;
      let s = 0, s2 = 0, r = 0, g = 0, b = 0;
      for (let p = 0; p < d.length; p += 4) { const l = d[p] * .3 + d[p + 1] * .59 + d[p + 2] * .11; s += l; s2 += l * l; r += d[p]; g += d[p + 1]; b += d[p + 2]; }
      const n = d.length / 4, m = s / n;
      dense.push({ t: meta.mediaTime, b: m, sd: Math.sqrt(Math.max(0, s2 / n - m * m)), rgb: [r / n, g / n, b / n] });
    } catch { /* ignore */ }
    video.requestVideoFrameCallback(sample);
  };
  if (rvfc) video.requestVideoFrameCallback(sample);
  return {
    dense,
    async next(t) {
      if (document.hidden || !rvfc) {
        video.pause();
        await seek(video, Math.min(t, dur - 0.01));
        return t;
      }
      if (video.paused || video.ended) {
        if (video.ended || video.currentTime > t + 0.1 || t - video.currentTime > 0.5) await seek(video, Math.min(t, dur - 0.01));
        video.playbackRate = RATE;
        try { await video.play(); } catch { await seek(video, Math.min(t, dur - 0.01)); return t; }
      }
      return new Promise(resolve => {
        let done = false;
        const finish = v => { if (!done) { done = true; video.removeEventListener('ended', onEnd); resolve(v); } };
        const onEnd = () => finish(dur);
        const cb = (_, meta) => {
          if (done) return;
          if (meta.mediaTime >= t - 0.5 / FPS) finish(meta.mediaTime);
          else video.requestVideoFrameCallback(cb);
        };
        video.addEventListener('ended', onEnd);
        video.requestVideoFrameCallback(cb);
        // Safety net if frames stop arriving (e.g. the tab was hidden mid-wait)
        setTimeout(async () => { if (!done) { video.pause(); await seek(video, Math.min(t, dur - 0.01)); finish(t); } }, 1500);
      });
    },
    hold() { video.pause(); },
    stop() { sampling = false; video.pause(); video.playbackRate = 1; }
  };
}

let facePromise = null;
export function loadFace() {
  if (!facePromise) {
    facePromise = (async () => {
      const { FilesetResolver, FaceDetector } = await import(`${MP_VISION}/vision_bundle.mjs`);
      const files = await FilesetResolver.forVisionTasks(`${MP_VISION}/wasm`);
      return FaceDetector.createFromOptions(files, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'CPU' },
        runningMode: 'IMAGE',
        minDetectionConfidence: 0.5
      });
    })();
    facePromise.catch(() => { facePromise = null; });
  }
  return facePromise;
}

// Laplacian variance (sharpness) over a region of a grayscale frame
function lapVar(g, w, x0, y0, x1, y1) {
  let s = 0, s2 = 0, n = 0;
  for (let y = Math.max(1, y0); y < Math.min(y1, (g.length / w) - 1); y++) {
    for (let x = Math.max(1, x0); x < Math.min(x1, w - 1); x++) {
      const p = y * w + x;
      const v = 4 * g[p] - g[p - 1] - g[p + 1] - g[p - w] - g[p + w];
      s += v; s2 += v * v; n++;
    }
  }
  return n ? s2 / n - (s / n) ** 2 : 0;
}

/* ---------- Text look: colour, box, background ---------- */
function textLook(img, box) {
  const { data, width: W, height: H } = img;
  const x0 = Math.max(0, Math.floor(box.x * W)), x1 = Math.min(W, Math.ceil((box.x + box.w) * W));
  const y0 = Math.max(0, Math.floor(box.y * H)), y1 = Math.min(H, Math.ceil((box.y + box.h) * H));
  const ch = Math.max(1, y1 - y0);
  const px = (x, y) => { const p = (clamp(y, 0, H - 1) * W + clamp(x, 0, W - 1)) * 4; return [data[p], data[p + 1], data[p + 2]]; };
  const stats = arr => {
    const m = [0, 1, 2].map(c => mean(arr.map(p => p[c])));
    const sd = arr.length ? mean([0, 1, 2].map(c => Math.sqrt(mean(arr.map(p => (p[c] - m[c]) ** 2))))) : 999;
    return { m, sd };
  };
  const inner = [];
  const step = Math.max(1, Math.round(Math.sqrt(((x1 - x0) * (y1 - y0)) / 2000)));
  for (let y = y0; y < y1; y += step) for (let x = x0; x < x1; x += step) inner.push(px(x, y));
  // Rows just above and below the glyphs (middle 60% of the width, clear of rounded box corners): background
  const edge = [];
  const ex0 = Math.round(x0 + (x1 - x0) * 0.2), ex1 = Math.round(x1 - (x1 - x0) * 0.2);
  const off = Math.max(1, Math.round(ch * 0.04));
  for (let x = ex0; x < ex1; x += Math.max(1, step)) for (let k = 0; k < 2; k++) { edge.push(px(x, y0 - off - k)); edge.push(px(x, y1 + off + k)); }
  const edgeS = stats(edge), edgeM = edgeS.m;
  const lum = p => p[0] * .3 + p[1] * .59 + p[2] * .11;
  let c1 = inner.reduce((a, p) => lum(p) < lum(a) ? p : a, inner[0]);
  let c2 = inner.reduce((a, p) => lum(p) > lum(a) ? p : a, inner[0]);
  let g1 = [], g2 = [];
  for (let it = 0; it < 6; it++) {
    g1 = []; g2 = [];
    for (const p of inner) (colorDist(p, c1) <= colorDist(p, c2) ? g1 : g2).push(p);
    if (g1.length) c1 = stats(g1).m;
    if (g2.length) c2 = stats(g2).m;
  }
  const bgFirst = colorDist(c1, edgeM) <= colorDist(c2, edgeM);
  const [bgPix, text] = bgFirst ? [g1, c2] : [g2, c1];
  const bg = stats(bgPix);
  // Solid background behind the text: either the rows around it are one colour, or the letter gaps are
  const edgeSolid = edgeS.sd < 24 && bg.sd < 30 && colorDist(bg.m, edgeM) < 45 && bgPix.length > inner.length * 0.15;
  const gapSolid = bg.sd < 12 && bgPix.length > inner.length * 0.45;
  const solid = edgeSolid || gapSolid;
  const bgColor = edgeSolid ? edgeM : bg.m;
  // Far ring (well outside the text): same colour means a flat background, different means a box
  const far = [];
  const d = Math.round(1.2 * ch);
  for (let x = x0 - d; x < x1 + d; x += Math.max(1, step * 2)) { far.push(px(x, y0 - d)); far.push(px(x, y1 + d)); }
  for (let y = y0 - d; y < y1 + d; y += Math.max(1, step * 2)) { far.push(px(x0 - d, y)); far.push(px(x1 + d, y)); }
  const farS = stats(far);
  const flat = solid && farS.sd < 24 && colorDist(farS.m, bgColor) < 30;
  return { color: hex(text), boxed: solid && !flat, boxColor: solid && !flat ? hex(bgColor) : null, bg: flat ? hex(bgColor) : null };
}

/* ---------- Pass 1: scan every analysis frame ---------- */
export async function scanVideo(video, onProgress) {
  const dur = video.duration, vw = video.videoWidth, vh = video.videoHeight;
  const L = Math.max(vw, vh);
  const aw = Math.round(256 * vw / L), ah = Math.round(256 * vh / L);
  const k = Math.min(1, 1280 / L);
  const ow = Math.round(vw * k), oh = Math.round(vh * k);
  const [sm, smx] = makeCanvas(aw, ah);
  const [big, bigx] = makeCanvas(ow, oh);
  const N = Math.max(1, Math.ceil((dur - 0.02) * FPS));
  const n = aw * ah;
  const gray = [], col = [];
  const cv = new Float32Array(N), bright = new Float32Array(N), sharp = new Float32Array(N);
  const face = new Array(N).fill(null);
  const samples = [];
  const cw = aw >> 2, chh = ah >> 2;

  let detector = null;
  try { detector = await loadFace(); } catch { detector = null; }
  let ocrError = null;
  try { await loadOcr(); } catch (e) { ocrError = e.message || 'text reader failed to load'; }
  resetTextCache();

  let prevHSV = null, prevI = 0, skipped = 0;
  const stepper = makeStepper(video, dur);
  let lastOcr = -OCR_EVERY;
  for (let target = 0; target < N;) {
    const mt = await stepper.next(target / FPS);
    const i = clamp(Math.round(mt * FPS), target, N - 1);
    smx.drawImage(video, 0, 0, aw, ah);
    const d = smx.getImageData(0, 0, aw, ah).data;
    const g = new Uint8Array(n), hsv = new Uint8Array(n * 3);
    let vsum = 0;
    for (let p = 0; p < n; p++) {
      const r = d[p * 4], gg = d[p * 4 + 1], b = d[p * 4 + 2];
      g[p] = (r * 77 + gg * 150 + b * 29) >> 8;
      const mx = Math.max(r, gg, b), mn = Math.min(r, gg, b), dd = mx - mn;
      let hh = 0;
      if (dd) hh = mx === r ? 43 * (gg - b) / dd : mx === gg ? 85 + 43 * (b - r) / dd : 171 + 43 * (r - gg) / dd;
      hsv[p * 3] = ((Math.round(hh) % 256) + 256) % 256;
      hsv[p * 3 + 1] = mx ? Math.round(dd * 255 / mx) : 0;
      hsv[p * 3 + 2] = mx;
      vsum += mx;
    }
    if (prevHSV) {
      let dh = 0, ds = 0, dv = 0;
      for (let p = 0; p < n; p++) {
        const a = Math.abs(hsv[p * 3] - prevHSV[p * 3]);
        dh += Math.min(a, 256 - a);
        ds += Math.abs(hsv[p * 3 + 1] - prevHSV[p * 3 + 1]);
        dv += Math.abs(hsv[p * 3 + 2] - prevHSV[p * 3 + 2]);
      }
      cv[i] = (dh + ds + dv) / (3 * n) / Math.max(1, i - prevI);
    }
    prevI = i;
    prevHSV = hsv;
    bright[i] = vsum / n;
    sharp[i] = lapVar(g, aw, 0, 0, aw, ah);
    gray[i] = g;
    const c = new Uint8Array(cw * chh * 3);
    for (let y = 0; y < chh; y++) for (let x = 0; x < cw; x++) {
      let sr = 0, sg = 0, sb = 0;
      for (let yy = 0; yy < 4; yy++) for (let xx = 0; xx < 4; xx++) { const p = ((y * 4 + yy) * aw + x * 4 + xx) * 4; sr += d[p]; sg += d[p + 1]; sb += d[p + 2]; }
      const q = (y * cw + x) * 3;
      c[q] = sr >> 4; c[q + 1] = sg >> 4; c[q + 2] = sb >> 4;
    }
    col[i] = c;

    if (detector) {
      try {
        const res = detector.detect(sm);
        let best = null;
        for (const det of res.detections || []) {
          const bb = det.boundingBox, sc = det.categories?.[0]?.score ?? 0;
          if (bb && sc >= 0.5 && (!best || bb.width * bb.height > best.w * best.h * aw * ah)) best = { x: bb.originX / aw, y: bb.originY / ah, w: bb.width / aw, h: bb.height / ah };
        }
        face[i] = best;
      } catch { /* keep going without a face for this frame */ }
    }

    // Frames skipped by playback reuse the previous frame's data
    for (let j = target; j < i; j++) {
      gray[j] = gray[target - 1] || g; col[j] = col[target - 1] || col[i] || null;
      bright[j] = j > 0 ? bright[j - 1] : bright[i]; sharp[j] = j > 0 ? sharp[j - 1] : sharp[i]; face[j] = j > 0 ? face[j - 1] : face[i]; cv[j] = NaN;
      skipped++;
    }
    if (!ocrError && i - lastOcr >= OCR_EVERY) {
      lastOcr = i;
      stepper.hold();
      bigx.drawImage(video, 0, 0, ow, oh);
      let lines = [];
      try { lines = await readText(big); } catch (e) { ocrError = e.message || 'text reader failed'; }
      if (lines.length) {
        const img = bigx.getImageData(0, 0, ow, oh);
        for (const l of lines) Object.assign(l, textLook(img, l.box));
      }
      samples.push({ i, t: i / FPS, lines });
    }
    onProgress?.(i / N);
    target = i + 1;
  }
  stepper.stop();
  const dense = stepper.dense.sort((a, b) => a.t - b.t);
  return { dur, vw, vh, aw, ah, N, gray, col, cw, chh, cv, bright, sharp, face, samples, faceOk: !!detector, ocrError, skipped, dense };
}

/* ---------- Pass 2: turn frame data into events ---------- */
const textSim = (a, b) => {
  a = a.toLowerCase(); b = b.toLowerCase();
  if (a === b) return 1;
  const m = a.length, n = b.length;
  if (!m || !n) return 0;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) {
    const cur = [i];
    for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return 1 - prev[n] / Math.max(m, n);
};
export { textSim };
const iou = (a, b) => {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
};
const union = boxes => {
  const x0 = Math.min(...boxes.map(b => b.x)), y0 = Math.min(...boxes.map(b => b.y));
  const x1 = Math.max(...boxes.map(b => b.x + b.w)), y1 = Math.max(...boxes.map(b => b.y + b.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
};
const isCaps = s => /[A-Z]/.test(s) && s === s.toUpperCase();

function blocksOf(lines) {
  const sorted = [...lines].filter(l => /[\p{L}\p{N}]{1,}/u.test(l.text) && l.text.replace(/[^\p{L}\p{N}$%]/gu, '').length >= 1).sort((a, b) => a.box.y - b.box.y);
  const blocks = [];
  for (const l of sorted) {
    const b = blocks.find(b => {
      const last = b.lines[b.lines.length - 1];
      const gap = l.box.y - (last.box.y + last.box.h);
      const hmax = Math.max(l.box.h, last.box.h);
      const overlapX = Math.min(l.box.x + l.box.w, b.x1) - Math.max(l.box.x, b.x0);
      return gap < 0.6 * hmax && gap > -0.6 * hmax && (overlapX > 0 || Math.abs(l.box.x + l.box.w / 2 - (b.x0 + b.x1) / 2) < 0.1);
    });
    if (b) { b.lines.push(l); b.x0 = Math.min(b.x0, l.box.x); b.x1 = Math.max(b.x1, l.box.x + l.box.w); }
    else blocks.push({ lines: [l], x0: l.box.x, x1: l.box.x + l.box.w });
  }
  return blocks.map(b => ({
    lines: b.lines,
    text: b.lines.map(l => l.text).join(' / '),
    box: union(b.lines.map(l => l.box)),
    conf: mean(b.lines.map(l => l.conf)),
    primary: b.lines.reduce((m, l) => l.box.h > m.box.h ? l : m)
  }));
}

export function buildEvents(scan) {
  const { N, dur, aw, ah, gray, col, cw, chh, cv, bright, sharp, face, samples } = scan;
  const T = i => i / FPS;

  const regionDiff = (i, R) => {
    if (i <= 0 || i >= N) return 0;
    const x0 = clamp(Math.floor(R.x * aw), 0, aw - 1), x1 = clamp(Math.ceil((R.x + R.w) * aw), x0 + 1, aw);
    const y0 = clamp(Math.floor(R.y * ah), 0, ah - 1), y1 = clamp(Math.ceil((R.y + R.h) * ah), y0 + 1, ah);
    const a = gray[i], b = gray[i - 1];
    let s = 0, c = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const p = y * aw + x; s += Math.abs(a[p] - b[p]); c++; }
    return c ? s / c : 0;
  };
  const colDiff = (i, j) => { const a = col[i], b = col[j]; let s = 0; for (let p = 0; p < a.length; p++) s += Math.abs(a[p] - b[p]); return s / a.length; };
  const frameColor = i => { const c = col[i]; const m = [0, 0, 0]; for (let p = 0; p < c.length; p += 3) { m[0] += c[p]; m[1] += c[p + 1]; m[2] += c[p + 2]; } return hex(m.map(v => v / (c.length / 3))); };

  /* Flashes: a short burst of much brighter frames */
  const solid = i => sharp[i] < 4 && (bright[i] >= 200 || bright[i] <= 30);
  const flashes = [];
  for (let i = 1; i < N; i++) {
    let j = i;
    if (bright[i] - bright[i - 1] >= 35 && bright[i] >= 1.25 * bright[i - 1]) {
      while (j < N && j - i < 6 && bright[j] >= bright[i - 1] + 25) j++;
    } else if (solid(i) && !solid(i - 1)) {
      while (j < N && j - i < 6 && solid(j)) j++;
    }
    if (j > i && j < N && j - i <= 5) {
      const dark = bright[i] <= 30;
      flashes.push({ a: i, b: j, start: T(i), end: T(j), color: dark ? 'black' : colorName(frameColor(i)), dip: dark });
      i = j;
    }
  }
  // Short flashes seen only in the every-frame brightness samples
  const dense = scan.dense || [];
  const flat = d => d.sd < 12 && (d.b >= 240 || d.b <= 15);
  for (let k = 1; k < dense.length; k++) {
    const d = dense[k], p = dense[k - 1];
    const jump = d.b - p.b >= 35 && d.b >= 1.25 * p.b;
    if (!jump && !(flat(d) && !(flat(p) && Math.abs(p.b - d.b) < 10))) continue;
    let j = k;
    while (j < dense.length && dense[j].t - d.t <= 0.25 && (flat(d) ? flat(dense[j]) : dense[j].b >= p.b + 25)) j++;
    if (j >= dense.length || dense[j].t - d.t > 0.25) continue;
    const a = Math.min(N - 1, Math.round(d.t * FPS)), b = Math.min(N - 1, Math.max(a + 1, Math.round(dense[j].t * FPS)));
    if (!flashes.some(f => Math.abs(f.a - a) <= 2)) {
      const dark = d.b <= 30;
      flashes.push({ a, b, start: d.t, end: dense[j].t, color: dark ? 'black' : colorName(hex(d.rgb)), dip: dark });
    }
    k = j;
  }
  flashes.sort((x, y) => x.a - y.a);
  const inFlash = i => flashes.find(f => i >= f.a && i <= f.b);

  /* Hard cuts (content change spikes, like PySceneDetect's adaptive detector) */
  let cutIdx = [];
  const cutInfo = new Map(); // i -> { weak, gap }
  for (let i = 1; i < N; i++) {
    if (isNaN(cv[i])) continue;
    const nb = [i - 3, i - 2, i - 1, i + 1, i + 2, i + 3].filter(j => j >= 1 && j < N && !isNaN(cv[j])).map(j => cv[j]).slice(0, 4);
    const avg = mean(nb);
    if ((cv[i] >= 18 && cv[i] >= 3 * (avg + 0.5)) || cv[i] >= 55) {
      let gap = 1;
      while (i - gap >= 1 && isNaN(cv[i - gap])) gap++;
      cutIdx.push(i);
      cutInfo.set(i, { weak: cv[i] < 55 && (cv[i] < 25 || cv[i] < 4 * (avg + 0.5)), gap });
    }
  }
  // A flash counts as one transition: drop cuts inside it, keep one at its end if the scene changed
  const flashCuts = new Set(), flashUnsure = [];
  for (const f of flashes) {
    cutIdx = cutIdx.filter(i => i < f.a || i > f.b);
    if (f.a <= 0) continue;
    const change = colDiff(f.a - 1, Math.min(N - 1, f.b));
    if (change > 22) { cutIdx.push(f.b); flashCuts.add(f.b); cutInfo.set(f.b, { weak: change < 30, gap: 1 }); }
    else if (change > 15) flashUnsure.push(f);
  }
  cutIdx = [...new Set(cutIdx)].sort((a, b) => a - b).filter((i, k, arr) => k === 0 || i - arr[k - 1] >= 3);
  // The same face in the same place, at nearly the same size, on both sides is a zoom or movement, not a cut
  const notCuts = [];
  cutIdx = cutIdx.filter(i => {
    if (flashCuts.has(i)) return true;
    const a = face[i - 1] || face[i - 2], b = face[i] || face[i + 1];
    if (!a || !b) return true;
    const r = b.h / a.h, dx = (b.x + b.w / 2) - (a.x + a.w / 2), dy = (b.y + b.h / 2) - (a.y + a.h / 2);
    const same = r > 0.88 && r < 1.13 && Math.hypot(dx, dy) < 0.06;
    if (same && cv[i] >= 30) notCuts.push(T(i)); // big change but the same face: maybe a cut after all
    return !same;
  });

  /* Shots and shot types */
  const bounds = [0, ...cutIdx, N];
  const shots = [];
  for (let s = 0; s < bounds.length - 1; s++) {
    const a = bounds[s], b = bounds[s + 1];
    if (b <= a) continue;
    const faces = face.slice(a, b);
    const faceFrac = faces.filter(Boolean).length / (b - a);
    const mid = col[Math.floor((a + b) / 2)];
    const counts = new Map();
    for (let p = 0; p < mid.length; p += 3) { const key = (mid[p] >> 5) * 64 + (mid[p + 1] >> 5) * 8 + (mid[p + 2] >> 5); counts.set(key, (counts.get(key) || 0) + 1); }
    const top3 = [...counts.values()].sort((x, y) => y - x).slice(0, 3).reduce((x, y) => x + y, 0) / (mid.length / 3);
    const type = scan.faceOk && faceFrac >= 0.5 ? 'talking head' : top3 >= 0.45 ? 'graphic' : scan.faceOk ? 'b-roll' : 'footage';
    const unsure = type === 'graphic' && (top3 < 0.55 || (scan.faceOk && faceFrac >= 0.3));
    shots.push({ n: shots.length, a, b, start: T(a), end: b >= N ? dur : T(b), type, faceFrac, unsure });
  }
  const shotAt = i => shots.find(s => i >= s.a && i < s.b) || shots[shots.length - 1];

  /* Face size per frame (gaps filled), for zooms and punch-ins */
  const fh = new Float32Array(N).fill(NaN);
  face.forEach((f, i) => { if (f) fh[i] = f.h; });
  for (const s of shots) {
    const idx = [];
    for (let i = s.a; i < s.b; i++) if (!isNaN(fh[i])) idx.push(i);
    if (!idx.length) continue;
    for (let i = s.a; i < s.b; i++) {
      if (!isNaN(fh[i])) continue;
      const prev = idx.filter(j => j < i).pop(), next = idx.find(j => j > i);
      fh[i] = prev === undefined ? fh[next] : next === undefined ? fh[prev] : fh[prev] + (fh[next] - fh[prev]) * (i - prev) / (next - prev);
    }
  }
  const smoothH = i => median([fh[i - 1], fh[i], fh[i + 1]].filter(v => v !== undefined && !isNaN(v)));

  const zooms = [];
  for (const s of shots) {
    if (s.type !== 'talking head' || s.b - s.a < 4) continue;
    const used = new Uint8Array(s.b - s.a);
    for (let pass = 0; pass < 4; pass++) {
      let best = null;
      for (let i = s.a; i < s.b - 3; i++) {
        if (used[i - s.a]) continue;
        for (let j = i + 3; j <= Math.min(s.b - 1, i + FPS); j++) {
          if (used[j - s.a]) break;
          const r = smoothH(j) / smoothH(i);
          if (!isFinite(r) || Math.abs(Math.log(r)) < Math.log(1.08)) continue;
          let mono = 0;
          for (let q = i + 1; q <= j; q++) mono += Math.sign(smoothH(q) - smoothH(q - 1)) === Math.sign(r - 1) ? 1 : 0;
          if (mono < (j - i) * 0.6) continue;
          if (!best || Math.abs(Math.log(r)) > Math.abs(Math.log(best.r))) best = { i, j, r };
        }
      }
      if (!best) break;
      for (let q = best.i; q <= best.j; q++) used[q - s.a] = 1;
      zooms.push(zoomOf(best.i, best.j, best.r));
    }
    // A steady push over the shot, timed from where the face size really starts and stops changing
    const total = smoothH(s.b - 1) / smoothH(s.a);
    if (isFinite(total) && Math.abs(Math.log(total)) >= Math.log(1.1) && s.end - s.start > 1.2 && !zooms.some(z => z.a >= s.a && z.b <= s.b)) {
      const L = q => Math.abs(Math.log(smoothH(q) / smoothH(s.a))), tot = Math.abs(Math.log(total));
      let i0 = s.a, i1 = s.b - 1;
      while (i0 < i1 && L(i0) < 0.05 * tot) i0++;
      while (i1 > i0 && L(i1) > 0.95 * tot) i1--;
      i0 = Math.max(s.a, i0 - 1); i1 = Math.min(s.b - 1, i1 + 1);
      zooms.push(zoomOf(i0, i1, smoothH(i1) / smoothH(i0)));
    }
  }
  function zoomOf(i, j, r) {
    let seen = 0;
    for (let q = i; q <= j; q++) if (face[q]) seen++;
    const dir = r > 1 ? 'in' : 'out';
    return {
      type: 'zoom', dir, start: T(i), end: T(j), a: i, b: j,
      from: dir === 'out' ? Math.round(100 / r) : 100, to: dir === 'out' ? 100 : Math.round(100 * r),
      unsure: Math.abs(Math.log(r)) < Math.log(1.12) || seen < 0.7 * (j - i + 1)
    };
  }
  // Zooms of 3 seconds or more are left out
  for (let k = zooms.length - 1; k >= 0; k--) if (zooms[k].end - zooms[k].start >= 3) zooms.splice(k, 1);
  zooms.sort((a, b) => a.start - b.start);

  /* Transitions at each cut */
  const transitions = [];
  for (let k = 1; k < shots.length; k++) {
    const prev = shots[k - 1], cur = shots[k], i = cur.a;
    const flash = flashCuts.has(i) ? flashes.find(f => f.b === i) : null;
    // Face or no face on each side: look at 3 frames just outside the cut (outside the flash if there is one)
    const pre = flash ? flash.a : i;
    const hasFace = (a, b) => { const fs = face.slice(Math.max(0, a), Math.min(N, b)); return fs.filter(Boolean).length * 2 > fs.length; };
    const faceBefore = scan.faceOk ? hasFace(pre - 3, pre) : null, faceAfter = scan.faceOk ? hasFace(i, i + 3) : null;
    let faceScale = null;
    if (faceBefore && faceAfter) {
      const before = median([fh[pre - 3], fh[pre - 2], fh[pre - 1]].filter(v => v !== undefined && !isNaN(v)));
      const after = median([fh[i], fh[i + 1], fh[i + 2]].filter(v => v !== undefined && !isNaN(v)));
      if (before && after) faceScale = after / before;
    }
    const info = cutInfo.get(i) || { weak: false, gap: 1 };
    const faceFlicker = scan.faceOk && (faceBefore !== hasFace(pre - 6, pre - 3) || faceAfter !== hasFace(i + 3, i + 6));
    transitions.push({ type: 'cut', t: T(i), i, from: prev, to: cur, flash, faceBefore, faceAfter, faceScale, weak: info.weak, gap: info.gap, faceFlicker });
  }
  const flashOnly = flashes.filter(f => !flashCuts.has(f.b));

  /* On-screen text: track blocks across text-reading samples */
  const events = [];
  let active = [];
  samples.forEach((s, k) => {
    const blocks = blocksOf(s.lines);
    const used = new Set();
    for (const ev of active) {
      let best = -1, score = 0;
      blocks.forEach((b, bi) => {
        if (used.has(bi)) return;
        const na = ev.text.toLowerCase().replace(/[^a-z0-9$%]/g, ''), nb = b.text.toLowerCase().replace(/[^a-z0-9$%]/g, '');
        const contains = na.length >= 3 && nb.length >= 3 && (na.includes(nb) || nb.includes(na));
        const sim = Math.max(textSim(ev.text, b.text), contains ? 0.8 : 0), ov = Math.max(iou(ev.box, b.box), contains ? iou(union([ev.box, b.box]), b.box) * 0.5 + 0.2 : 0);
        const close = Math.hypot(ev.box.x + ev.box.w / 2 - b.box.x - b.box.w / 2, ev.box.y + ev.box.h / 2 - b.box.y - b.box.h / 2) < 0.08;
        if (sim >= 0.6 && (ov >= 0.2 || close) && sim + ov > score) { best = bi; score = sim + ov; }
      });
      if (best < 0) { ev.missed++; continue; }
      used.add(best);
      const b = blocks[best];
      ev.last = k; ev.missed = 0; ev.seen.push({ k, t: s.t, block: b });
      if (b.conf > ev.conf || b.lines.length > ev.rep.lines.length) { ev.text = b.text; ev.conf = Math.max(ev.conf, b.conf); ev.box = b.box; ev.rep = b; }
      // Same text, clearly different colour (seen twice in a row) = a colour change
      const c = b.primary.color;
      if (colorDist(c, ev.curColor) > 90 && textSim(b.text, ev.text) >= 0.85) {
        if (ev.pending && colorDist(ev.pending.c, c) < 60) { ev.changes.push({ t: ev.pending.t, k: ev.pending.k, from: ev.curColor, to: c }); ev.curColor = c; ev.pending = null; }
        else ev.pending = { c, t: s.t, k };
      } else ev.pending = null;
    }
    active = active.filter(ev => { if (ev.missed > 1) { events.push(ev); return false; } return true; });
    blocks.forEach((b, bi) => {
      if (!used.has(bi)) active.push({ first: k, last: k, seen: [{ k, t: s.t, block: b }], text: b.text, conf: b.conf, box: b.box, rep: b, missed: 0, changes: [], curColor: b.primary.color, pending: null });
    });
  });
  events.push(...active);

  const textEvents = [];
  for (const ev of events) {
    if (ev.seen.length === 1 && ev.conf < 0.85) continue;
    const R = { x: ev.box.x - ev.box.w * 0.08, y: ev.box.y - ev.box.h * 0.15, w: ev.box.w * 1.16, h: ev.box.h * 1.3 };
    const firstF = samples[ev.first].i, prevF = ev.first > 0 ? samples[ev.first - 1].i : 0;
    const lastF = samples[ev.last].i, nextF = ev.last + 1 < samples.length ? samples[ev.last + 1].i : N - 1;
    const refF = ev.seen.length > 1 ? ev.seen[1].block && samples[ev.seen[1].k].i : firstF;
    const mask = textMask(gray[refF], aw, ah, ev.rep.lines);
    const dist = (f, ref, m) => { if (!m.length) return 0; const a = gray[f], b = gray[ref]; let s = 0; for (const p of m) s += Math.abs(a[p] - b[p]); return s / m.length; };
    // Entrance: within the window before the first sighting, find where the text's own pixels arrive
    let lowest = ev.first === 0 ? 0 : Math.max(0, prevF - 2);
    for (const tr of transitions) if (tr.i > lowest && tr.i <= firstF) lowest = tr.i; // text that starts in a new shot can't start before the cut
    const dIn = [];
    for (let f = lowest; f <= refF; f++) dIn.push(dist(f, refF, mask));
    const dMax = Math.max(...dIn);
    const pin = dIn.map(d => dMax < 8 ? 1 : 1 - d / dMax); // 0 = text absent (as at the window start), 1 = settled
    let start = lowest, settle = refF;
    for (let f = refF; f >= lowest; f--) { if (pin[f - lowest] < 0.08) { start = f + 1; break; } start = f; }
    for (let f = start; f <= refF; f++) { if (pin[f - lowest] >= 0.88) { settle = f; break; } }
    if (ev.first === 0 && start <= 1) { start = 0; }
    // Exit: walk forward from the last sighting until the text starts to change, then until it is gone
    const refE = lastF;
    const maskE = textMask(gray[refE], aw, ah, ev.rep.lines);

    let limit = Math.min(N - 1, nextF + 2);
    for (const tr of transitions) if (tr.i > refE && tr.i < limit) limit = tr.i;
    for (const fl of flashes) if (fl.a > refE && fl.a < limit) limit = fl.a;
    const dOut = [];
    for (let f = refE; f <= limit; f++) dOut.push(dist(f, refE, maskE));
    const eMax = Math.max(...dOut);
    const pout = dOut.map(d => eMax < 8 ? 1 : 1 - d / eMax);
    let endF = limit;
    for (let f = refE; f <= limit; f++) { if (pout[f - refE] < 0.88) { endF = f; break; } }
    if (ev.last + 1 >= samples.length) endF = N;
    const shot = shotAt(start);
    const rep = ev.rep, prim = rep.primary;
    textEvents.push({
      type: 'text', text: ev.text, start: T(start), settle: T(settle), end: endF >= N ? dur : T(endF), a: start, s: settle, e: Math.min(endF, N - 1),
      box: ev.box, region: R, lines: rep.lines.length, caps: isCaps(ev.text), shot,
      lineData: rep.lines.map(l => ({ text: l.text, pos: l.pos || [] })), conf: ev.conf, sightings: ev.seen.length,
      timingUnsure: ev.first > 0 && dMax < 8,
      color: prim.color, heightPct: prim.box.h, centerY: ev.box.y + ev.box.h / 2, centerX: ev.box.x + ev.box.w / 2,
      boxed: !!prim.boxed, boxColor: prim.boxColor,
      changes: ev.changes.map(c => ({ t: samples[c.k].t, from: c.from, to: c.to }))
    });
  }
  textEvents.sort((a, b) => a.start - b.start);
  // Text that stays on screen almost the whole video is a persistent overlay (handle, watermark)
  for (const ev of textEvents) ev.persistent = ev.end - ev.start > 0.8 * dur && dur > 4;

  /* Graphic screens: how each one builds step by step */
  const screens = [];
  for (const s of shots.filter(s => s.type === 'graphic')) {
    const inside = textEvents.filter(e => e.a >= s.a - 1 && e.a < s.b);
    const steps = [];
    const initial = inside.filter(e => e.start - s.start < 0.35);
    steps.push({ t: s.start, i: s.a, settleI: Math.max(s.a, ...initial.map(e => e.s), s.a + 1), what: initial.length ? `appears with ${initial.map(e => `"${e.text}"`).join(', ')}` : 'appears' });
    for (const e of inside) if (!initial.includes(e)) steps.push({ t: e.start, i: e.a, settleI: e.s, what: `+ "${e.text}"`, text: e });
    for (const e of inside) for (const c of e.changes) steps.push({ t: c.t, i: Math.round(c.t * FPS), settleI: Math.round(c.t * FPS) + 1, what: `"${e.text}" changes color (${colorName(c.from)} → ${colorName(c.to)})`, text: e });
    // Non-text visual changes that then hold still (icons, arrows, shapes)
    for (let f = s.a + 2; f < s.b - 2; f++) {
      if (inFlash(f) || inFlash(f + 1)) continue;
      const g1 = gray[f], g0 = gray[f - 1];
      let changed = 0;
      for (let p = 0; p < g1.length; p++) if (Math.abs(g1[p] - g0[p]) > 20) changed++;
      if (changed / g1.length < 0.01) continue;
      if (steps.some(st => Math.abs(st.i - f) <= 4)) { f += 2; continue; }
      steps.push({ t: T(f), i: f, settleI: Math.min(s.b - 1, f + 3), what: 'shape or icon change (no text)' });
      f += 4;
    }
    steps.sort((x, y) => x.t - y.t);
    let heroI = Math.max(s.a, s.b - 2);
    while (heroI > s.a && (inFlash(heroI) || solid(heroI))) heroI--;
    const c = col[heroI];
    const counts = new Map();
    for (let p = 0; p < c.length; p += 3) {
      const key = (c[p] >> 4) * 256 + (c[p + 1] >> 4) * 16 + (c[p + 2] >> 4);
      const e = counts.get(key) || { n: 0, r: 0, g: 0, b: 0 };
      e.n++; e.r += c[p]; e.g += c[p + 1]; e.b += c[p + 2];
      counts.set(key, e);
    }
    const palette = [...counts.values()].sort((x, y) => y.n - x.n).slice(0, 3).map(e => ({ hex: hex([e.r / e.n, e.g / e.n, e.b / e.n]), share: e.n / (c.length / 3) }));
    screens.push({ shot: s, start: s.start, end: s.end, steps, heroI, bg: palette[0].hex, texts: inside, unsure: s.unsure });
  }

  return { shots, transitions, flashes: flashOnly, zooms, textEvents, screens, notCuts, flashUnsure };
}

// Thumbnail pixels that belong to the glyphs: inside each line's box and close to that line's text colour
function textMask(g, aw, ah, lines) {
  const mask = [];
  for (const l of lines) {
    const [r, gg, b] = rgbOf(l.color);
    const lum = r * .3 + gg * .59 + b * .11;
    const x0 = clamp(Math.floor(l.box.x * aw), 0, aw - 1), x1 = clamp(Math.ceil((l.box.x + l.box.w) * aw), x0 + 1, aw);
    const y0 = clamp(Math.floor(l.box.y * ah), 0, ah - 1), y1 = clamp(Math.ceil((l.box.y + l.box.h) * ah), y0 + 1, ah);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const p = y * aw + x; if (Math.abs(g[p] - lum) < 28) mask.push(p); }
  }
  return mask;
}

/* ---------- Which frames to export, and the sheets ---------- */
const MAX_SHEETS = 5, MAX_SIDE = 1568, GAP = 8, HEAD = 44, ROWHEAD = 30, PER_ROW = 6;
const cellSize = portrait => portrait ? [232, 412] : [240, 135];

// Entrance strips for the first 2 text appearances after each cut, a build strip for each graphic screen.
// Over budget, strips for looks already pictured earlier in the batch are dropped first.
export function planImages(reelNo, ev, scan) {
  const portrait = scan.vh >= scan.vw;
  const T = i => i / FPS;
  const evenPick = (arr, k) => arr.length <= k ? arr : Array.from({ length: k }, (_, j) => arr[Math.round(j * (arr.length - 1) / (k - 1))]);
  const strips = [];
  for (const sh of ev.shots) {
    if (sh.type === 'graphic') continue;
    ev.textEvents.filter(e => !e.persistent && e.shot === sh).slice(0, 2).forEach((e, k) => {
      const from = Math.max(0, e.a - 2), to = Math.min(scan.N - 1, Math.max(e.s, e.a + 1) + 2, e.a + 10);
      const idx = [];
      for (let i = from; i <= to; i++) idx.push(i);
      const times = evenPick(idx, PER_ROW).map(T);
      if (times.length < 2) return;
      const fresh = !e.look || e.look.first === e;
      strips.push({ t: e.start, times, title: `Text "${e.text}" appears · ${fmt(e.start)}`, key: `text:${e.start}`, rank: fresh ? 1 : k === 0 ? 3 : 4 });
    });
  }
  for (const sc of ev.screens) {
    let pts = evenPick(sc.steps, PER_ROW - 1).map(stp => ({ t: T(Math.min(stp.settleI, sc.shot.b - 1)), label: stp.what }));
    if (!pts.length || T(sc.heroI) > pts[pts.length - 1].t) pts.push({ t: T(sc.heroI), label: 'fully built' });
    pts.sort((x, y) => x.t - y.t);
    pts = pts.filter((p, k) => k === 0 || p.t - pts[k - 1].t > 0.05).slice(-PER_ROW);
    if (pts.length < 2 && sc.shot.b - sc.shot.a > 2) {
      const mid = T(Math.floor((sc.shot.a + sc.shot.b) / 2));
      if (!pts.length || Math.abs(mid - pts[0].t) > 0.05) { pts.push({ t: mid, label: 'middle' }); pts.sort((x, y) => x.t - y.t); }
    }
    if (pts.length < 2) continue;
    strips.push({ t: sc.start, times: pts.map(p => p.t), labels: pts.map(p => p.label), title: `Graphic screen ${fmt(sc.start)}–${fmt(sc.end)} · build`, key: `graphic:${sc.start}`, rank: sc.firstHere ? 0 : 2 });
  }

  const SMALL = cellSize(portrait);
  const perSheet = Math.max(1, Math.floor((MAX_SIDE - HEAD) / (ROWHEAD + SMALL[1] + GAP)));
  const ranked = [...strips].sort((x, y) => x.rank - y.rank || x.t - y.t);
  const kept = ranked.slice(0, MAX_SHEETS * perSheet).sort((x, y) => x.t - y.t);
  const omitted = ranked.slice(MAX_SHEETS * perSheet).sort((x, y) => x.t - y.t).map(x => x.title);
  // Spread the strips evenly over as few sheets as possible
  const count = Math.ceil(kept.length / perSheet);
  const sheets = [];
  for (let k = 0, at = 0; k < count; k++) { const n = Math.floor(kept.length / count) + (k < kept.length % count ? 1 : 0); sheets.push({ items: kept.slice(at, at + n) }); at += n; }
  sheets.forEach((sh, k) => {
    sh.name = `reel${String(reelNo).padStart(2, '0')}_s${k + 1}.jpg`;
    sh.title = `Reel ${reelNo} · Sheet ${k + 1} of ${sheets.length}`;
  });
  return { sheets, omitted, portrait };
}

export async function renderSheets(video, plan) {
  const SMALL = cellSize(plan.portrait);
  // Grab every needed frame once, in time order
  const times = new Set();
  for (const s of plan.sheets) for (const it of s.items) it.times.forEach(t => times.add(Math.round(t * 1000) / 1000));
  const frames = new Map();
  const [fc, fx] = makeCanvas(SMALL[0] * 2, SMALL[1] * 2);
  for (const t of [...times].sort((a, b) => a - b)) {
    await seek(video, Math.min(t + 0.02, video.duration - 0.01)); // just past the frame boundary, so the frame shown starts at t
    fx.drawImage(video, 0, 0, fc.width, fc.height);
    frames.set(t, await createImageBitmap(fc));
  }
  const get = t => frames.get(Math.round(t * 1000) / 1000);

  const out = [];
  for (const s of plan.sheets) {
    const cells = [];
    const W = PER_ROW * SMALL[0] + (PER_ROW + 1) * GAP, H = HEAD + s.items.length * (ROWHEAD + SMALL[1] + GAP);
    const rowTitles = [];
    s.items.forEach((it, r) => {
      const y = HEAD + r * (ROWHEAD + SMALL[1] + GAP);
      rowTitles.push({ y, text: it.title });
      it.times.forEach((t, j) => cells.push({ x: GAP + j * (SMALL[0] + GAP), y: y + ROWHEAD, w: SMALL[0], h: SMALL[1], t, label: it.labels ? it.labels[j] : `${j + 1}/${it.times.length}`, item: it }));
    });
    const [c, x] = makeCanvas(W, H);
    x.fillStyle = '#111'; x.fillRect(0, 0, W, H);
    x.fillStyle = '#fff'; x.font = '600 22px system-ui, sans-serif'; x.textBaseline = 'middle';
    x.fillText(s.title, GAP + 4, HEAD / 2);
    x.fillStyle = '#ddd'; x.font = '600 16px system-ui, sans-serif';
    for (const rt of rowTitles) x.fillText(ellipsize(x, rt.text, W - 2 * GAP), GAP + 2, rt.y + ROWHEAD / 2);
    cells.forEach((cell, k) => {
      cell.n = k + 1;
      const bmp = get(cell.t);
      if (bmp) x.drawImage(bmp, cell.x, cell.y, cell.w, cell.h);
      const time = `#${cell.n} ${fmt(cell.t)}`;
      x.font = '700 15px system-ui, sans-serif';
      x.fillStyle = 'rgba(0,0,0,.8)'; x.fillRect(cell.x + 4, cell.y + 4, x.measureText(time).width + 12, 24);
      x.fillStyle = '#ffde59'; x.fillText(time, cell.x + 10, cell.y + 16);
      x.font = '600 13px system-ui, sans-serif';
      x.fillStyle = 'rgba(0,0,0,.75)'; x.fillRect(cell.x, cell.y + cell.h - 24, cell.w, 24);
      x.fillStyle = '#fff'; x.fillText(ellipsize(x, cell.label, cell.w - 16), cell.x + 8, cell.y + cell.h - 12);
    });
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.86));
    // Where each strip landed: "reel01_s2 #1–6"
    const refs = [];
    for (const it of s.items) {
      const ns = cells.filter(cl => cl.item === it).map(cl => cl.n);
      if (ns.length) refs.push({ key: it.key, ref: `${s.name.replace(/\.jpg$/, '')} #${ns[0]}–${ns[ns.length - 1]}` });
    }
    out.push({ name: s.name, title: s.title, blob, url: URL.createObjectURL(blob), refs, rows: s.items.map(it => ({ title: it.title, cells: cells.filter(cl => cl.item === it).map(cl => ({ n: cl.n, t: cl.t, label: cl.label })) })), width: W, height: H });
  }
  frames.forEach(b => b.close());
  return out;
}

function ellipsize(ctx, text, max) {
  if (ctx.measureText(text).width <= max) return text;
  let s = text;
  while (s.length > 1 && ctx.measureText(s + '…').width > max) s = s.slice(0, -1);
  return s + '…';
}
