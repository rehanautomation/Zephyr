// On-screen text reading: PaddleOCR (PP-OCRv4) detection + recognition, run in the browser with ONNX Runtime
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';

const MODELS = 'https://cdn.jsdelivr.net/npm/@gutenye/ocr-models@1.4.2/assets/';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let loading = null;
export function loadOcr() {
  if (!loading) {
    loading = (async () => {
      const [det, rec, keys] = await Promise.all([
        ort.InferenceSession.create(MODELS + 'ch_PP-OCRv4_det_infer.onnx'),
        ort.InferenceSession.create(MODELS + 'ch_PP-OCRv4_rec_infer.onnx'),
        fetch(MODELS + 'ppocr_keys_v1.txt').then(r => { if (!r.ok) throw new Error('Could not load the text reader'); return r.text(); })
      ]);
      const dict = keys.split('\n');
      if (dict[dict.length - 1] === '') dict.pop();
      dict.push(' ');
      return { det, rec, dict };
    })();
    loading.catch(() => { loading = null; });
  }
  return loading;
}

const detCanvas = document.createElement('canvas');
const detCtx = detCanvas.getContext('2d', { willReadFrequently: true });
const recCanvas = document.createElement('canvas');
const recCtx = recCanvas.getContext('2d', { willReadFrequently: true });
const sigCanvas = Object.assign(document.createElement('canvas'), { width: 48, height: 12 });
const sigCtx = sigCanvas.getContext('2d', { willReadFrequently: true });
let cache = []; // text read in the previous frame, reused when the same crop shows up again

export function resetTextCache() { cache = []; }

function signature(canvas, x, y, w, h) {
  sigCtx.drawImage(canvas, x, y, w, h, 0, 0, 48, 12);
  const d = sigCtx.getImageData(0, 0, 48, 12).data;
  const g = new Float32Array(576);
  for (let i = 0; i < 576; i++) g[i] = d[i * 4] * .3 + d[i * 4 + 1] * .59 + d[i * 4 + 2] * .11;
  return g;
}

function sameCrop(a, b) {
  if (Math.abs(a.x - b.x) > 0.01 || Math.abs(a.y - b.y) > 0.01 || Math.abs(a.w - b.w) > 0.02 || Math.abs(a.h - b.h) > 0.01) return false;
  let diff = 0;
  for (let i = 0; i < 576; i++) diff += Math.abs(a.sig[i] - b.sig[i]);
  return diff / 576 < 6;
}

/**
 * Reads text in a frame. Returns lines with boxes as fractions of the frame:
 * box = padded box (good for cropping), core = tight glyph box (good for measuring size).
 */
export async function readText(canvas, { detLong = 512, minConf = 0.6 } = {}) {
  const { det, rec, dict } = await loadOcr();
  const W = canvas.width, H = canvas.height;
  const s = detLong / Math.max(W, H);
  const dw = Math.max(32, Math.round(W * s / 32) * 32), dh = Math.max(32, Math.round(H * s / 32) * 32);
  detCanvas.width = dw; detCanvas.height = dh;
  detCtx.drawImage(canvas, 0, 0, dw, dh);
  const px = detCtx.getImageData(0, 0, dw, dh).data;
  const n = dw * dh;
  const input = new Float32Array(3 * n);
  // PaddleOCR detection expects BGR planes normalised with ImageNet mean/std
  for (let i = 0; i < n; i++) {
    input[i] = (px[i * 4 + 2] / 255 - 0.485) / 0.229;
    input[n + i] = (px[i * 4 + 1] / 255 - 0.456) / 0.224;
    input[2 * n + i] = (px[i * 4] / 255 - 0.406) / 0.225;
  }
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, dh, dw]) });
  const prob = out[det.outputNames[0]].data;
  const boxes = findBoxes(prob, dw, dh);

  const lines = [];
  const nextCache = [];
  for (const b of boxes) {
    const bx = b.x0 / dw * W, by = b.y0 / dh * H, bw = (b.x1 - b.x0) / dw * W, bh = (b.y1 - b.y0) / dh * H;
    if (bh < 6 || bw < 6) continue;
    const key = { x: bx / W, y: by / H, w: bw / W, h: bh / H, sig: signature(canvas, bx, by, bw, bh) };
    const hit = cache.find(c => sameCrop(c, key));
    const res = hit ? hit.res : await recognize(rec, dict, canvas, bx, by, bw, bh);
    nextCache.push({ ...key, res });
    if (!res.text.trim() || res.conf < minConf) continue;
    const lead = res.text.length - res.text.trimStart().length, text = res.text.trim();
    lines.push({
      text,
      conf: res.conf,
      pos: res.pos.slice(lead, lead + text.length),
      box: { x: bx / W, y: by / H, w: bw / W, h: bh / H },
      core: { x: b.cx0 / dw, y: b.cy0 / dh, w: (b.cx1 - b.cx0) / dw, h: (b.cy1 - b.cy0) / dh }
    });
  }
  cache = nextCache;
  return lines;
}

// DB post-processing: threshold the probability map, take connected regions, expand ("unclip") each box
function findBoxes(prob, w, h) {
  const seen = new Uint8Array(w * h);
  const boxes = [];
  const stack = [];
  for (let start = 0; start < w * h; start++) {
    if (seen[start] || prob[start] < 0.3) continue;
    let x0 = w, y0 = h, x1 = 0, y1 = 0, sum = 0, count = 0;
    stack.push(start);
    seen[start] = 1;
    while (stack.length) {
      const p = stack.pop();
      const x = p % w, y = (p / w) | 0;
      sum += prob[p]; count++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (x > 0 && !seen[p - 1] && prob[p - 1] >= 0.3) { seen[p - 1] = 1; stack.push(p - 1); }
      if (x < w - 1 && !seen[p + 1] && prob[p + 1] >= 0.3) { seen[p + 1] = 1; stack.push(p + 1); }
      if (y > 0 && !seen[p - w] && prob[p - w] >= 0.3) { seen[p - w] = 1; stack.push(p - w); }
      if (y < h - 1 && !seen[p + w] && prob[p + w] >= 0.3) { seen[p + w] = 1; stack.push(p + w); }
    }
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    if (count < 8 || bh < 3 || sum / count < 0.55) continue;
    const d = (bw * bh * 1.5) / (2 * (bw + bh));
    boxes.push({
      x0: Math.max(0, x0 - d), y0: Math.max(0, y0 - d), x1: Math.min(w, x1 + 1 + d), y1: Math.min(h, y1 + 1 + d),
      cx0: x0, cy0: y0, cx1: x1 + 1, cy1: y1 + 1
    });
  }
  return boxes.sort((a, b) => a.y0 - b.y0 || a.x0 - b.x0);
}

async function recognize(rec, dict, canvas, x, y, w, h) {
  const th = 48;
  const tw = Math.min(1280, Math.max(16, Math.round(th * w / h / 8) * 8));
  recCanvas.width = tw; recCanvas.height = th;
  recCtx.drawImage(canvas, x, y, w, h, 0, 0, tw, th);
  const px = recCtx.getImageData(0, 0, tw, th).data;
  const n = tw * th;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i] = px[i * 4 + 2] / 127.5 - 1;
    input[n + i] = px[i * 4 + 1] / 127.5 - 1;
    input[2 * n + i] = px[i * 4] / 127.5 - 1;
  }
  const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, th, tw]) });
  const t = out[rec.outputNames[0]];
  const [, steps, classes] = t.dims;
  const data = t.data;
  // pos = where each character sits along the line (in model steps), used later to find missing spaces
  let text = '', confSum = 0, chars = 0, prev = -1;
  const pos = [];
  for (let s = 0; s < steps; s++) {
    let best = 0, bestP = -Infinity;
    const off = s * classes;
    for (let c = 0; c < classes; c++) if (data[off + c] > bestP) { bestP = data[off + c]; best = c; }
    if (best !== 0 && best !== prev) {
      const ch = dict[best - 1] ?? '';
      text += ch; confSum += bestP; chars++;
      for (let k = 0; k < ch.length; k++) pos.push([s, s]);
    } else if (best !== 0 && pos.length) pos[pos.length - 1][1] = s;
    prev = best;
  }
  return { text, conf: chars ? confSum / chars : 0, pos: pos.map(([a, b]) => (a + b) / 2) };
}
