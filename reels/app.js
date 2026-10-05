// Reels page: queue, progress, results and the two delivery buttons (Copy text / Get images)
import { openVideo, scanVideo, buildEvents, planImages, renderSheets, fmt } from './vision.js';
import { decodeAudio, findBumps, wavClip } from './audio.js';
import { groupLooks, cleanTranscript, packageText, bumpList } from './report.js';

const ICON = {
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
  spin: '<svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M21 12a9 9 0 1 1-6.2-8.56"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2.5"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>'
};
const STEP_LABEL = { scan: 'Video scan', sound: 'Sound bumps', transcript: 'Transcript', images: 'Images' };
const IMAGE_LIMIT = 20;

const $ = s => document.querySelector(s);
const ui = {
  drop: $('#drop'), file: $('#file'), quality: $('#quality'), bar: $('#bar'), count: $('#count'), warn: $('#warn'),
  copyText: $('#copyText'), getImages: $('#getImages'), zip: $('#zip'), clear: $('#clear'), list: $('#list'), toast: $('#toast')
};

const registry = { looks: [], screens: [] }; // similar-looking text and graphic screens across the batch
const reels = [];
let counter = 0;
let queue = Promise.resolve();

/* ---------- Small helpers ---------- */
let toastTimer;
function toast(msg, ms = 2600) {
  ui.toast.textContent = msg;
  ui.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove('show'), ms);
}
const pad2 = n => String(n).padStart(2, '0');
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

async function writeClipboard(text) {
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = Object.assign(document.createElement('textarea'), { value: text });
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    if (!ok) throw new Error('Copy failed');
  }
}

async function toPng(blob) {
  const bmp = await createImageBitmap(blob);
  const c = Object.assign(document.createElement('canvas'), { width: bmp.width, height: bmp.height });
  c.getContext('2d').drawImage(bmp, 0, 0);
  bmp.close();
  return new Promise(r => c.toBlob(r, 'image/png'));
}

/* ---------- Speech-to-text worker ---------- */
let worker = null;
const jobs = new Map();
let modelProgress = null;
function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./whisper-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.type === 'model') {
      modelProgress = data.progress;
      for (const r of reels) if (r.state.transcript === 'run' && !r.transcribing) setStep(r, 'transcript', 'run', `downloading speech model (one time) ${Math.round(data.progress)}%`);
      return;
    }
    const job = jobs.get(data.id);
    if (!job) return;
    if (data.type === 'status') job.onStatus(data.text);
    else { jobs.delete(data.id); data.type === 'done' ? job.resolve(data) : job.reject(new Error(data.error)); }
  };
  worker.onerror = e => {
    for (const [, job] of jobs) job.reject(new Error(e.message || 'speech engine failed to load'));
    jobs.clear();
    worker = null;
  };
  return worker;
}
function transcribe(r, audio, quality) {
  return new Promise((resolve, reject) => {
    const id = `${r.n}-${Date.now()}`;
    jobs.set(id, { resolve, reject, onStatus: text => { r.transcribing = true; setStep(r, 'transcript', 'run', text.toLowerCase()); } });
    getWorker().postMessage({ id, audio, quality }, [audio.buffer]);
  });
}

/* ---------- Reel cards ---------- */
function addCard(r) {
  const el = document.createElement('article');
  el.className = 'item';
  el.innerHTML = `
    <img class="poster" alt="">
    <div class="item-main">
      <div class="item-head">
        <div><p class="name"></p><p class="info">Waiting…</p></div>
        <div class="item-actions">
          <button class="btn small copy-reel" type="button" disabled>${ICON.copy}<span>Copy this reel</span></button>
          <button class="remove" type="button" aria-label="Remove video" title="Remove">${ICON.x}</button>
        </div>
      </div>
      <ul class="steps">${Object.keys(STEP_LABEL).map(k => `<li data-k="${k}"><span class="dot"></span><span class="txt"></span></li>`).join('')}</ul>
      <div class="sheets"></div>
    </div>`;
  el.querySelector('.name').textContent = `Reel ${r.n} · ${r.name}`;
  el.querySelector('.remove').addEventListener('click', () => removeReel(r));
  el.querySelector('.copy-reel').addEventListener('click', e => copyText([r], e.currentTarget));
  r.el = el;
  for (const k of Object.keys(STEP_LABEL)) setStep(r, k, 'wait');
  ui.list.append(el);
}

function setStep(r, key, s, text) {
  r.state[key] = s;
  const li = r.el.querySelector(`[data-k="${key}"]`);
  li.dataset.s = s;
  li.querySelector('.dot').innerHTML = s === 'done' ? ICON.check : s === 'run' ? ICON.spin : s === 'error' ? ICON.alert : '';
  li.querySelector('.txt').textContent = text ? `${STEP_LABEL[key]}: ${text}` : STEP_LABEL[key];
  refresh();
}

function renderSheetsUI(r) {
  const box = r.el.querySelector('.sheets');
  box.replaceChildren(...r.sheets.map(sh => {
    const fig = document.createElement('figure');
    fig.innerHTML = `<a target="_blank" title="${sh.name}"><img alt=""></a><figcaption><span></span><button class="btn tiny" type="button">${ICON.copy}<span>Copy</span></button></figcaption>`;
    fig.querySelector('a').href = sh.url;
    fig.querySelector('img').src = sh.url;
    fig.querySelector('img').alt = sh.title;
    fig.querySelector('figcaption span').textContent = sh.title.split(' · ').pop();
    const btn = fig.querySelector('button');
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': toPng(sh.blob) })]);
        btn.classList.add('done');
        btn.querySelector('span').textContent = 'Copied';
        setTimeout(() => { btn.classList.remove('done'); btn.querySelector('span').textContent = 'Copy'; }, 1500);
        toast(`${sh.name} copied. Paste it into Claude`);
      } catch {
        toast('Your browser blocked copying images. Use Get images instead');
      }
    });
    return fig;
  }));
}

function removeReel(r) {
  r.removed = true;
  r.el.remove();
  for (const sh of r.sheets) URL.revokeObjectURL(sh.url);
  reels.splice(reels.indexOf(r), 1);
  refresh();
}

function refresh() {
  ui.bar.hidden = !reels.length;
  const busy = reels.filter(r => Object.values(r.state).some(s => s === 'run' || s === 'wait')).length;
  const images = reels.reduce((s, r) => s + r.sheets.length, 0);
  ui.count.textContent = `${plural(reels.length, 'reel')} · batch total: ${plural(images, 'image')}${busy ? ` · ${busy} working` : ' · ready'}`;
  ui.warn.hidden = images <= IMAGE_LIMIT;
  if (images > IMAGE_LIMIT) {
    const per = Math.max(1, Math.floor(IMAGE_LIMIT / Math.max(1, images / reels.length)));
    ui.warn.textContent = `${images} images is a lot for one Claude message (about ${IMAGE_LIMIT} works best). Split it: use each reel's "Copy this reel" button and do about ${per} reel${per === 1 ? '' : 's'} per chat.`;
  }
  const ready = reels.filter(r => r.ev);
  ui.copyText.disabled = !ready.length;
  ui.getImages.disabled = ui.zip.disabled = !images;
  for (const r of reels) {
    r.el.querySelector('.copy-reel').disabled = !r.ev;
    r.el.querySelector('.info').textContent = r.dur ? `${fmt(r.dur)} · ${r.vw}×${r.vh}${r.sheets.length ? ` · ${plural(r.sheets.length, 'image')}` : ''}` : r.el.querySelector('.info').textContent;
  }
}

/* ---------- Processing ---------- */
function addFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith('video/') && !/\.(mp4|mov|webm|m4v|mkv)$/i.test(file.name)) continue;
    const r = { n: ++counter, file, name: file.name, state: {}, sheets: [], omitted: [], bumpClips: new Map() };
    reels.push(r);
    addCard(r);
    queue = queue.then(() => processReel(r));
  }
}

async function processReel(r) {
  if (r.removed) return;
  let video = null, step = 'scan';
  try {
    setStep(r, 'scan', 'run', 'loading (first time downloads the analysis models)');
    video = await openVideo(r.file);
    r.dur = video.duration; r.vw = video.videoWidth; r.vh = video.videoHeight;
    const pc = Object.assign(document.createElement('canvas'), { width: 180, height: Math.round(180 * r.vh / r.vw) });
    pc.getContext('2d').drawImage(video, 0, 0, pc.width, pc.height);
    r.el.querySelector('.poster').src = pc.toDataURL('image/jpeg', 0.7);
    const scan = await scanVideo(video, p => setStep(r, 'scan', 'run', `${Math.round(p * 100)}%`));
    if (r.removed) return;
    r.ocrError = scan.ocrError;
    r.scanFaceOk = scan.faceOk;
    r.ev = buildEvents(scan);
    const texts = r.ev.textEvents.filter(e => !e.persistent).length;
    setStep(r, 'scan', 'done', `${plural(r.ev.transitions.length, 'cut')} · ${plural(texts, 'text')} · ${plural(r.ev.zooms.length, 'zoom')} · ${plural(r.ev.screens.length, 'graphic screen')}`);

    step = 'sound';
    setStep(r, 'sound', 'run', 'listening');
    let audio = null;
    try { audio = await decodeAudio(r.file); } catch { audio = null; }
    if (audio) {
      try {
        // Clips for the strongest candidates; the voice filter picks which ones are listed once speech is ready
        r.bumps = findBumps(audio.pcm16);
        for (const b of [...r.bumps].sort((x, y) => y.flux - x.flux).slice(0, 60)) {
          r.bumpClips.set(b.t, { name: `reel${pad2(r.n)}_bump_${b.t.toFixed(2)}s.wav`, blob: wavClip(audio.mono, audio.sr, b.t) });
        }
        setStep(r, 'sound', 'done', `${plural(r.bumps.length, 'bump')} found (voice filtered after the transcript)`);
      } catch (e) {
        r.soundError = `sound check failed (${e.message})`;
        setStep(r, 'sound', 'error', 'sound check failed');
      }
      startTranscript(r, audio.pcm16);
    } else {
      r.soundError = 'no audio track in this video';
      r.transcriptError = 'no audio in this video';
      setStep(r, 'sound', 'error', 'no audio in this video');
      setStep(r, 'transcript', 'error', 'no audio to transcribe');
    }

    step = 'images';
    setStep(r, 'images', 'run', 'picking frames');
    groupLooks(r, registry);
    const plan = planImages(r.n, r.ev, scan);
    r.omitted = plan.omitted;
    scan.gray = scan.col = null; // free memory before the next reel
    r.sheets = await renderSheets(video, plan);
    renderSheetsUI(r);
    setStep(r, 'images', 'done', `${plural(r.sheets.length, 'image')}`);
  } catch (e) {
    setStep(r, step, 'error', e.message || 'something went wrong');
    for (const k of Object.keys(STEP_LABEL)) if (r.state[k] === 'wait') setStep(r, k, 'error', 'skipped');
  } finally {
    if (video) { URL.revokeObjectURL(video.src); video.removeAttribute('src'); video.load(); }
  }
}

function startTranscript(r, pcm16) {
  setStep(r, 'transcript', 'run', modelProgress !== null && modelProgress >= 100 ? 'waiting' : 'loading speech model (one-time download)');
  transcribe(r, pcm16.slice(), ui.quality.value).then(res => {
    r.transcript = cleanTranscript(res);
    const n = r.transcript.words.length;
    if (!r.removed) {
      setStep(r, 'transcript', 'done', n ? plural(n, 'word') : 'no speech detected');
      const b = bumpList(r);
      if (b && r.state.sound === 'done') setStep(r, 'sound', 'done', `${plural(b.list.length, 'bump')} off the voice`);
    }
  }).catch(e => {
    r.transcriptError = /fetch|network|load/i.test(e.message) ? "couldn't download the speech model (check your internet)" : e.message;
    if (!r.removed) setStep(r, 'transcript', 'error', r.transcriptError);
  });
}

/* ---------- Delivery ---------- */
const allRefs = list => list.flatMap(r => r.sheets.flatMap(sh => sh.refs.map(x => ({ ...x, reel: r.n }))));

async function copyText(list, btn) {
  const ready = list.filter(r => r.ev);
  if (!ready.length) return;
  try {
    await writeClipboard(packageText(ready, allRefs(ready)));
    const pending = ready.some(r => r.state.transcript === 'run' || r.state.images === 'run');
    toast(pending ? 'Copied, but some reels are still working. Copy again when everything shows a check' : `Copied. Paste into Claude, then add the ${plural(ready.reduce((s, r) => s + r.sheets.length, 0), 'image')}`);
    if (btn) {
      const span = btn.querySelector('span');
      const old = span ? span.textContent : btn.textContent;
      btn.classList.add('done');
      if (span) span.textContent = 'Copied'; else btn.textContent = 'Copied';
      setTimeout(() => { btn.classList.remove('done'); if (span) span.textContent = old; else btn.textContent = old; }, 1600);
    }
  } catch {
    toast('Your browser blocked clipboard access');
  }
}

function batchFiles() {
  const files = [];
  for (const r of reels) for (const sh of r.sheets) files.push({ name: sh.name, blob: sh.blob });
  const ready = reels.filter(r => r.ev);
  if (ready.length) files.push({ name: 'for-claude.txt', blob: new Blob([packageText(ready, allRefs(ready))], { type: 'text/plain' }) });
  for (const r of ready) for (const b of bumpList(r)?.list || []) { const clip = r.bumpClips.get(b.t); if (clip) files.push({ dir: 'sounds', name: clip.name, blob: clip.blob }); }
  return files;
}

function stamp() {
  const d = new Date();
  return `zephyr-reels-${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}-${pad2(d.getHours())}${pad2(d.getMinutes())}`;
}

async function saveToFolder() {
  const files = batchFiles();
  const images = files.filter(f => f.name.endsWith('.jpg')).length;
  if (!window.showDirectoryPicker) return downloadZip(files);
  let root;
  try {
    root = await window.showDirectoryPicker({ id: 'zephyr-reels', mode: 'readwrite', startIn: 'downloads' });
  } catch (e) {
    if (e.name === 'AbortError') return;
    return downloadZip(files);
  }
  try {
    const dir = await root.getDirectoryHandle(stamp(), { create: true });
    for (const f of files) {
      const d = f.dir ? await dir.getDirectoryHandle(f.dir, { create: true }) : dir;
      const w = await (await d.getFileHandle(f.name, { create: true })).createWritable();
      await w.write(f.blob);
      await w.close();
    }
    toast(`Saved ${plural(images, 'image')} to "${root.name}/${dir.name}". Open that folder, select all the images and drag them into Claude`, 7000);
  } catch {
    downloadZip(files);
  }
}

// Minimal ZIP writer (no compression; JPEGs are already compressed)
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(buf) { let c = 0xFFFFFFFF; for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
async function downloadZip(files = batchFiles()) {
  const enc = new TextEncoder();
  const parts = [], central = [];
  let offset = 0;
  for (const f of files) {
    const data = new Uint8Array(await f.blob.arrayBuffer());
    const name = enc.encode(f.dir ? `${f.dir}/${f.name}` : f.name), crc = crc32(data), size = data.length;
    const loc = new DataView(new ArrayBuffer(30));
    loc.setUint32(0, 0x04034b50, true); loc.setUint16(4, 20, true); loc.setUint16(12, 0x21, true);
    loc.setUint32(14, crc, true); loc.setUint32(18, size, true); loc.setUint32(22, size, true); loc.setUint16(26, name.length, true);
    parts.push(loc, name, data);
    const cen = new DataView(new ArrayBuffer(46));
    cen.setUint32(0, 0x02014b50, true); cen.setUint16(4, 20, true); cen.setUint16(6, 20, true); cen.setUint16(14, 0x21, true);
    cen.setUint32(16, crc, true); cen.setUint32(20, size, true); cen.setUint32(24, size, true); cen.setUint16(28, name.length, true); cen.setUint32(42, offset, true);
    central.push(cen, name);
    offset += 30 + name.length + size;
  }
  const cenSize = central.reduce((s, p) => s + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
  end.setUint32(12, cenSize, true); end.setUint32(16, offset, true);
  const url = URL.createObjectURL(new Blob([...parts, ...central, end], { type: 'application/zip' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: `${stamp()}.zip` });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  toast('Zip downloaded. Unzip it, select all the images and drag them into Claude', 5000);
}

/* ---------- Wire up ---------- */
ui.file.addEventListener('change', () => { addFiles(ui.file.files); ui.file.value = ''; });
ui.drop.addEventListener('dragover', e => { e.preventDefault(); ui.drop.classList.add('over'); });
ui.drop.addEventListener('dragleave', () => ui.drop.classList.remove('over'));
ui.drop.addEventListener('drop', e => { e.preventDefault(); ui.drop.classList.remove('over'); addFiles(e.dataTransfer.files); });
ui.copyText.addEventListener('click', e => copyText(reels, e.currentTarget));
ui.getImages.addEventListener('click', saveToFolder);
ui.zip.addEventListener('click', () => downloadZip());
ui.clear.addEventListener('click', () => {
  for (const r of [...reels]) removeReel(r);
  registry.looks.length = 0;
  registry.screens.length = 0;
  counter = 0;
});
