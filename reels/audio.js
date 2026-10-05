// Sound: decode the audio and find sudden bumps of non-voice sound (no labels; the times and strength are all we report)

export async function decodeAudio(file) {
  const buf = await file.arrayBuffer();
  const audio = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(buf);
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const ch = audio.getChannelData(c);
    for (let i = 0; i < ch.length; i++) mono[i] += ch[i] / audio.numberOfChannels;
  }
  // 16 kHz copy for the speech model and the bump finder
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(audio.duration * 16000)), 16000);
  const src = off.createBufferSource();
  src.buffer = audio;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { mono, sr: audio.sampleRate, pcm16: rendered.getChannelData(0).slice() };
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

const med = (a, tmp) => { for (let i = 0; i < a.length; i++) tmp[i] = a[i]; tmp.sort(); return tmp[tmp.length >> 1]; };

// Sudden sounds on the non-voice part of the audio. Voice (and held music notes) is mostly harmonic:
// steady lines across time in the spectrum. Hits, pops and whooshes are percussive: short and spread
// across many frequencies. Median filtering splits the two (harmonic/percussive separation); onsets are
// then found on the percussive part only. Bumps that still land inside spoken words are flagged later.
export function findBumps(x, sr = 16000) {
  const size = 512, hop = 160, bins = size / 2;
  const frames = Math.max(0, Math.floor((x.length - size) / hop) + 1);
  const win = Float32Array.from({ length: size }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / size));
  const re = new Float32Array(size), im = new Float32Array(size);
  const mag = Array.from({ length: frames }, () => new Float32Array(bins));
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < size; i++) { re[i] = x[f * hop + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < bins; k++) mag[f][k] = Math.hypot(re[k], im[k]);
  }
  const HT = 17, HF = 17;
  const colT = new Float32Array(HT), tmpT = new Float32Array(HT), colF = new Float32Array(HF), tmpF = new Float32Array(HF);
  let prev = null;
  const flux = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    const perc = new Float32Array(bins);
    for (let k = 1; k < bins; k++) {
      for (let j = 0; j < HT; j++) colT[j] = mag[clampI(f + j - (HT >> 1), frames)][k];
      for (let j = 0; j < HF; j++) colF[j] = mag[f][clampI(k + j - (HF >> 1), bins)];
      const h = med(colT, tmpT), p = med(colF, tmpF);
      const m = mag[f][k] * (p * p) / (h * h + p * p + 1e-12);
      perc[k] = Math.log1p(10 * m);
    }
    if (prev) { let s = 0; for (let k = 1; k < bins; k++) s += Math.max(0, perc[k] - prev[k]); flux[f] = s; }
    prev = perc;
  }
  let max = 1e-9;
  for (let f = 0; f < frames; f++) if (flux[f] > max) max = flux[f];
  const bumps = [];
  let last = -100;
  for (let f = 1; f < frames - 1; f++) {
    const a = Math.max(0, f - 25), b = Math.min(frames, f + 26);
    const local = Array.from(flux.slice(a, b)).sort((p, q) => p - q);
    const th = local[local.length >> 1] * 1.5 + 0.15 * max;
    let peak = true;
    for (let g = Math.max(0, f - 5); g <= Math.min(frames - 1, f + 5); g++) if (flux[g] > flux[f]) { peak = false; break; }
    if (peak && flux[f] > th && f - last >= 12) { bumps.push({ t: Math.round((f * hop + size / 2) / sr * 100) / 100, flux: flux[f] / max }); last = f; }
  }
  return bumps;
}
const clampI = (i, n) => i < 0 ? 0 : i >= n ? n - 1 : i;

// A short WAV clip of a bump, so it can be heard
export function wavClip(mono, sr, t, dur = 0.6) {
  const a = Math.max(0, Math.round((t - 0.05) * sr));
  const data = mono.subarray(a, Math.min(mono.length, a + Math.round(dur * sr)));
  const buf = new ArrayBuffer(44 + data.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + data.length * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, data.length * 2, true);
  for (let i = 0; i < data.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, data[i])) * 32767, true);
  return new Blob([buf], { type: 'audio/wav' });
}
