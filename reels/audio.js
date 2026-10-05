// Sound: decode the audio, find sudden sounds, label them with YAMNet (whoosh, pop, ding...), detect background music
const MP_AUDIO = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@1.0.1';
const YAMNET = 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite';

// Sound-effect-like YAMNet classes (AudioSet names); speech and music classes are excluded below
const SFX = /whoosh|swish|swoosh|^pop$|click|ding|bell|chime|cash register|thump|thud|boom|bang|explosion|slam|knock|tap|clang|clank|clink|glass|shatter|crack|snap|whip|zip|beep|bleep|tick|buzz|boing|zing|spring|camera|clap|typing|rustle|siren|alarm|coin|jingle|rattle|crunch|crumpl|squeak|splash|drip|thunder|gunshot|punch|sound effect|wind|wood block|cowbell|gong|xylophone|glockenspiel|marimba|chirp|sine wave|tone|ringtone|whistl|horn|hammer|smash|scratch|slap|thwack|chop|ping|electronic tuner|synthetic|noise/i;
const NOT_SFX = /music|speech|singing|song|conversation|narration|silence|babbling|choir|rapping|hip hop|beatbox|chant|humming|inside, small room/i;

export async function decodeAudio(file) {
  const buf = await file.arrayBuffer();
  const audio = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(buf);
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const ch = audio.getChannelData(c);
    for (let i = 0; i < ch.length; i++) mono[i] += ch[i] / audio.numberOfChannels;
  }
  // 16 kHz copy for the speech and sound models
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(audio.duration * 16000)), 16000);
  const src = off.createBufferSource();
  src.buffer = audio;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { mono, sr: audio.sampleRate, pcm16: rendered.getChannelData(0).slice() };
}

let clsPromise = null;
export function loadSoundClassifier() {
  if (!clsPromise) {
    clsPromise = (async () => {
      const { FilesetResolver, AudioClassifier } = await import(`${MP_AUDIO}/audio_bundle.mjs`);
      const files = await FilesetResolver.forAudioTasks(`${MP_AUDIO}/wasm`);
      return AudioClassifier.createFromOptions(files, { baseOptions: { modelAssetPath: YAMNET }, maxResults: 15, scoreThreshold: 0.01 });
    })();
    clsPromise.catch(() => { clsPromise = null; });
  }
  return clsPromise;
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

// Spectral-flux onsets: moments where new sound energy suddenly appears
export function findOnsets(x, sr = 16000) {
  const size = 1024, hop = 160;
  const frames = Math.max(0, Math.floor((x.length - size) / hop) + 1);
  const win = Float32Array.from({ length: size }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / size));
  const re = new Float32Array(size), im = new Float32Array(size);
  let prev = null;
  const flux = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < size; i++) { re[i] = x[f * hop + i] * win[i]; im[i] = 0; }
    fft(re, im);
    const mag = new Float32Array(size / 2);
    let s = 0;
    for (let k = 1; k < size / 2; k++) {
      mag[k] = Math.log1p(10 * Math.hypot(re[k], im[k]));
      if (prev) s += Math.max(0, mag[k] - prev[k]);
    }
    flux[f] = s;
    prev = mag;
  }
  const max = Math.max(...flux, 1e-9);
  const onsets = [];
  let last = -100;
  for (let f = 1; f < frames - 1; f++) {
    const a = Math.max(0, f - 25), b = Math.min(frames, f + 26);
    const local = Array.from(flux.slice(a, b)).sort((p, q) => p - q);
    const th = local[local.length >> 1] * 1.5 + 0.06 * max;
    let peak = true;
    for (let g = Math.max(0, f - 5); g <= Math.min(frames - 1, f + 5); g++) if (flux[g] > flux[f]) { peak = false; break; }
    if (peak && flux[f] > th && f - last >= 8) { onsets.push({ t: (f * hop + size / 2) / sr, strength: flux[f] / max }); last = f; }
  }
  return onsets;
}

const simplify = name => name.split(',')[0].trim().toLowerCase();

export async function analyzeSound(pcm16) {
  const onsets = findOnsets(pcm16);
  const cls = await loadSoundClassifier();
  const clip = t => {
    const a = Math.max(0, Math.round((t - 0.05) * 16000));
    const out = new Float32Array(15600);
    out.set(pcm16.subarray(a, a + 15600));
    return out;
  };
  const hits = [];
  for (const o of onsets) {
    const cats = cls.classify(clip(o.t), 16000)?.[0]?.classifications?.[0]?.categories || [];
    const speech = cats.find(c => c.categoryName === 'Speech')?.score ?? 0;
    const sfx = cats.find(c => SFX.test(c.categoryName) && !NOT_SFX.test(c.categoryName));
    const generic = sfx && /sound effect|noise|sine wave|tone|synthetic|electronic|static|hum/i.test(sfx.categoryName);
    if (sfx && (generic ? sfx.score >= 0.15 : (sfx.score >= 0.12 || (sfx.score >= 0.05 && speech < 0.4)))) {
      hits.push({ t: Math.round(o.t * 100) / 100, label: simplify(sfx.categoryName), score: Math.round(sfx.score * 100) / 100, speech: Math.round(speech * 100) / 100, strength: o.strength });
    } else if (o.strength > 0.45 && speech < 0.15) {
      hits.push({ t: Math.round(o.t * 100) / 100, label: 'hit (unclear type)', score: 0, speech: Math.round(speech * 100) / 100, strength: o.strength });
    }
  }
  // One hit per sound: merge hits closer than 0.2 s
  const merged = [];
  for (const h of hits) {
    const prev = merged[merged.length - 1];
    if (prev && h.t - prev.t < 0.2) { if (h.score > prev.score) merged[merged.length - 1] = h; }
    else merged.push(h);
  }

  // Background music: YAMNet's "Music" score over ~1 s windows
  const windows = (cls.classify(pcm16, 16000) || []).map(r => {
    const cats = r.classifications?.[0]?.categories || [];
    return { t: (r.timestampMs ?? 0) / 1000, music: cats.find(c => c.categoryName === 'Music')?.score ?? 0, speech: cats.find(c => c.categoryName === 'Speech')?.score ?? 0 };
  });
  const step = windows.length > 1 ? windows[1].t - windows[0].t : 0.975;
  const segments = [];
  for (const w of windows) {
    if (w.music < 0.15) continue;
    const last = segments[segments.length - 1];
    if (last && w.t - last.end <= step + 0.01) last.end = w.t + step;
    else segments.push({ start: w.t, end: w.t + step });
  }
  const rms = (a, b) => { let s = 0; const i0 = Math.max(0, Math.floor(a * 16000)), i1 = Math.min(pcm16.length, Math.floor(b * 16000)); for (let i = i0; i < i1; i++) s += pcm16[i] ** 2; return i1 > i0 ? Math.sqrt(s / (i1 - i0)) : 0; };
  const musicOnly = windows.filter(w => w.music >= 0.3 && w.speech < 0.15).map(w => rms(w.t, w.t + step));
  const speechWin = windows.filter(w => w.speech >= 0.5).map(w => rms(w.t, w.t + step));
  const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
  const underVoiceDb = musicOnly.length && speechWin.length && avg(speechWin) > 0 ? Math.round(20 * Math.log10(avg(musicOnly) / avg(speechWin))) : null;
  const total = pcm16.length / 16000;
  const coverage = segments.reduce((s, g) => s + Math.min(total, g.end) - g.start, 0) / (total || 1);
  return { hits: merged, music: { present: coverage > 0.15, segments, coverage, underVoiceDb } };
}

// A short WAV clip of a sound effect, so it can be heard and matched
export function wavClip(mono, sr, t, dur = 0.8) {
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
