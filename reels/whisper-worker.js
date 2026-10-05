// Speech-to-text (Whisper) in a background worker so the page stays responsive
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js';

const MODELS = { fast: 'Xenova/whisper-base', accurate: 'Xenova/whisper-small' };
const loaded = {};

function getAsr(quality) {
  const id = MODELS[quality] || MODELS.fast;
  if (!loaded[id]) {
    loaded[id] = pipeline('automatic-speech-recognition', id, {
      device: 'wasm',
      dtype: 'q8',
      progress_callback: p => { if (p.status === 'progress_total') postMessage({ type: 'model', progress: p.progress }); }
    });
    loaded[id].catch(() => { delete loaded[id]; });
  }
  return loaded[id];
}

async function handle({ id, audio, quality }) {
  try {
    const run = await getAsr(quality);
    postMessage({ id, type: 'status', text: 'Transcribing' });
    const opts = { chunk_length_s: 30, stride_length_s: 5 };
    let out, words = true;
    try {
      out = await run(audio, { ...opts, return_timestamps: 'word' });
    } catch {
      words = false; // fall back to phrase-level timestamps
      out = await run(audio, { ...opts, return_timestamps: true });
    }
    postMessage({ id, type: 'done', words, chunks: (out.chunks || []).map(c => ({ text: c.text, timestamp: c.timestamp })) });
  } catch (e) {
    postMessage({ id, type: 'error', error: String(e && e.message || e) });
  }
}

let chain = Promise.resolve();
self.onmessage = e => { chain = chain.then(() => handle(e.data)); };
