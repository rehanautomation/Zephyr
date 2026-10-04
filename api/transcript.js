// GET /api/transcript?id=VIDEO_ID  ->  { id, title, author, transcript, words }
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';

function fail(status, message) { const e = new Error(message); e.status = status; return e; }

function decode(s) {
  return s
    .replace(/<[^>]+>/g, '')
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(+n))
    .replace(/\s+/g, ' ').trim();
}

async function getTranscript(id) {
  // 1. Watch page -> innertube key
  const html = await fetch(`https://www.youtube.com/watch?v=${id}&hl=en`, {
    headers: { 'User-Agent': UA, 'Accept-Language': 'en-US,en;q=0.9', Cookie: 'CONSENT=YES+1' }
  }).then(r => r.text());
  const key = (html.match(/"INNERTUBE_API_KEY":"([^"]+)"/) || [])[1];

  // 2. Player data via the Android client (caption URLs work without extra tokens)
  const player = await fetch(`https://www.youtube.com/youtubei/v1/player${key ? `?key=${key}` : ''}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'User-Agent': UA },
    body: JSON.stringify({ context: { client: { clientName: 'ANDROID', clientVersion: '20.10.38' } }, videoId: id })
  }).then(r => r.json());

  const ps = player.playabilityStatus || {};
  if (ps.status && ps.status !== 'OK') {
    throw fail(ps.status === 'LOGIN_REQUIRED' ? 429 : 404, ps.reason || 'Video unavailable');
  }

  const title = player.videoDetails?.title || '';
  const author = player.videoDetails?.author || '';
  const tracks = player.captions?.playerCaptionsTracklistRenderer?.captionTracks || [];
  if (!tracks.length) throw fail(404, 'This video has no captions');

  const isEn = t => (t.languageCode || '').startsWith('en');
  const track =
    tracks.find(t => isEn(t) && t.kind !== 'asr') ||
    tracks.find(isEn) ||
    tracks.find(t => t.kind !== 'asr') ||
    tracks[0];

  // 3. Caption XML -> plain text
  const xml = await fetch(track.baseUrl.replace(/&fmt=\w+/, ''), { headers: { 'User-Agent': UA } }).then(r => r.text());
  const parts = [...xml.matchAll(/<(?:text|p)\b[^>]*>([\s\S]*?)<\/(?:text|p)>/g)].map(m => decode(m[1])).filter(Boolean);
  if (!parts.length) throw fail(502, 'YouTube returned an empty transcript');

  const transcript = parts.join(' ');
  return { id, title, author, transcript, words: transcript.split(' ').length };
}

// Title + channel from YouTube oEmbed (not bot-blocked)
async function getMeta(id) {
  try {
    const r = await fetch(`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${id}&format=json`);
    if (!r.ok) return {};
    const d = await r.json();
    return { title: d.title || '', author: d.author_name || '' };
  } catch { return {}; }
}

// Transcript via Supadata (YouTube blocks Vercel servers directly)
async function getViaSupadata(id, key) {
  const [r, meta] = await Promise.all([
    fetch(`https://api.supadata.ai/v1/youtube/transcript?videoId=${id}&text=true`, { headers: { 'x-api-key': key } }),
    getMeta(id)
  ]);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw fail(r.status === 404 ? 404 : r.status, d.message || d.error || `Transcript service error ${r.status}`);
  const transcript = String(d.content || '').replace(/\s+/g, ' ').trim();
  if (!transcript) throw fail(404, 'This video has no captions');
  return { id, title: meta.title || '', author: meta.author || '', transcript, words: transcript.split(' ').length };
}

export default async function handler(req, res) {
  const id = String(req.query.id || '').trim();
  if (!/^[\w-]{11}$/.test(id)) return res.status(400).json({ error: 'Not a valid YouTube link' });
  const key = process.env.SUPADATA_API_KEY;

  let lastErr;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const out = key ? await getViaSupadata(id, key) : await getTranscript(id);
      res.setHeader('Cache-Control', 's-maxage=86400, stale-while-revalidate');
      return res.status(200).json(out);
    } catch (e) {
      lastErr = e;
      if (e.status === 404) break; // no point retrying a video with no captions
    }
  }
  res.status(lastErr.status || 502).json({ error: lastErr.message || 'Could not reach YouTube' });
}
