// GET /api/channel?url=CHANNEL_LINK  ->  { channel: { id, title, avatar }, videos: [{ id, title, ago, days, duration, views }], months, partial, source }
// Lists the channel's uploads (Videos tab, newest first) from the last MONTHS months.
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';
const CONTEXT = { client: { clientName: 'WEB', clientVersion: '2.20250101.00.00', hl: 'en', gl: 'US' } };
const VIDEOS_TAB = 'EgZ2aWRlb3PyBgQKAjoA';
const MONTHS = 6;
const MAX_PAGES = 15; // 30 videos a page; keeps the request well inside the function time limit

function fail(status, message) { const e = new Error(message); e.status = status; return e; }

async function yt(endpoint, body) {
  const r = await fetch(`https://www.youtube.com/youtubei/v1/${endpoint}?prettyPrint=false`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'User-Agent': UA, 'Accept-Language': 'en-US,en;q=0.9' },
    body: JSON.stringify({ context: CONTEXT, ...body })
  });
  if (r.status === 404) throw fail(404, 'Channel not found');
  if (!r.ok) throw fail(502, `YouTube error ${r.status}`);
  return r.json();
}

// Accepts youtube.com/@handle, /channel/UC…, /c/name, /user/name (any trailing path) or a bare @handle
function normalize(input) {
  const s = String(input || '').trim();
  const bare = s.match(/^@[\w.\-%]{2,}$/);
  if (bare) return { url: `https://www.youtube.com/${bare[0]}` };
  const m = s.match(/(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/(@[\w.\-%]+|channel\/(UC[\w-]{22})|c\/[^/?#\s]+|user\/[^/?#\s]+)/i);
  if (!m) return null;
  return { url: `https://www.youtube.com/${m[1]}`, id: m[2] };
}

async function resolveId(ref) {
  if (ref.id) return ref.id;
  const d = await yt('navigation/resolve_url', { url: ref.url });
  const id = d.endpoint?.browseEndpoint?.browseId;
  if (!id || !id.startsWith('UC')) throw fail(404, 'Channel not found');
  return id;
}

// "3 weeks ago", "Streamed 2 months ago" -> approximate age in days
const UNIT_DAYS = { second: 1 / 86400, minute: 1 / 1440, hour: 1 / 24, day: 1, week: 7, month: 30.44, year: 365.25 };
function ageDays(text) {
  const m = String(text || '').match(/(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago/i);
  return m ? +m[1] * UNIT_DAYS[m[2].toLowerCase()] : null;
}

const txt = v => v?.content ?? v?.simpleText ?? v?.runs?.map(r => r.text).join('') ?? '';

// YouTube serves either the newer lockupViewModel or the older videoRenderer; read both
function toVideo(item) {
  const c = item.richItemRenderer?.content || item;
  const lv = c.lockupViewModel;
  if (lv && lv.contentType === 'LOCKUP_CONTENT_TYPE_VIDEO') {
    const meta = lv.metadata?.lockupMetadataViewModel;
    const parts = (meta?.metadata?.contentMetadataViewModel?.metadataRows || []).flatMap(r => r.metadataParts || []).map(p => txt(p.text));
    const badges = (lv.contentImage?.thumbnailViewModel?.overlays || []).flatMap(o => o.thumbnailBottomOverlayViewModel?.badges || o.thumbnailOverlayBadgeViewModel?.thumbnailBadges || []);
    const duration = badges.map(b => b.thumbnailBadgeViewModel?.text || '').find(t => /^\d+(:\d\d)+$/.test(t)) || '';
    const ago = parts.find(p => /ago$/i.test(p)) || '';
    return { id: lv.contentId, title: txt(meta?.title), ago, duration, views: parts.find(p => /views?$/i.test(p)) || '' };
  }
  const vr = c.videoRenderer;
  if (vr) {
    return { id: vr.videoId, title: txt(vr.title), ago: txt(vr.publishedTimeText), duration: txt(vr.lengthText), views: txt(vr.shortViewCountText) };
  }
  return null;
}

function continuationOf(items) {
  return items.find(i => i.continuationItemRenderer)?.continuationItemRenderer?.continuationEndpoint?.continuationCommand?.token;
}

async function listViaInnertube(channelId, cutoff) {
  const first = await yt('browse', { browseId: channelId, params: VIDEOS_TAB });
  const meta = first.metadata?.channelMetadataRenderer || {};
  const channel = { id: channelId, title: meta.title || '', avatar: meta.avatar?.thumbnails?.at(-1)?.url || '' };
  const tab = (first.contents?.twoColumnBrowseResultsRenderer?.tabs || []).find(t => t.tabRenderer?.selected && /videos/i.test(t.tabRenderer?.title || ''));
  let items = tab?.tabRenderer?.content?.richGridRenderer?.contents;
  if (!items) return { channel, videos: [], partial: false }; // channel has no Videos tab

  const videos = [];
  for (let page = 1; ; page++) {
    let reachedCutoff = false;
    for (const item of items) {
      const v = toVideo(item);
      if (!v?.id) continue;
      const days = ageDays(v.ago);
      if (days == null) continue; // upcoming premiere or live right now: nothing to transcribe yet
      if (days > cutoff) { reachedCutoff = true; break; }
      videos.push({ ...v, days: Math.round(days * 10) / 10 });
    }
    const token = continuationOf(items);
    if (reachedCutoff || !token) return { channel, videos, partial: false };
    if (page >= MAX_PAGES) return { channel, videos, partial: true };
    const next = await yt('browse', { continuation: token });
    items = (next.onResponseReceivedActions || []).flatMap(a => a.appendContinuationItemsAction?.continuationItems || []);
    if (!items.length) return { channel, videos, partial: false };
  }
}

// Fallback: the channel's RSS feed (exact dates, but only the latest 15 uploads)
async function listViaRss(channelId, cutoff) {
  const r = await fetch(`https://www.youtube.com/feeds/videos.xml?channel_id=${channelId}`, { headers: { 'User-Agent': UA } });
  if (r.status === 404) throw fail(404, 'Channel not found');
  if (!r.ok) throw fail(502, `YouTube error ${r.status}`);
  const xml = await r.text();
  const decode = s => s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'");
  const title = decode((xml.match(/<title>([^<]*)<\/title>/) || [])[1] || '');
  const videos = [...xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)].map(([, e]) => {
    const published = (e.match(/<published>([^<]+)/) || [])[1];
    const days = (Date.now() - Date.parse(published)) / 86400000;
    return {
      id: (e.match(/<yt:videoId>([^<]+)/) || [])[1],
      title: decode((e.match(/<title>([^<]*)/) || [])[1] || ''),
      published, days: Math.round(days * 10) / 10, ago: '', duration: '', views: ''
    };
  }).filter(v => v.id && v.days <= cutoff);
  return { channel: { id: channelId, title, avatar: '' }, videos, partial: videos.length >= 15 };
}

export default async function handler(req, res) {
  const ref = normalize(req.query.url);
  if (!ref) return res.status(400).json({ error: 'Not a YouTube channel link' });
  const cutoff = MONTHS * 30.44 + 1;

  try {
    const channelId = await resolveId(ref);
    let out, source = 'youtube';
    try {
      out = await listViaInnertube(channelId, cutoff);
    } catch (e) {
      if (e.status === 404) throw e;
      out = await listViaRss(channelId, cutoff);
      source = 'rss';
    }
    res.setHeader('Cache-Control', 's-maxage=900, stale-while-revalidate=3600');
    return res.status(200).json({ ...out, months: MONTHS, source });
  } catch (e) {
    return res.status(e.status || 502).json({ error: e.status ? e.message : 'Could not reach YouTube' });
  }
}
