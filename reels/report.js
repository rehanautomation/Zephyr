// Turns the measurements into the text package for Claude: timeline, CHECK THESE, image index. The tool measures, the chat judges.
import { fmt, colorDist } from './vision.js';

const pct0 = v => `${Math.round(v * 100)}%`;
const s2 = t => t.toFixed(2);
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const median = arr => { if (!arr.length) return 0; const s = [...arr].sort((a, b) => a - b); return s[s.length >> 1]; };

/* ---------- Similar-looking text and graphic screens, grouped across the batch (only used to pick pictures) ---------- */
function sameLook(a, b) {
  if ((a.shot.type === 'graphic') !== (b.shot.type === 'graphic')) return false;
  return Math.abs(Math.log(a.heightPct / b.heightPct)) < 0.3 && Math.abs(a.centerY - b.centerY) < 0.12 && a.boxed === b.boxed &&
    (!a.boxed || colorDist(a.boxColor, b.boxColor) < 60) && a.caps === b.caps && Math.abs(a.lines - b.lines) <= 1 && colorDist(a.color, b.color) < 90;
}

export function groupLooks(reel, registry) {
  for (const e of reel.ev.textEvents) {
    if (e.persistent) continue;
    let look = registry.looks.find(l => sameLook(l.first, e));
    if (!look) { look = { first: e }; registry.looks.push(look); }
    e.look = look;
  }
  for (const sc of reel.ev.screens) {
    sc.firstHere = !registry.screens.some(bg => colorDist(bg, sc.bg) < 30);
    if (sc.firstHere) registry.screens.push(sc.bg);
  }
}

/* ---------- Transcript: merged numbers, no zero-length words, compact lines ---------- */
export function cleanTranscript(res) {
  if (!res) return null;
  let words = [];
  if (res.words) {
    words = res.chunks.map(c => ({ t: c.timestamp[0] ?? 0, e: c.timestamp[1] ?? c.timestamp[0] ?? 0, w: c.text.trim() })).filter(w => w.w);
  } else {
    for (const c of res.chunks) {
      const ws = c.text.trim().split(/\s+/).filter(Boolean);
      const a = c.timestamp[0] ?? 0, b = c.timestamp[1] ?? a + ws.length * 0.3;
      ws.forEach((w, k) => words.push({ t: a + (b - a) * k / ws.length, e: a + (b - a) * (k + 1) / ws.length, w }));
    }
  }
  // Merge numbers the model split apart ("$300" + ",000", "50" + "%", "$" + "300")
  const merged = [];
  for (const w of words) {
    const p = merged[merged.length - 1];
    if (p && ((/^[,.]\d/.test(w.w) && /\d$/.test(p.w)) || (/^%/.test(w.w) && /\d$/.test(p.w)) || (p.w === '$' && /^\d/.test(w.w)))) { p.w += w.w; p.e = Math.max(p.e, w.e); continue; }
    merged.push({ ...w });
  }
  merged.forEach((w, k) => {
    const next = merged[k + 1];
    if (w.e <= w.t + 0.01) w.e = next && next.t > w.t ? Math.min(next.t, w.t + 0.15) : w.t + 0.15;
  });
  // One line per sentence: "start word · start word", pauses marked
  const lines = [];
  let cur = [], lineT = 0;
  merged.forEach((w, k) => {
    if (!cur.length) lineT = w.t;
    cur.push(`${s2(w.t)} ${w.w}`);
    const next = merged[k + 1];
    const gap = next ? next.t - w.e : 0;
    const endSentence = /[.?!]$/.test(w.w) || gap >= 0.6 || cur.length >= 18 || !next;
    if (gap >= 0.2 && !endSentence) cur.push(`[pause ${gap.toFixed(1)}s]`);
    if (endSentence) { lines.push({ t: lineT, s: cur.join(' · ') + (next && gap >= 0.2 ? ` · [pause ${gap.toFixed(1)}s]` : '') }); cur = []; }
  });
  return { words: merged, lines, estimated: !res.words };
}

/* ---------- Joined words in on-screen text ("MAKEMONEY" → "MAKE MONEY") ---------- */
const normW = w => w.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
const isAlnum = c => /[\p{L}\p{N}]/u.test(c);

// Insert spaces into tok after the given counts of letters/digits
function cutAt(tok, sizes) {
  let out = '', seen = 0, k = 0, need = sizes[0];
  for (const c of tok) {
    if (k < sizes.length - 1 && seen === need && isAlnum(c)) { out += ' '; k++; need += sizes[k]; }
    out += c;
    if (isAlnum(c)) seen++;
  }
  return out;
}

// Spoken words right after each other that spell the token exactly
function bySpeech(tok, spoken) {
  const target = normW(tok);
  for (let k = 0; k < spoken.length; k++) {
    let joined = '';
    const sizes = [];
    for (let j = k; j < spoken.length && joined.length < target.length; j++) {
      if (!spoken[j]) break;
      joined += spoken[j];
      sizes.push(spoken[j].length);
    }
    if (joined === target) return cutAt(tok, sizes);
  }
  return null;
}

// A gap between two letters clearly wider than the line's usual letter step
function byGaps(tok, pos, step) {
  if (!step || pos.length !== tok.length) return tok;
  const sizes = [];
  let n = 0;
  for (let j = 0; j < tok.length; j++) {
    if (j > 0 && isAlnum(tok[j - 1]) && isAlnum(tok[j]) && !(/\d/.test(tok[j - 1]) && /\d/.test(tok[j]))) {
      const d = pos[j] - pos[j - 1];
      if (d >= 1.8 * step && d - step >= 1.5 && n) { sizes.push(n); n = 0; }
    }
    if (isAlnum(tok[j])) n++;
  }
  sizes.push(n);
  return sizes.length > 1 ? cutAt(tok, sizes) : tok;
}

export function splitText(ev, words) {
  const spoken = (words || []).filter(w => w.t >= ev.start - 1.5 && w.t <= ev.end + 1.5).map(w => normW(w.w));
  const guesses = [];
  const lines = (ev.lineData || [{ text: ev.text, pos: [] }]).map(({ text, pos }) => {
    const diffs = [];
    for (let j = 1; j < text.length; j++) if (isAlnum(text[j - 1]) && isAlnum(text[j]) && pos[j] !== undefined) diffs.push(pos[j] - pos[j - 1]);
    const step = diffs.length >= 4 ? median(diffs) : 0;
    return text.replace(/\S+/g, (tok, i0) => {
      if (normW(tok).length < 4) return tok;
      const heard = bySpeech(tok, spoken);
      if (heard) return heard;
      const split = byGaps(tok, pos.slice(i0, i0 + tok.length), step);
      if (split !== tok) guesses.push(`"${tok}" → "${split}"`);
      return split;
    });
  });
  return { text: lines.join(' / '), guesses };
}

/* ---------- Sound bumps (voice filtered out using the word times) ---------- */
export function bumpList(reel) {
  if (!reel.bumps) return null;
  const words = reel.transcript?.words;
  const wordAt = b => words?.find(w => b.t >= w.t - 0.08 && b.t <= w.e + 0.05) || null; // a word's sound starts a little before its time
  const all = reel.bumps.map(b => ({ ...b, word: wordAt(b) }));
  const inVoice = median(all.filter(b => b.word).map(b => b.flux));
  // Inside a word, only a bump much stronger than the voice's own bumps counts
  let list = all.filter(b => !b.word || (b.flux >= 2 * inVoice && b.flux >= 0.3));
  const dropped = Math.max(0, list.length - 30);
  list = list.sort((a, b) => b.flux - a.flux).slice(0, 30).sort((a, b) => a.t - b.t);
  const top = Math.max(1e-9, ...list.map(b => b.flux));
  for (const b of list) { b.strength = Math.round(100 * b.flux / top); b.file = reel.bumpClips?.get(b.t)?.name || null; }
  return { list, dropped, voiceChecked: !!words };
}

/* ---------- Text package ---------- */
export function intro(n, images, clips) {
  return [
    `Here are ${plural(n, 'reel')}, measured by a tool. ${plural(images, 'picture')}${clips ? ` and ${plural(clips, 'sound clip')}` : ''} are attached as separate files.`,
    `The tool only measures. It doesn't judge. You work out what the editor did and why.`,
    ``,
    `Give me ONE playbook across all reels:`,
    `(1) when the editor cuts, adds text, zooms, shows a graphic or adds a sound, tied to what is being said;`,
    `(2) how the text looks and moves (judge this from the pictures);`,
    `(3) what they leave unedited;`,
    `(4) a template I can apply to a new script.`,
    ``,
    `How to read it:`,
    `• Times are m:ss.cc, accurate to about ±0.1s.`,
    `• TIMELINE: everything in order. SAID = speech, each word with its start time ([pause] = 0.2s or more of silence). CUT = a new shot; "face → no face" says if a face is on screen just before and just after. TEXT = on-screen text, when it leaves, and where it sits (% of the frame). ZOOM = the face gets bigger or smaller within one shot (only zooms under 3s are listed). GRAPHIC = a flat, designed screen. BUMP = a sudden sound that isn't the voice; strength 0–100 (100 = the strongest in that reel), with its 0.6s .wav clip.`,
    `• CHECK THESE: measurements the tool isn't sure about. Look at them in the pictures or clips before relying on them.`,
    `• IMAGES: each row is one strip of frames, either a text appearing (from just before until it settles) or a graphic screen building up. Every frame has its number and time printed in yellow.`
  ].join('\n');
}

const where = e => `center ${pct0(e.centerX)} from left, ${pct0(e.centerY)} from top · ${pct0(e.box.w)} wide, ${pct0(e.box.h)} tall`;

export function reelText(reel, allRefs) {
  const ev = reel.ev;
  const refs = allRefs.filter(r => r.reel === reel.n);
  const img = key => { const r = refs.filter(x => x.key === key).map(x => x.ref); return r.length ? ` · [img ${r.join(', ')}]` : ''; };
  const words = reel.transcript?.words;
  const texts = ev.textEvents.filter(e => !e.persistent);
  const bumps = bumpList(reel);
  const items = [], checks = [];
  const check = (t, s) => checks.push({ t, s });

  // Speech
  for (const l of reel.transcript?.lines || []) items.push({ t: l.t, s: `SAID ${l.s}` });

  // Cuts
  for (const tr of ev.transitions) {
    const face = tr.faceBefore === null ? 'face check unavailable' : `${tr.faceBefore ? 'face' : 'no face'} → ${tr.faceAfter ? 'face' : 'no face'}`;
    const scale = tr.faceScale && Math.abs(Math.log(tr.faceScale)) >= Math.log(1.1) ? `, face ${Math.round(Math.abs(tr.faceScale - 1) * 100)}% ${tr.faceScale > 1 ? 'bigger' : 'smaller'}` : '';
    const flash = tr.flash ? ` · after a ${tr.flash.dip ? 'black' : tr.flash.color} flash (${fmt(tr.flash.start)}, ${(tr.flash.end - tr.flash.start).toFixed(2)}s)` : '';
    items.push({ t: tr.t, s: `CUT · ${face}${scale}${flash}` });
    if (tr.weak) check(tr.t, tr.flash ? 'CUT after a flash: the scene only changed a little, so it may be the same shot' : 'CUT: small picture change, may be fast movement instead of a cut');
    if (tr.gap >= 2) check(tr.t, `CUT: frames were skipped here, so the time may be off by up to ${(tr.gap / 15).toFixed(2)}s`);
    if (tr.faceFlicker) check(tr.t, 'CUT: the face was found on and off near this cut, so "face/no face" may be wrong');
  }
  for (const t of ev.notCuts || []) check(t, 'big picture change with the same face in the same place, counted as NOT a cut');
  for (const f of ev.flashUnsure || []) check(f.start, 'flash with a small scene change, counted as NOT a cut');
  for (const f of ev.flashes) items.push({ t: f.start, s: `FLASH ${f.dip ? 'black' : f.color}, ${(f.end - f.start).toFixed(2)}s, no new shot` });

  // On-screen text
  const persistent = ev.textEvents.filter(e => e.persistent);
  for (const e of texts) {
    const { text, guesses } = splitText(e, words);
    items.push({ t: e.start, s: `TEXT "${text}" until ${fmt(e.end)} · ${where(e)}${e.shot.type === 'graphic' ? ' · on a graphic screen' : ''}${img(`text:${e.start}`)}` });
    if (guesses.length) check(e.start, `TEXT: joined words split by letter spacing, not confirmed by speech: ${guesses.join(', ')}`);
    if (e.conf < 0.85) check(e.start, `TEXT "${text}": read with low confidence, spelling may be off`);
    if (e.sightings === 1) check(e.start, `TEXT "${text}": seen only briefly, may be a misread`);
    if (e.timingUnsure) check(e.start, `TEXT "${text}": start time unsure`);
  }

  // Zooms
  for (const z of ev.zooms) {
    items.push({ t: z.start, s: `ZOOM ${z.dir}, face ${z.from}%→${z.to}% over ${(z.end - z.start).toFixed(2)}s` });
    if (z.unsure) check(z.start, 'ZOOM: small, or the face was lost for part of it');
  }

  // Graphic screens
  for (const sc of ev.screens) {
    items.push({ t: sc.start, s: `GRAPHIC screen until ${fmt(sc.end)}${img(`graphic:${sc.start}`)}` });
    for (const st of sc.steps.slice(1)) if (!st.text || st.what.includes('changes color')) items.push({ t: st.t, s: `GRAPHIC step: ${st.what}` });
    if (sc.unsure) check(sc.start, 'GRAPHIC: may be normal video rather than a designed screen');
  }

  // Sound bumps
  if (bumps) for (const b of bumps.list) {
    items.push({ t: b.t, s: `BUMP strength ${b.strength}${b.file ? ` · ${b.file}` : ''}` });
    const why = [];
    if (b.word) why.push(`lands on the spoken word "${b.word.w}", may be the voice`);
    if (b.strength < 25) why.push('faint');
    if (why.length) check(b.t, `BUMP: ${why.join('; ')}`);
  }

  items.sort((a, b) => a.t - b.t);
  checks.sort((a, b) => a.t - b.t);
  if (!reel.scanFaceOk) checks.unshift({ t: null, s: 'The face finder did not load, so face checks and zooms are missing.' });
  if (reel.ocrError) checks.unshift({ t: null, s: `On-screen text could not be read (${reel.ocrError}).` });
  if (reel.transcript?.estimated) checks.unshift({ t: null, s: 'Word times are estimates (only phrase times were available).' });
  if (bumps && !bumps.voiceChecked) checks.unshift({ t: null, s: 'Speech was not ready, so BUMPs were not checked against the voice.' });

  const out = [];
  out.push(`=== REEL ${reel.n}: ${reel.name} ===`);
  out.push(`Length ${fmt(reel.dur)} · ${reel.vw}×${reel.vh} · ${plural(ev.transitions.length, 'cut')} · ${plural(texts.length, 'text')} · ${plural(ev.zooms.length, 'zoom')} · ${plural(ev.screens.length, 'graphic screen')} · ${bumps ? plural(bumps.list.length, 'bump') : 'no sound'}`);
  if (persistent.length) out.push(`Always on screen: ${persistent.map(e => `"${e.text}" (${where(e)})`).join('; ')}`);
  if (!reel.transcript) out.push(reel.transcriptError ? `Speech: not available (${reel.transcriptError}).` : 'Speech: still processing when copied.');
  else if (!reel.transcript.words.length) out.push('Speech: none detected.');
  if (reel.soundError) out.push(`Sound: ${reel.soundError}`);
  if (bumps?.dropped) out.push(`${plural(bumps.dropped, 'weaker bump')} not listed.`);

  out.push('', 'TIMELINE');
  out.push(...(items.length ? items.map(i => `${fmt(i.t)} ${i.s}`) : ['(nothing found)']));

  out.push('', 'CHECK THESE');
  out.push(...(checks.length ? checks.map(c => `- ${c.t === null ? '' : `${fmt(c.t)} `}${c.s}`) : ['(nothing)']));

  out.push('', 'IMAGES');
  if (!reel.sheets.length) out.push('(no images for this reel)');
  for (const sh of reel.sheets) {
    out.push(sh.name);
    for (const row of sh.rows) {
      const c = row.cells;
      if (/build$/.test(row.title)) out.push(`  ${row.title}: ${c.map(x => `#${x.n} ${fmt(x.t)} ${x.label}`).join(' · ')}`);
      else out.push(`  ${row.title}: #${c[0].n}–${c[c.length - 1].n}, ${fmt(c[0].t)} to ${fmt(c[c.length - 1].t)}`);
    }
  }
  if (reel.omitted.length) out.push(`Not pictured (5-sheet limit): ${reel.omitted.join('; ')}`);
  return out.join('\n');
}

export function packageText(reels, allRefs) {
  const images = reels.reduce((s, r) => s + r.sheets.length, 0);
  const clips = reels.reduce((s, r) => s + (bumpList(r)?.list.filter(b => b.file).length || 0), 0);
  return [intro(reels.length, images, clips), ...reels.map(r => reelText(r, allRefs))].join('\n\n');
}
