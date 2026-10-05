// Turns analysis results into the text package for Claude: styles, timeline, graphics, sound, transcript, image index
import { fmt, colorName, colorDist, textSim } from './vision.js';

const pct = v => `${(v * 100).toFixed(1)}%`;
const pct0 = v => `${Math.round(v * 100)}%`;
const s2 = t => t.toFixed(2);
const letter = n => { let s = ''; n++; while (n > 0) { const r = (n - 1) % 26; s = String.fromCharCode(65 + r) + s; n = Math.floor((n - 1) / 26); } return s; };
const col = h => h ? `${colorName(h)} ${h}` : '';

/* ---------- Styles and graphic screens, grouped across the whole batch ---------- */
function features(e) {
  return { h: e.heightPct, y: e.centerY, x: e.centerX, boxed: e.boxed, boxColor: e.boxColor, caps: e.caps, lines: e.lines, color: e.color, graphic: e.shot.type === 'graphic', words: e.text.split(/\s|\//).filter(Boolean).length };
}
function sameStyle(a, b) {
  if (a.graphic !== b.graphic) return false;
  if (a.graphic) return colorDist(a.color, b.color) < 70 && Math.abs(Math.log(a.h / b.h)) < 0.7;
  return Math.abs(Math.log(a.h / b.h)) < 0.3 && Math.abs(a.y - b.y) < 0.12 && a.boxed === b.boxed &&
    (!a.boxed || colorDist(a.boxColor, b.boxColor) < 60) && a.caps === b.caps && Math.abs(a.lines - b.lines) <= 1;
}

export function assignStyles(reel, registry) {
  const use = new Map();
  for (const e of reel.ev.textEvents) {
    if (e.persistent) continue;
    const f = features(e);
    let st = registry.styles.find(s => sameStyle(s.f, f));
    if (!st) { st = { id: letter(registry.styles.length), f, firstReel: reel.n, first: e, uses: [], colors: new Map(), entrances: new Map(), exits: new Map(), images: [] }; registry.styles.push(st); }
    e.style = st;
    st.uses.push({ reel: reel.n, e });
    st.colors.set(e.color, (st.colors.get(e.color) || 0) + 1);
    for (const c of e.colors) if (c !== e.color) st.colors.set(c, (st.colors.get(c) || 0) + 0.01);
    st.entrances.set(e.entrance, (st.entrances.get(e.entrance) || 0) + 1);
    st.exits.set(e.exit, (st.exits.get(e.exit) || 0) + 1);
    if (!use.has(st)) use.set(st, { id: st.id, style: st, firstHere: st.firstReel === reel.n, first: st.firstReel === reel.n ? st.first : null, events: [], differentLater: [] });
    use.get(st).events.push(e);
  }
  // A later use that animates differently from the first use gets its own strip
  for (const u of use.values()) {
    if (!u.firstHere) continue;
    const seen = new Set([u.first.entranceKind]);
    for (const e of u.events.slice(1)) if (!seen.has(e.entranceKind) && e.entranceKind !== 'subtle') { seen.add(e.entranceKind); u.differentLater.push(e); }
  }
  reel.styleUse = [...use.values()];
  for (const sc of reel.ev.screens) {
    let g = registry.screens.find(x => colorDist(x.bg, sc.bg) < 30);
    if (!g) { g = { id: `G${registry.screens.length + 1}`, bg: sc.bg, firstReel: reel.n }; registry.screens.push(g); sc.firstHere = true; }
    else sc.firstHere = false;
    sc.id = g.id;
    sc.group = g;
    for (const e of sc.texts) e.screenId = e.screenId || g.id;
  }
}

/* ---------- Transcript: clean up, fix from on-screen text, compact format ---------- */
const norm = x => x.toLowerCase().replace(/[^\p{L}\p{N}$%]/gu, '');

function adaptCase(screen, heard) {
  const trail = (heard.match(/[.,!?;:]+$/) || [''])[0];
  let w = screen.replace(/[.,!?;:]+$/, '');
  if (/[A-Z]/.test(w) && w === w.toUpperCase() && heard !== heard.toUpperCase()) w = /^[A-Z]/.test(heard) ? w[0] + w.slice(1).toLowerCase() : w.toLowerCase();
  return w + trail;
}

export function cleanTranscript(res, textEvents) {
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
  // No zero-length words
  merged.forEach((w, k) => {
    const next = merged[k + 1];
    if (w.e <= w.t + 0.01) w.e = next && next.t > w.t ? Math.min(next.t, w.t + 0.15) : w.t + 0.15;
  });
  // Prefer the editor's on-screen words when they overlap in time and closely match what was heard
  for (const ev of textEvents) {
    if (ev.persistent || ev.shot.type === 'graphic') continue;
    const O = ev.text.split(/\s+|\//).map(x => x.trim()).filter(x => norm(x));
    if (!O.length || O.length > 12) continue;
    const idx = merged.map((_, k) => k).filter(k => merged[k].t >= ev.start - 0.35 && merged[k].t <= ev.end + 0.1);
    if (!idx.length) continue;
    const n = idx.length, m = O.length, GAP = 0.6;
    const D = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: m + 1 }, (_, j) => (i + j) * GAP));
    for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) {
      const sub = 1 - textSim(norm(merged[idx[i - 1]].w), norm(O[j - 1]));
      D[i][j] = Math.min(D[i - 1][j - 1] + sub, D[i - 1][j] + GAP, D[i][j - 1] + GAP);
    }
    const pairs = [];
    for (let i = n, j = m; i > 0 && j > 0;) {
      const sub = 1 - textSim(norm(merged[idx[i - 1]].w), norm(O[j - 1]));
      if (Math.abs(D[i][j] - (D[i - 1][j - 1] + sub)) < 1e-9) { pairs.push([idx[i - 1], O[j - 1], 1 - sub]); i--; j--; }
      else if (Math.abs(D[i][j] - (D[i - 1][j] + GAP)) < 1e-9) i--;
      else j--;
    }
    const good = pairs.filter(p => p[2] >= 0.5);
    if (good.length < Math.max(1, Math.ceil(Math.min(n, m) * 0.5))) continue;
    for (const [k, o, sim] of good) {
      const w = merged[k];
      const loose = x => x.toLowerCase().replace(/[.,!?;:"]+/g, '');
      if (loose(w.w) !== loose(o) && !w.fixed) { w.fixed = w.w; w.w = adaptCase(o, w.w); }
    }
  }
  // Compact lines: one per sentence, "start word · start word", pauses marked
  const lines = [];
  let cur = [];
  merged.forEach((w, k) => {
    cur.push(`${s2(w.t)} ${w.w}${w.fixed ? ` (screen fix, heard "${w.fixed}")` : ''}`);
    const next = merged[k + 1];
    const gap = next ? next.t - w.e : 0;
    const endSentence = /[.?!]$/.test(w.w) || gap >= 0.6 || cur.length >= 18 || !next;
    if (gap >= 0.2 && !endSentence) cur.push(`[pause ${gap.toFixed(1)}s]`);
    if (endSentence) { lines.push(cur.join(' · ') + (next && gap >= 0.2 ? ` · [pause ${gap.toFixed(1)}s]` : '')); cur = []; }
  });
  return { words: merged, lines, fixes: merged.filter(w => w.fixed).length };
}

const spokenAt = (reel, t, span = 0.9) => {
  const w = reel.transcript?.words;
  if (!w) return '';
  const said = w.filter(x => x.t >= t - 0.15 && x.t <= t + span).slice(0, 6).map(x => x.w).join(' ');
  return said ? ` | "${said}"` : '';
};
const speakingAt = (reel, t) => reel.transcript?.words?.some(x => t >= x.t - 0.05 && t <= x.e + 0.05);

/* ---------- Text package ---------- */
export function intro(n, totalImages) {
  return [
    `Here are ${n} reel${n === 1 ? '' : 's'}. The text package is below. ${totalImages} image${totalImages === 1 ? ' is' : 's are'} attached separately, named by reel and sheet (reel01_s1_styles.jpg …); every frame in them has its number and exact time printed in yellow.`,
    `Each style is shown in images once (its settled look + its entrance animation); the timeline lists every use.`,
    ``,
    `Work out the editor's decisions and give me ONE combined playbook across all reels:`,
    `(1) style specs I can rebuild: colors, size, position, animation, timing;`,
    `(2) rules for when each style, graphic, zoom and SFX is used, tied to what is being said;`,
    `(3) what the editor deliberately left unedited;`,
    `(4) a template I can apply to a new script.`,
    ``,
    `How to read the package:`,
    `• STYLE CATALOG: every text style found across the batch, with measured color, size (% of frame height), position (% from left/top), box, case and animation.`,
    `• QUICK STATS: cuts, shot lengths, talking head vs graphic screens, number of text events, SFX, music.`,
    `• TIMELINE: every event in time order (cuts, zooms, flashes, text, graphic steps, sound effects); "quotes" are the words spoken at that moment; [img …] points to the picture.`,
    `• GRAPHIC SCREENS: background, elements and the step-by-step build, with the spoken word that triggers each step.`,
    `• SOUND: sound effects (label + model confidence, paired with the nearest visual event) and background music.`,
    `• TRANSCRIPT: "start-time word" pairs; [pause] = silence of 0.2s or more; "(screen fix)" = corrected from the editor's on-screen text.`,
    `• IMAGES: what every numbered frame shows. Times are accurate to about ±0.1s.`
  ].join('\n');
}

function styleName(st) {
  const f = st.f;
  if (f.graphic) return 'graphic-screen text';
  const big = f.h > 0.07, mid = f.h > 0.035;
  const kind = f.lines > 1 ? (big ? 'headline stack' : 'multi-line caption') : (f.words <= 2 && mid ? 'word-by-word caption' : big ? 'big headline' : mid ? 'caption' : 'small label');
  return `${f.boxed ? 'boxed ' : ''}${kind}`;
}
const topEntries = (map, k = 3) => [...map.entries()].sort((a, b) => b[1] - a[1]).slice(0, k).map(([v]) => v);

export function styleSpec(st, imageRefs) {
  const f = st.f;
  const colors = topEntries(st.colors, 3).map(h => col(h)).join(', ');
  const parts = [
    `Style ${st.id} — ${styleName(st)}: ${f.lines} line${f.lines > 1 ? 's' : ''}, ${f.caps ? 'ALL CAPS' : 'mixed case'}, main text ${pct(f.h)} of frame height, centered at ${pct0(f.x)} from left / ${pct0(f.y)} from top`,
    `text color ${colors}`,
    f.boxed ? `on a box ${col(f.boxColor)}` : f.graphic ? 'on the graphic background' : 'no box (on the video)',
    `enters: ${topEntries(st.entrances, 2).join(' / ')}`,
    `exits: ${topEntries(st.exits, 2).join(' / ')}`,
    `used ${st.uses.length}× in the batch, first in Reel ${st.firstReel} at ${fmt(st.first.start)}`
  ];
  let refs = imageRefs.filter(r => r.key === `hero:${st.id}` || r.key === `strip:${st.id}`).map(r => r.ref);
  if (!refs.length && st.first.screenId) refs = imageRefs.filter(r => r.key === `graphic:${st.first.screenId}`).map(r => `${r.ref} (graphic build)`);
  if (refs.length) parts.push(`images: ${refs.join(', ')}`);
  return parts.join('; ');
}

export function reelText(reel, allRefs) {
  const ev = reel.ev;
  const refs = allRefs.filter(r => r.reel === reel.n);
  const refFor = key => refs.filter(r => r.key === key).map(r => r.ref).join(', ');
  const out = [];
  out.push(`=== REEL ${reel.n}: ${reel.name} ===`);
  out.push(`Length ${fmt(reel.dur)} · ${reel.vw}×${reel.vh}${reel.notes.length ? ` · notes: ${reel.notes.join('; ')}` : ''}`);

  // Quick stats
  const shotTime = type => ev.shots.filter(s => s.type === type).reduce((a, s) => a + (s.end - s.start), 0);
  const talking = shotTime('talking head') / reel.dur, graphic = shotTime('graphic') / reel.dur;
  const texts = ev.textEvents.filter(e => !e.persistent);
  let sfx = reel.sound?.hits || [];
  for (const h of sfx) h.pair = null;
  const visualTimes = [...ev.transitions.map(t => t.t), ...ev.zooms.map(z => z.start), ...ev.flashes.map(f => f.start), ...texts.map(e => e.start), ...ev.screens.flatMap(sc => sc.steps.map(st => st.t))];
  sfx = sfx.filter(h => h.score > 0 || visualTimes.some(t => Math.abs(t - h.t) <= 0.15));
  out.push('', 'QUICK STATS');
  out.push(`${ev.transitions.length} cuts (average shot ${(reel.dur / ev.shots.length).toFixed(1)}s) · ${pct0(talking)} talking head / ${pct0(graphic)} graphic screens${talking + graphic < 0.97 ? ` / ${pct0(1 - talking - graphic)} other footage` : ''} · ${texts.length} text events in ${reel.styleUse.length} style${reel.styleUse.length === 1 ? '' : 's'} (${reel.styleUse.map(u => u.id).join(', ') || 'none'}) · ${ev.zooms.length} zoom${ev.zooms.length === 1 ? '' : 's'} · ${sfx.length} SFX · music: ${reel.sound ? (reel.sound.music.present ? 'yes' : 'no') : 'unknown'}`);
  const persistent = ev.textEvents.filter(e => e.persistent);
  if (persistent.length) out.push(`Always on screen: ${persistent.map(e => `"${e.text}" (${col(e.color)}, at ${pct0(e.centerX)} / ${pct0(e.centerY)})`).join(', ')}`);

  // Timeline
  const items = [];
  for (const z of ev.zooms) items.push({ t: z.start, s: `ZOOM-${z.dir.toUpperCase()} ~${z.from}%→${z.to}%${z.blur ? ' + motion blur' : ''}${z.slow ? ' (slow push)' : ''}, ${(z.end - z.start).toFixed(2)}s${spokenAt(reel, z.start)}${refFor(`zoom:${z.start}`) ? ` · [img ${refFor(`zoom:${z.start}`)}]` : ''}` });
  for (const tr of ev.transitions) {
    const to = tr.to.type === 'graphic' ? `graphic ${ev.screens.find(sc => sc.shot === tr.to)?.id || ''}`.trim() : tr.to.type;
    const img = refFor(`cut:${tr.t}`);
    items.push({ t: tr.t, s: `CUT → ${to}${tr.kinds.length ? ` (${tr.kinds.join(', ')})` : ''}${spokenAt(reel, tr.t)}${img ? ` · [img ${img}]` : ''}` });
  }
  for (const f of ev.flashes) items.push({ t: f.start, s: `${f.dip ? 'DIP TO BLACK' : `FLASH (${f.color})`}, ${(f.end - f.start).toFixed(2)}s${spokenAt(reel, f.start)}` });
  for (const e of texts) {
    if (e.shot.type === 'graphic') continue; // listed under the graphic's build steps
    const id = e.style?.id;
    const img = id && e === e.style.first ? refFor(`strip:${id}`) : refFor(`repeat:${e.start}`);
    items.push({ t: e.start, s: `TEXT Style ${id} "${e.text}" · in: ${e.entrance} · out: ${e.exit} at ${fmt(e.end)}${spokenAt(reel, e.start)}${img ? ` · [img ${img}]` : ''}` });
  }
  for (const sc of ev.screens) for (const st of sc.steps) {
    const styleIds = [...new Set(sc.texts.filter(e => st.what.includes(`"${e.text}"`)).map(e => e.style?.id).filter(Boolean))];
    items.push({ t: st.t, s: `GRAPHIC ${sc.id}: ${st.what}${styleIds.length ? ` (Style ${styleIds.join(', ')})` : ''}${spokenAt(reel, st.t, 0.6)}` });
  }
  const visual = [...ev.transitions.map(t => ({ t: t.t, what: 'CUT' })), ...ev.zooms.map(z => ({ t: z.start, what: `ZOOM-${z.dir.toUpperCase()}` })),
    ...ev.flashes.map(f => ({ t: f.start, what: 'FLASH' })), ...texts.map(e => ({ t: e.start, what: `TEXT "${e.text}"` })),
    ...ev.screens.flatMap(sc => sc.steps.map(st => ({ t: st.t, what: `${sc.id} step` })))];
  for (const h of sfx) h.pair = visual.filter(v => Math.abs(v.t - h.t) <= 0.15).sort((a, b) => Math.abs(a.t - h.t) - Math.abs(b.t - h.t))[0];
  sfx = sfx.filter(h => h.score > 0 || h.pair);
  for (const h of sfx) {
    const pair = h.pair;
    items.push({ t: h.t, s: `SFX ${h.label}${h.score ? ` (${h.score.toFixed(2)})` : ''}${pair ? ` ↔ ${fmt(pair.t)} ${pair.what}` : ''}${speakingAt(reel, h.t) ? ' · during speech' : ''}` });
  }
  items.sort((a, b) => a.t - b.t);
  out.push('', 'TIMELINE');
  out.push(...items.map(i => `${fmt(i.t)} ${i.s}`));

  // Styles used here
  out.push('', 'STYLES USED IN THIS REEL (full specs in the STYLE CATALOG)');
  for (const u of reel.styleUse) out.push(`Style ${u.id}: ${u.events.length}× at ${u.events.map(e => fmt(e.start)).join(', ')}${u.firstHere ? '' : ` · same as Reel ${u.style.firstReel} Style ${u.id}`}`);
  if (!reel.styleUse.length) out.push('(no on-screen text found)');

  // Graphic screens
  if (ev.screens.length) {
    out.push('', 'GRAPHIC SCREENS');
    for (const sc of ev.screens) {
      out.push(`${sc.id} (${fmt(sc.start)}–${fmt(sc.end)})${sc.firstHere ? '' : ` · same look as ${sc.group.id} from Reel ${sc.group.firstReel}`} · background ${col(sc.bg)} · main colors ${sc.palette.map(p => `${p.hex} ${pct0(p.share)}`).join(', ')}`);
      if (sc.texts.length) out.push(`  Text: ${sc.texts.map(e => `"${e.text}" (${col(e.color)}, ${pct(e.heightPct)} tall, at ${pct0(e.centerY)} from top)`).join('; ')}`);
      out.push('  Build:');
      for (const st of sc.steps) out.push(`   ${fmt(st.t)} ${st.what}${spokenAt(reel, st.t, 0.6)}`);
      const img = refFor(`graphic:${sc.id}`);
      if (img) out.push(`  Images: ${img}`);
    }
  }

  // Sound
  out.push('', 'SOUND');
  if (!reel.sound) out.push(reel.soundError || 'No audio found.');
  else {
    if (sfx.length) for (const h of sfx) out.push(`${fmt(h.t)} ${h.label}${h.score ? ` (${h.score.toFixed(2)})` : ''}${h.pair ? ` ↔ ${fmt(h.pair.t)} ${h.pair.what}` : ' (no visual event within 0.15s)'}${reel.sfxFiles?.[h.t] ? ` · clip: ${reel.sfxFiles[h.t]}` : ''}`);
    else out.push('No clear sound effects found.');
    const m = reel.sound.music;
    out.push(m.present
      ? `Music: yes, ${m.segments.map(g => `${fmt(g.start)}–${fmt(Math.min(reel.dur, g.end))}`).join(', ')} (${pct0(m.coverage)} of the reel)${m.underVoiceDb !== null ? `, about ${Math.abs(m.underVoiceDb)} dB ${m.underVoiceDb <= 0 ? 'under' : 'over'} the voice` : ''}`
      : 'Music: none detected.');
  }

  // Transcript
  out.push('', `TRANSCRIPT${reel.transcript?.fixes ? ` (${reel.transcript.fixes} word${reel.transcript.fixes === 1 ? '' : 's'} corrected from on-screen text)` : ''}`);
  if (reel.transcript) out.push(...(reel.transcript.lines.length ? reel.transcript.lines : ['(no speech detected)']));
  else out.push(reel.transcriptError ? `Not available: ${reel.transcriptError}` : 'Still processing when copied.');

  // Image index
  out.push('', 'IMAGES');
  if (!reel.sheets.length) out.push('(no images for this reel)');
  for (const sh of reel.sheets) {
    out.push(`${sh.name} — ${sh.title.split(' · ').slice(2).join(' · ')}`);
    let lastTitle = null;
    for (const c of sh.cells) {
      if (c.title && c.title !== lastTitle) { out.push(`  ${c.title}`); lastTitle = c.title; }
      out.push(`   #${c.n} ${fmt(c.t)} ${c.label}`);
    }
  }
  if (reel.omitted.length) out.push(`Not pictured (image budget): ${reel.omitted.join('; ')}`);
  return out.join('\n');
}

export function packageText(reels, registry, allRefs) {
  const total = reels.reduce((s, r) => s + r.sheets.length, 0);
  const usedStyles = registry.styles.filter(st => st.uses.some(u => reels.some(r => r.n === u.reel)));
  const catalog = ['STYLE CATALOG', ...(usedStyles.length ? usedStyles.map(st => styleSpec(st, allRefs)) : ['(no on-screen text found)'])].join('\n');
  return [intro(reels.length, total), catalog, ...reels.map(r => reelText(r, allRefs))].join('\n\n');
}
