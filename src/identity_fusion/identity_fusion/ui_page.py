"""HTML page served by ui_node (kept in its own module: no data_files needed)."""

INDEX_HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HRI Visión - ¿Quién habla?</title>
<style>
:root {
  --bg: #f4f5f7; --card: #ffffff; --text: #14181f; --muted: #6a7280;
  --line: #e1e4ea; --accent: #2f6fed; --ok: #1f9d55; --warn: #d98a00; --bad: #d64545;
  --track: #e8ebf0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --card: #161b22; --text: #e6edf3; --muted: #8b95a3;
    --line: #262d38; --accent: #5b8cff; --ok: #3ccf7a; --warn: #f0b429; --bad: #ff6b6b;
    --track: #232a35;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; align-items: center; justify-content: space-between;
  padding: 14px 24px; border-bottom: 1px solid var(--line); background: var(--card); }
h1 { font-size: 18px; margin: 0; font-weight: 650; letter-spacing: .2px; }
h1 span { color: var(--muted); font-weight: 400; margin-left: 8px; font-size: 14px; }
.pill { font-size: 12px; padding: 4px 10px; border-radius: 999px;
  background: var(--track); color: var(--muted); }
.pill.on { background: color-mix(in srgb, var(--ok) 18%, transparent); color: var(--ok); }
.pill.off { background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); }
main { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(320px, 1fr);
  gap: 18px; padding: 18px 24px; max-width: 1500px; margin: 0 auto; }
@media (max-width: 960px) { main { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; }
.card h2 { margin: 0 0 10px; font-size: 12px; text-transform: uppercase;
  letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.col { display: flex; flex-direction: column; gap: 18px; min-width: 0; }
.video { position: relative; background: #000; border-radius: 10px; overflow: hidden; aspect-ratio: 4/3; }
.video img { width: 100%; height: 100%; object-fit: contain; display: block; }
.video .nosig { position: absolute; inset: 0; display: none; align-items: center;
  justify-content: center; color: #9aa3b2; font-size: 15px; background: #0b0e13; }
.video.nosignal .nosig { display: flex; }
.chips { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; min-height: 34px; }
.chips .none { color: var(--muted); font-size: 13px; }
.face { display: grid; grid-template-columns: minmax(0, 1.1fr) auto minmax(90px, 1fr); align-items: center; gap: 12px;
  padding: 8px 12px; border-radius: 10px; border: 1px solid var(--line); background: var(--bg); font-size: 14px; }
.face.speaking { border-color: var(--ok); background: color-mix(in srgb, var(--ok) 12%, transparent); }
.face .id { color: var(--muted); font-size: 12px; margin-left: 6px; }
.face .sc { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 13px; margin-left: 8px; }
.lip { font-size: 12px; font-weight: 650; padding: 3px 9px; border-radius: 6px; white-space: nowrap;
  background: var(--track); color: var(--muted); }
.lip.on { background: color-mix(in srgb, var(--ok) 20%, transparent); color: var(--ok); }
.lipbar { position: relative; height: 8px; background: var(--track); border-radius: 5px; }
.lipbar > i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 5px; background: var(--muted); transition: width .2s; }
.face.speaking .lipbar > i { background: var(--ok); }
.lipbar > u { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--text); opacity: .45; }
.who { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.who .name { font-size: 30px; font-weight: 700; line-height: 1.1; }
.badge { padding: 5px 12px; border-radius: 8px; font-size: 13px; font-weight: 650; white-space: nowrap; }
.badge.responder { background: color-mix(in srgb, var(--ok) 18%, transparent); color: var(--ok); }
.badge.preguntar { background: color-mix(in srgb, var(--warn) 20%, transparent); color: var(--warn); }
.badge.negar { background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); }
.badge.idle { background: var(--track); color: var(--muted); }
.meter { margin-top: 14px; }
.meter .row { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted); margin-bottom: 4px; }
.meter .row b { color: var(--text); font-variant-numeric: tabular-nums; }
.bar { position: relative; height: 10px; background: var(--track); border-radius: 6px; }
.bar > i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; background: var(--accent); transition: width .25s; }
.bar > u { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--muted); opacity: .6; }
.bar.main { height: 14px; } .bar.main > i { background: var(--ok); }
.bar.main.preguntar > i { background: var(--warn); } .bar.main.negar > i { background: var(--bad); }
.text { font-size: 17px; min-height: 26px; white-space: pre-wrap; word-wrap: break-word; }
.text.empty { color: var(--muted); font-size: 15px; }
.hist { max-height: 240px; overflow: auto; display: flex; flex-direction: column; gap: 10px; }
.hist .item { border-top: 1px solid var(--line); padding-top: 8px; font-size: 13px; }
.hist .item:first-child { border-top: 0; padding-top: 0; }
.hist .meta { color: var(--muted); margin-bottom: 2px; }
.hist .q { font-weight: 600; } .hist .a { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Reto Avanzado Mario Casas Donjuan<span>HRI / Visión · RoBorregos 2026</span></h1>
  <div><span class="pill" id="fps">cámara…</span> <span class="pill" id="conn">conectando…</span></div>
</header>
<main>
  <div class="col">
    <div class="card">
      <h2>Cámara y labios</h2>
      <div class="video nosignal" id="video">
        <img id="cam" src="/video" alt="">
        <div class="nosig">Sin señal de cámara</div>
      </div>
      <div class="chips" id="chips"></div>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>Quién habló</h2>
      <div class="who"><div class="name" id="who">—</div><div class="badge idle" id="state">Esperando</div></div>
      <div class="meter">
        <div class="row"><span>Confianza combinada</span><b id="conf">—</b></div>
        <div class="bar main" id="confbar"><i style="width:0%"></i><u id="thr1"></u><u id="thr2"></u></div>
      </div>
      <div class="meter">
        <div class="row"><span>Cara</span><b id="faceTxt">—</b></div>
        <div class="bar"><i id="faceBar" style="width:0%"></i></div>
      </div>
      <div class="meter">
        <div class="row"><span>Voz</span><b id="voiceTxt">—</b></div>
        <div class="bar"><i id="voiceBar" style="width:0%"></i></div>
      </div>
    </div>
    <div class="card"><h2>Pregunta</h2><div class="text empty" id="q">Esperando a que alguien hable…</div></div>
    <div class="card"><h2>Respuesta</h2><div class="text empty" id="a">—</div></div>
    <div class="card"><h2>Historial</h2><div class="hist" id="hist"></div></div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const LABEL = { responder: 'Responder', preguntar: 'Confirmar', negar: 'Negar acceso' };
const pct = (x) => (x == null ? '—' : Math.round(x * 100) + '%');
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

function setBar(id, v) { $(id).style.width = Math.max(0, Math.min(1, v || 0)) * 100 + '%'; }

function render(s) {
  $('conn').textContent = 'conectado'; $('conn').className = 'pill on';
  const live = s.fps > 0.3;
  $('fps').textContent = live ? 'cámara ' + s.fps.toFixed(1) + ' fps' : 'sin cámara';
  $('video').classList.toggle('nosignal', !live);

  const chips = $('chips'); chips.replaceChildren();
  if (!s.faces.length) chips.appendChild(el('span', 'none', 'Sin caras en cuadro'));
  const LIP_MAX = 0.15;
  s.faces.forEach((f) => {
    const row = el('div', 'face' + (f.speaking ? ' speaking' : ''));
    const left = el('div');
    left.appendChild(el('b', null, f.name));
    if (f.id) left.appendChild(el('span', 'id', 'ID ' + f.id));
    if (f.known) left.appendChild(el('span', 'sc', pct(f.score)));
    row.appendChild(left);
    row.appendChild(el('span', 'lip' + (f.speaking ? ' on' : ''),
      !f.has_lips ? 'labios: —' : (f.speaking ? 'HABLANDO' : 'callado')));
    const bar = el('div', 'lipbar');
    const fill = el('i'); fill.style.width = Math.min(1, (f.activity || 0) / LIP_MAX) * 100 + '%';
    const mark = el('u'); mark.style.left = Math.min(1, s.speaking_threshold / LIP_MAX) * 100 + '%';
    bar.appendChild(fill); bar.appendChild(mark);
    row.appendChild(bar);
    chips.appendChild(row);
  });

  const t1 = s.thresholds.respond, t2 = s.thresholds.ask;
  $('thr1').style.left = t1 * 100 + '%'; $('thr2').style.left = t2 * 100 + '%';

  const turn = s.turns.length ? s.turns[s.turns.length - 1] : null;
  if (!turn) return;
  $('who').textContent = turn.name || 'No identificado';
  $('state').textContent = LABEL[turn.state] || turn.state;
  $('state').className = 'badge ' + turn.state;
  $('conf').textContent = pct(turn.confidence);
  $('confbar').className = 'bar main ' + turn.state;
  $('confbar').firstElementChild.style.width = Math.min(1, turn.confidence || 0) * 100 + '%';
  const fs = (turn.faces || []).filter((f) => f.person === (turn.voice && turn.voice.speaker)).map((f) => f.score);
  const face = fs.length ? Math.max(...fs) : null;
  $('faceTxt').textContent = pct(face); setBar('faceBar', face);
  const v = turn.voice ? turn.voice.score : null;
  $('voiceTxt').textContent = turn.voice ? turn.voice.name + ' · ' + pct(v) : '—'; setBar('voiceBar', v);

  const q = $('q'); q.textContent = turn.question || '—'; q.className = 'text' + (turn.question ? '' : ' empty');
  const a = $('a');
  if (turn.answer) { a.textContent = turn.answer; a.className = 'text'; }
  else { a.textContent = turn.state === 'responder' ? 'Pensando…' : '—'; a.className = 'text empty'; }

  const h = $('hist'); h.replaceChildren();
  s.turns.slice(0, -1).reverse().forEach((t) => {
    const i = el('div', 'item');
    i.appendChild(el('div', 'meta', t.time + ' · ' + (t.name || 'No identificado') + ' · ' + (LABEL[t.state] || t.state) + ' · ' + pct(t.confidence)));
    i.appendChild(el('div', 'q', t.question || '—'));
    i.appendChild(el('div', 'a', t.answer || '—'));
    h.appendChild(i);
  });
}

async function tick() {
  try {
    const r = await fetch('/state', { cache: 'no-store' });
    render(await r.json());
  } catch (e) {
    $('conn').textContent = 'sin conexión'; $('conn').className = 'pill off';
  }
  setTimeout(tick, 350);
}
tick();
</script>
</body>
</html>
"""
