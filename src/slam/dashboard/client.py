def build_hud_html(ctx) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Triton ROV · Pilot HUD</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#05080f; --surface:rgba(10,15,24,.82); --surface-2:rgba(14,20,32,.94);
  --border:rgba(100,140,200,.14); --border-hi:rgba(100,160,255,.32);
  --text:#dce8f8; --muted:#5a7aa0; --label:#8aaacf; --accent:#2563eb;
  --accent-glow:rgba(37,99,235,.4); --cyan:#06b6d4; --amber:#f59e0b;
  --red:#ef4444; --green:#22c55e; --mono:'DM Mono',monospace; --display:'Syne',sans-serif;
  --panel-radius:10px; --drag-h:18px;
}}
html,body{{width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:var(--mono)}}
body::before{{content:'';position:fixed;inset:0;background:
  radial-gradient(ellipse 80vw 55vh at 20% -10%, rgba(37,99,235,.12) 0%, transparent 70%),
  radial-gradient(ellipse 60vw 50vh at 90% 110%, rgba(6,182,212,.08) 0%, transparent 65%);pointer-events:none;z-index:0}}
#topbar{{position:fixed;top:0;left:0;right:0;height:46px;background:var(--surface-2);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:20px;padding:0 18px;z-index:100;backdrop-filter:blur(12px)}}
.topbar-logo{{font-family:var(--display);font-weight:800;font-size:15px;letter-spacing:2px;text-transform:uppercase;display:flex;align-items:center;gap:8px}}
.topbar-logo::before{{content:'';display:block;width:8px;height:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan);animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.5;transform:scale(.85)}}}}
.topbar-sep{{width:1px;height:22px;background:var(--border)}}
.topbar-stat{{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}}
.topbar-stat b{{color:var(--text);font-weight:500}} .topbar-stat.live b{{color:var(--green)}} .topbar-right{{margin-left:auto;display:flex;align-items:center;gap:10px}}
#workspace{{position:fixed;top:46px;left:0;right:0;bottom:0;z-index:1;overflow:hidden}}
.panel{{position:absolute;background:var(--surface);border:1px solid var(--border);border-radius:var(--panel-radius);backdrop-filter:blur(16px);overflow:hidden;transition:border-color 120ms ease,box-shadow 120ms ease;will-change:transform}}
.panel:hover{{border-color:var(--border-hi)}} .panel.dragging,.panel.resizing{{border-color:rgba(37,99,235,.5);box-shadow:0 0 0 1px rgba(37,99,235,.18),0 20px 50px rgba(0,0,0,.5);transition:none}}
.drag-handle{{position:absolute;top:0;left:0;right:0;height:var(--drag-h);cursor:grab;z-index:10;background:linear-gradient(180deg, rgba(255,255,255,.03) 0%, transparent 100%);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 10px;gap:6px}}
.drag-handle::before{{content:'⠿';font-size:10px;color:var(--muted);opacity:.5;pointer-events:none;letter-spacing:1px}} .drag-handle:active{{cursor:grabbing}}
.panel-title{{font-family:var(--display);font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:var(--muted);pointer-events:none;user-select:none}}
.panel-body{{position:absolute;top:var(--drag-h);left:0;right:0;bottom:0;overflow:hidden}} .resize-se{{position:absolute;right:0;bottom:0;width:14px;height:14px;cursor:se-resize;z-index:10;background:linear-gradient(135deg, transparent 50%, rgba(100,160,255,.18) 50%)}}
#panel-video{{left:12px;top:12px;width:calc(100% - 344px);height:calc(100% - 24px);min-width:400px;min-height:280px}}
.video-wrap{{width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#000}}
#hud-canvas{{width:100%;height:100%;display:block;object-fit:contain}} #hud-source{{display:none}}
#rec-badge{{position:absolute;top:calc(var(--drag-h) + 10px);right:12px;display:none;background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.5);border-radius:6px;padding:4px 10px;font-size:11px;font-weight:500;letter-spacing:2px;color:var(--red);align-items:center;gap:6px;z-index:5;animation:recBlink 1.2s ease infinite}}
#rec-badge.visible{{display:flex}} #rec-dot{{width:7px;height:7px;border-radius:50%;background:var(--red)}}
@keyframes recBlink{{0%,100%{{opacity:1}}50%{{opacity:.55}}}}
#panel-telemetry{{right:12px;top:12px;width:308px;height:218px;min-width:220px;min-height:160px}}
#panel-camera{{right:12px;top:242px;width:308px;height:224px;min-width:220px;min-height:180px}}
#panel-controls{{right:12px;top:478px;width:308px;min-width:220px;min-height:60px;height:68px}}
.telem-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1px;height:100%;background:var(--border)}}
.telem-cell{{background:var(--surface);padding:12px 14px;display:flex;flex-direction:column;justify-content:space-between}}
.telem-cell:nth-child(1){{border-radius:0 0 0 calc(var(--panel-radius) - 1px)}} .telem-cell:nth-child(4){{border-radius:0 0 calc(var(--panel-radius) - 1px) 0}}
.telem-label{{font-size:9px;font-weight:500;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:4px}}
.telem-value{{font-family:var(--display);font-size:22px;font-weight:700;color:var(--text);line-height:1}} .telem-value.good{{color:var(--green)}} .telem-value.warn{{color:var(--amber)}} .telem-value.crit{{color:var(--red)}}
.telem-sub{{font-size:9px;color:var(--muted);margin-top:4px}}
.telem-bar{{height:3px;border-radius:2px;background:var(--border);margin-top:6px;overflow:hidden}} .telem-bar-fill{{height:100%;border-radius:2px;background:var(--accent);transition:width 300ms ease}} .telem-bar-fill.good{{background:var(--green)}} .telem-bar-fill.warn{{background:var(--amber)}}
.cam-body{{padding:14px 16px;display:flex;flex-direction:column;gap:16px;height:100%}} .cam-row{{display:flex;flex-direction:column;gap:6px}} .cam-row-header{{display:flex;justify-content:space-between;align-items:center}}
.cam-name{{font-size:9px;font-weight:500;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted)}} .cam-val{{font-family:var(--display);font-size:13px;font-weight:700;color:var(--text)}} .cam-val span{{color:var(--muted);font-size:10px;font-weight:400}}
input[type=range]{{-webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:2px;background:var(--border);outline:none;cursor:pointer}} input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--text);border:2px solid var(--accent);box-shadow:0 0 6px var(--accent-glow);cursor:grab;transition:transform 100ms ease,box-shadow 100ms ease}} input[type=range]::-webkit-slider-thumb:active{{cursor:grabbing;transform:scale(1.2);box-shadow:0 0 12px var(--accent-glow)}}
input[type=range]#wb-slider::-webkit-slider-thumb{{border-color:var(--cyan);box-shadow:0 0 6px rgba(6,182,212,.4)}} input[type=range]#exp-slider::-webkit-slider-thumb{{border-color:var(--amber);box-shadow:0 0 6px rgba(245,158,11,.4)}} input[type=range]#iso-slider::-webkit-slider-thumb{{border-color:var(--red);box-shadow:0 0 6px rgba(239,68,68,.4)}}
.ctrl-body{{padding:0 14px;display:flex;align-items:center;gap:10px;height:100%}}
.btn{{font-family:var(--display);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;border:1px solid var(--border);border-radius:6px;padding:8px 14px;cursor:pointer;transition:all 120ms ease;background:rgba(255,255,255,.04);color:var(--label);white-space:nowrap}}
.btn:hover{{background:rgba(255,255,255,.08);border-color:var(--border-hi);color:var(--text)}} .btn.active{{background:rgba(37,99,235,.18);border-color:rgba(37,99,235,.5);color:var(--accent);box-shadow:0 0 0 1px rgba(37,99,235,.12)}} .btn.reset{{background:rgba(255,255,255,.03);margin-left:auto}}
#state-badge{{position:absolute;top:calc(var(--drag-h) + 10px);left:12px;font-family:var(--display);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;padding:5px 12px;border-radius:6px;border:1px solid var(--border);background:rgba(10,15,24,.7);color:var(--muted);z-index:5;transition:all 300ms ease}}
#state-badge.good{{border-color:rgba(34,197,94,.4);color:var(--green);background:rgba(34,197,94,.1)}} #state-badge.warn{{border-color:rgba(245,158,11,.4);color:var(--amber);background:rgba(245,158,11,.1)}}
::-webkit-scrollbar{{width:4px;height:4px}} ::-webkit-scrollbar-track{{background:transparent}} ::-webkit-scrollbar-thumb{{background:var(--border-hi);border-radius:2px}}
</style>
</head>
<body>
<div id="topbar">
  <div class="topbar-logo">Triton ROV</div><div class="topbar-sep"></div>
  <div class="topbar-stat live"><span>FEED</span> <b id="tb-feed">LIVE</b></div><div class="topbar-sep"></div>
  <div class="topbar-stat"><span>MISSION</span> <b id="tb-state">IDLE</b></div><div class="topbar-sep"></div>
  <div class="topbar-stat"><span>SCORE</span> <b id="tb-score">1.00</b></div>
</div>
<div id="workspace">
  <div class="panel" id="panel-video"><div class="drag-handle"><span class="panel-title">Pilot View</span></div><div class="panel-body"><div class="video-wrap"><canvas id="hud-canvas"></canvas><img id="hud-source" src="/stream" alt=""></div></div><div id="state-badge">IDLE</div><div id="rec-badge"><div id="rec-dot"></div>REC</div><div class="resize-se"></div></div>
  <div class="panel" id="panel-telemetry"><div class="drag-handle"><span class="panel-title">Mission Telemetry</span></div><div class="panel-body"><div class="telem-grid">
    <div class="telem-cell"><div><div class="telem-label">Score</div><div class="telem-value" id="t-score">1.00</div><div class="telem-sub">target &gt; {ctx.target_score_good:.2f}</div></div><div class="telem-bar"><div class="telem-bar-fill" id="t-score-bar" style="width:100%"></div></div></div>
    <div class="telem-cell"><div><div class="telem-label">Blur (Lapl.)</div><div class="telem-value" id="t-blur">0.0</div><div class="telem-sub">target &gt; {ctx.laplacian_pass_threshold:.1f}</div></div><div class="telem-bar"><div class="telem-bar-fill" id="t-blur-bar" style="width:0%"></div></div></div>
    <div class="telem-cell"><div><div class="telem-label">Depth</div><div class="telem-value" id="t-depth">0.0<span>%</span></div><div class="telem-sub">target &gt; {ctx.target_depth_pct:.1f}%</div></div><div class="telem-bar"><div class="telem-bar-fill" id="t-depth-bar" style="width:0%"></div></div></div>
    <div class="telem-cell"><div><div class="telem-label">Recording</div><div class="telem-value" id="t-rec">NO</div><div class="telem-sub">target: YES while live</div></div></div>
  </div></div><div class="resize-se"></div></div>
  <div class="panel" id="panel-camera"><div class="drag-handle"><span class="panel-title">Camera Controls</span></div><div class="panel-body"><div class="cam-body">
    <div class="cam-row"><div class="cam-row-header"><div class="cam-name" style="color:var(--cyan)">White Balance</div><div class="cam-val"><span id="wbVal">4600</span><span> K</span></div></div><input type="range" id="wb-slider" min="2500" max="8000" step="100" value="4600" oninput="document.getElementById('wbVal').textContent=this.value;fetch('/set_wb?v='+this.value);"></div>
    <div class="cam-row"><div class="cam-row-header"><div class="cam-name" style="color:var(--amber)">Exposure</div><div class="cam-val"><span id="expVal">{ctx.exposure_time_us}</span><span> μs</span></div></div><input type="range" id="exp-slider" min="1000" max="33000" step="500" value="{ctx.exposure_time_us}" oninput="document.getElementById('expVal').textContent=this.value;fetch('/set_exp?v='+this.value);"></div>
    <div class="cam-row"><div class="cam-row-header"><div class="cam-name" style="color:var(--red)">ISO Sensitivity</div><div class="cam-val"><span id="isoVal">{ctx.iso_sensitivity}</span></div></div><input type="range" id="iso-slider" min="100" max="1600" step="100" value="{ctx.iso_sensitivity}" oninput="document.getElementById('isoVal').textContent=this.value;fetch('/set_iso?v='+this.value);"></div>
  </div></div><div class="resize-se"></div></div>
  <div class="panel" id="panel-controls"><div class="drag-handle"><span class="panel-title">View</span></div><div class="panel-body"><div class="ctrl-body"><button id="annotations-toggle" class="btn active" type="button">Annotations</button><button id="reset-layout-btn" class="btn reset" type="button">Reset Layout</button></div></div><div class="resize-se"></div></div>
</div>
<script>
(function(){{
  const canvas = document.getElementById('hud-canvas');
  const ctx2d = canvas.getContext('2d');
  const src = document.getElementById('hud-source');
  const targetScore = {ctx.target_score_good};
  const targetBlur = {ctx.laplacian_pass_threshold};
  const targetDepth = {ctx.target_depth_pct};
  const layoutKey = 'triton_hud_layout_v2';
  const layoutEndpoint = '/layout';
  const workspace = document.getElementById('workspace');
  const annBtn = document.getElementById('annotations-toggle');
  const resetBtn = document.getElementById('reset-layout-btn');
  let annotationsEnabled = true;
  let zTop = 10;
  let telem = {{state:'IDLE', score:1, blur:0, depth_pct:0, message:'AWAITING GRAVITY CALIBRATION', recording:false}};

  function drawHud() {{
    const w = canvas.clientWidth || 640;
    const h = canvas.clientHeight || 360;
    if (canvas.width !== w || canvas.height !== h) {{ canvas.width = w; canvas.height = h; }}
    try {{ ctx2d.drawImage(src, 0, 0, w, h); }} catch (_) {{}}
    if (annotationsEnabled) {{
      ctx2d.save();
      ctx2d.strokeStyle = 'rgba(255,255,255,.75)';
      ctx2d.lineWidth = 1;
      ctx2d.setLineDash([4,4]);
      const cx = w/2, cy = h/2;
      ctx2d.beginPath(); ctx2d.moveTo(cx-18, cy); ctx2d.lineTo(cx+18, cy); ctx2d.moveTo(cx, cy-18); ctx2d.lineTo(cx, cy+18); ctx2d.stroke();
      ctx2d.setLineDash([]); ctx2d.strokeStyle = 'rgba(255,255,255,.9)'; ctx2d.lineWidth = 1.5;
      ctx2d.beginPath(); ctx2d.arc(cx, cy, 6, 0, Math.PI*2); ctx2d.stroke(); ctx2d.restore();
      if (telem.message) {{
        ctx2d.save(); ctx2d.fillStyle = 'rgba(0,0,0,.6)'; ctx2d.fillRect(0, h-52, w, 52);
        ctx2d.font = 'bold 20px "Syne", sans-serif'; ctx2d.fillStyle = '#ef4444'; ctx2d.textAlign = 'center';
        ctx2d.fillText(telem.message, w/2, h-18); ctx2d.restore();
      }}
    }}
    requestAnimationFrame(drawHud);
  }}

  function colorClass(v, t) {{ return v >= t ? 'good' : v >= t*0.6 ? 'warn' : 'crit'; }}
  function setBar(el, val, max) {{ const pct = Math.min(100, Math.max(0, (val / max) * 100)); el.style.width = pct + '%'; el.className = 'telem-bar-fill ' + (pct >= 66 ? 'good' : pct >= 33 ? 'warn' : ''); }}
  async function pollTelemetry() {{
    try {{
      const r = await fetch('/telemetry', {{cache:'no-store'}});
      if (r.ok) {{
        telem = await r.json();
        document.getElementById('tb-state').textContent = telem.state || 'IDLE';
        document.getElementById('tb-score').textContent = Number(telem.score||0).toFixed(2);
        const score = Number(telem.score||0); const blur = Number(telem.blur||0); const depth = Number(telem.depth_pct||0) * 100;
        const scoreEl = document.getElementById('t-score'); scoreEl.textContent = score.toFixed(2); scoreEl.className = 'telem-value ' + colorClass(score, targetScore);
        const blurEl = document.getElementById('t-blur'); blurEl.textContent = blur.toFixed(1); blurEl.className = 'telem-value ' + colorClass(blur, targetBlur);
        const depthEl = document.getElementById('t-depth'); depthEl.innerHTML = depth.toFixed(1) + '<span>%</span>'; depthEl.className = 'telem-value ' + colorClass(depth, targetDepth);
        setBar(document.getElementById('t-score-bar'), score, 1); setBar(document.getElementById('t-blur-bar'), blur, targetBlur * 1.5); setBar(document.getElementById('t-depth-bar'), depth, 100);
        const recEl = document.getElementById('t-rec'); const recBadge = document.getElementById('rec-badge');
        if (telem.recording) {{ recEl.textContent = 'YES'; recEl.className = 'telem-value good'; recBadge.classList.add('visible'); }}
        else {{ recEl.textContent = 'NO'; recEl.className = 'telem-value crit'; recBadge.classList.remove('visible'); }}
        const sb = document.getElementById('state-badge'); sb.textContent = telem.state || 'IDLE'; sb.className = telem.state === 'GOOD' ? 'good' : telem.state === 'CAPTURING' ? 'warn' : '';
      }}
    }} catch (_) {{}}
    setTimeout(pollTelemetry, 200);
  }}

  function getTranslate(el) {{ const m = /translate\\(([-\\d.]+)px,\\s*([-\\d.]+)px\\)/.exec(el.style.transform || ''); return {{x:m?+m[1]:0, y:m?+m[2]:0}}; }}
  function setTranslate(el, x, y) {{ el.style.transform = `translate(${{Math.round(x)}}px, ${{Math.round(y)}}px)`; }}
  function clamp(el) {{
    const pr = workspace.getBoundingClientRect(); const er = el.getBoundingClientRect(); const t = getTranslate(el);
    const minW = parseInt(el.dataset.minW || 200); const minH = parseInt(el.dataset.minH || 100);
    const maxW = Math.max(minW, pr.width - 8); const maxH = Math.max(minH, pr.height - 8);
    el.style.width = Math.min(Math.max(minW, er.width), maxW) + 'px';
    el.style.height = Math.min(Math.max(minH, er.height), maxH) + 'px';
    const re = el.getBoundingClientRect();
    let tx = t.x + Math.max(0, pr.left + 4 - re.left) + Math.min(0, pr.right - 4 - re.right);
    let ty = t.y + Math.max(0, pr.top + 4 - re.top) + Math.min(0, pr.bottom - 4 - re.bottom);
    setTranslate(el, tx, ty);
  }}
  function edgeDir(e, el) {{
    const r = el.getBoundingClientRect(); const dx = e.clientX - r.left; const dy = e.clientY - r.top;
    const rw = r.width, rh = r.height; let d = ''; const edge = 8;
    if (dy <= edge) d += 'n'; else if (dy >= rh - edge) d += 's'; if (dx <= edge) d += 'w'; else if (dx >= rw - edge) d += 'e'; return d;
  }}
  function saveLayout() {{
    const layout = {{}}; document.querySelectorAll('.panel').forEach(p => {{ if (p.id) layout[p.id] = {{width:p.style.width||'', height:p.style.height||'', transform:p.style.transform||'translate(0px,0px)'}}; }});
    const s = JSON.stringify(layout); try {{ localStorage.setItem(layoutKey, s); }} catch (_) {{}}
    fetch(layoutEndpoint, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:s, keepalive:true}}).catch(()=>{{}});
  }}
  async function loadLayout() {{
    let layout = null;
    try {{ const r = await fetch(layoutEndpoint, {{cache:'no-store'}}); if (r.ok) {{ const d = await r.json(); if (d && Object.keys(d).length) layout = d; }} }} catch (_) {{}}
    if (!layout) {{ try {{ const raw = localStorage.getItem(layoutKey); if (raw) layout = JSON.parse(raw); }} catch (_) {{}} }}
    if (layout) {{ Object.entries(layout).forEach(([id, s]) => {{ const el = document.getElementById(id); if (!el) return; if (s.width) el.style.width = s.width; if (s.height) el.style.height = s.height; if (s.transform) el.style.transform = s.transform; clamp(el); }}); }}
  }}
  function makePanel(el) {{
    const r = el.getBoundingClientRect(); el.style.width = r.width + 'px'; el.style.height = r.height + 'px'; el.style.transform = el.style.transform || 'translate(0px,0px)';
    let mode = null, dir = ''; let sx=0, sy=0, sw=0, sh=0, stx=0, sty=0; const minW = parseInt(el.dataset.minW || 200); const minH = parseInt(el.dataset.minH || 100);
    const onMove = e => {{
      if (!mode) {{ const d = edgeDir(e, el); el.style.cursor = d ? 'se-resize' : 'default'; return; }}
      const dx = e.clientX - sx, dy = e.clientY - sy;
      if (mode === 'drag') {{
        const pr = workspace.getBoundingClientRect(); const er = el.getBoundingClientRect();
        const nx = Math.min(pr.width - er.width - 4, Math.max(4 - (er.left - pr.left - stx), stx + dx));
        const ny = Math.min(pr.height - er.height - 4, Math.max(4 - (er.top - pr.top - sty), sty + dy));
        setTranslate(el, nx, ny);
      }} else {{
        let nw=sw, nh=sh, ntx=stx, nty=sty; if (dir.includes('e')) nw = Math.max(minW, sw + dx); if (dir.includes('s')) nh = Math.max(minH, sh + dy);
        if (dir.includes('w')) {{ nw = Math.max(minW, sw - dx); ntx += sw - nw; }} if (dir.includes('n')) {{ nh = Math.max(minH, sh - dy); nty += sh - nh; }}
        el.style.width = nw + 'px'; el.style.height = nh + 'px'; setTranslate(el, ntx, nty); clamp(el);
      }}
    }};
    const onUp = () => {{ if (mode) {{ clamp(el); saveLayout(); }} mode = null; dir = ''; el.classList.remove('dragging','resizing'); document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); }};
    el.addEventListener('mousedown', e => {{
      if (e.button !== 0) return;
      if (e.target.closest('input,button,label')) return;
      const d = edgeDir(e, el); const t = getTranslate(el); const r = el.getBoundingClientRect();
      sx=e.clientX; sy=e.clientY; stx=t.x; sty=t.y; sw=r.width; sh=r.height; el.style.zIndex = ++zTop;
      const isHandle = !!e.target.closest('.drag-handle');
      if (d && !isHandle) {{ mode = 'resize'; dir = d; el.classList.add('resizing'); }}
      else if (isHandle || (!d && !e.target.closest('input,button,label,canvas'))) {{ mode = 'drag'; el.classList.add('dragging'); }}
      else return;
      e.preventDefault(); e.stopPropagation(); document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp);
    }});
    el.addEventListener('mousemove', onMove); el.addEventListener('mouseleave', () => {{ if (!mode) el.style.cursor='default'; }});
  }}

  document.querySelectorAll('.panel').forEach(makePanel);
  loadLayout(); window.addEventListener('beforeunload', saveLayout);
  resetBtn.addEventListener('click', () => {{ document.querySelectorAll('.panel').forEach(p => {{ p.style.width=''; p.style.height=''; p.style.transform=''; clamp(p); }}); try {{ localStorage.removeItem(layoutKey); }} catch(_){{}} fetch(layoutEndpoint, {{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}',keepalive:true}}).catch(()=>{{}}); }});
  annBtn.addEventListener('click', () => {{ annotationsEnabled = !annotationsEnabled; annBtn.classList.toggle('active', annotationsEnabled); annBtn.textContent = 'Annotations'; }});
  requestAnimationFrame(drawHud); pollTelemetry();
}})();
</script>
</body>
</html>"""
