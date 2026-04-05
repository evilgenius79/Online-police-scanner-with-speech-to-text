/**
 * Police Scanner – frontend application
 *
 * Responsibilities:
 *   1. Live audio: WebSocket → Web Audio API scheduler → speaker
 *   2. Frequency-spectrum visualiser on <canvas id="visualizer">
 *   3. WebSocket events: new_clip, transcript_ready, transmission_start/end
 *   4. Clips list: paginated REST API, date filter, prepend-on-new-clip
 *   5. Search: debounced full-text search with <mark> highlighting
 *   6. Status polling: updates header + stats card every 15 s
 */
'use strict';

/* ═══════════════════════════════════════════════════════════════
   Configuration
═══════════════════════════════════════════════════════════════ */
const WS_URL     = `ws://${location.host}/ws/audio`;
const API_BASE   = '/api';
const SAMPLE_RATE = 16000;
const SEARCH_DEBOUNCE_MS = 450;

/* ═══════════════════════════════════════════════════════════════
   Module state
═══════════════════════════════════════════════════════════════ */
let audioCtx      = null;   // AudioContext (created on user interaction)
let gainNode      = null;   // master volume
let analyserNode  = null;   // feeds the visualiser
let ws            = null;   // WebSocket
let wsReconnectTimer = null;
let nextPlayTime  = 0;      // next scheduled playback time in audioCtx
let wsConnected   = false;
let vizAnimId     = null;   // requestAnimationFrame handle

// Clip browser state
let currentPage   = 1;
let currentDate   = null;   // null = all, "YYYY-MM-DD" = filtered
let isSearching   = false;
let currentQuery  = '';
let searchTimer   = null;

/* ═══════════════════════════════════════════════════════════════
   Public namespace  (called from HTML onclick= attributes)
═══════════════════════════════════════════════════════════════ */
const App = {
  toggleAudio,
  setVolume,
  onSearchInput,
  doSearch,
  clearSearch,
  loadPage,
};
window.App = App;

/* ═══════════════════════════════════════════════════════════════
   Initialisation
═══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  loadClips(1, null);
  loadDates();
  pollStatus();
  setInterval(pollStatus, 15_000);
});

/* ═══════════════════════════════════════════════════════════════
   Audio context + WebSocket
═══════════════════════════════════════════════════════════════ */
function initAudioContext() {
  if (audioCtx) return;
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'interactive' });
  gainNode = audioCtx.createGain();
  gainNode.gain.value = parseFloat(document.getElementById('volume').value);
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 512;
  analyserNode.smoothingTimeConstant = 0.6;
  gainNode.connect(analyserNode);
  analyserNode.connect(audioCtx.destination);
  startVisualiser();
}

function toggleAudio() {
  if (wsConnected) {
    disconnectAudio();
  } else {
    connectAudio();
  }
}

function connectAudio() {
  initAudioContext();

  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }

  clearTimeout(wsReconnectTimer);
  openWebSocket();
}

function disconnectAudio() {
  wsConnected = false;
  if (ws) {
    ws.onclose = null;  // suppress auto-reconnect
    ws.close();
    ws = null;
  }
  setWsStatus(false);
}

function openWebSocket() {
  if (ws) {
    ws.onclose = null;
    ws.close();
  }

  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    wsConnected = true;
    setWsStatus(true);
    nextPlayTime = 0;  // reset playback clock on reconnect
    console.info('[Scanner] WebSocket connected');
  };

  ws.onclose = () => {
    wsConnected = false;
    setWsStatus(false);
    console.info('[Scanner] WebSocket closed – reconnecting in 3 s');
    wsReconnectTimer = setTimeout(openWebSocket, 3000);
  };

  ws.onerror = (err) => {
    console.warn('[Scanner] WebSocket error', err);
  };

  ws.onmessage = handleWsMessage;
}

/* ═══════════════════════════════════════════════════════════════
   WebSocket message handler
═══════════════════════════════════════════════════════════════ */
function handleWsMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    scheduleAudio(event.data);
    return;
  }

  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }

  switch (msg.type) {
    case 'transmission_start':
      setTransmitting(true);
      break;

    case 'transmission_end':
      setTransmitting(false);
      break;

    case 'new_clip':
      if (!isSearching && currentPage === 1 && currentDate === null) {
        prependClip(msg.clip);
      }
      break;

    case 'transcript_ready':
      updateTranscriptInDom(msg.clip_id, msg.transcript);
      break;

    case 'ping':
      break;  // server keepalive, no action needed

    default:
      break;
  }
}

/* ═══════════════════════════════════════════════════════════════
   PCM scheduling (Web Audio API)
   Each 30 ms PCM frame arrives as a raw Int16 ArrayBuffer.
   We convert to Float32, wrap in an AudioBuffer, and schedule it
   slightly ahead so there are no gaps between frames.
═══════════════════════════════════════════════════════════════ */
function scheduleAudio(buffer) {
  if (!audioCtx) return;

  const int16 = new Int16Array(buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }

  const audioBuffer = audioCtx.createBuffer(1, float32.length, SAMPLE_RATE);
  audioBuffer.copyToChannel(float32, 0);

  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(gainNode);

  // Keep a small lookahead buffer to prevent underruns without adding
  // noticeable latency.  100 ms should cover scheduling jitter.
  const LOOKAHEAD = 0.10;   // seconds
  const now = audioCtx.currentTime;

  if (nextPlayTime < now + LOOKAHEAD) {
    // Gap detected (reconnect, tab was hidden, etc.) – resync clock.
    nextPlayTime = now + LOOKAHEAD;
  }

  source.start(nextPlayTime);
  nextPlayTime += audioBuffer.duration;
}

/* ═══════════════════════════════════════════════════════════════
   Visualiser
═══════════════════════════════════════════════════════════════ */
function startVisualiser() {
  if (vizAnimId !== null) return;  // already running

  const canvas = document.getElementById('visualizer');
  const ctx = canvas.getContext('2d');

  // The canvas physical size must match its CSS display size for sharp drawing.
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width  = Math.round(rect.width  * devicePixelRatio);
    canvas.height = Math.round(rect.height * devicePixelRatio);
  }
  resizeCanvas();
  new ResizeObserver(resizeCanvas).observe(canvas);

  const dataArray = new Uint8Array(256);  // half of analyser.fftSize

  function draw() {
    vizAnimId = requestAnimationFrame(draw);

    const W = canvas.width;
    const H = canvas.height;

    if (analyserNode) {
      analyserNode.getByteFrequencyData(dataArray);
    }

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0e15';
    ctx.fillRect(0, 0, W, H);

    const barCount = dataArray.length;
    const barW = W / barCount;

    for (let i = 0; i < barCount; i++) {
      const v = dataArray[i] / 255;
      const barH = v * H;

      // Colour: dim green when low, bright green → yellow at peaks
      const lightness = 25 + v * 40;
      const hue = 120 - v * 40;  // 120 = green, 80 = yellow-green
      ctx.fillStyle = `hsl(${hue}, 100%, ${lightness}%)`;

      ctx.fillRect(
        i * barW,
        H - barH,
        Math.max(barW - 1, 1),
        barH,
      );
    }
  }

  draw();
}

/* ═══════════════════════════════════════════════════════════════
   UI helpers
═══════════════════════════════════════════════════════════════ */
function setWsStatus(online) {
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  const btn   = document.getElementById('btn-connect');

  if (online) {
    dot.className = 'status-dot status-dot--online';
    label.textContent = 'Connected';
    btn.textContent   = 'Disconnect';
    btn.classList.replace('btn--primary', 'btn--danger');
  } else {
    dot.className = 'status-dot status-dot--offline';
    label.textContent = wsConnected ? 'Reconnecting…' : 'Disconnected';
    btn.textContent   = 'Connect Audio';
    btn.classList.replace('btn--danger', 'btn--primary');
  }
}

function setTransmitting(active) {
  const badge = document.getElementById('live-badge');
  const bar   = document.getElementById('tx-bar');
  const label = document.getElementById('tx-label');

  if (active) {
    badge.className   = 'badge badge--live';
    badge.textContent = 'LIVE';
    bar.className     = 'tx-bar tx-bar--active';
    label.textContent = 'TRANSMISSION';
  } else {
    badge.className   = 'badge badge--off';
    badge.textContent = 'OFF AIR';
    bar.className     = 'tx-bar tx-bar--silent';
    label.textContent = 'MONITORING';
  }
}

function setVolume(v) {
  if (gainNode) gainNode.gain.value = parseFloat(v);
}

/* ═══════════════════════════════════════════════════════════════
   Status polling
═══════════════════════════════════════════════════════════════ */
async function pollStatus() {
  try {
    const data = await apiFetch('/api/status');
    document.getElementById('stat-total').textContent   = data.total_clips ?? '—';
    document.getElementById('stat-pending').textContent = data.pending_transcription ?? '—';
    document.getElementById('stat-clients').textContent = data.ws_clients ?? '—';
    document.getElementById('stat-model').textContent   = data.model_loaded ? 'Yes' : 'Loading…';
  } catch {
    // Network error – don't crash the UI
  }
}

/* ═══════════════════════════════════════════════════════════════
   Date browser
═══════════════════════════════════════════════════════════════ */
async function loadDates() {
  const container = document.getElementById('date-list');
  try {
    const dates = await apiFetch('/api/dates');
    if (!dates.length) {
      container.innerHTML = '<p class="empty-msg">No recordings yet.</p>';
      return;
    }

    container.innerHTML = '';

    // "All" button
    const allBtn = document.createElement('button');
    allBtn.className = 'date-btn' + (currentDate === null ? ' active' : '');
    allBtn.innerHTML = '<span>All dates</span>';
    allBtn.addEventListener('click', () => {
      currentDate = null;
      setActiveDateBtn(allBtn);
      loadClips(1, null);
    });
    container.appendChild(allBtn);

    for (const item of dates) {
      const btn = document.createElement('button');
      btn.className = 'date-btn' + (item.date === currentDate ? ' active' : '');
      btn.dataset.date = item.date;
      btn.innerHTML = `
        <span>${formatDateLabel(item.date)}</span>
        <span class="date-btn-count">${item.count}</span>
      `;
      btn.addEventListener('click', () => {
        currentDate = item.date;
        isSearching = false;
        document.getElementById('search-input').value = '';
        document.getElementById('btn-clear-search').style.display = 'none';
        setActiveDateBtn(btn);
        loadClips(1, item.date);
        document.getElementById('clips-title').textContent = formatDateLabel(item.date);
      });
      container.appendChild(btn);
    }
  } catch (err) {
    container.innerHTML = '<p class="empty-msg">Could not load dates.</p>';
    console.error('[Scanner] loadDates:', err);
  }
}

function setActiveDateBtn(activeBtn) {
  document.querySelectorAll('.date-btn').forEach(b => b.classList.remove('active'));
  activeBtn.classList.add('active');
}

function formatDateLabel(dateStr) {
  // dateStr = "YYYY-MM-DD"
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const d = new Date(dateStr + 'T00:00:00');
  const diff = Math.round((today - d) / 86_400_000);

  if (diff === 0) return 'Today';
  if (diff === 1) return 'Yesterday';

  return d.toLocaleDateString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric',
    year: d.getFullYear() !== today.getFullYear() ? 'numeric' : undefined,
  });
}

/* ═══════════════════════════════════════════════════════════════
   Clips loading
═══════════════════════════════════════════════════════════════ */
function loadPage(page) {
  if (isSearching) {
    doSearch(page);
  } else {
    loadClips(page, currentDate);
  }
}

async function loadClips(page, date) {
  isSearching = false;
  currentPage = page;
  currentDate = date;

  const params = new URLSearchParams({ page, per_page: 20 });
  if (date) params.set('date', date);

  renderClipsLoading();
  try {
    const data = await apiFetch(`/api/clips?${params}`);
    renderClips(data);
  } catch (err) {
    renderClipsError(err);
  }
}

/* ═══════════════════════════════════════════════════════════════
   Search
═══════════════════════════════════════════════════════════════ */
function onSearchInput() {
  clearTimeout(searchTimer);
  const q = document.getElementById('search-input').value.trim();
  if (!q) {
    clearSearch();
    return;
  }
  searchTimer = setTimeout(() => doSearch(1), SEARCH_DEBOUNCE_MS);
}

async function doSearch(page = 1) {
  const q = document.getElementById('search-input').value.trim();
  if (!q) {
    clearSearch();
    return;
  }

  isSearching = true;
  currentQuery = q;
  currentPage  = page;
  currentDate  = null;

  document.getElementById('btn-clear-search').style.display = '';
  document.getElementById('clips-title').textContent = `Search: "${q}"`;

  const params = new URLSearchParams({ q, page, per_page: 20 });
  renderClipsLoading();
  try {
    const data = await apiFetch(`/api/search?${params}`);
    renderClips(data);
  } catch (err) {
    renderClipsError(err);
  }
}

function clearSearch() {
  isSearching = false;
  currentQuery = '';
  document.getElementById('search-input').value = '';
  document.getElementById('btn-clear-search').style.display = 'none';
  document.getElementById('clips-title').textContent = 'Recent Transmissions';
  loadClips(1, null);
}

/* ═══════════════════════════════════════════════════════════════
   Clip rendering
═══════════════════════════════════════════════════════════════ */
function renderClipsLoading() {
  document.getElementById('clips-list').innerHTML = '<p class="empty-msg">Loading…</p>';
  document.getElementById('clips-count').textContent = '';
  document.getElementById('pagination').innerHTML = '';
}

function renderClipsError(err) {
  console.error('[Scanner] clips error:', err);
  document.getElementById('clips-list').innerHTML =
    '<p class="empty-msg">Could not load clips. Is the server running?</p>';
}

function renderClips(data) {
  const list    = document.getElementById('clips-list');
  const countEl = document.getElementById('clips-count');
  const pagEl   = document.getElementById('pagination');

  if (!data.clips.length) {
    list.innerHTML = '<p class="empty-msg">No transmissions found.</p>';
    countEl.textContent = '';
    pagEl.innerHTML = '';
    return;
  }

  countEl.textContent = `${data.total.toLocaleString()} clip${data.total !== 1 ? 's' : ''}`;
  list.innerHTML = '';
  for (const clip of data.clips) {
    list.appendChild(buildClipCard(clip));
  }

  renderPagination(data.page, data.pages);
}

function buildClipCard(clip) {
  const tpl  = document.getElementById('clip-card-tpl');
  const node = tpl.content.cloneNode(true);
  const card = node.querySelector('.clip-card');

  card.dataset.id = clip.id;
  card.querySelector('.clip-time').textContent     = formatDateTime(clip.start_time);
  card.querySelector('.clip-duration').textContent = formatDuration(clip.duration);

  const textEl    = card.querySelector('.transcript-text');
  const pendingEl = card.querySelector('.transcript-pending');

  if (clip.transcript === null) {
    // STT not yet complete
    textEl.classList.add('hidden');
    pendingEl.classList.remove('hidden');
  } else if (clip.transcript === '') {
    // Whisper found no speech
    textEl.classList.add('transcript-empty');
    textEl.textContent = '[No speech detected]';
  } else if (clip.snippet) {
    // Search result with highlighted snippet
    textEl.innerHTML = sanitiseSnippet(clip.snippet);
  } else {
    textEl.textContent = clip.transcript;
  }

  card.querySelector('.clip-audio').src = clip.audio_url;
  return card;
}

function prependClip(clip) {
  const list = document.getElementById('clips-list');

  // Remove "no transmissions" placeholder if present
  const empty = list.querySelector('.empty-msg');
  if (empty) empty.remove();

  const card = buildClipCard(clip);
  card.querySelector('.clip-card').classList.add('clip-card--new');
  list.prepend(card);
}

function updateTranscriptInDom(clipId, transcript) {
  const card = document.querySelector(`.clip-card[data-id="${clipId}"]`);
  if (!card) return;

  const textEl    = card.querySelector('.transcript-text');
  const pendingEl = card.querySelector('.transcript-pending');

  pendingEl.classList.add('hidden');
  textEl.classList.remove('hidden');

  if (transcript) {
    textEl.classList.remove('transcript-empty');
    textEl.textContent = transcript;
  } else {
    textEl.classList.add('transcript-empty');
    textEl.textContent = '[No speech detected]';
  }
}

/* ═══════════════════════════════════════════════════════════════
   Pagination
═══════════════════════════════════════════════════════════════ */
function renderPagination(currentPg, totalPages) {
  const nav = document.getElementById('pagination');
  nav.innerHTML = '';
  if (totalPages <= 1) return;

  function makeBtn(label, page, active = false, disabled = false) {
    const btn = document.createElement('button');
    btn.className = 'page-btn' + (active ? ' page-btn--active' : '');
    btn.textContent = label;
    btn.disabled = disabled;
    if (!disabled && !active) {
      btn.addEventListener('click', () => loadPage(page));
    }
    return btn;
  }

  nav.appendChild(makeBtn('‹', currentPg - 1, false, currentPg === 1));

  const pages = paginationRange(currentPg, totalPages);
  let prev = null;
  for (const pg of pages) {
    if (prev !== null && pg - prev > 1) {
      const dots = document.createElement('span');
      dots.className = 'page-btn';
      dots.textContent = '…';
      dots.style.cursor = 'default';
      nav.appendChild(dots);
    }
    nav.appendChild(makeBtn(pg, pg, pg === currentPg));
    prev = pg;
  }

  nav.appendChild(makeBtn('›', currentPg + 1, false, currentPg === totalPages));
}

function paginationRange(current, total) {
  // Always show first, last, and up to 3 pages around current.
  const pages = new Set([1, total]);
  for (let p = Math.max(1, current - 2); p <= Math.min(total, current + 2); p++) {
    pages.add(p);
  }
  return Array.from(pages).sort((a, b) => a - b);
}

/* ═══════════════════════════════════════════════════════════════
   Formatting helpers
═══════════════════════════════════════════════════════════════ */
function formatDateTime(isoStr) {
  // isoStr from Python: "2024-01-15T14:30:22.123456"
  const d = new Date(isoStr.replace(' ', 'T'));
  if (isNaN(d)) return isoStr;
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  });
}

function formatDuration(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

/**
 * Sanitise a snippet string that may contain <mark>…</mark> tags (from FTS5).
 * All other HTML is escaped to prevent XSS.
 */
function sanitiseSnippet(text) {
  if (!text) return '';
  // 1. Escape ALL HTML entities.
  const escaped = text
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#x27;');
  // 2. Re-allow only the specific <mark> / </mark> tags we emit server-side.
  return escaped
    .replace(/&lt;mark&gt;/g,  '<mark>')
    .replace(/&lt;\/mark&gt;/g, '</mark>');
}

/* ═══════════════════════════════════════════════════════════════
   Fetch helper
═══════════════════════════════════════════════════════════════ */
async function apiFetch(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
  }
  return resp.json();
}
