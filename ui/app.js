const runBtn = document.getElementById('runBtn');
const streamBtn = document.getElementById('streamBtn');
const stopBtn = document.getElementById('stopBtn');
const queryEl = document.getElementById('query');
const sessionEl = document.getElementById('session');
const streamEl = document.getElementById('stream');
const resultEl = document.getElementById('result');
const copyBtn = document.getElementById('copyJson');
const downloadBtn = document.getElementById('downloadJson');
const recsTable = document.getElementById('recsTable');
const recsEmpty = document.getElementById('recsEmpty');
const statusEl = document.getElementById('status');
const summaryInfo = document.getElementById('summaryInfo');
const summaryBody = document.getElementById('summaryBody');

let currentEventSource = null;
let lastResponse = null;
let latestValidated = undefined;
let latestRecs = [];

async function pingServer() {
  try {
    const res = await fetch('/query?query=ping');
    statusEl.textContent = res.ok ? 'online' : 'offline';
  } catch {
    statusEl.textContent = 'offline';
  }
}

function getParams() {
  const query = queryEl.value.trim();
  const session_id = sessionEl.value.trim() || undefined;
  return { query, session_id };
}

function toUrl(params) {
  const search = new URLSearchParams();
  if (params.query) search.set('query', params.query);
  if (params.session_id) search.set('session_id', params.session_id);
  return search.toString();
}

async function runOnce() {
  stopStreaming();
  streamEl.textContent = '';
  resultEl.textContent = 'Running...';
  const params = getParams();
  if (!params.query) {
    resultEl.textContent = 'Please enter a query.';
    return;
  }
  const url = `/query?${toUrl(params)}`;
  try {
    const res = await fetch(url);
    const data = await res.json();
    lastResponse = data;
    resultEl.textContent = JSON.stringify(data, null, 2);
    renderSummary(data);
    renderRecommendations(data);
  } catch (e) {
    resultEl.textContent = `Error: ${e?.message || e}`;
  }
}

function startStreaming() {
  stopStreaming();
  streamEl.textContent = '';
  resultEl.textContent = '';
  const params = getParams();
  if (!params.query) {
    streamEl.textContent = 'Please enter a query.';
    return;
  }
  const url = `/stream?${toUrl(params)}`;
  currentEventSource = new EventSource(url);
  stopBtn.disabled = false;
  streamBtn.disabled = true;
  runBtn.disabled = true;

  currentEventSource.addEventListener('update', (e) => {
    try {
      const data = JSON.parse(e.data);
      streamEl.textContent += JSON.stringify(data) + '\n';
      streamEl.scrollTop = streamEl.scrollHeight;
      // Track validated state seen in any event
      if (typeof data.validated === 'boolean') {
        latestValidated = data.validated;
      }
      // Try to render summary and recommendations opportunistically during streaming
      // If the event already contains partial or final fields, this will show them.
      try { renderSummary(data); } catch {}
      try { renderRecommendations(data, { streaming: true }); } catch {}
    } catch {
      streamEl.textContent += e.data + '\n';
    }
  });
  currentEventSource.addEventListener('error', (e) => {
    // Some browsers fire 'error' on normal SSE disconnects without data.
    // If there's no data, treat it as a benign network event unless we've never received anything.
    const hasData = e && typeof e.data === 'string' && e.data.length > 0;
    if (hasData) {
      try {
        const data = JSON.parse(e.data);
        streamEl.textContent += 'ERROR: ' + JSON.stringify(data) + '\n';
      } catch {
        streamEl.textContent += 'ERROR: ' + (e?.message || 'unknown') + '\n';
      }
    }
    // Do not auto-stop here; wait for explicit 'end' or user stop.
  });
  // Handle explicit server-completion event
  currentEventSource.addEventListener('end', () => {
    stopStreaming();
  });
  // Not all servers send 'end' events; we rely on user to Stop for now.
}

async function stopStreaming() {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  stopBtn.disabled = true;
  streamBtn.disabled = false;
  runBtn.disabled = false;
  // After stopping a stream, fetch the final result once to populate recommendations
  const params = getParams();
  if (params.query) {
    try { await fetchFinalResult(params); } catch {}
  }
}

runBtn.addEventListener('click', runOnce);
streamBtn.addEventListener('click', startStreaming);
stopBtn.addEventListener('click', stopStreaming);

pingServer();
setInterval(pingServer, 5000);
async function fetchFinalResult(params) {
  const url = `/query?${toUrl(params)}`;
  const res = await fetch(url);
  const data = await res.json();
  lastResponse = data;
  if (typeof data.validated === 'boolean') {
    latestValidated = data.validated;
  }
  resultEl.textContent = JSON.stringify(data, null, 2);
  renderSummary(data);
  renderRecommendations(data, { streaming: false });
}

function renderRecommendations(data, opts = {}) {
  try {
    const streaming = !!opts.streaming;
    const tbodyOld = recsTable.querySelector('tbody');
    const tbody = document.createElement('tbody');

    const baseRecs = (data?.comparison?.ranked_recommendations) || [];
    let source = 'ranked';
    let recs = baseRecs;

    if (!recs.length) {
      // Fallback A: example_collections from results.queries or analysis.queries
      const ex = [];
      const queries = (data?.results?.queries || data?.analysis?.queries || []);
      for (const q of queries) {
        const arr = q?.example_collections || [];
        for (const n of arr) if (n) ex.push(String(n));
      }
      const unique = Array.from(new Set(ex)).slice(0, 5);
      if (unique.length) {
        source = 'examples';
        recs = unique.map((name, i) => ({ collection: name, rank: i + 1, why: 'from example_collections' }));
      }
    }

    if (!recs.length) {
      // Fallback B: related_collections concept IDs (top level)
      const relTop = (data?.related_collections || []).map((r) => r?.concept_id).filter(Boolean);
      // Fallback C: per-query related_collections (strings of concept ids)
      const qrels = [];
      const queries = (data?.results?.queries || data?.analysis?.queries || []);
      for (const q of queries) {
        for (const cid of (q?.related_collections || [])) {
          if (cid) qrels.push(String(cid));
        }
      }
      const merged = Array.from(new Set([...relTop, ...qrels])).slice(0, 5);
      if (merged.length) {
        source = 'related';
        recs = merged.map((cid, i) => ({ collection: cid, rank: i + 1, why: 'related collection' }));
      }
    }

    const info = document.getElementById('recsInfo');
    if (!recs.length) {
      // During streaming, do not overwrite existing recs UI if current event has none
      if (streaming) {
        return;
      }
      // If we previously had recommendations (from stream), keep showing them
      if (latestRecs && latestRecs.length) {
        recs = latestRecs;
        source = 'ranked';
      } else {
        recsEmpty.style.display = 'block';
        recsTable.style.display = 'none';
        if (info) {
          const validatedFromData = data?.validated;
          const validated = typeof validatedFromData === 'boolean' ? validatedFromData : (typeof latestValidated === 'boolean' ? latestValidated : !!lastResponse?.validated);
          info.style.display = 'block';
          info.textContent = validated ? 'No recommendations available for this query.' : 'No recommendations (query not validated). Try adding dates, region, or keywords.';
        }
        // Replace tbody to clear any old listeners
        recsTable.replaceChild(tbody, tbodyOld);
        return;
      }
    }

    // Persist latest non-empty recommendations
    if (recs && recs.length) {
      latestRecs = recs.slice(0, 5);
    }
    if (info) {
      info.style.display = 'block';
      info.textContent = source === 'ranked' ? 'Top recommendations' : source === 'examples' ? 'Suggestions from example collections' : 'Suggestions from related collections';
    }

    recsEmpty.style.display = 'none';
    recsTable.style.display = 'table';
    for (const r of recs) {
      const tr = document.createElement('tr');
      const name = String(r.collection || 'Unknown');
      const rank = r.rank ?? '';
      const why = r.why || '';
      const { url, label, copyLabel, copyValue } = cmrLinkForNameOrId(name);
      tr.innerHTML = `
        <td>${rank}</td>
        <td>
          <a href="${url}" target="_blank" rel="noopener">${escapeHtml(label)}</a>
        </td>
        <td>
          <button class="secondary" data-action="open" data-url="${url}">Open</button>
          <button class="secondary" data-action="copy" data-name="${escapeAttr(copyValue)}">${copyLabel}</button>
        </td>
        <td>${escapeHtml(why)}</td>
      `;
      tbody.appendChild(tr);
    }

    // Replace tbody to avoid accumulating listeners
    tbody.addEventListener('click', onRecsAction);
    recsTable.replaceChild(tbody, tbodyOld);
  } catch {}
}

function renderSummary(data) {
  try {
    const text = data?.synthesis || '';
    if (!text) {
      if (summaryInfo) {
        summaryInfo.style.display = 'block';
        summaryInfo.textContent = 'No summary available yet.';
      }
      if (summaryBody) summaryBody.innerHTML = '';
      return;
    }
    if (summaryInfo) {
      summaryInfo.style.display = 'block';
      summaryInfo.textContent = 'Why these datasets were selected';
    }
    // Support simple markdown headings in synthesis. Minimal sanitization via escapeHtml then reinsert basic headings.
    const escaped = escapeHtml(text);
    const withHeadings = escaped
      .replaceAll(/^###\s+(.+)$/gm, '<h3>$1</h3>')
      .replaceAll(/^####\s+(.+)$/gm, '<h4>$1</h4>')
      .replaceAll(/^\*\*([^*]+)\*\*/gm, '<strong>$1</strong>')
      .replaceAll(/^\-\s+/gm, '&#8226; ');
    if (summaryBody) summaryBody.innerHTML = `<div class="summary-content">${withHeadings.replaceAll('\n', '<br/>')}</div>`;
  } catch {}
}

function onRecsAction(e) {
  const btn = e.target.closest('button[data-action]');
  if (!btn) return;
  const action = btn.getAttribute('data-action');
  if (action === 'open') {
    const url = btn.getAttribute('data-url');
    if (url) window.open(url, '_blank');
  } else if (action === 'copy') {
    const name = btn.getAttribute('data-name') || '';
    navigator.clipboard?.writeText(name);
  }
}

function cmrLinkForNameOrId(nameOrId) {
  const s = String(nameOrId || '');
  const conceptId = /^C\d{3,}-[A-Z0-9_]+/i.test(s) ? s : null;
  if (conceptId) {
    // Open directly in Earthdata Search granules view for the collection
    const url = `https://search.earthdata.nasa.gov/search/granules?p=${encodeURIComponent(conceptId)}`;
    return { url, label: conceptId, copyLabel: 'Copy ID', copyValue: conceptId };
  }
  // Fallback: open a friendly Earthdata Search query by name/keyword
  const url = `https://search.earthdata.nasa.gov/search?q=${encodeURIComponent(s)}`;
  return { url, label: s, copyLabel: 'Copy name', copyValue: s };
}

function escapeHtml(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function escapeAttr(s) {
  return escapeHtml(s).replaceAll('"', '&quot;');
}

copyBtn.addEventListener('click', async () => {
  if (!lastResponse) return;
  const text = JSON.stringify(lastResponse, null, 2);
  await navigator.clipboard?.writeText(text);
});

downloadBtn.addEventListener('click', () => {
  if (!lastResponse) return;
  const blob = new Blob([JSON.stringify(lastResponse, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'cmr_agent_result.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});


