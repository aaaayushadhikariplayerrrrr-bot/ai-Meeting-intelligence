/**
 * liveTranslation.js — Live microphone → local Whisper → local translation.
 *
 * Depends on globals: WhisperEngine, TranslationEngine (loaded before this file).
 *
 * Pipeline (all on-device):
 *   Microphone → MediaRecorder → 5s chunks → Web Audio decode
 *   → local Whisper ASR → recognised text → local OPUS-MT → translated text
 *
 * PRIVACY: Microphone audio never leaves the browser.
 * No cloud ASR or cloud translation API is used.
 */
var LiveTranslation = (function () {
  var CHUNK_SEC   = 5;
  var SAMPLE_RATE = 16000;

  var _micStream  = null;
  var _recorder   = null;
  var _isRunning  = false;
  var _chunks     = [];
  var _winTimer   = null;
  var _segIdx     = 0;

  // ─── Render ──────────────────────────────────────────────────────────────

  function renderLiveTranslationPage() {
    var pairs     = TranslationEngine.SUPPORTED_PAIRS;
    var pairOpts  = pairs.map(function (p, i) {
      return '<option value="' + p.src + ':' + p.tgt + '"' + (i === 0 ? ' selected' : '') + '>' + p.label + '</option>';
    }).join('');

    var modelOpts = Object.keys(WhisperEngine.MODEL_OPTIONS).map(function (k) {
      var m = WhisperEngine.MODEL_OPTIONS[k];
      return '<option value="' + k + '">' + m.label + '</option>';
    }).join('');

    return [
      /* Privacy badges */
      '<div class="row" style="margin-bottom:14px;flex-wrap:wrap;gap:6px">',
        '<span class="b g">⛨ LOCAL ONLY</span>',
        '<span class="b g">MIC → DEVICE → TRANSCRIPT</span>',
        '<span class="b" style="color:var(--r);border-color:var(--r)">✗ NO CLOUD ASR</span>',
        '<span class="b" style="color:var(--r);border-color:var(--r)">✗ NO CLOUD TRANSLATION</span>',
      '</div>',

      /* Controls */
      '<div class="card" style="margin-bottom:14px">',
        '<div class="grid g3" style="margin-bottom:14px">',
          '<div>',
            '<label style="display:block;margin-bottom:4px;color:var(--m);font-size:12px">LANGUAGE PAIR</label>',
            '<select id="lt-pair" style="width:100%">', pairOpts, '</select>',
          '</div>',
          '<div>',
            '<label style="display:block;margin-bottom:4px;color:var(--m);font-size:12px">WHISPER MODEL</label>',
            '<select id="lt-model" style="width:100%">', modelOpts, '</select>',
          '</div>',
          '<div>',
            '<label style="display:block;margin-bottom:4px;color:var(--m);font-size:12px">TRANSLATION</label>',
            '<label style="display:flex;gap:8px;align-items:center;margin-top:8px">',
              '<input type="checkbox" class="tog" id="lt-translate-on" checked>',
              '<span>Translate output</span>',
            '</label>',
          '</div>',
        '</div>',
        '<div class="row sp" style="margin:0">',
          '<div class="row" style="margin:0;gap:10px">',
            '<button class="pri" id="lt-start">▶ Start</button>',
            '<button id="lt-stop" disabled>■ Stop</button>',
            '<span id="lt-mic-status" class="muted" style="font-size:12px">Microphone: not started</span>',
          '</div>',
          '<span id="lt-runtime-status" style="font-size:12px;color:var(--m)"></span>',
        '</div>',
      '</div>',

      /* Loading indicator */
      '<div id="lt-loading" style="display:none;margin-bottom:14px">',
        '<div class="card" style="border-color:var(--c)">',
          '<div class="row" style="margin:0"><span class="pulse" style="color:var(--c)">●</span>',
          '<b id="lt-loading-msg" style="margin-left:8px">Loading…</b></div>',
          '<div class="bar" style="margin-top:8px">',
            '<i id="lt-loading-bar" style="width:0%;display:block;height:100%;background:var(--c);transition:width .4s"></i>',
          '</div>',
        '</div>',
      '</div>',

      /* Two-column live transcript */
      '<div class="grid" style="grid-template-columns:1fr 1fr;gap:14px">',
        '<div class="card">',
          '<div class="row sp" style="margin:0 0 8px">',
            '<h2 style="margin:0" id="lt-src-label">Recognised Speech</h2>',
            '<span id="lt-whisper-badge" class="b c" style="font-size:11px">Whisper local</span>',
          '</div>',
          '<div class="tx" id="lt-src-tx" style="height:340px"></div>',
        '</div>',
        '<div class="card">',
          '<div class="row sp" style="margin:0 0 8px">',
            '<h2 style="margin:0" id="lt-tgt-label">Translation</h2>',
            '<span class="b g" style="font-size:11px">Local OPUS-MT</span>',
          '</div>',
          '<div class="tx" id="lt-tgt-tx" style="height:340px"></div>',
        '</div>',
      '</div>',

      /* Pipeline diagram */
      '<div class="card" style="margin-top:14px">',
        '<b style="font-size:12px;color:var(--m)">LIVE PIPELINE (ALL ON-DEVICE)</b>',
        '<div class="flow" style="margin-top:8px;grid-template-columns:repeat(auto-fit,minmax(70px,1fr));font-size:11px">',
          '<div>🎙 Mic</div><span style="color:var(--c)">→</span>',
          '<div>🌊 Web Audio</div><span style="color:var(--c)">→</span>',
          '<div>✂ 5s chunks</div><span style="color:var(--c)">→</span>',
          '<div>🧠 Whisper</div><span style="color:var(--c)">→</span>',
          '<div>🔤 Text</div><span style="color:var(--c)">→</span>',
          '<div>🌐 OPUS-MT</div><span style="color:var(--c)">→</span>',
          '<div>📋 Output</div>',
        '</div>',
      '</div>'
    ].join('');
  }

  // ─── Attach handlers ─────────────────────────────────────────────────────

  function attachLiveTranslationHandlers() {
    var startBtn = document.getElementById('lt-start');
    var stopBtn  = document.getElementById('lt-stop');
    if (!startBtn) return;

    // Reset state from any previous session
    _isRunning = false; _segIdx = 0;

    startBtn.addEventListener('click', startLive);
    stopBtn.addEventListener('click',  stopLive);

    document.getElementById('lt-pair').addEventListener('change', updateColLabels);
    updateColLabels();
  }

  function updateColLabels() {
    var pair = (document.getElementById('lt-pair') || {}).value || 'hi:en';
    var parts = pair.split(':');
    var src = parts[0], tgt = parts[1];
    var NAMES = { en: 'English', hi: 'Hindi', fr: 'French', es: 'Spanish', auto: 'Auto-detect', mul: 'Multi' };
    var srcEl = document.getElementById('lt-src-label');
    var tgtEl = document.getElementById('lt-tgt-label');
    if (srcEl) srcEl.textContent = (NAMES[src] || src) + ' (Recognised)';
    if (tgtEl) tgtEl.textContent = (NAMES[tgt] || tgt) + ' (Translated)';
  }

  // ─── Start / Stop ─────────────────────────────────────────────────────────

  function startLive() {
    var startBtn    = document.getElementById('lt-start');
    var stopBtn     = document.getElementById('lt-stop');
    var micStatus   = document.getElementById('lt-mic-status');
    var modelKey    = (document.getElementById('lt-model')        || {}).value || 'tiny';
    var pair        = (document.getElementById('lt-pair')         || {}).value || 'hi:en';
    var doTranslate = (document.getElementById('lt-translate-on') || {}).checked !== false;
    var parts       = pair.split(':');
    var srcLang     = parts[0], tgtLang = parts[1];

    startBtn.disabled = true;

    // Show notice if WASM unavailable
    var rtEl = document.getElementById('lt-runtime-status');
    if (!WhisperEngine.isWhisperWasmAvailable() && rtEl) {
      rtEl.innerHTML = '⚠ Using Web Speech API fallback — may use browser cloud engine';
      rtEl.style.color = 'var(--a)';
    }

    setLoading(true, 'Loading Whisper model…', 0);

    WhisperEngine.loadWhisper(modelKey, function (info) {
      setLoading(true, info.msg || '', info.pct || 0);
    }).then(function () {
      if (doTranslate) {
        setLoading(true, 'Loading translation model…', 80);
        return TranslationEngine.loadTransformers(function (msg) { setLoading(true, msg, 90); });
      }
    }).catch(function (err) {
      appendError('Model load failed: ' + err.message);
      startBtn.disabled = false;
      setLoading(false);
      return null; // continue without translation
    }).then(function () {
      setLoading(true, 'Requesting microphone…', 95);
      return navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true }
      });
    }).then(function (stream) {
      setLoading(false);
      _micStream = stream;
      _isRunning = true;
      _segIdx    = 0;

      if (micStatus) micStatus.innerHTML = '<span class="b g" style="font-size:11px">● Microphone active</span>';
      if (stopBtn)   stopBtn.disabled  = false;

      updateBadge();

      _chunks   = [];
      _recorder = new MediaRecorder(stream);
      _recorder.ondataavailable = function (e) { if (e.data.size > 0) _chunks.push(e.data); };
      _recorder.onstop = function () {
        if (!_isRunning) return;
        var blob = new Blob(_chunks, { type: _recorder.mimeType });
        _chunks  = [];
        processChunk(blob, srcLang, tgtLang, doTranslate);
      };

      startWindow();
    }).catch(function (err) {
      setLoading(false);
      appendError('Could not start: ' + err.message);
      startBtn.disabled = false;
    });
  }

  function startWindow() {
    if (!_isRunning) return;
    _chunks = [];
    _recorder.start();
    _winTimer = setTimeout(function () {
      if (_recorder && _recorder.state === 'recording') _recorder.stop();
      if (_isRunning) startWindow();
    }, CHUNK_SEC * 1000);
  }

  function stopLive() {
    _isRunning = false;
    clearTimeout(_winTimer);
    try { if (_recorder && _recorder.state === 'recording') _recorder.stop(); } catch (_) {}
    if (_micStream) { _micStream.getTracks().forEach(function (t) { t.stop(); }); _micStream = null; }
    _recorder = null;

    var startBtn  = document.getElementById('lt-start');
    var stopBtn   = document.getElementById('lt-stop');
    var micStatus = document.getElementById('lt-mic-status');
    if (startBtn)  startBtn.disabled  = false;
    if (stopBtn)   stopBtn.disabled   = true;
    if (micStatus) micStatus.innerHTML = '<span class="muted">Microphone stopped</span>';
  }

  // ─── Chunk processing ─────────────────────────────────────────────────────

  function processChunk(blob, srcLang, tgtLang, doTranslate) {
    blob.arrayBuffer().then(function (ab) {
      var AudioCtx = window.AudioContext || window.webkitAudioContext;
      var ctx      = new AudioCtx({ sampleRate: SAMPLE_RATE });
      return ctx.decodeAudioData(ab).then(function (decoded) {
        ctx.close();
        var samples = decoded.getChannelData(0);
        // Skip silence
        var sum = 0;
        for (var i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
        var rms = Math.sqrt(sum / samples.length);
        if (rms < 0.005) return;

        return WhisperEngine.transcribeChunk(decoded, srcLang === 'auto' ? 'auto' : srcLang, function () {})
          .then(function (results) {
            var srcText = results.map(function (r) { return r.text; }).join(' ').trim();
            if (!srcText) return;

            if (doTranslate && srcLang !== tgtLang) {
              return TranslationEngine.translateText(srcText, srcLang, tgtLang, function () {})
                .then(function (tgtText) { appendSegment(srcText, tgtText); })
                .catch(function (err)    { appendSegment(srcText, '[Translation error: ' + err.message + ']'); });
            } else {
              appendSegment(srcText, '');
            }
          });
      }).catch(function (err) {
        ctx.close();
        appendError('Chunk decode error: ' + err.message);
      });
    }).catch(function (err) {
      appendError('Chunk error: ' + err.message);
    });
  }

  // ─── UI helpers ──────────────────────────────────────────────────────────

  function appendSegment(src, tgt) {
    var srcTx = document.getElementById('lt-src-tx');
    var tgtTx = document.getElementById('lt-tgt-tx');
    if (!srcTx || !tgtTx) return;
    var now = new Date().toLocaleTimeString();
    srcTx.insertAdjacentHTML('beforeend',
      '<p style="margin:0 0 8px;border-bottom:1px solid var(--bd);padding-bottom:6px">' +
      '<span class="muted" style="font-size:11px">' + now + '</span><br>' + escHtml(src) + '</p>');
    tgtTx.insertAdjacentHTML('beforeend',
      '<p style="margin:0 0 8px;border-bottom:1px solid var(--bd);padding-bottom:6px">' +
      '<span class="muted" style="font-size:11px">' + now + '</span><br>' + (tgt ? escHtml(tgt) : '<span class="muted">—</span>') + '</p>');
    srcTx.scrollTop = srcTx.scrollHeight;
    tgtTx.scrollTop = tgtTx.scrollHeight;
  }

  function appendError(msg) {
    var el = document.getElementById('lt-src-tx');
    if (el) el.insertAdjacentHTML('beforeend',
      '<p style="color:var(--r);font-size:12px">⚠ ' + escHtml(msg) + '</p>');
  }

  function setLoading(show, msg, pct) {
    var div    = document.getElementById('lt-loading');
    var msgEl  = document.getElementById('lt-loading-msg');
    var barEl  = document.getElementById('lt-loading-bar');
    if (!div) return;
    div.style.display = show ? 'block' : 'none';
    if (msgEl) msgEl.textContent = msg || '';
    if (barEl) barEl.style.width = (pct || 0) + '%';
  }

  function updateBadge() {
    var badge = document.getElementById('lt-whisper-badge');
    if (!badge) return;
    if (WhisperEngine.isWhisperWasmAvailable()) {
      badge.textContent = '✓ Whisper WASM'; badge.className = 'b g';
    } else {
      badge.textContent = '⚠ Web Speech fallback'; badge.className = 'b a';
    }
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  return {
    renderLiveTranslationPage:    renderLiveTranslationPage,
    attachLiveTranslationHandlers: attachLiveTranslationHandlers
  };
})();
