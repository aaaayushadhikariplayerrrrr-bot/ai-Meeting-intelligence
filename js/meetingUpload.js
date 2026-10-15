/**
 * meetingUpload.js — Upload Meeting workflow.
 *
 * Depends on globals: AudioStore, FFmpegPipeline, VADProcessor, WhisperEngine
 * (all loaded as classic scripts before this file).
 *
 * Pipeline (100% local — no audio leaves the device):
 *   File API → ffmpeg.wasm → VAD → Whisper WASM → Transcript → IDB cleanup
 */
var MeetingUpload = (function () {

  // ─── Render ──────────────────────────────────────────────────────────────

  function renderUploadPage() {
    var modelOpts = Object.keys(WhisperEngine.MODEL_OPTIONS).map(function (k) {
      var m = WhisperEngine.MODEL_OPTIONS[k];
      return '<option value="' + k + '">' + m.label + '</option>';
    }).join('');

    return [
      /* Privacy architecture banner */
      '<div class="card" style="border-color:var(--g);margin-bottom:14px">',
        '<div class="row sp" style="margin:0">',
          '<b>⛨ Local-Only Processing Architecture</b>',
          '<span class="b g">DEVICE ONLY — No audio leaves your device</span>',
        '</div>',
        '<div class="flow" style="margin-top:10px;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));font-size:12px">',
          '<div>📁 File API</div><span style="color:var(--c)">→</span>',
          '<div>🔧 FFmpeg<br><small>extract / convert</small></div><span style="color:var(--c)">→</span>',
          '<div>👂 VAD<br><small>speech detect</small></div><span style="color:var(--c)">→</span>',
          '<div>🧠 Whisper<br><small>local ASR</small></div><span style="color:var(--c)">→</span>',
          '<div>📝 Transcript</div><span style="color:var(--c)">→</span>',
          '<div style="border-color:var(--r);color:var(--r)">🗑 Audio<br>Deleted</div>',
        '</div>',
        '<p class="muted" style="margin:8px 0 0;font-size:12px">',
          'Preprocessing converts format &amp; sample-rate only — it is <b>not</b> anonymisation. ',
          'Privacy comes from keeping all audio local.',
        '</p>',
      '</div>',

      /* Drop zone */
      '<div class="card" id="um-dropzone" style="border:2px dashed var(--bd);text-align:center;padding:32px;cursor:pointer">',
        '<div style="font-size:32px;margin-bottom:10px">📂</div>',
        '<b>Drop your meeting file here, or click to browse</b>',
        '<p class="muted">Audio: WAV, MP3, M4A, FLAC, OGG &nbsp;·&nbsp; Video: MP4, WebM, MOV, MKV</p>',
        '<input type="file" id="um-fileinput" accept="audio/*,video/*,.wav,.mp3,.m4a,.flac,.ogg,.mp4,.webm,.mov,.mkv" style="display:none">',
        '<button class="pri" id="um-browse" style="margin-top:10px">Choose File</button>',
      '</div>',

      /* Model + language selection */
      '<div class="card" style="margin-top:14px">',
        '<div class="row sp" style="margin:0 0 10px">',
          '<div>',
            '<b>Whisper Model</b>',
            '<p class="muted" style="margin:4px 0 0;font-size:12px">Downloaded once, cached in your browser. Larger = more accurate but slower.</p>',
          '</div>',
          '<div class="row" style="margin:0;gap:8px">',
            '<select id="um-model">', modelOpts, '</select>',
            '<span id="um-cache-status" class="muted" style="font-size:12px"></span>',
          '</div>',
        '</div>',
        '<div class="row" style="margin:0">',
          '<select id="um-lang">',
            '<option value="auto">Auto-detect language</option>',
            '<option value="en">English</option>',
            '<option value="hi">Hindi</option>',
          '</select>',
          '<span class="muted" style="font-size:12px">Whisper is multilingual — English / Hindi / Hinglish all supported</span>',
        '</div>',
      '</div>',

      /* File info (shown after selection) */
      '<div id="um-fileinfo" style="display:none;margin-top:14px"></div>',

      /* Pipeline progress */
      '<div id="um-progress" style="display:none;margin-top:14px"></div>',

      /* Transcript output */
      '<div id="um-transcript" style="display:none;margin-top:14px"></div>',

      /* Privacy status */
      '<div id="um-privacy-status" style="display:none;margin-top:14px"></div>'
    ].join('');
  }

  // ─── Attach handlers ─────────────────────────────────────────────────────

  function attachUploadHandlers() {
    var dropzone  = document.getElementById('um-dropzone');
    var fileinput = document.getElementById('um-fileinput');
    var browse    = document.getElementById('um-browse');
    var modelSel  = document.getElementById('um-model');
    if (!dropzone) return;

    updateCacheStatus(modelSel.value);
    modelSel.addEventListener('change', function () { updateCacheStatus(modelSel.value); });

    browse.addEventListener('click', function (e) { e.stopPropagation(); fileinput.click(); });
    dropzone.addEventListener('click', function () { fileinput.click(); });
    fileinput.addEventListener('change', function (e) {
      if (e.target.files[0]) handleFileSelected(e.target.files[0]);
    });

    dropzone.addEventListener('dragover', function (e) {
      e.preventDefault(); dropzone.style.borderColor = 'var(--c)';
    });
    dropzone.addEventListener('dragleave', function () {
      dropzone.style.borderColor = 'var(--bd)';
    });
    dropzone.addEventListener('drop', function (e) {
      e.preventDefault(); dropzone.style.borderColor = 'var(--bd)';
      var file = e.dataTransfer.files[0];
      if (file) handleFileSelected(file);
    });
  }

  // ─── File selected ────────────────────────────────────────────────────────

  function handleFileSelected(file) {
    document.getElementById('um-progress').style.display     = 'none';
    document.getElementById('um-transcript').style.display   = 'none';
    document.getElementById('um-privacy-status').style.display = 'none';

    var sizeMB  = (file.size / 1048576).toFixed(1);
    var isVideo = /\.(mp4|mov|mkv|webm)$/i.test(file.name) || file.type.startsWith('video/');

    var infoDiv = document.getElementById('um-fileinfo');
    infoDiv.style.display = 'block';
    infoDiv.innerHTML = [
      '<div class="card">',
        '<div class="row sp" style="margin:0">',
          '<div>',
            '<b>', escHtml(file.name), '</b>',
            '<p class="muted" style="margin:4px 0 0">', sizeMB, ' MB',
            isVideo ? ' · Video (audio will be extracted)' : '',
            '</p>',
          '</div>',
          '<button class="pri" id="um-start">▶ Start Processing</button>',
        '</div>',
      '</div>'
    ].join('');

    document.getElementById('um-start').addEventListener('click', function () {
      startPipeline(file);
    });
  }

  // ─── Pipeline ─────────────────────────────────────────────────────────────

  var STAGES = [
    'Reading file locally',
    'Extracting & converting audio',
    'Detecting speech regions',
    'Loading Whisper model',
    'Transcribing',
    'Cleaning up'
  ];

  function startPipeline(file) {
    var startBtn    = document.getElementById('um-start');
    var progressDiv = document.getElementById('um-progress');
    var modelKey    = (document.getElementById('um-model') || {}).value || 'tiny';
    var lang        = (document.getElementById('um-lang') || {}).value || 'auto';

    if (startBtn) startBtn.disabled = true;
    progressDiv.style.display = 'block';

    var currentStage = 0;
    var sessionId    = 'meet_' + Date.now();
    var tempIds      = [];

    function setStage(idx, detail) {
      currentStage = idx;
      var pct = Math.round(idx / STAGES.length * 100);
      progressDiv.innerHTML = [
        '<div class="card">',
          '<div class="row sp" style="margin:0 0 10px">',
            '<b>Processing Pipeline</b>',
            '<span class="b c">' + pct + '%</span>',
          '</div>',
          STAGES.map(function (s, i) {
            var icon  = i < currentStage ? '✓' : (i === currentStage ? '●' : '○');
            var color = i < currentStage ? 'var(--g)' : (i === currentStage ? 'var(--t)' : 'var(--m)');
            var anim  = i === currentStage ? 'animation:pulse 1.2s infinite;' : '';
            return '<div style="display:flex;gap:8px;align-items:center;padding:3px 0;font-size:13px">' +
              '<span style="color:' + color + ';width:16px;' + anim + '">' + icon + '</span>' +
              '<span style="color:' + color + '">' + s + '</span>' +
              '</div>';
          }).join(''),
          '<div class="bar" style="margin-top:10px"><i style="width:' + pct + '%;display:block;height:100%;background:var(--c);transition:width .4s"></i></div>',
          detail ? '<p class="muted" style="margin:8px 0 0;font-size:12px">' + escHtml(String(detail)) + '</p>' : '',
        '</div>'
      ].join('');
    }

    setStage(0, 'Reading file via File API (no upload)…');

    var wavBlob, audioBuffer, speechSegments;

    // STEP 1: preprocess with FFmpeg.wasm
    Promise.resolve().then(function () {
      setStage(1, 'Starting FFmpeg.wasm…');
      return FFmpegPipeline.preprocessAudio(file, file.name, function (msg) {
        setStage(1, msg);
      });
    }).catch(function (ffErr) {
      // FFmpeg unavailable — try Web Audio API decode directly
      console.warn('[Upload] FFmpeg failed, using Web Audio fallback:', ffErr.message);
      setStage(1, 'FFmpeg unavailable — using Web Audio API…');
      return decodeToWav(file);
    }).then(function (wav) {
      wavBlob = wav;
      return AudioStore.storeAudio(sessionId + '_wav', wavBlob);
    }).then(function (id) {
      tempIds.push(id);
      setStage(2, 'Running voice activity detection…');
      return VADProcessor.detectSpeech(wavBlob, function (msg) { setStage(2, msg); });
    }).then(function (segs) {
      speechSegments = segs;
      if (!speechSegments.length) throw new Error('No speech detected in audio. Please check the file.');
      setStage(3, 'Loading Whisper model (' + modelKey + ')…');
      return WhisperEngine.loadWhisper(modelKey, function (info) {
        setStage(3, info.msg || '');
      });
    }).then(function () {
      setStage(4, 'Decoding audio for chunking…');
      return VADProcessor.decodeWavBlob(wavBlob);
    }).then(function (ab) {
      audioBuffer = ab;
      // Transcribe each speech chunk
      var allSegments = [];
      var i = 0;

      function nextChunk() {
        if (i >= speechSegments.length) return Promise.resolve(allSegments);
        var seg   = speechSegments[i];
        var label = 'Segment ' + (i+1) + '/' + speechSegments.length +
                    ' (' + seg.startSec.toFixed(1) + 's – ' + seg.endSec.toFixed(1) + 's)';
        setStage(4, label);

        var chunkBlob = VADProcessor.sliceAudioBuffer(audioBuffer, seg.startSec, seg.endSec);
        var chunkId   = sessionId + '_chunk' + i;
        tempIds.push(chunkId);

        return AudioStore.storeAudio(chunkId, chunkBlob)
          .then(function () { return VADProcessor.decodeWavBlob(chunkBlob); })
          .then(function (chunkAB) {
            return WhisperEngine.transcribeChunk(chunkAB, lang, function (info) {
              setStage(4, label + ' — ' + (info.msg || ''));
            });
          }).then(function (results) {
            results.forEach(function (r) {
              allSegments.push({
                startSec: seg.startSec + (r.startSec || 0),
                endSec:   seg.startSec + (r.endSec   || 0),
                text:     r.text
              });
            });
            i++;
            return nextChunk();
          });
      }

      return nextChunk();
    }).then(function (allSegments) {
      setStage(5, 'Deleting temporary audio…');
      return AudioStore.deleteAudio.apply(AudioStore, tempIds).then(function () {
        return allSegments;
      });
    }).then(function (allSegments) {
      renderTranscript(allSegments, file.name, lang);
      showPrivacyStatus(tempIds.length);
    }).catch(function (err) {
      // Always clean up audio on error
      AudioStore.deleteAudio.apply(AudioStore, tempIds).catch(function () {});
      document.getElementById('um-progress').innerHTML = [
        '<div class="card" style="border-color:var(--r)">',
          '<b style="color:var(--r)">⚠ Processing failed</b>',
          '<p>', escHtml(err.message), '</p>',
          '<p class="muted" style="font-size:12px">',
            'Temporary audio has been deleted. No data was sent to any server.',
          '</p>',
          '<button onclick="location.reload()">Try again</button>',
        '</div>'
      ].join('');
      if (startBtn) startBtn.disabled = false;
      console.error('[Pipeline]', err);
    });
  }

  // ─── Transcript renderer ──────────────────────────────────────────────────

  function renderTranscript(segments, filename, lang) {
    var div = document.getElementById('um-transcript');
    if (!div) return;

    var segHtml = segments.length
      ? segments.map(function (s) {
          return '<div style="margin-bottom:12px">' +
            '<span class="muted" style="font-size:12px;font-family:monospace">[' +
            fmtTime(s.startSec) + ' – ' + fmtTime(s.endSec) + ']</span>' +
            '<p style="margin:4px 0 0;font-family:Consolas,monospace;font-size:13px">' +
            escHtml(s.text) + '</p>' +
            '</div>';
        }).join('<hr style="border:none;border-top:1px solid var(--bd)">')
      : '<p class="muted">No speech segments produced — audio may be silent.</p>';

    div.style.display = 'block';
    div.innerHTML = [
      '<div class="card">',
        '<div class="row sp" style="margin:0 0 12px">',
          '<div>',
            '<h2 style="margin:0">Transcript</h2>',
            '<p class="muted" style="margin:4px 0 0;font-size:12px">',
              escHtml(filename), ' · ', segments.length, ' segment(s) · Language: ', lang,
            '</p>',
          '</div>',
          '<div class="row" style="margin:0">',
            '<button id="um-copy-tx">Copy</button>',
            '<button id="um-export-tx">Export .txt</button>',
          '</div>',
        '</div>',
        '<div class="tx" id="um-tx-body" style="height:auto;max-height:460px">', segHtml, '</div>',
      '</div>'
    ].join('');

    var txText = segments.map(function (s) {
      return '[' + fmtTime(s.startSec) + ' – ' + fmtTime(s.endSec) + ']\n' + s.text;
    }).join('\n\n');

    document.getElementById('um-copy-tx').addEventListener('click', function () {
      navigator.clipboard.writeText(txText).then(function () { memoraToast('Transcript copied'); });
    });
    document.getElementById('um-export-tx').addEventListener('click', function () {
      var blob = new Blob([txText], { type: 'text/plain' });
      var a    = document.createElement('a');
      a.href   = URL.createObjectURL(blob);
      a.download = filename.replace(/\.[^.]+$/, '') + '_transcript.txt';
      a.click(); URL.revokeObjectURL(a.href);
    });
  }

  function showPrivacyStatus(numDeleted) {
    var div = document.getElementById('um-privacy-status');
    if (!div) return;
    div.style.display = 'block';
    div.innerHTML = [
      '<div class="card" style="border-color:var(--g)">',
        '<div class="row" style="margin:0 0 8px;flex-wrap:wrap;gap:6px">',
          '<span class="b g">⛨ PROCESSED LOCALLY</span>',
          '<span class="b g">✓ TEMPORARY AUDIO DELETED</span>',
          '<span class="b" style="color:var(--r);border-color:var(--r)">✗ NO CLOUD CALLS</span>',
        '</div>',
        '<p class="muted" style="margin:0;font-size:12px">',
          numDeleted, ' temporary audio entry(ies) deleted from IndexedDB. ',
          'Transcript is retained. No audio was sent to any backend or third-party service.',
        '</p>',
      '</div>'
    ].join('');
  }

  // ─── Utilities ────────────────────────────────────────────────────────────

  function updateCacheStatus(modelKey) {
    var el = document.getElementById('um-cache-status');
    if (!el) return;
    WhisperEngine.isModelCached(modelKey).then(function (cached) {
      el.textContent = cached ? '✓ Cached locally' : '⬇ Will download on first use';
      el.style.color = cached ? 'var(--g)' : 'var(--a)';
    }).catch(function () {});
  }

  /** Decode any audio/video file to 16 kHz mono WAV Blob using Web Audio API. */
  function decodeToWav(file) {
    return file.arrayBuffer().then(function (ab) {
      var AudioCtx = window.AudioContext || window.webkitAudioContext;
      var ctx      = new AudioCtx({ sampleRate: 16000 });
      return ctx.decodeAudioData(ab).then(function (decoded) {
        ctx.close();
        // Mix down to mono
        var ch0 = decoded.getChannelData(0);
        var out  = new Float32Array(ch0.length);
        for (var i = 0; i < ch0.length; i++) {
          var v = ch0[i];
          for (var c = 1; c < decoded.numberOfChannels; c++) {
            v += decoded.getChannelData(c)[i];
          }
          out[i] = v / decoded.numberOfChannels;
        }
        return VADProcessor.pcmToWavBlob(out, decoded.sampleRate);
      }).catch(function (e) {
        ctx.close(); throw e;
      });
    });
  }

  function fmtTime(sec) {
    if (sec == null || isNaN(sec)) return '00:00.0';
    var m = Math.floor(sec / 60);
    var s = sec % 60;
    return ('0' + m).slice(-2) + ':' + ('0' + s.toFixed(1)).slice(-4);
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function memoraToast(msg) {
    if (window._memoraToast) { window._memoraToast(msg); return; }
    var t = document.getElementById('toast');
    if (t) { t.textContent = msg; t.className = 'on'; setTimeout(function () { t.className = ''; }, 1800); }
  }

  return { renderUploadPage: renderUploadPage, attachUploadHandlers: attachUploadHandlers };
})();
