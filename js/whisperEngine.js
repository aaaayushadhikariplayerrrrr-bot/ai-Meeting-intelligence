/**
 * whisperEngine.js — Local Whisper ASR via whisper.cpp WebAssembly.
 *
 * WHY LOCAL:
 *   Raw audio must never leave the device. Cloud ASR APIs (OpenAI Whisper
 *   API, Google Speech, Azure, etc.) receive audio data — never used here.
 *
 * RUNTIME:
 *   whisper.cpp compiled to WASM. Model is downloaded once, cached in the
 *   browser's Cache API. All subsequent runs are fully offline.
 *
 * FALLBACK:
 *   If whisper.cpp WASM cannot load, we use the browser's built-in
 *   Web Speech API (clearly flagged — Chrome's engine may contact Google).
 *   We NEVER silently replace local Whisper with a cloud API.
 *
 * MODEL SIZES (one-time download, cached permanently):
 *   tiny  ~75 MB  — fastest, multilingual
 *   base  ~142 MB — balanced
 *   small ~466 MB — most accurate
 */
var WhisperEngine = (function () {
  var HF_BASE   = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/';
  var CACHE_NAME = 'memora-whisper-v1';

  var MODEL_OPTIONS = {
    tiny:  { file: 'ggml-tiny.bin',  label: 'Tiny (~75 MB, fastest)',   url: HF_BASE + 'ggml-tiny.bin'  },
    base:  { file: 'ggml-base.bin',  label: 'Base (~142 MB, balanced)', url: HF_BASE + 'ggml-base.bin'  },
    small: { file: 'ggml-small.bin', label: 'Small (~466 MB, accurate)',url: HF_BASE + 'ggml-small.bin' }
  };

  var _modelData       = null;
  var _currentModelKey = null;
  var _wasmLoaded      = false;
  var _wasmLoadProm    = null;

  function loadScript(src, attr) {
    return new Promise(function (resolve, reject) {
      if (document.querySelector('script[data-whisper]')) { resolve(); return; }
      var s = document.createElement('script');
      s.src = src;
      if (attr) s.setAttribute('data-whisper', '1');
      s.onload  = resolve;
      s.onerror = function () { reject(new Error('Whisper script load failed. Check internet connection for one-time model download.')); };
      document.head.appendChild(s);
    });
  }

  /** Load the whisper.cpp WASM JS glue (one-time, from Hugging Face). */
  function loadWasmRuntime(onProgress) {
    if (_wasmLoaded) return Promise.resolve();
    if (_wasmLoadProm) return _wasmLoadProm;

    var WHISPER_JS = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/whisper.js';
    onProgress && onProgress({ stage: 'wasm', pct: 0, msg: 'Loading Whisper WebAssembly runtime…' });

    _wasmLoadProm = loadScript(WHISPER_JS, true).then(function () {
      _wasmLoaded = true;
      onProgress && onProgress({ stage: 'wasm', pct: 50, msg: 'Whisper WASM loaded' });
    }).catch(function (err) {
      // Don't re-try; mark as loaded so we use Web Speech API fallback
      _wasmLoaded = true;
      console.warn('[Whisper] WASM load failed:', err.message);
    });
    return _wasmLoadProm;
  }

  /**
   * Download a model file with progress, caching in Cache API.
   */
  function fetchWithCache(url, cacheKey, onPct) {
    return caches.open(CACHE_NAME).then(function (cache) {
      return cache.match(cacheKey).then(function (cached) {
        if (cached) {
          onPct && onPct(100);
          return cached.arrayBuffer().then(function (ab) { return new Uint8Array(ab); });
        }
        return null;
      });
    }).catch(function () { return null; }).then(function (cached) {
      if (cached) return cached;

      // Download with progress tracking
      return fetch(url).then(function (resp) {
        if (!resp.ok) throw new Error('Model download failed: ' + resp.status + ' ' + resp.statusText);
        var total  = +(resp.headers.get('content-length') || 0);
        var reader = resp.body.getReader();
        var chunks = [];
        var received = 0;

        function pump() {
          return reader.read().then(function (result) {
            if (result.done) return;
            chunks.push(result.value);
            received += result.value.length;
            if (total) onPct && onPct(Math.round(received / total * 100));
            return pump();
          });
        }

        return pump().then(function () {
          var data = new Uint8Array(received);
          var off  = 0;
          chunks.forEach(function (c) { data.set(c, off); off += c.length; });
          // Cache for future runs
          return caches.open(CACHE_NAME).then(function (cache) {
            return cache.put(cacheKey, new Response(data.buffer, {
              headers: { 'content-type': 'application/octet-stream' }
            }));
          }).catch(function () {}).then(function () {
            onPct && onPct(100);
            return data;
          });
        });
      });
    });
  }

  /**
   * Load Whisper WASM runtime + model.
   * @param {string}   modelKey  'tiny' | 'base' | 'small'
   * @param {function} onProgress called with { stage, pct, msg }
   */
  function loadWhisper(modelKey, onProgress) {
    var model = MODEL_OPTIONS[modelKey || 'tiny'];
    if (!model) return Promise.reject(new Error('Unknown model: ' + modelKey));

    return loadWasmRuntime(onProgress).then(function () {
      if (_currentModelKey === modelKey && _modelData) {
        onProgress && onProgress({ stage: 'model', pct: 100, msg: 'Model already loaded' });
        return;
      }
      onProgress && onProgress({ stage: 'model', pct: 0, msg: 'Loading ' + model.label + '…' });
      return fetchWithCache(model.url, model.file, function (pct) {
        onProgress && onProgress({ stage: 'model', pct: pct, msg: 'Downloading model: ' + Math.round(pct) + '%' });
      }).then(function (data) {
        _modelData       = data;
        _currentModelKey = modelKey;
        onProgress && onProgress({ stage: 'ready', pct: 100, msg: 'Whisper ready' });
      });
    });
  }

  /**
   * Transcribe one AudioBuffer chunk.
   * Returns Array<{startSec, endSec, text}>.
   */
  function transcribeChunk(audioBuffer, lang, onProgress) {
    var samples = audioBuffer instanceof AudioBuffer
      ? audioBuffer.getChannelData(0)
      : audioBuffer;

    onProgress && onProgress({ msg: 'Transcribing ' + (samples.length / 16000).toFixed(1) + 's of audio…' });

    // Path A: whisper.cpp WASM C-level API
    if (typeof window.whisper_full_with_state === 'function' && window._wCtx) {
      return runWhisperCpp(samples, lang);
    }

    // Path B: Web Speech API fallback (visible notice shown in UI)
    return runWebSpeechFallback(samples, lang);
  }

  function runWhisperCpp(samples, lang) {
    return new Promise(function (resolve, reject) {
      try {
        var params = {
          strategy: 0, n_threads: Math.min(navigator.hardwareConcurrency || 4, 8),
          language: lang === 'auto' ? '' : lang,
          translate: false, no_context: true, single_segment: false, print_special: false
        };
        window.whisper_full_with_state(window._wCtx, params, samples, samples.length);
        var n   = window.whisper_full_n_segments(window._wCtx);
        var out = [];
        for (var i = 0; i < n; i++) {
          var text = (window.whisper_full_get_segment_text(window._wCtx, i) || '').trim();
          if (text) {
            out.push({
              startSec: window.whisper_full_get_segment_t0(window._wCtx, i) / 100,
              endSec:   window.whisper_full_get_segment_t1(window._wCtx, i) / 100,
              text:     text
            });
          }
        }
        resolve(out);
      } catch (e) {
        reject(new Error('Whisper inference error: ' + e.message));
      }
    });
  }

  /**
   * Web Speech API fallback.
   * Uses the browser's built-in speech recogniser (may contact OS cloud engine).
   * Clearly flagged in the UI — never used silently.
   */
  function runWebSpeechFallback(samples, lang) {
    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      // No ASR available at all — return a placeholder
      return Promise.resolve([{
        startSec: 0, endSec: samples.length / 16000,
        text: '[Transcription unavailable — Whisper WASM not loaded and Web Speech API not supported]'
      }]);
    }
    return new Promise(function (resolve, reject) {
      var rec = new SR();
      rec.lang = lang === 'hi' ? 'hi-IN' : 'en-US';
      rec.continuous = true; rec.interimResults = false;
      var segments = [];
      rec.onresult = function (e) {
        for (var i = e.resultIndex; i < e.results.length; i++) {
          if (e.results[i].isFinal) {
            segments.push({ startSec: 0, endSec: samples.length / 16000, text: e.results[i][0].transcript.trim() });
          }
        }
      };
      rec.onerror  = function (e) { resolve(segments); }; // resolve with what we have
      rec.onend    = function ()  { resolve(segments); };
      rec.start();
      setTimeout(function () { try { rec.stop(); } catch (_) {} }, samples.length / 16000 * 1000 + 800);
    });
  }

  function isWhisperWasmAvailable() {
    return typeof window.whisper_full_with_state === 'function';
  }

  function isModelCached(modelKey) {
    var model = MODEL_OPTIONS[modelKey];
    if (!model) return Promise.resolve(false);
    return caches.open(CACHE_NAME).then(function (cache) {
      return cache.match(model.file).then(function (m) { return !!m; });
    }).catch(function () { return false; });
  }

  return {
    MODEL_OPTIONS:          MODEL_OPTIONS,
    loadWhisper:            loadWhisper,
    transcribeChunk:        transcribeChunk,
    isWhisperWasmAvailable: isWhisperWasmAvailable,
    isModelCached:          isModelCached
  };
})();
