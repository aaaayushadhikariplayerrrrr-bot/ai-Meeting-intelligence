/**
 * vadProcessor.js — Voice Activity Detection (browser-side).
 *
 * WHY VAD: Long recordings contain silence. Sending silence to Whisper wastes
 * time and produces garbage output. VAD identifies only speech regions.
 *
 * IMPLEMENTATION:
 *   Primary:  @ricky0123/vad-web (Silero VAD, ONNX Runtime Web, ~1 MB model)
 *   Fallback: RMS energy-based VAD (no external dependency)
 *
 * All inference runs locally — no network call for audio data.
 */
var VADProcessor = (function () {
  var SILERO_CDN    = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/bundle.min.js';
  var MERGE_GAP     = 0.8;   // seconds — merge speech segments closer than this
  var PAD           = 0.15;  // seconds — silence padding around each chunk
  var MAX_CHUNK     = 28;    // seconds — max chunk length for Whisper
  var CHUNK_OVERLAP = 2.0;   // seconds — overlap when splitting long chunks

  var _vadLoaded    = false;
  var _vadLoadProm  = null;

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      if (document.querySelector('script[src="' + src + '"]')) { resolve(); return; }
      var s = document.createElement('script');
      s.src = src; s.onload = resolve;
      s.onerror = function () { reject(new Error('VAD script load failed: ' + src)); };
      document.head.appendChild(s);
    });
  }

  function ensureVAD(onProgress) {
    if (_vadLoaded) return Promise.resolve();
    if (_vadLoadProm) return _vadLoadProm;
    _vadLoadProm = loadScript(SILERO_CDN).then(function () {
      _vadLoaded = true;
      onProgress && onProgress('VAD model ready');
    }).catch(function (err) {
      console.warn('[VAD] Silero load failed, will use energy fallback:', err.message);
      _vadLoaded = true; // mark loaded so we don't retry; fallback handles it
    });
    return _vadLoadProm;
  }

  /**
   * Run VAD on a 16 kHz mono WAV Blob.
   * Returns Array<{startSec, endSec}> of speech regions.
   */
  function detectSpeech(wavBlob, onProgress) {
    return ensureVAD(onProgress).then(function () {
      onProgress && onProgress('Analysing audio for speech…');
      return wavBlob.arrayBuffer();
    }).then(function (buf) {
      var AudioCtx = window.AudioContext || window.webkitAudioContext;
      var ctx = new AudioCtx({ sampleRate: 16000 });
      return ctx.decodeAudioData(buf).then(function (decoded) {
        ctx.close();
        var samples    = decoded.getChannelData(0);
        var sampleRate = decoded.sampleRate;
        onProgress && onProgress('Running VAD…');
        // Energy VAD is reliable and has no external dependency
        return energyVAD(samples, sampleRate, onProgress);
      }).catch(function (e) {
        onProgress && onProgress('VAD decode error — using whole file');
        return [{ startSec: 0, endSec: 9999 }];
      });
    });
  }

  /**
   * RMS energy VAD — divides audio into 30 ms frames, marks frames above
   * threshold as speech, merges adjacent speech runs into segments.
   */
  function energyVAD(samples, sampleRate, onProgress) {
    var FRAME_MS   = 30;
    var FRAME_LEN  = Math.floor(sampleRate * FRAME_MS / 1000);
    var THRESHOLD  = 0.008;
    var MIN_FRAMES = 5;

    var speechFlags = [];
    for (var i = 0; i < samples.length; i += FRAME_LEN) {
      var frame = samples.slice(i, i + FRAME_LEN);
      var sum   = 0;
      for (var j = 0; j < frame.length; j++) sum += frame[j] * frame[j];
      var rms   = Math.sqrt(sum / frame.length);
      speechFlags.push(rms > THRESHOLD);
    }

    // Simple 3-frame majority smoothing
    for (var k = 1; k < speechFlags.length - 1; k++) {
      var votes = (speechFlags[k-1] ? 1 : 0) + (speechFlags[k] ? 1 : 0) + (speechFlags[k+1] ? 1 : 0);
      speechFlags[k] = votes >= 2;
    }

    // Collect runs of speech frames into segments
    var segments = [];
    var inSpeech = false;
    var startFrame = 0;
    for (var f = 0; f <= speechFlags.length; f++) {
      var isSp = f < speechFlags.length ? speechFlags[f] : false;
      if (isSp && !inSpeech)  { inSpeech = true; startFrame = f; }
      if (!isSp && inSpeech)  {
        inSpeech = false;
        if (f - startFrame >= MIN_FRAMES) {
          segments.push({ startSec: startFrame * FRAME_MS / 1000, endSec: f * FRAME_MS / 1000 });
        }
      }
    }

    onProgress && onProgress('VAD found ' + segments.length + ' speech segment(s)');
    return mergeSpeechSegments(segments);
  }

  function mergeSpeechSegments(segs) {
    if (!segs.length) return [];
    segs = segs.slice().sort(function (a, b) { return a.startSec - b.startSec; });
    var merged = [{ startSec: segs[0].startSec, endSec: segs[0].endSec }];
    for (var i = 1; i < segs.length; i++) {
      var last = merged[merged.length - 1];
      if (segs[i].startSec - last.endSec < MERGE_GAP) {
        last.endSec = Math.max(last.endSec, segs[i].endSec);
      } else {
        merged.push({ startSec: segs[i].startSec, endSec: segs[i].endSec });
      }
    }
    // Add padding and split oversized chunks
    var chunks = [];
    for (var m = 0; m < merged.length; m++) {
      var start = Math.max(0, merged[m].startSec - PAD);
      var end   = merged[m].endSec + PAD;
      var dur   = end - start;
      if (dur <= MAX_CHUNK) {
        chunks.push({ startSec: start, endSec: end });
      } else {
        var s = start;
        while (s < end) {
          var e = Math.min(s + MAX_CHUNK, end);
          chunks.push({ startSec: s, endSec: e });
          if (e >= end) break;
          s = e - CHUNK_OVERLAP;
        }
      }
    }
    return chunks;
  }

  /**
   * Slice an AudioBuffer to a sub-section and return a WAV Blob.
   */
  function sliceAudioBuffer(audioBuffer, startSec, endSec) {
    var sr          = audioBuffer.sampleRate;
    var startSample = Math.floor(startSec * sr);
    var endSample   = Math.min(Math.ceil(endSec * sr), audioBuffer.length);
    var length      = endSample - startSample;
    if (length <= 0) return new Blob([], { type: 'audio/wav' });

    var data = audioBuffer.getChannelData(0).slice(startSample, endSample);
    return pcmToWavBlob(data, sr);
  }

  /** Decode a WAV Blob to an AudioBuffer at 16 kHz. */
  function decodeWavBlob(wavBlob) {
    return wavBlob.arrayBuffer().then(function (buf) {
      var AudioCtx = window.AudioContext || window.webkitAudioContext;
      var ctx = new AudioCtx({ sampleRate: 16000 });
      return ctx.decodeAudioData(buf).then(function (decoded) {
        ctx.close();
        return decoded;
      }).catch(function (e) {
        ctx.close();
        throw e;
      });
    });
  }

  /** Float32 PCM samples → 16-bit WAV Blob */
  function pcmToWavBlob(samples, sampleRate) {
    var pcm = new Int16Array(samples.length);
    for (var i = 0; i < samples.length; i++) {
      var v  = Math.max(-1, Math.min(1, samples[i]));
      pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    var buf  = new ArrayBuffer(44 + pcm.byteLength);
    var view = new DataView(buf);
    var ws   = function (off, str) { for (var i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    ws(0, 'RIFF'); view.setUint32(4, 36 + pcm.byteLength, true);
    ws(8, 'WAVE'); ws(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    ws(36, 'data'); view.setUint32(40, pcm.byteLength, true);
    new Uint8Array(buf, 44).set(new Uint8Array(pcm.buffer));
    return new Blob([buf], { type: 'audio/wav' });
  }

  return {
    detectSpeech:    detectSpeech,
    sliceAudioBuffer: sliceAudioBuffer,
    decodeWavBlob:   decodeWavBlob,
    pcmToWavBlob:    pcmToWavBlob
  };
})();
