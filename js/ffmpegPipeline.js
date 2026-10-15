/**
 * ffmpegPipeline.js — Local audio preprocessing using ffmpeg.wasm.
 *
 * WHY ffmpeg.wasm:
 *   The user's file may be MP3, M4A, MP4, WebM, etc. Local Whisper needs
 *   16 kHz mono PCM WAV. ffmpeg.wasm runs full FFmpeg inside WebAssembly —
 *   no audio leaves the browser.
 *
 * NOTE: Preprocessing converts format/sample-rate only. It is NOT
 * anonymisation. Privacy comes from keeping all audio local.
 */
var FFmpegPipeline = (function () {
  var FFMPEG_CDN   = 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/umd/ffmpeg.js';
  var FFMPEG_CORE  = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.js';
  var FFMPEG_WASM  = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.wasm';

  var _ffmpeg      = null;
  var _loadPromise = null;

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      if (document.querySelector('script[src="' + src + '"]')) { resolve(); return; }
      var s    = document.createElement('script');
      s.src    = src;
      s.onload = resolve;
      s.onerror = function () { reject(new Error('Failed to load script: ' + src)); };
      document.head.appendChild(s);
    });
  }

  /** Load ffmpeg.wasm once. */
  function loadFFmpeg(onProgress) {
    if (_ffmpeg) return Promise.resolve(_ffmpeg);
    if (_loadPromise) return _loadPromise;

    _loadPromise = loadScript(FFMPEG_CDN).then(function () {
      onProgress && onProgress('Loading FFmpeg core…');
      var FFmpeg = (window.FFmpegWASM || window.FFmpeg || {}).FFmpeg;
      if (!FFmpeg) throw new Error('FFmpeg.wasm library not found after script load.');
      _ffmpeg = new FFmpeg();
      return _ffmpeg.load({ coreURL: FFMPEG_CORE, wasmURL: FFMPEG_WASM });
    }).then(function () {
      onProgress && onProgress('FFmpeg ready');
      return _ffmpeg;
    });

    return _loadPromise;
  }

  /**
   * Full preprocessing pipeline:
   *   1. Write raw bytes into ffmpeg virtual FS
   *   2. Extract audio (handles video containers)
   *   3. Resample → 16 000 Hz
   *   4. Convert → mono
   *   5. Loudness normalise (EBU R128)
   *   6. Output 16-bit PCM WAV
   */
  function preprocessAudio(blob, inputName, onProgress) {
    return loadFFmpeg(onProgress).then(function (ffmpeg) {
      var ext     = (inputName.match(/\.[^.]+$/) || ['.bin'])[0].toLowerCase();
      var inFile  = 'input' + ext;
      var outFile = 'output.wav';

      onProgress && onProgress('Extracting audio…');
      return blob.arrayBuffer().then(function (buf) {
        return ffmpeg.writeFile(inFile, new Uint8Array(buf));
      }).then(function () {
        onProgress && onProgress('Converting & normalising audio…');
        /* FFmpeg flags:
           -vn          : drop video
           -ac 1        : mono
           -ar 16000    : 16 kHz (Whisper requirement)
           -af loudnorm : EBU R128 loudness normalisation
           -acodec pcm_s16le : 16-bit signed PCM
           -f wav       : WAV container                       */
        return ffmpeg.exec(['-i', inFile, '-vn', '-ac', '1', '-ar', '16000',
                            '-af', 'loudnorm', '-acodec', 'pcm_s16le', '-f', 'wav', outFile]);
      }).then(function () {
        onProgress && onProgress('Reading processed audio…');
        return ffmpeg.readFile(outFile);
      }).then(function (out) {
        var wavBlob = new Blob([out.buffer], { type: 'audio/wav' });
        // Clean up virtual FS
        try { ffmpeg.deleteFile(inFile);  } catch (_) {}
        try { ffmpeg.deleteFile(outFile); } catch (_) {}
        onProgress && onProgress('Audio preprocessing complete');
        return wavBlob;
      });
    });
  }

  /** Get audio duration in seconds via Web Audio API. */
  function getAudioDuration(blob) {
    return blob.arrayBuffer().then(function (buf) {
      var ctx = new (window.AudioContext || window.webkitAudioContext)();
      return ctx.decodeAudioData(buf).then(function (decoded) {
        ctx.close();
        return decoded.duration;
      }).catch(function () { ctx.close(); return null; });
    });
  }

  return { loadFFmpeg: loadFFmpeg, preprocessAudio: preprocessAudio, getAudioDuration: getAudioDuration };
})();
