/**
 * translationEngine.js — Local neural machine translation.
 *
 * WHY LOCAL:
 *   No text/audio may be sent to cloud translation APIs (Google Translate,
 *   DeepL, OpenAI). This module runs Transformers.js + OPUS-MT ONNX models
 *   entirely in the browser. Models are cached after first download.
 *
 * RUNTIME: Xenova/transformers (Transformers.js) + Helsinki-NLP OPUS-MT
 */
var TranslationEngine = (function () {
  var TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1/dist/transformers.min.js';

  var MT_MODELS = {
    'hi-en':  'Xenova/opus-mt-hi-en',
    'en-hi':  'Xenova/opus-mt-en-hi',
    'en-fr':  'Xenova/opus-mt-en-fr',
    'fr-en':  'Xenova/opus-mt-fr-en',
    'en-es':  'Xenova/opus-mt-en-es',
    'es-en':  'Xenova/opus-mt-es-en',
    'mul-en': 'Xenova/opus-mt-mul-en'
  };

  var _loaded       = false;
  var _loadProm     = null;
  var _pipelines    = {};

  var SUPPORTED_PAIRS = [
    { src: 'hi', tgt: 'en', label: 'Hindi → English' },
    { src: 'en', tgt: 'hi', label: 'English → Hindi' },
    { src: 'en', tgt: 'fr', label: 'English → French' },
    { src: 'fr', tgt: 'en', label: 'French → English' },
    { src: 'en', tgt: 'es', label: 'English → Spanish' },
    { src: 'es', tgt: 'en', label: 'Spanish → English' },
    { src: 'auto', tgt: 'en', label: 'Auto-detect → English' }
  ];

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      if (document.querySelector('script[data-transformers]')) { resolve(); return; }
      var s = document.createElement('script');
      s.src = src; s.dataset.transformers = '1';
      s.onload  = resolve;
      s.onerror = function () { reject(new Error('Transformers.js failed to load. Check internet connection.')); };
      document.head.appendChild(s);
    });
  }

  function loadTransformers(onProgress) {
    if (_loaded) return Promise.resolve();
    if (_loadProm) return _loadProm;
    onProgress && onProgress('Loading Transformers.js…');
    _loadProm = loadScript(TRANSFORMERS_CDN).then(function () {
      _loaded = true;
      var tf  = window.transformers || window.Transformers;
      if (tf && tf.env) {
        tf.env.allowLocalModels = false;
        tf.env.useBrowserCache  = true;
      }
      onProgress && onProgress('Transformers.js ready');
    });
    return _loadProm;
  }

  /**
   * Translate text. All inference runs locally via ONNX Runtime Web.
   */
  function translateText(text, srcLang, tgtLang, onProgress) {
    if (!text || !text.trim()) return Promise.resolve('');
    if (srcLang === tgtLang) return Promise.resolve(text);

    var pairKey = srcLang === 'auto' ? 'mul-en' : (srcLang + '-' + tgtLang);
    var modelId  = MT_MODELS[pairKey];

    if (!modelId) {
      return Promise.reject(new Error('No local translation model for ' + srcLang + '→' + tgtLang));
    }

    return loadTransformers(onProgress).then(function () {
      var tf = window.transformers || window.Transformers;
      if (!tf) throw new Error('Transformers.js not available');

      if (_pipelines[pairKey]) return _pipelines[pairKey];

      onProgress && onProgress('Loading translation model (' + pairKey + ')…');
      return tf.pipeline('translation', modelId, {
        progress_callback: function (info) {
          if (info.status === 'downloading') {
            var pct = info.loaded && info.total ? Math.round(info.loaded / info.total * 100) : 0;
            onProgress && onProgress('Downloading translation model: ' + pct + '%');
          }
        }
      }).then(function (pipe) {
        _pipelines[pairKey] = pipe;
        onProgress && onProgress('Translation model ready');
        return pipe;
      });
    }).then(function (pipe) {
      return pipe(text, { max_new_tokens: 512 });
    }).then(function (result) {
      return (result && result[0]) ? (result[0].translation_text || result[0].generated_text || '') : '';
    });
  }

  return {
    loadTransformers:  loadTransformers,
    translateText:     translateText,
    SUPPORTED_PAIRS:   SUPPORTED_PAIRS
  };
})();
