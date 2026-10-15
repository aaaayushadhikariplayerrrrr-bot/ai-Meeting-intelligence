/**
 * appModules.js — Wires the new local-processing pages into MEMORA's app.js.
 *
 * All dependencies (AudioStore, FFmpegPipeline, VADProcessor, WhisperEngine,
 * TranslationEngine, MeetingUpload, LiveTranslation) are already loaded as
 * classic scripts before this file runs — no import statements needed.
 *
 * WHAT THIS DOES:
 *   1. Patches window.P with the real page renderers (replacing stubs)
 *   2. If the user is already on #upload or #livetranslate, re-renders immediately
 *   3. On every hashchange, re-renders + attaches handlers for module pages
 *   4. Injects the "Upload Meeting" button into the Meetings page
 *   5. Exposes window._memoraToast and window._clearAllTempAudio
 */
(function () {
  var MODULE_PAGES = {
    upload:        { render: function () { return MeetingUpload.renderUploadPage();            }, attach: function () { MeetingUpload.attachUploadHandlers();            } },
    livetranslate: { render: function () { return LiveTranslation.renderLiveTranslationPage(); }, attach: function () { LiveTranslation.attachLiveTranslationHandlers(); } }
  };

  // ── 1. Patch window.P immediately ────────────────────────────────────────
  // app.js already ran and set window.P before this script was parsed.
  Object.keys(MODULE_PAGES).forEach(function (key) {
    if (window.P) window.P[key] = MODULE_PAGES[key].render;
  });

  // ── 2. Core: render + attach if we're on a module page ───────────────────
  function activateModulePage() {
    var hash = location.hash.slice(1) || 'overview';
    var page = MODULE_PAGES[hash];
    if (!page) return;

    var view = document.getElementById('view');
    if (!view) return;

    // Replace whatever is in #view (stubs or previous content) with real page
    view.innerHTML = page.render();

    // Update breadcrumb + title
    var titles = { upload: 'Upload Meeting', livetranslate: 'Live Translation' };
    var crumb  = document.getElementById('crumb');
    var ptitle = document.getElementById('ptitle');
    if (crumb)  crumb.textContent  = 'MEMORA / ' + (titles[hash] || hash);
    if (ptitle) ptitle.textContent = titles[hash] || hash;

    // Nav highlight
    document.querySelectorAll('#nav a').forEach(function (a) {
      a.classList.toggle('on', a.dataset.k === hash);
    });

    // Attach interactive handlers
    page.attach();

    // MEMORA enter animation
    view.classList.remove('enter');
    void view.offsetWidth;
    view.classList.add('enter');
  }

  // Run immediately (classic scripts run synchronously; app.js already ran)
  activateModulePage();

  // ── 3. Re-activate on every hash change ──────────────────────────────────
  // app.js's go() runs on hashchange and sets innerHTML to the real renderer
  // (now patched). We still need to attach() afterwards.
  window.addEventListener('hashchange', function () {
    // Use setTimeout(0) so app.js's go() finishes its innerHTML write first
    setTimeout(activateModulePage, 0);
  });

  // ── 4. Inject Upload Meeting button into Meetings page ───────────────────
  var _observer = new MutationObserver(function () {
    if (location.hash !== '#meetings') return;
    var view = document.getElementById('view');
    if (!view) return;
    var row = view.querySelector('.row');
    if (!row || row.querySelector('[data-upload-btn]')) return;
    var btn = document.createElement('button');
    btn.className = 'pri';
    btn.setAttribute('data-upload-btn', '1');
    btn.textContent = '⬆ Upload Meeting';
    btn.onclick = function () { location.hash = 'upload'; };
    var newMeetBtn = row.querySelector('[data-modal="newm"]');
    if (newMeetBtn) newMeetBtn.insertAdjacentElement('afterend', btn);
  });
  var viewEl = document.getElementById('view') || document.body;
  _observer.observe(viewEl, { childList: true });

  // ── 5. Global utilities ───────────────────────────────────────────────────
  window._memoraToast = function (msg) {
    var t = document.getElementById('toast');
    if (t) { t.textContent = msg; t.className = 'on'; setTimeout(function () { t.className = ''; }, 1800); }
  };

  window._clearAllTempAudio = function () {
    AudioStore.clearAllAudio().then(function () {
      window._memoraToast('All temporary audio cleared from IndexedDB');
    });
  };

  // Wire "Delete raw data now" in Retention page to also clear IndexedDB
  document.addEventListener('click', function (e) {
    if (e.target.closest && e.target.closest('[data-modal="delraw"]')) {
      document.addEventListener('click', function handler(e2) {
        if (e2.target.closest && e2.target.closest('.modal [data-close]') && e2.target.dataset.toast) {
          AudioStore.clearAllAudio().catch(function () {});
          document.removeEventListener('click', handler);
        }
      });
    }
  });
})();
