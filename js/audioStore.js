/**
 * audioStore.js — IndexedDB wrapper for temporary audio storage.
 *
 * WHY IndexedDB: Large audio files can't go in localStorage (5 MB limit,
 * synchronous). IndexedDB handles binary blobs locally. All entries are
 * explicitly deleted after successful transcription.
 *
 * PRIVACY: Audio never leaves the browser. IndexedDB is origin-local.
 */
var AudioStore = (function () {
  var DB_NAME = 'memoraAudio';
  var STORE_NAME = 'tempAudio';
  var DB_VERSION = 1;
  var _db = null;

  function openDB() {
    if (_db) return Promise.resolve(_db);
    return new Promise(function (resolve, reject) {
      var req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = function (e) {
        var db = e.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        }
      };
      req.onsuccess = function (e) { _db = e.target.result; resolve(_db); };
      req.onerror   = function (e) { reject(e.target.error); };
    });
  }

  /** Store a Blob under id. Returns id. */
  function storeAudio(id, blob) {
    return openDB().then(function (db) {
      return new Promise(function (resolve, reject) {
        var tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).put({ id: id, blob: blob });
        tx.oncomplete = function () { resolve(id); };
        tx.onerror    = function (e) { reject(e.target.error); };
      });
    });
  }

  /** Load blob by id. Resolves null if missing. */
  function loadAudio(id) {
    return openDB().then(function (db) {
      return new Promise(function (resolve, reject) {
        var tx  = db.transaction(STORE_NAME, 'readonly');
        var req = tx.objectStore(STORE_NAME).get(id);
        req.onsuccess = function (e) { resolve(e.target.result ? e.target.result.blob : null); };
        req.onerror   = function (e) { reject(e.target.error); };
      });
    });
  }

  /** Delete one or more audio entries. Call after successful transcription. */
  function deleteAudio() {
    var ids = Array.prototype.slice.call(arguments);
    return openDB().then(function (db) {
      return new Promise(function (resolve, reject) {
        var tx    = db.transaction(STORE_NAME, 'readwrite');
        var store = tx.objectStore(STORE_NAME);
        ids.forEach(function (id) { store.delete(id); });
        tx.oncomplete = function () { resolve(); };
        tx.onerror    = function (e) { reject(e.target.error); };
      });
    });
  }

  /** Delete ALL temporary audio entries. */
  function clearAllAudio() {
    return openDB().then(function (db) {
      return new Promise(function (resolve, reject) {
        var tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).clear();
        tx.oncomplete = function () { resolve(); };
        tx.onerror    = function (e) { reject(e.target.error); };
      });
    });
  }

  return { storeAudio: storeAudio, loadAudio: loadAudio, deleteAudio: deleteAudio, clearAllAudio: clearAllAudio };
})();
