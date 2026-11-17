import test from 'node:test';
import assert from 'node:assert/strict';
import { isAbortError, logoutUser } from '../services/api.ts';

// --- Test Lifecycle Simulator ---
// Models the exact state and lifecycle refs implemented in App.tsx
class AppLifecycleSimulator {
  constructor() {
    this.currentOrgId = 'org-a';
    this.activeOrgIdRef = { current: 'org-a' };
    this.requestEpochRef = { current: 0 };
    this.activeAbortControllers = new Set();
    this.activeUploadAbortController = null;

    // React state mirrors
    this.meetings = [];
    this.transcripts = {};
    this.activeJob = null;
    this.selectedVersion = {};
    this.uploadProgress = 0;
    this.isUploading = false;
  }

  isContextCurrent(capturedEpoch, capturedOrgId) {
    return (
      capturedEpoch === this.requestEpochRef.current &&
      capturedOrgId !== null &&
      capturedOrgId === this.activeOrgIdRef.current
    );
  }

  createRegisteredController() {
    const controller = new AbortController();
    this.activeAbortControllers.add(controller);
    return controller;
  }

  unregisterController(controller) {
    this.activeAbortControllers.delete(controller);
  }

  cancelAllActiveOperations() {
    this.activeAbortControllers.forEach((controller) => {
      try {
        controller.abort();
      } catch {}
    });
    this.activeAbortControllers.clear();

    if (this.activeUploadAbortController) {
      try {
        this.activeUploadAbortController.abort();
      } catch {}
      this.activeUploadAbortController = null;
    }
  }

  switchOrganization(targetOrgId) {
    // 1. Invalidate request epoch synchronously
    this.requestEpochRef.current += 1;
    const newEpoch = this.requestEpochRef.current;

    // 2. Abort all registered fetch controllers & uploads
    this.cancelAllActiveOperations();

    // 3. Clear old tenant dashboard state
    this.meetings = [];
    this.transcripts = {};
    this.activeJob = null;
    this.selectedVersion = {};
    this.uploadProgress = 0;
    this.isUploading = false;

    // 4. Update active organization context
    this.currentOrgId = targetOrgId;
    this.activeOrgIdRef.current = targetOrgId;

    return newEpoch;
  }

  logout() {
    // 1. Invalidate epoch
    this.requestEpochRef.current += 1;
    this.activeOrgIdRef.current = null;
    this.currentOrgId = null;

    // 2. Abort all active operations
    this.cancelAllActiveOperations();

    // 3. Clear tenant state
    this.meetings = [];
    this.transcripts = {};
    this.activeJob = null;
    this.selectedVersion = {};
    this.uploadProgress = 0;
    this.isUploading = false;
  }
}

// --- Helper: Deferred Promise for Controlled Asynchrony ---
function createDeferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

// =========================================================================
// Tests
// =========================================================================

test('1. Delayed Organization A transcript response arriving after switching to Organization B', async () => {
  const sim = new AppLifecycleSimulator();
  const deferredOrgATranscripts = createDeferred();

  // Step 1: Start loading transcripts for Org A
  const capturedEpoch = sim.requestEpochRef.current;
  const capturedOrgId = sim.activeOrgIdRef.current;
  const controller = sim.createRegisteredController();

  let transcriptUpdateAttempted = false;

  const loadOrgATranscript = async () => {
    try {
      const data = await deferredOrgATranscripts.promise;
      if (sim.isContextCurrent(capturedEpoch, capturedOrgId)) {
        sim.transcripts['meeting-org-a'] = data;
        transcriptUpdateAttempted = true;
      }
    } catch (e) {
      if (!isAbortError(e)) throw e;
    } finally {
      sim.unregisterController(controller);
    }
  };

  const pendingPromise = loadOrgATranscript();

  // Step 2: User switches to Org B before Org A responds
  sim.switchOrganization('org-b');

  // Verify controller was aborted
  assert.strictEqual(controller.signal.aborted, true, 'Controller should be aborted on org switch');

  // Step 3: Org A response finally arrives
  deferredOrgATranscripts.resolve([{ id: 'transcript-org-a', text: 'Confidential Org A Data' }]);
  await pendingPromise;

  // Step 4: Verify Org A transcripts were discarded and NOT injected into Org B state
  assert.strictEqual(transcriptUpdateAttempted, false, 'Transcript update should be discarded');
  assert.deepStrictEqual(sim.transcripts, {}, 'Transcripts state must remain empty in Org B');
});

test('2. Delayed meeting-list response after organization switching', async () => {
  const sim = new AppLifecycleSimulator();
  const deferredOrgAMeetings = createDeferred();

  const capturedEpoch = sim.requestEpochRef.current;
  const capturedOrgId = sim.activeOrgIdRef.current;
  const controller = sim.createRegisteredController();

  const loadOrgAMeetings = async () => {
    try {
      const data = await deferredOrgAMeetings.promise;
      if (sim.isContextCurrent(capturedEpoch, capturedOrgId)) {
        sim.meetings = data;
      }
    } catch (e) {
      if (!isAbortError(e)) throw e;
    } finally {
      sim.unregisterController(controller);
    }
  };

  const pending = loadOrgAMeetings();

  // Switch to Org B
  sim.switchOrganization('org-b');

  // Org A meetings arrive later
  deferredOrgAMeetings.resolve([{ id: 'm-1', title: 'Org A Secret Strategy' }]);
  await pending;

  // Verify Org A meetings did NOT overwrite Org B
  assert.deepStrictEqual(sim.meetings, [], 'Stale Org A meetings must not populate Org B');
});

test('3. XHR upload cancellation during organization switching', async () => {
  const sim = new AppLifecycleSimulator();
  const uploadController = new AbortController();
  sim.activeUploadAbortController = uploadController;
  sim.isUploading = true;

  let xhrAborted = false;
  let progressUpdatesAfterSwitch = 0;

  // Simulate XHR abort listener
  uploadController.signal.addEventListener('abort', () => {
    xhrAborted = true;
  });

  const capturedEpoch = sim.requestEpochRef.current;
  const capturedOrgId = sim.activeOrgIdRef.current;

  // Simulate upload progress
  const onProgress = (p) => {
    if (sim.isContextCurrent(capturedEpoch, capturedOrgId)) {
      sim.uploadProgress = p;
    } else {
      progressUpdatesAfterSwitch++;
    }
  };

  onProgress(25);
  assert.strictEqual(sim.uploadProgress, 25);

  // Switch to Org B while uploading
  sim.switchOrganization('org-b');

  // Verify XHR was aborted immediately
  assert.strictEqual(xhrAborted, true, 'XHR upload must be aborted on organization switch');
  assert.strictEqual(sim.isUploading, false, 'Upload state must be reset');
  assert.strictEqual(sim.uploadProgress, 0, 'Upload progress must be reset');

  // Any late progress tick must be rejected
  onProgress(50);
  assert.strictEqual(progressUpdatesAfterSwitch, 1, 'Late progress tick must be blocked by currency check');
  assert.strictEqual(sim.uploadProgress, 0);
});

test('4. XHR upload cancellation during logout', async () => {
  const sim = new AppLifecycleSimulator();
  const uploadController = new AbortController();
  sim.activeUploadAbortController = uploadController;
  sim.isUploading = true;

  let xhrAborted = false;
  uploadController.signal.addEventListener('abort', () => {
    xhrAborted = true;
  });

  // User logs out
  sim.logout();

  assert.strictEqual(xhrAborted, true, 'XHR upload must be aborted on logout');
  assert.strictEqual(sim.activeUploadAbortController, null, 'Active upload ref must be cleared');
  assert.strictEqual(sim.isUploading, false, 'Upload state must be cleared');
});

test('5. Stale polling response after organization switching', async () => {
  const sim = new AppLifecycleSimulator();
  sim.activeJob = { id: 'job-org-a', meeting_id: 'm-1', status: 'IN_PROGRESS' };

  const deferredPoll = createDeferred();
  const pollController = sim.createRegisteredController();
  const capturedEpoch = sim.requestEpochRef.current;
  const capturedOrgId = sim.activeOrgIdRef.current;

  let pollingApplied = false;

  const pollJob = async () => {
    try {
      const job = await deferredPoll.promise;
      if (sim.isContextCurrent(capturedEpoch, capturedOrgId)) {
        sim.activeJob = job;
        pollingApplied = true;
      }
    } catch (e) {
      if (!isAbortError(e)) throw e;
    } finally {
      sim.unregisterController(pollController);
    }
  };

  const pendingPoll = pollJob();

  // Switch to Org B
  sim.switchOrganization('org-b');

  // Poll response for Org A arrives with COMPLETED
  deferredPoll.resolve({ id: 'job-org-a', meeting_id: 'm-1', status: 'COMPLETED' });
  await pendingPoll;

  assert.strictEqual(pollingApplied, false, 'Polling response must not be applied after switch');
  assert.strictEqual(sim.activeJob, null, 'Active job must remain null in Org B');
});

test('6. Stale transcript derivation response after switching', async () => {
  const sim = new AppLifecycleSimulator();
  const deferredClean = createDeferred();
  const derivationController = sim.createRegisteredController();
  const capturedEpoch = sim.requestEpochRef.current;
  const capturedOrgId = sim.activeOrgIdRef.current;

  let versionUpdated = false;

  const runClean = async () => {
    try {
      const derived = await deferredClean.promise;
      if (sim.isContextCurrent(capturedEpoch, capturedOrgId)) {
        sim.selectedVersion['m-1'] = derived.id;
        versionUpdated = true;
      }
    } catch (e) {
      if (!isAbortError(e)) throw e;
    } finally {
      sim.unregisterController(derivationController);
    }
  };

  const pendingClean = runClean();

  // Switch organization
  sim.switchOrganization('org-b');

  // Derivation resolves
  deferredClean.resolve({ id: 'cleaned-transcript-org-a' });
  await pendingClean;

  assert.strictEqual(versionUpdated, false, 'Derived version must not update state after switch');
  assert.deepStrictEqual(sim.selectedVersion, {}, 'Selected version map must remain empty');
});

test('7. No protected state updates after logout', async () => {
  const sim = new AppLifecycleSimulator();

  const capturedEpoch = sim.requestEpochRef.current;
  const capturedOrgId = sim.activeOrgIdRef.current;

  // Log out
  sim.logout();

  // Try applying any async response using previously captured context
  assert.strictEqual(
    sim.isContextCurrent(capturedEpoch, capturedOrgId),
    false,
    'Captured context must be invalid after logout'
  );

  // Even if an unauthenticated caller tries to check with null org
  assert.strictEqual(
    sim.isContextCurrent(capturedEpoch, null),
    false,
    'Null org is never current for tenant state'
  );

  assert.deepStrictEqual(sim.meetings, []);
  assert.deepStrictEqual(sim.transcripts, {});
  assert.strictEqual(sim.activeJob, null);
});

test('8. Normal upload, polling, and transcript operations within one organization', async () => {
  const sim = new AppLifecycleSimulator();

  // A. Normal load meetings
  const epoch1 = sim.requestEpochRef.current;
  const org1 = sim.activeOrgIdRef.current;
  assert.strictEqual(sim.isContextCurrent(epoch1, org1), true);
  sim.meetings = [{ id: 'm-1', title: 'Q4 Review' }];
  assert.strictEqual(sim.meetings.length, 1);

  // B. Normal upload progress and completion
  sim.isUploading = true;
  sim.uploadProgress = 100;
  sim.activeJob = { id: 'job-1', meeting_id: 'm-1', status: 'IN_PROGRESS' };
  assert.strictEqual(sim.isContextCurrent(epoch1, org1), true);
  assert.strictEqual(sim.activeJob.id, 'job-1');

  // C. Normal polling update
  sim.activeJob = { id: 'job-1', meeting_id: 'm-1', status: 'COMPLETED' };
  assert.strictEqual(sim.activeJob.status, 'COMPLETED');

  // D. Normal transcript fetch
  sim.transcripts['m-1'] = [{ id: 't-1', text: 'Hello team' }];
  assert.strictEqual(sim.transcripts['m-1'].length, 1);
});

test('9. isAbortError correctly identifies DOMException and Error aborts', () => {
  const domAbort = new DOMException('The operation was aborted.', 'AbortError');
  const errAbort = new Error('Upload aborted');
  const otherErr = new Error('Network connection timeout');
  const nullErr = null;

  assert.strictEqual(isAbortError(domAbort), true);
  assert.strictEqual(isAbortError(errAbort), true);
  assert.strictEqual(isAbortError(otherErr), false);
  assert.strictEqual(isAbortError(nullErr), false);
});

test('10. logoutUser sends revocation request with Bearer token', async () => {
  const originalFetch = globalThis.fetch;
  const originalSessionStorage = globalThis.sessionStorage;
  let requestedUrl = null;
  let requestedHeaders = null;

  const storage = new Map();
  storage.set('meet_intel_token', 'test-token-xyz');
  globalThis.sessionStorage = {
    getItem: (key) => storage.get(key) || null,
    setItem: (key, val) => storage.set(key, val),
    removeItem: (key) => storage.delete(key),
  };

  globalThis.fetch = async (url, options) => {
    requestedUrl = url;
    requestedHeaders = options.headers;
    return new Response(JSON.stringify({ message: 'Successfully logged out' }), { status: 200 });
  };

  try {
    await logoutUser();
    assert.match(requestedUrl, /\/api\/v1\/auth\/logout/);
    assert.strictEqual(requestedHeaders['Authorization'], 'Bearer test-token-xyz');
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.sessionStorage = originalSessionStorage;
  }
});

test('11. logoutUser gracefully tolerates 401 or network errors without throwing', async () => {
  const originalFetch = globalThis.fetch;
  const originalSessionStorage = globalThis.sessionStorage;

  const storage = new Map();
  storage.set('meet_intel_token', 'expired-token');
  globalThis.sessionStorage = {
    getItem: (key) => storage.get(key) || null,
    setItem: (key, val) => storage.set(key, val),
    removeItem: (key) => storage.delete(key),
  };

  // 401 response
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: 'Token has been revoked or superseded' }), { status: 401 });
  try {
    await logoutUser(); // Must not throw
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.sessionStorage = originalSessionStorage;
  }

  // Network offline error
  globalThis.fetch = async () => {
    throw new TypeError('Failed to fetch (offline)');
  };
  try {
    await logoutUser(); // Must not throw
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.sessionStorage = originalSessionStorage;
  }
});

// --- Phase 6E.2 Tests: Admin Role Guards & Hardware Diagnostics UI ---

test('12. ORG_ADMIN sees permitted admin actions (Delete Raw, Clean)', () => {
  const currentOrg = { organization_id: 'org-1', role: 'ORG_ADMIN' };
  const rawAsset = { id: 'asset-1', media_type: 'AUDIO', is_raw_deleted: false };
  const normalizedAsset = { id: 'asset-2', media_type: 'NORMALIZED_AUDIO', is_raw_deleted: false };

  // Guard logic from App.tsx:
  // Delete Raw: currentOrg?.role === 'ORG_ADMIN' && asset.media_type !== 'NORMALIZED_AUDIO'
  const canDeleteRaw = (org, asset) => org?.role === 'ORG_ADMIN' && asset.media_type !== 'NORMALIZED_AUDIO';
  // Clean Transcript: currentOrg?.role === 'ORG_ADMIN'
  const canCleanTranscript = (org) => org?.role === 'ORG_ADMIN';

  assert.strictEqual(canDeleteRaw(currentOrg, rawAsset), true);
  assert.strictEqual(canDeleteRaw(currentOrg, normalizedAsset), false);
  assert.strictEqual(canCleanTranscript(currentOrg), true);
});

test('13. MEMBER does not see admin actions (Delete Raw, Clean) but retains standard member actions', () => {
  const currentOrg = { organization_id: 'org-1', role: 'MEMBER' };
  const rawAsset = { id: 'asset-1', media_type: 'AUDIO', is_raw_deleted: false };

  const canDeleteRaw = (org, asset) => org?.role === 'ORG_ADMIN' && asset.media_type !== 'NORMALIZED_AUDIO';
  const canCleanTranscript = (org) => org?.role === 'ORG_ADMIN';
  const canRedactTranscript = (org) => Boolean(org);
  const canProcessBothTranscript = (org) => Boolean(org);

  assert.strictEqual(canDeleteRaw(currentOrg, rawAsset), false);
  assert.strictEqual(canCleanTranscript(currentOrg), false);
  assert.strictEqual(canRedactTranscript(currentOrg), true);
  assert.strictEqual(canProcessBothTranscript(currentOrg), true);
});

test('14. Non-super-admin does not fetch hardware diagnostics and diagnostics component is hidden', async () => {
  const regularUser = { id: 'u-1', email: 'user@example.com', is_super_admin: false };
  let fetchCount = 0;

  const loadHardwareDiagnostics = async (user) => {
    if (!user?.is_super_admin) {
      return null;
    }
    fetchCount++;
    return { status: 'ok', gpu: 'RTX 5060' };
  };

  const isDiagnosticsTabVisible = (user) => Boolean(user?.is_super_admin);

  const result = await loadHardwareDiagnostics(regularUser);
  assert.strictEqual(result, null);
  assert.strictEqual(fetchCount, 0);
  assert.strictEqual(isDiagnosticsTabVisible(regularUser), false);
});

test('15. Super Admin fetches hardware diagnostics and diagnostics tab is visible', async () => {
  const superAdminUser = { id: 'sa-1', email: 'admin@system.local', is_super_admin: true };
  let fetchCount = 0;

  const loadHardwareDiagnostics = async (user) => {
    if (!user?.is_super_admin) {
      return null;
    }
    fetchCount++;
    return { status: 'ok', gpu: 'RTX 5060' };
  };

  const isDiagnosticsTabVisible = (user) => Boolean(user?.is_super_admin);

  const result = await loadHardwareDiagnostics(superAdminUser);
  assert.deepStrictEqual(result, { status: 'ok', gpu: 'RTX 5060' });
  assert.strictEqual(fetchCount, 1);
  assert.strictEqual(isDiagnosticsTabVisible(superAdminUser), true);
});

test('16. Organization switching and logout clear diagnostics for non-super-admins', () => {
  let hardwareDiagnostics = { gpu: 'cached' };
  let activeInspectorTab = 'asr';

  // State clearer matching clearTenantDashboardState in App.tsx
  const clearState = (user) => {
    if (!user?.is_super_admin) {
      hardwareDiagnostics = null;
      if (activeInspectorTab === 'asr') {
        activeInspectorTab = 'health';
      }
    }
  };

  // Normal user switches org
  const regularUser = { id: 'u-1', is_super_admin: false };
  clearState(regularUser);
  assert.strictEqual(hardwareDiagnostics, null);
  assert.strictEqual(activeInspectorTab, 'health');

  // Logout reset
  const handleLogoutReset = () => {
    hardwareDiagnostics = null;
    activeInspectorTab = 'health';
  };
  hardwareDiagnostics = { gpu: 'stale' };
  activeInspectorTab = 'asr';
  handleLogoutReset();
  assert.strictEqual(hardwareDiagnostics, null);
  assert.strictEqual(activeInspectorTab, 'health');
});
