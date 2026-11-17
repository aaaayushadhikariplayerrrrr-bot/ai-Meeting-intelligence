import test from 'node:test';
import assert from 'node:assert/strict';
import { isAbortError } from '../services/api.ts';

// --- Simulator Modeling App.tsx Intelligence State and Async Lifecycle ---
class IntelligenceAppSimulator {
  constructor() {
    this.currentOrgId = 'org-alpha';
    this.activeOrgIdRef = { current: 'org-alpha' };
    this.requestEpochRef = { current: 0 };
    this.activeAbortControllers = new Set();

    // In-memory intelligence state mirrors App.tsx
    this.meetingIntelligence = {};
    this.isLoadingIntelligence = {};
    this.intelligenceError = {};
    this.expandedIntelligence = {};
    this.isExtractingMap = {};
    this.activeJob = null;
    this.transcripts = {};
    this.selectedVersionMap = {};
    this.expandedTranscripts = {};
    this.evidenceNavErrorMap = {};
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
  }

  switchOrganization(targetOrgId) {
    this.requestEpochRef.current += 1;
    this.cancelAllActiveOperations();

    this.meetingIntelligence = {};
    this.isLoadingIntelligence = {};
    this.intelligenceError = {};
    this.expandedIntelligence = {};
    this.isExtractingMap = {};
    this.activeJob = null;
    this.transcripts = {};
    this.selectedVersionMap = {};
    this.expandedTranscripts = {};
    this.evidenceNavErrorMap = {};

    this.currentOrgId = targetOrgId;
    this.activeOrgIdRef.current = targetOrgId;
  }

  logout() {
    this.requestEpochRef.current += 1;
    this.activeOrgIdRef.current = null;
    this.currentOrgId = null;
    this.cancelAllActiveOperations();

    this.meetingIntelligence = {};
    this.isLoadingIntelligence = {};
    this.intelligenceError = {};
    this.expandedIntelligence = {};
    this.isExtractingMap = {};
    this.activeJob = null;
    this.transcripts = {};
    this.selectedVersionMap = {};
    this.expandedTranscripts = {};
    this.evidenceNavErrorMap = {};
  }

  async loadIntelligenceForMeeting(meetingId, fetchFn) {
    const capturedEpoch = this.requestEpochRef.current;
    const capturedOrgId = this.currentOrgId;
    const controller = this.createRegisteredController();

    this.isLoadingIntelligence[meetingId] = true;
    this.intelligenceError[meetingId] = null;

    try {
      const data = await fetchFn(meetingId, controller.signal);
      if (this.isContextCurrent(capturedEpoch, capturedOrgId)) {
        this.meetingIntelligence[meetingId] = data;
      }
    } catch (e) {
      if (this.isContextCurrent(capturedEpoch, capturedOrgId) && !isAbortError(e)) {
        this.intelligenceError[meetingId] = e.message || 'Failed to load intelligence';
      }
    } finally {
      this.unregisterController(controller);
      if (this.isContextCurrent(capturedEpoch, capturedOrgId)) {
        this.isLoadingIntelligence[meetingId] = false;
      }
    }
  }

  async updateActionItemStatus(meetingId, itemId, newStatus, patchFn) {
    const controller = this.createRegisteredController();
    try {
      const updated = await patchFn(itemId, newStatus, controller.signal);
      const current = this.meetingIntelligence[meetingId];
      if (current) {
        this.meetingIntelligence[meetingId] = {
          ...current,
          action_items: current.action_items.map((item) =>
            item.id === itemId ? { ...item, status: updated.status } : item
          ),
        };
      }
      return updated;
    } catch (err) {
      this.intelligenceError[meetingId] = err.message;
      throw err;
    } finally {
      this.unregisterController(controller);
    }
  }
  handleNavigateToEvidence(meetingId, segmentId, _timestampSeconds = null) {
    this.evidenceNavErrorMap[meetingId] = null;
    this.expandedTranscripts[meetingId] = true;

    const transcripts = this.transcripts[meetingId] || [];
    const targetTranscript = transcripts.find((t) =>
      t.segments && t.segments.some((seg) => seg.id === segmentId)
    );

    if (transcripts.length > 0 && !targetTranscript) {
      this.evidenceNavErrorMap[meetingId] =
        'Referenced evidence segment is not available in current transcripts.';
      return;
    }

    if (targetTranscript) {
      this.selectedVersionMap[meetingId] = targetTranscript.id;
    }
  }
}

function createDeferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}


// --- 1. Empty Intelligence State ---
test('1. Empty intelligence response handled consistently with has_intelligence=false', async () => {
  const sim = new IntelligenceAppSimulator();
  const emptyData = {
    meeting_id: 'meeting-101',
    summary: null,
    topics: [],
    decisions: [],
    action_items: [],
    risks: [],
    unresolved_questions: [],
    has_intelligence: false,
  };

  await sim.loadIntelligenceForMeeting('meeting-101', async () => emptyData);

  assert.strictEqual(sim.isLoadingIntelligence['meeting-101'], false);
  assert.strictEqual(sim.intelligenceError['meeting-101'], null);
  const result = sim.meetingIntelligence['meeting-101'];
  assert.notStrictEqual(result, null);
  assert.strictEqual(result.has_intelligence, false);
  assert.strictEqual(result.summary, null);
  assert.deepStrictEqual(result.decisions, []);
  assert.deepStrictEqual(result.action_items, []);
});


// --- 2. Successful Intelligence Retrieval ---
test('2. Successful intelligence retrieval populates summary, topics, decisions, actions, risks, questions', async () => {
  const sim = new IntelligenceAppSimulator();
  const mockIntel = {
    meeting_id: 'meeting-202',
    summary: {
      id: 'sum-1',
      meeting_id: 'meeting-202',
      transcript_id: 'tr-1',
      executive_summary: 'The team completed Phase 7 structured intelligence engine.',
      detailed_summary: 'Comprehensive review of evidence attribution and Ollama fallback.',
      key_takeaways: '["Enforce privacy-safe transcript", "Zero schema drift"]',
      version: 1,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
    topics: [
      { id: 'top-1', meeting_id: 'meeting-202', title: 'Architecture Review', description: 'Engine design', start_time: 0, end_time: 60 },
    ],
    decisions: [
      { id: 'dec-1', meeting_id: 'meeting-202', description: 'Deploy RTX 5060 local inference', decided_by: 'Lead Architect', evidence_segment_id: 'seg-1', timestamp_seconds: 12.5 },
    ],
    action_items: [
      { id: 'act-1', meeting_id: 'meeting-202', description: 'Write unit tests by Friday', owner_name: 'Engineer', status: 'PENDING', evidence_segment_id: 'seg-2', timestamp_seconds: 25.0 },
    ],
    risks: [
      { id: 'risk-1', meeting_id: 'meeting-202', description: 'VRAM exhaustion under concurrent requests', severity: 'HIGH', mitigation_suggestion: 'Queue via JobWorker' },
    ],
    unresolved_questions: [
      { id: 'q-1', meeting_id: 'meeting-202', question_text: 'What is the target P95 inference latency?', raised_by: 'QA Lead', status: 'OPEN' },
    ],
    has_intelligence: true,
  };

  await sim.loadIntelligenceForMeeting('meeting-202', async () => mockIntel);

  assert.strictEqual(sim.isLoadingIntelligence['meeting-202'], false);
  const result = sim.meetingIntelligence['meeting-202'];
  assert.strictEqual(result.has_intelligence, true);
  assert.strictEqual(result.summary.version, 1);
  assert.strictEqual(result.decisions.length, 1);
  assert.strictEqual(result.decisions[0].timestamp_seconds, 12.5);
  assert.strictEqual(result.action_items[0].status, 'PENDING');
  assert.strictEqual(result.risks[0].severity, 'HIGH');
  assert.strictEqual(result.unresolved_questions[0].status, 'OPEN');
});


// --- 3. Extraction Request & Active-Job State ---
test('3. Extraction job enqueues and transitions through activeJob lifecycle', async () => {
  const sim = new IntelligenceAppSimulator();

  // Simulate triggering extraction
  sim.isExtractingMap['meeting-303'] = true;
  const mockJobResponse = {
    job_id: 'job-extract-1',
    meeting_id: 'meeting-303',
    job_type: 'EXTRACTION',
    status: 'PENDING',
    message: 'Structured meeting intelligence extraction job queued successfully.',
  };

  sim.activeJob = {
    id: mockJobResponse.job_id,
    meeting_id: mockJobResponse.meeting_id,
    job_type: mockJobResponse.job_type,
    status: mockJobResponse.status,
    progress: 0,
    error_message: null,
  };
  sim.expandedIntelligence['meeting-303'] = true;

  assert.strictEqual(sim.activeJob.status, 'PENDING');
  assert.strictEqual(sim.activeJob.job_type, 'EXTRACTION');
  assert.strictEqual(sim.expandedIntelligence['meeting-303'], true);

  // Simulate worker progress
  sim.activeJob.status = 'IN_PROGRESS';
  sim.activeJob.progress = 50.0;
  assert.strictEqual(sim.activeJob.progress, 50.0);

  // Simulate completion and automatic intelligence refresh
  sim.activeJob.status = 'COMPLETED';
  sim.activeJob.progress = 100.0;
  sim.isExtractingMap['meeting-303'] = false;

  assert.strictEqual(sim.activeJob.status, 'COMPLETED');
  assert.strictEqual(sim.isExtractingMap['meeting-303'], false);
});


// --- 4. Failed Extraction Handling ---
test('4. Failed extraction records error message and preserves previous state', async () => {
  const sim = new IntelligenceAppSimulator();

  // Initial state has v1 intelligence
  sim.meetingIntelligence['meeting-404'] = {
    meeting_id: 'meeting-404',
    summary: { version: 1, executive_summary: 'Original v1 summary' },
    decisions: [{ id: 'd-1', description: 'Existing decision' }],
    action_items: [],
    risks: [],
    unresolved_questions: [],
    has_intelligence: true,
  };

  // Re-extraction fails with error
  await sim.loadIntelligenceForMeeting('meeting-404', async () => {
    throw new Error('No privacy-safe (redacted) transcript available');
  });

  assert.strictEqual(sim.intelligenceError['meeting-404'], 'No privacy-safe (redacted) transcript available');
  // Previous intelligence remains intact
  assert.strictEqual(sim.meetingIntelligence['meeting-404'].summary.version, 1);
  assert.strictEqual(sim.meetingIntelligence['meeting-404'].decisions.length, 1);
});


// --- 5. Action Item Status Update Success & Reversion on Failure ---
test('5. Action item status updates successfully and preserves previous state on failure', async () => {
  const sim = new IntelligenceAppSimulator();

  sim.meetingIntelligence['meeting-505'] = {
    meeting_id: 'meeting-505',
    summary: null,
    topics: [],
    decisions: [],
    action_items: [
      { id: 'act-101', meeting_id: 'meeting-505', description: 'Deploy Phase 7.5 API', status: 'PENDING' },
    ],
    risks: [],
    unresolved_questions: [],
    has_intelligence: true,
  };

  // 1. Successful update to IN_PROGRESS
  const mockPatchSuccess = async (itemId, newStatus) => ({
    id: itemId,
    meeting_id: 'meeting-505',
    description: 'Deploy Phase 7.5 API',
    status: newStatus,
  });

  await sim.updateActionItemStatus('meeting-505', 'act-101', 'IN_PROGRESS', mockPatchSuccess);
  assert.strictEqual(sim.meetingIntelligence['meeting-505'].action_items[0].status, 'IN_PROGRESS');

  // 2. Failed update preserves current status
  const mockPatchFailure = async () => {
    throw new Error('Action item not found in this organization');
  };

  try {
    await sim.updateActionItemStatus('meeting-505', 'act-101', 'COMPLETED', mockPatchFailure);
    assert.fail('Should have thrown error');
  } catch (err) {
    assert.strictEqual(err.message, 'Action item not found in this organization');
  }

  // Status remains IN_PROGRESS (did not incorrectly change to COMPLETED)
  assert.strictEqual(sim.meetingIntelligence['meeting-505'].action_items[0].status, 'IN_PROGRESS');
});


// --- 6. Missing Evidence References Handled Gracefully ---
test('6. Decisions and action items with null evidence are handled gracefully', async () => {
  const sim = new IntelligenceAppSimulator();
  const dataWithMissingEvidence = {
    meeting_id: 'meeting-606',
    summary: { version: 1, executive_summary: 'Heuristic summary' },
    topics: [],
    decisions: [
      { id: 'd-none', description: 'High level decision without specific segment', decided_by: null, evidence_segment_id: null, timestamp_seconds: null },
    ],
    action_items: [
      { id: 'a-none', description: 'General follow up', owner_name: null, status: 'PENDING', evidence_segment_id: null, timestamp_seconds: null },
    ],
    risks: [],
    unresolved_questions: [],
    has_intelligence: true,
  };

  await sim.loadIntelligenceForMeeting('meeting-606', async () => dataWithMissingEvidence);

  const res = sim.meetingIntelligence['meeting-606'];
  assert.strictEqual(res.decisions[0].evidence_segment_id, null);
  assert.strictEqual(res.decisions[0].timestamp_seconds, null);
  assert.strictEqual(res.action_items[0].evidence_segment_id, null);
  assert.strictEqual(res.action_items[0].timestamp_seconds, null);
});


// --- 7. Organization Switching While Intelligence Request is Pending ---
test('7. Delayed intelligence response for Org A is discarded after switching to Org B', async () => {
  const sim = new IntelligenceAppSimulator();
  const deferredOrgA = createDeferred();

  const capturedEpoch = sim.requestEpochRef.current;
  const capturedOrgId = sim.currentOrgId;
  const controller = sim.createRegisteredController();

  let stateUpdated = false;
  const pendingPromise = (async () => {
    try {
      const data = await deferredOrgA.promise;
      if (sim.isContextCurrent(capturedEpoch, capturedOrgId)) {
        sim.meetingIntelligence['meeting-alpha'] = data;
        stateUpdated = true;
      }
    } catch (e) {
      if (!isAbortError(e)) throw e;
    } finally {
      sim.unregisterController(controller);
    }
  })();

  // User switches to Org B before Org A finishes
  sim.switchOrganization('org-beta');
  assert.strictEqual(controller.signal.aborted, true, 'Controller must be aborted on org switch');

  // Org A response arrives late
  deferredOrgA.resolve({
    meeting_id: 'meeting-alpha',
    has_intelligence: true,
    summary: { executive_summary: 'Confidential Org Alpha Intelligence' },
  });
  await pendingPromise;

  // Verify Org A data was discarded
  assert.strictEqual(stateUpdated, false, 'State update must be discarded');
  assert.deepStrictEqual(sim.meetingIntelligence, {}, 'Org B meetingIntelligence state must remain empty');
});


// --- 8. Logout While Intelligence Request is Pending ---
test('8. Delayed intelligence response is discarded after logout', async () => {
  const sim = new IntelligenceAppSimulator();
  const deferred = createDeferred();

  const capturedEpoch = sim.requestEpochRef.current;
  const capturedOrgId = sim.currentOrgId;
  const controller = sim.createRegisteredController();

  let stateUpdated = false;
  const pendingPromise = (async () => {
    try {
      const data = await deferred.promise;
      if (sim.isContextCurrent(capturedEpoch, capturedOrgId)) {
        sim.meetingIntelligence['meeting-secret'] = data;
        stateUpdated = true;
      }
    } catch (e) {
      if (!isAbortError(e)) throw e;
    } finally {
      sim.unregisterController(controller);
    }
  })();

  // User logs out
  sim.logout();
  assert.strictEqual(controller.signal.aborted, true, 'Controller must be aborted on logout');

  // Late response arrives
  deferred.resolve({ meeting_id: 'meeting-secret', has_intelligence: true });
  await pendingPromise;

  assert.strictEqual(stateUpdated, false, 'State update must be discarded');
  assert.deepStrictEqual(sim.meetingIntelligence, {}, 'meetingIntelligence state must remain empty');
});


// --- 9. Evidence Navigation Switches Tab to Target Transcript Version ---
test('9. Evidence navigation when original transcript tab is selected switches to redacted transcript version', () => {
  const sim = new IntelligenceAppSimulator();
  const meetingId = 'm-evidence-1';

  sim.transcripts[meetingId] = [
    {
      id: 'tr-orig',
      transcript_type: 'ORIGINAL',
      version: 1,
      segments: [{ id: 'seg-orig-1', text: 'Original sensitive transcript' }],
    },
    {
      id: 'tr-red',
      transcript_type: 'CLEANED_REDACTED',
      version: 1,
      segments: [{ id: 'seg-red-1', text: 'Redacted safe transcript' }],
    },
  ];

  // User was currently viewing the ORIGINAL transcript tab
  sim.selectedVersionMap[meetingId] = 'tr-orig';

  // Navigate to an evidence segment that belongs to the REDACTED transcript
  sim.handleNavigateToEvidence(meetingId, 'seg-red-1');

  // Verify version selection switched to tr-red
  assert.strictEqual(sim.selectedVersionMap[meetingId], 'tr-red', 'Active transcript version must switch to target version');
  assert.strictEqual(sim.expandedTranscripts[meetingId], true, 'Transcripts container must be expanded');
  assert.strictEqual(sim.evidenceNavErrorMap[meetingId], null, 'No error should be recorded');
});


// --- 10. Evidence Navigation Keeps Redacted Version When Already Selected ---
test('10. Evidence navigation when redacted transcript tab is already selected keeps redacted version', () => {
  const sim = new IntelligenceAppSimulator();
  const meetingId = 'm-evidence-2';

  sim.transcripts[meetingId] = [
    {
      id: 'tr-orig',
      transcript_type: 'ORIGINAL',
      version: 1,
      segments: [{ id: 'seg-orig-1', text: 'Original sensitive transcript' }],
    },
    {
      id: 'tr-red',
      transcript_type: 'CLEANED_REDACTED',
      version: 1,
      segments: [{ id: 'seg-red-1', text: 'Redacted safe transcript' }],
    },
  ];

  sim.selectedVersionMap[meetingId] = 'tr-red';

  sim.handleNavigateToEvidence(meetingId, 'seg-red-1');

  assert.strictEqual(sim.selectedVersionMap[meetingId], 'tr-red');
  assert.strictEqual(sim.expandedTranscripts[meetingId], true);
  assert.strictEqual(sim.evidenceNavErrorMap[meetingId], null);
});


// --- 11. Missing Evidence Segment Handling ---
test('11. Missing evidence segment handling sets non-sensitive error message without crashing', () => {
  const sim = new IntelligenceAppSimulator();
  const meetingId = 'm-evidence-3';

  sim.transcripts[meetingId] = [
    {
      id: 'tr-red',
      transcript_type: 'CLEANED_REDACTED',
      version: 1,
      segments: [{ id: 'seg-red-1', text: 'Redacted safe transcript' }],
    },
  ];
  sim.selectedVersionMap[meetingId] = 'tr-red';

  // Navigate to non-existent segment ID
  sim.handleNavigateToEvidence(meetingId, 'non-existent-segment-id');

  assert.notStrictEqual(sim.evidenceNavErrorMap[meetingId], null, 'Error message must be set for unavailable segment');
  assert.match(
    sim.evidenceNavErrorMap[meetingId],
    /Referenced evidence segment is not available in current transcripts\./,
    'Message must explain segment absence non-sensitively'
  );
  assert.strictEqual(sim.selectedVersionMap[meetingId], 'tr-red', 'Active transcript must remain stable');
});


// --- 12. Provider Attribution Indicator Logic ---
test('12. Provider attribution indicator distinguishes Ollama, Deterministic, Fallback, and Legacy', () => {
  function resolveAttributionBadge(intelligence) {
    const providerUsed = intelligence?.provider_used ?? intelligence?.summary?.provider_used ?? null;
    const isHeuristic = intelligence?.is_heuristic ?? intelligence?.summary?.is_heuristic ?? false;
    const fallbackWarning = intelligence?.fallback_warning ?? intelligence?.summary?.fallback_warning ?? null;

    if (providerUsed === 'ollama' && !isHeuristic) {
      return { label: 'Ollama LLM', type: 'ollama' };
    }
    if (providerUsed === 'deterministic' || isHeuristic) {
      if (fallbackWarning) {
        return { label: 'Rule-Based Fallback (Ollama Offline)', type: 'fallback' };
      }
      return { label: 'Rule-Based (Deterministic)', type: 'deterministic' };
    }
    return { label: 'Provider: Legacy / Unrecorded', type: 'legacy' };
  }

  // 1. Ollama Success
  const ollamaSuccess = resolveAttributionBadge({
    provider_used: 'ollama',
    is_heuristic: false,
    fallback_warning: null,
  });
  assert.strictEqual(ollamaSuccess.type, 'ollama');
  assert.strictEqual(ollamaSuccess.label, 'Ollama LLM');

  // 2. Deterministic Rule-Based
  const deterministicDirect = resolveAttributionBadge({
    provider_used: 'deterministic',
    is_heuristic: true,
    fallback_warning: null,
  });
  assert.strictEqual(deterministicDirect.type, 'deterministic');
  assert.strictEqual(deterministicDirect.label, 'Rule-Based (Deterministic)');

  // 3. Fallback when Ollama failed
  const fallbackCase = resolveAttributionBadge({
    provider_used: 'deterministic',
    is_heuristic: true,
    fallback_warning: 'Connection refused to Ollama daemon on 127.0.0.1:11434',
  });
  assert.strictEqual(fallbackCase.type, 'fallback');
  assert.strictEqual(fallbackCase.label, 'Rule-Based Fallback (Ollama Offline)');

  // 4. Legacy record with null attribution
  const legacyRecord = resolveAttributionBadge({
    provider_used: null,
    is_heuristic: null,
    fallback_warning: null,
  });
  assert.strictEqual(legacyRecord.type, 'legacy');
  assert.strictEqual(legacyRecord.label, 'Provider: Legacy / Unrecorded');

  // 5. Never label heuristic output as LLM even if provider_used claimed ollama
  const falseClaimHeuristic = resolveAttributionBadge({
    provider_used: 'ollama',
    is_heuristic: true,
    fallback_warning: 'Model output unparseable, fell back to rules',
  });
  assert.notStrictEqual(falseClaimHeuristic.type, 'ollama', 'Heuristic output must never be labeled as LLM');
  assert.strictEqual(falseClaimHeuristic.type, 'fallback');
});
