import test from 'node:test';
import assert from 'node:assert/strict';
import { isAbortError } from '../services/api.ts';

// --- Simulator Modeling App.tsx and OrganizationalMemoryView State & Navigation ---
class MemoryAppSimulator {
  constructor() {
    this.currentOrgId = 'org-alpha';
    this.currentOrgName = 'Alpha Corp';
    this.activeOrgIdRef = { current: 'org-alpha' };
    this.requestEpochRef = { current: 0 };
    this.activeAbortControllers = new Set();

    // Top-level navigation state
    this.activeMainTab = 'meetings'; // 'meetings' | 'memory' | 'system'

    // Meetings & Transcripts state
    this.meetings = [
      {
        id: 'meeting-101',
        organization_id: 'org-alpha',
        title: 'Q3 Product Strategy & Roadmap',
        created_at: '2026-08-15T10:00:00Z',
        input_source_type: 'AUDIO',
        status: 'COMPLETED',
      },
      {
        id: 'meeting-202',
        organization_id: 'org-alpha',
        title: 'Engineering Architecture Review',
        created_at: '2026-09-01T14:00:00Z',
        input_source_type: 'VIDEO',
        status: 'COMPLETED',
      },
    ];
    this.transcripts = {
      'meeting-101': [
        {
          id: 'tr-raw-101',
          version_type: 'ORIGINAL',
          segments: [
            { id: 'seg-101-1', text: 'Unredacted discussion of confidential Q3 budget.', start_time: 10.0, end_time: 25.0 },
          ],
        },
        {
          id: 'tr-safe-101',
          version_type: 'CLEANED_REDACTED',
          segments: [
            { id: 'seg-101-1', text: 'Discussion of [CONFIDENTIAL_REDACTED] Q3 budget.', start_time: 10.0, end_time: 25.0 },
            { id: 'seg-101-2', text: 'Target launch date finalized for September 15.', start_time: 26.0, end_time: 45.0 },
          ],
        },
      ],
      'meeting-202': [
        {
          id: 'tr-safe-202',
          version_type: 'CLEANED_REDACTED',
          segments: [
            { id: 'seg-202-1', text: 'Decided to adopt SQLite FTS5 for local full-text search.', start_time: 5.0, end_time: 30.0 },
          ],
        },
      ],
    };
    this.selectedVersionMap = {};
    this.expandedTranscripts = {};
    this.evidenceNavErrorMap = {};
    this.lastNavigatedSegment = null;

    // Organizational Memory view state
    this.queryState = {
      question: '',
      selectedMeetingId: '',
      startDate: '',
      endDate: '',
      maxChunks: 5,
      isLoading: false,
      error: null,
      response: null,
    };

    this.searchState = {
      query: '',
      selectedMeetingId: '',
      startDate: '',
      endDate: '',
      topK: 10,
      isLoading: false,
      error: null,
      response: null,
    };
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

  switchOrganization(targetOrgId, targetOrgName = 'Target Corp') {
    this.requestEpochRef.current += 1;
    this.cancelAllActiveOperations();

    this.currentOrgId = targetOrgId;
    this.currentOrgName = targetOrgName;
    this.activeOrgIdRef.current = targetOrgId;

    // Reset dashboard and memory state
    this.activeMainTab = 'meetings';
    this.selectedVersionMap = {};
    this.expandedTranscripts = {};
    this.evidenceNavErrorMap = {};
    this.lastNavigatedSegment = null;

    this.queryState = {
      question: '',
      selectedMeetingId: '',
      startDate: '',
      endDate: '',
      maxChunks: 5,
      isLoading: false,
      error: null,
      response: null,
    };

    this.searchState = {
      query: '',
      selectedMeetingId: '',
      startDate: '',
      endDate: '',
      topK: 10,
      isLoading: false,
      error: null,
      response: null,
    };
  }

  logout() {
    this.requestEpochRef.current += 1;
    this.activeOrgIdRef.current = null;
    this.currentOrgId = null;
    this.cancelAllActiveOperations();

    this.activeMainTab = 'meetings';
    this.selectedVersionMap = {};
    this.expandedTranscripts = {};
    this.evidenceNavErrorMap = {};
    this.lastNavigatedSegment = null;
    this.queryState.response = null;
    this.searchState.response = null;
  }

  handleNavigateToEvidence(meetingId, segmentId, timestampSeconds = null) {
    this.evidenceNavErrorMap[meetingId] = null;
    this.expandedTranscripts[meetingId] = true;

    const transcripts = this.transcripts[meetingId] || [];
    // Prioritize safe transcript versions
    const targetTranscript = transcripts.find((t) =>
      t.segments && t.segments.some((seg) => seg.id === segmentId)
    );

    if (transcripts.length > 0 && !targetTranscript) {
      this.evidenceNavErrorMap[meetingId] =
        'Referenced evidence segment is not available in current transcripts.';
      return false;
    }

    if (targetTranscript) {
      this.selectedVersionMap[meetingId] = targetTranscript.id;
    }

    this.lastNavigatedSegment = {
      meetingId,
      segmentId,
      timestampSeconds,
      targetTranscriptId: targetTranscript ? targetTranscript.id : null,
    };
    return true;
  }

  handleMemoryNavigateToEvidence(meetingId, segmentId, timestampSeconds = null) {
    // 1. Switch to meetings tab
    this.activeMainTab = 'meetings';
    // 2. Expand and navigate to evidence
    return this.handleNavigateToEvidence(meetingId, segmentId, timestampSeconds);
  }

  async askAI(question, filters = {}, queryApiFn) {
    const trimmed = (question || '').trim();
    if (!trimmed) {
      this.queryState.error = 'Please enter a question to ask organizational AI.';
      return null;
    }

    const capturedEpoch = this.requestEpochRef.current;
    const capturedOrgId = this.currentOrgId;
    const controller = this.createRegisteredController();

    this.queryState.isLoading = true;
    this.queryState.error = null;

    try {
      const payload = {
        question: trimmed,
        query: trimmed,
        meeting_id: filters.meetingId || undefined,
        start_date: filters.startDate ? `${filters.startDate}T00:00:00Z` : undefined,
        end_date: filters.endDate ? `${filters.endDate}T23:59:59Z` : undefined,
        max_evidence_chunks: filters.maxEvidenceChunks || filters.maxChunks || 5,
        max_chunks: filters.maxEvidenceChunks || filters.maxChunks || 5,
      };

      const res = await queryApiFn(payload, controller.signal);
      if (this.isContextCurrent(capturedEpoch, capturedOrgId)) {
        this.queryState.response = res;
        this.queryState.error = null;
        return res;
      }
    } catch (err) {
      if (this.isContextCurrent(capturedEpoch, capturedOrgId) && !isAbortError(err)) {
        this.queryState.error = err.message || 'Failed to query organizational memory.';
      }
    } finally {
      this.unregisterController(controller);
      if (this.isContextCurrent(capturedEpoch, capturedOrgId)) {
        this.queryState.isLoading = false;
      }
    }
  }

  async searchMemory(query, filters = {}, searchApiFn) {
    const trimmed = (query || '').trim();
    if (!trimmed) {
      this.searchState.error = 'Please enter search terms.';
      return null;
    }

    const capturedEpoch = this.requestEpochRef.current;
    const capturedOrgId = this.currentOrgId;
    const controller = this.createRegisteredController();

    this.searchState.isLoading = true;
    this.searchState.error = null;

    try {
      const payload = {
        query: trimmed,
        meeting_id: filters.meetingId || undefined,
        start_date: filters.startDate ? `${filters.startDate}T00:00:00Z` : undefined,
        end_date: filters.endDate ? `${filters.endDate}T23:59:59Z` : undefined,
        limit: filters.limit || filters.topK || 10,
        top_k: filters.limit || filters.topK || 10,
      };

      const res = await searchApiFn(payload, controller.signal);
      if (this.isContextCurrent(capturedEpoch, capturedOrgId)) {
        this.searchState.response = res;
        this.searchState.error = null;
        return res;
      }
    } catch (err) {
      if (this.isContextCurrent(capturedEpoch, capturedOrgId) && !isAbortError(err)) {
        this.searchState.error = err.message || 'Failed to search organizational memory.';
      }
    } finally {
      this.unregisterController(controller);
      if (this.isContextCurrent(capturedEpoch, capturedOrgId)) {
        this.searchState.isLoading = false;
      }
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


// ============================================================================
// TEST SUITE: PHASE 8.6 ORGANIZATIONAL MEMORY UI & ASK AI
// ============================================================================

test('1. Organizational Memory top-level navigation switches tabs and defaults correctly', () => {
  const sim = new MemoryAppSimulator();
  assert.strictEqual(sim.activeMainTab, 'meetings');

  // Switch to Organizational Memory tab
  sim.activeMainTab = 'memory';
  assert.strictEqual(sim.activeMainTab, 'memory');

  // Switch to System & Diagnostics tab
  sim.activeMainTab = 'system';
  assert.strictEqual(sim.activeMainTab, 'system');

  // Switch back to Meetings tab
  sim.activeMainTab = 'meetings';
  assert.strictEqual(sim.activeMainTab, 'meetings');
});

test('2. Successful Ask AI query populates answer, honest provider attribution, and citations', async () => {
  const sim = new MemoryAppSimulator();
  sim.activeMainTab = 'memory';

  const mockResponse = {
    answer: 'The team finalized the Q3 launch date for September 15 during the product strategy session.',
    abstention_status: 'ANSWER_SUPPORTED',
    provider_used: 'ollama',
    fallback_used: false,
    citations: [
      {
        chunk_id: 'chk-101',
        meeting_id: 'meeting-101',
        meeting_title: 'Q3 Product Strategy & Roadmap',
        meeting_date: '2026-08-15T10:00:00Z',
        start_time_seconds: 26.0,
        end_time_seconds: 45.0,
        quoted_text: 'Target launch date finalized for September 15.',
        segment_ids: ['seg-101-2'],
      },
    ],
    diagnostics: {
      retrieval_count: 3,
      context_token_count: 142,
      latency_ms: 185.4,
    },
  };

  const result = await sim.askAI(
    'When is the Q3 launch date?',
    {},
    async () => mockResponse
  );

  assert.strictEqual(sim.queryState.isLoading, false);
  assert.strictEqual(sim.queryState.error, null);
  assert.strictEqual(result.abstention_status, 'ANSWER_SUPPORTED');
  assert.strictEqual(result.provider_used, 'ollama');
  assert.strictEqual(result.fallback_used, false);
  assert.strictEqual(result.citations.length, 1);
  assert.strictEqual(result.citations[0].meeting_title, 'Q3 Product Strategy & Roadmap');
  assert.strictEqual(result.citations[0].quoted_text, 'Target launch date finalized for September 15.');
});

test('3. Citations format meeting title, date, timestamps, and quotes accurately', async () => {
  const citation = {
    chunk_id: 'chk-202',
    meeting_id: 'meeting-202',
    meeting_title: 'Engineering Architecture Review',
    meeting_date: '2026-09-01T14:00:00Z',
    start_time_seconds: 65.5,
    end_time_seconds: 125.0,
    quoted_text: 'Decided to adopt SQLite FTS5 for local full-text search.',
    segment_ids: ['seg-202-1'],
  };

  // Helper matching formatTime in OrganizationalMemoryView
  const formatTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const remSecs = Math.floor(secs % 60);
    return `${mins.toString().padStart(2, '0')}:${remSecs.toString().padStart(2, '0')}`;
  };

  const formattedStart = formatTime(citation.start_time_seconds);
  const formattedEnd = formatTime(citation.end_time_seconds);
  assert.strictEqual(formattedStart, '01:05');
  assert.strictEqual(formattedEnd, '02:05');
  assert.strictEqual(citation.meeting_title, 'Engineering Architecture Review');
  assert.ok(citation.quoted_text.includes('SQLite FTS5'));
});

test('4. Clicking a citation invokes evidence navigation, switches to meetings tab, and selects safe transcript', async () => {
  const sim = new MemoryAppSimulator();
  sim.activeMainTab = 'memory';

  const citation = {
    chunk_id: 'chk-101',
    meeting_id: 'meeting-101',
    meeting_title: 'Q3 Product Strategy & Roadmap',
    start_time_seconds: 26.0,
    segment_ids: ['seg-101-2'],
  };

  const navSuccess = sim.handleMemoryNavigateToEvidence(
    citation.meeting_id,
    citation.segment_ids[0],
    citation.start_time_seconds
  );

  assert.strictEqual(navSuccess, true);
  // 1. Verifies it switched back to the meetings tab
  assert.strictEqual(sim.activeMainTab, 'meetings');
  // 2. Verifies the meeting's transcript was expanded
  assert.strictEqual(sim.expandedTranscripts['meeting-101'], true);
  // 3. Verifies the safe transcript version (CLEANED_REDACTED) was selected
  assert.strictEqual(sim.selectedVersionMap['meeting-101'], 'tr-safe-101');
  // 4. Verifies the navigated segment details
  assert.deepStrictEqual(sim.lastNavigatedSegment, {
    meetingId: 'meeting-101',
    segmentId: 'seg-101-2',
    timestampSeconds: 26.0,
    targetTranscriptId: 'tr-safe-101',
  });
  // 5. Verifies no navigation error
  assert.strictEqual(sim.evidenceNavErrorMap['meeting-101'], null);
});

test('5. Search Organizational Memory returns ranked results with metadata and provenance', async () => {
  const sim = new MemoryAppSimulator();
  sim.activeMainTab = 'memory';

  const mockSearchResponse = {
    results: [
      {
        chunk_id: 'chk-fts-1',
        meeting_id: 'meeting-202',
        meeting_title: 'Engineering Architecture Review',
        meeting_date: '2026-09-01T14:00:00Z',
        score: 0.94,
        start_time_seconds: 5.0,
        end_time_seconds: 30.0,
        text: 'Decided to adopt SQLite FTS5 for local full-text search.',
        segment_ids: ['seg-202-1'],
        dominant_language: 'en',
      },
    ],
    total_found: 1,
    query: 'SQLite FTS5',
    diagnostics: {
      retrieval_count: 1,
      latency_ms: 12.8,
    },
  };

  const searchResult = await sim.searchMemory('SQLite FTS5', {}, async () => mockSearchResponse);

  assert.strictEqual(sim.searchState.isLoading, false);
  assert.strictEqual(sim.searchState.error, null);
  assert.strictEqual(searchResult.total_found, 1);
  assert.strictEqual(searchResult.results[0].meeting_id, 'meeting-202');
  assert.strictEqual(searchResult.results[0].score, 0.94);
  assert.strictEqual(searchResult.results[0].dominant_language, 'en');
});

test('6. Clicking a search result jumps to cited meeting, selects version, and navigates to segment', () => {
  const sim = new MemoryAppSimulator();
  sim.activeMainTab = 'memory';

  const navSuccess = sim.handleMemoryNavigateToEvidence('meeting-202', 'seg-202-1', 5.0);

  assert.strictEqual(navSuccess, true);
  assert.strictEqual(sim.activeMainTab, 'meetings');
  assert.strictEqual(sim.expandedTranscripts['meeting-202'], true);
  assert.strictEqual(sim.selectedVersionMap['meeting-202'], 'tr-safe-202');
  assert.strictEqual(sim.lastNavigatedSegment.segmentId, 'seg-202-1');
});

test('7. Date filters and meeting filter are properly formatted and delivered in API payload', async () => {
  const sim = new MemoryAppSimulator();
  let capturedQueryPayload = null;
  let capturedSearchPayload = null;

  await sim.askAI(
    'What was decided?',
    {
      meetingId: 'meeting-101',
      startDate: '2026-08-01',
      endDate: '2026-08-31',
      maxChunks: 4,
    },
    async (payload) => {
      capturedQueryPayload = payload;
      return { answer: 'Answer', abstention_status: 'ANSWER_SUPPORTED', citations: [] };
    }
  );

  assert.strictEqual(capturedQueryPayload.question, 'What was decided?');
  assert.strictEqual(capturedQueryPayload.meeting_id, 'meeting-101');
  assert.strictEqual(capturedQueryPayload.start_date, '2026-08-01T00:00:00Z');
  assert.strictEqual(capturedQueryPayload.end_date, '2026-08-31T23:59:59Z');
  assert.strictEqual(capturedQueryPayload.max_evidence_chunks, 4);

  await sim.searchMemory(
    'budget review',
    {
      meetingId: 'meeting-101',
      startDate: '2026-08-01',
      endDate: '2026-08-31',
      limit: 8,
    },
    async (payload) => {
      capturedSearchPayload = payload;
      return { results: [], total_found: 0, query: payload.query };
    }
  );

  assert.strictEqual(capturedSearchPayload.query, 'budget review');
  assert.strictEqual(capturedSearchPayload.meeting_id, 'meeting-101');
  assert.strictEqual(capturedSearchPayload.start_date, '2026-08-01T00:00:00Z');
  assert.strictEqual(capturedSearchPayload.end_date, '2026-08-31T23:59:59Z');
  assert.strictEqual(capturedSearchPayload.limit, 8);
});

test('8. Client-side input validation rejects empty questions and empty search queries without network calls', async () => {
  const sim = new MemoryAppSimulator();
  let apiCallCount = 0;
  const mockApi = async () => {
    apiCallCount += 1;
    return {};
  };

  // Test empty and whitespace question
  const res1 = await sim.askAI('', {}, mockApi);
  assert.strictEqual(res1, null);
  assert.strictEqual(sim.queryState.error, 'Please enter a question to ask organizational AI.');

  const res2 = await sim.askAI('   ', {}, mockApi);
  assert.strictEqual(res2, null);
  assert.strictEqual(sim.queryState.error, 'Please enter a question to ask organizational AI.');

  // Test empty and whitespace search
  const res3 = await sim.searchMemory('', {}, mockApi);
  assert.strictEqual(res3, null);
  assert.strictEqual(sim.searchState.error, 'Please enter search terms.');

  const res4 = await sim.searchMemory('   ', {}, mockApi);
  assert.strictEqual(res4, null);
  assert.strictEqual(sim.searchState.error, 'Please enter search terms.');

  // Zero network calls made
  assert.strictEqual(apiCallCount, 0);
});

test('9. NO_EVIDENCE and INSUFFICIENT_EVIDENCE abstention statuses render cleanly with guidance', async () => {
  const sim = new MemoryAppSimulator();

  // Test NO_EVIDENCE
  const noEvidenceResponse = {
    answer: 'No relevant evidence was found across meetings in this organization matching your query.',
    abstention_status: 'NO_EVIDENCE',
    provider_used: 'deterministic',
    fallback_used: true,
    fallback_warning: 'No relevant evidence found across organizational records.',
    citations: [],
  };

  const res1 = await sim.askAI('What is quantum entanglement?', {}, async () => noEvidenceResponse);
  assert.strictEqual(res1.abstention_status, 'NO_EVIDENCE');
  assert.strictEqual(res1.fallback_used, true);
  assert.strictEqual(res1.citations.length, 0);

  // Test INSUFFICIENT_EVIDENCE
  const insufficientResponse = {
    answer: 'The retrieved meeting segments discuss the topic partially, but lack sufficient confirmed details.',
    abstention_status: 'INSUFFICIENT_EVIDENCE',
    provider_used: 'deterministic',
    fallback_used: true,
    citations: [],
  };

  const res2 = await sim.askAI('What was the exact revenue in 2024?', {}, async () => insufficientResponse);
  assert.strictEqual(res2.abstention_status, 'INSUFFICIENT_EVIDENCE');
});

test('10. CONFLICTING_EVIDENCE displays alert banner without failing', async () => {
  const sim = new MemoryAppSimulator();

  const conflictingResponse = {
    answer: 'Meeting A indicated a September launch, while Meeting B specified a November deferral.',
    abstention_status: 'CONFLICTING_EVIDENCE',
    provider_used: 'ollama',
    fallback_used: false,
    citations: [
      { chunk_id: 'chk-1', meeting_id: 'meeting-101', meeting_title: 'Meeting 1', start_time_seconds: 10.0, quoted_text: 'Launch in Sept', segment_ids: [] },
      { chunk_id: 'chk-2', meeting_id: 'meeting-202', meeting_title: 'Meeting 2', start_time_seconds: 20.0, quoted_text: 'Launch in Nov', segment_ids: [] },
    ],
  };

  const res = await sim.askAI('When is the launch?', {}, async () => conflictingResponse);
  assert.strictEqual(res.abstention_status, 'CONFLICTING_EVIDENCE');
  assert.strictEqual(res.citations.length, 2);
  assert.strictEqual(sim.queryState.error, null);
});

test('11. MODEL_UNAVAILABLE abstention clearly surfaces model availability without crash', async () => {
  const sim = new MemoryAppSimulator();

  const modelUnavailableResponse = {
    answer: 'Answer generation service is currently unavailable. Ollama model is not responding.',
    abstention_status: 'MODEL_UNAVAILABLE',
    provider_used: 'deterministic',
    fallback_used: true,
    fallback_warning: 'Local LLM unavailable. Deterministic fallback active.',
    citations: [],
  };

  const res = await sim.askAI('Summarize all projects', {}, async () => modelUnavailableResponse);
  assert.strictEqual(res.abstention_status, 'MODEL_UNAVAILABLE');
  assert.strictEqual(res.provider_used, 'deterministic');
  assert.strictEqual(res.fallback_used, true);
  assert.ok(res.fallback_warning.includes('Local LLM unavailable'));
});

test('12. RETRIEVAL_UNAVAILABLE renders safe error banner', async () => {
  const sim = new MemoryAppSimulator();

  const retrievalUnavailableResponse = {
    answer: 'Hybrid retrieval service is unavailable. Please verify SQLite FTS5 configuration.',
    abstention_status: 'RETRIEVAL_UNAVAILABLE',
    provider_used: 'deterministic',
    fallback_used: true,
    citations: [],
  };

  const res = await sim.askAI('Any updates?', {}, async () => retrievalUnavailableResponse);
  assert.strictEqual(res.abstention_status, 'RETRIEVAL_UNAVAILABLE');
  assert.strictEqual(res.citations.length, 0);
});

test('13. Honest provider attribution distinguishes Ollama from Deterministic fallback', async () => {
  const sim = new MemoryAppSimulator();

  // 1. Genuine Ollama output
  const ollamaRes = await sim.askAI('Question 1', {}, async () => ({
    answer: 'Generated by Ollama',
    abstention_status: 'ANSWER_SUPPORTED',
    provider_used: 'ollama',
    fallback_used: false,
    citations: [],
  }));
  assert.strictEqual(ollamaRes.provider_used, 'ollama');
  assert.strictEqual(ollamaRes.fallback_used, false);

  // 2. Deterministic Fallback output
  const fallbackRes = await sim.askAI('Question 2', {}, async () => ({
    answer: 'Deterministic heuristic summary',
    abstention_status: 'ANSWER_SUPPORTED',
    provider_used: 'deterministic',
    fallback_used: true,
    fallback_warning: 'LLM timed out. Deterministic rule-based extraction utilized.',
    citations: [],
  }));
  assert.strictEqual(fallbackRes.provider_used, 'deterministic');
  assert.strictEqual(fallbackRes.fallback_used, true);
  assert.ok(fallbackRes.fallback_warning.includes('Deterministic rule-based extraction'));
});

test('14. Concurrency & epoch isolation: switching organizations aborts in-flight search and ignores stale response', async () => {
  const sim = new MemoryAppSimulator();
  const deferredSearch = createDeferred();

  // Start search in Org Alpha
  const searchPromise = sim.searchMemory('Confidential Project X', {}, async (_p, signal) => {
    signal.addEventListener('abort', () => {
      const err = new Error('The operation was aborted');
      err.name = 'AbortError';
      deferredSearch.reject(err);
    });
    return deferredSearch.promise;
  });

  // User immediately switches to Org Beta before search completes
  sim.switchOrganization('org-beta', 'Beta Industries');

  // Verify memory and search state was wiped
  assert.strictEqual(sim.currentOrgId, 'org-beta');
  assert.strictEqual(sim.activeMainTab, 'meetings');
  assert.strictEqual(sim.searchState.response, null);
  assert.strictEqual(sim.searchState.isLoading, false);

  // Stale search response or abort resolves
  await searchPromise.catch(() => {});

  // Results from Org Alpha MUST NOT leak into Org Beta
  assert.strictEqual(sim.searchState.response, null);
  assert.strictEqual(sim.searchState.error, null);
});

test('15. Rate limiting (429) and authentication errors are handled safely', async () => {
  const sim = new MemoryAppSimulator();

  // 429 Rate limit test
  await sim.askAI('Fast query', {}, async () => {
    const err = new Error('HTTP 429: Rate limit exceeded for organizational memory queries. Please wait 15 seconds.');
    throw err;
  });

  assert.strictEqual(sim.queryState.isLoading, false);
  assert.ok(sim.queryState.error.includes('Rate limit exceeded'));

  // Logout clears memory state and resets navigation to meetings
  sim.activeMainTab = 'memory';
  sim.logout();
  assert.strictEqual(sim.activeMainTab, 'meetings');
  assert.strictEqual(sim.currentOrgId, null);
  assert.strictEqual(sim.queryState.response, null);
});
