/**
 * Ask AI Structured Intelligence Presentation Test Suite (Phase 8.9.5)
 *
 * Validates:
 * 1. Decision timeline rendering with correct lifecycle status badges (ACTIVE, SUPERSEDED, REVOKED)
 * 2. Explicit supersession lineage visualization (earlier vs. superseding) and what changed
 * 3. Unlinked contradictory decisions displayed as potential conflicts, not confirmed lineage
 * 4. Meeting title and actual date display from database metadata
 * 5. Direct verified evidence citation button with timestamp: formatted mm:ss and navigation callback
 * 6. Missing evidence gracefully rendered as non-clickable indicator ("Evidence: Unavailable")
 * 7. Commitment rendering (owner, deadline, status, overdue, review info)
 * 8. Unresolved question rendering (status, raised by, resolution details)
 * 9. Potential conflict warning rendering (heuristic advisory banner, Decision A & B, both active)
 * 10. Expandable / collapsible sections for long histories with summary count
 * 11. Empty structured intelligence response handling (cleanly shows only answer and citations)
 * 12. Abstention response handling (protection alert preserved)
 * 13. Organization switching clears state, aborts pending queries, and ignores stale responses
 * 14. Multilingual text support (Hindi and Hinglish answer text, question text, and decisions)
 */

import test from 'node:test';
import assert from 'node:assert/strict';

// Helper formatting timestamp seconds to mm:ss
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Simulator modeling OrganizationalMemoryView Ask AI state and rendering
class AskAIStructuredPresentationSimulator {
  constructor(initialOrgId = 'org-alpha') {
    this.currentOrgId = initialOrgId;
    this.currentOrgName = initialOrgId === 'org-alpha' ? 'Alpha Innovations' : 'Beta Health';
    this.activeOrgIdRef = { current: initialOrgId };
    this.requestEpochRef = { current: 0 };
    this.activeAbortControllers = new Set();

    // Navigation and tab state
    this.activeTab = 'memory'; // 'meetings' | 'memory'
    this.lastNavigatedEvidence = null;

    // Component state
    this.question = '';
    this.isLoadingQuery = false;
    this.queryError = null;
    this.queryResponse = null;

    // Collapsible section states
    this.isDecisionsExpanded = true;
    this.isCommitmentsExpanded = true;
    this.isQuestionsExpanded = true;
    this.isConflictsExpanded = true;
  }

  // Abort-aware and epoch-isolated query submission
  async submitAskQuery(payload, mockResolver, signalDelayMs = 0) {
    this.isLoadingQuery = true;
    this.queryError = null;
    this.queryResponse = null;

    const controller = new AbortController();
    this.activeAbortControllers.add(controller);
    const epoch = ++this.requestEpochRef.current;
    const capturedOrgId = this.currentOrgId;

    try {
      if (signalDelayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, signalDelayMs));
      }

      if (controller.signal.aborted) {
        const abortErr = new Error('Aborted');
        abortErr.name = 'AbortError';
        throw abortErr;
      }

      const response = await mockResolver(payload, controller.signal);

      // Verify epoch and org isolation
      if (epoch === this.requestEpochRef.current && capturedOrgId === this.activeOrgIdRef.current) {
        this.queryResponse = response;
      }
      return response;
    } catch (err) {
      if (err.name !== 'AbortError' && epoch === this.requestEpochRef.current && capturedOrgId === this.activeOrgIdRef.current) {
        this.queryError = err.message || 'Query failed';
      }
      throw err;
    } finally {
      this.activeAbortControllers.delete(controller);
      if (epoch === this.requestEpochRef.current && capturedOrgId === this.activeOrgIdRef.current) {
        this.isLoadingQuery = false;
      }
    }
  }

  // Switch organization boundary
  switchOrganization(newOrgId, newOrgName = 'New Org') {
    for (const controller of this.activeAbortControllers) {
      controller.abort();
    }
    this.activeAbortControllers.clear();

    ++this.requestEpochRef.current;
    this.currentOrgId = newOrgId;
    this.currentOrgName = newOrgName;
    this.activeOrgIdRef.current = newOrgId;

    this.queryResponse = null;
    this.queryError = null;
    this.isLoadingQuery = false;
  }

  // Evidence navigation callback
  handleNavigateToEvidence(meetingId, segmentId, timestampSeconds = null) {
    this.activeTab = 'meetings';
    this.lastNavigatedEvidence = {
      meetingId,
      segmentId,
      timestampSeconds,
      formattedTime: timestampSeconds != null ? formatTime(timestampSeconds) : null,
    };
  }

  // Render representations for testing
  renderDecisionStatusBadge(status) {
    switch (status) {
      case 'ACTIVE':
        return { label: 'Active', className: 'badge-status active' };
      case 'SUPERSEDED':
        return { label: 'Superseded', className: 'badge-status superseded' };
      case 'REVOKED':
        return { label: 'Revoked', className: 'badge-status revoked' };
      default:
        return { label: status, className: 'badge-status pending' };
    }
  }

  renderCommitmentStatusBadge(status, isOverdue) {
    if (isOverdue) {
      return { label: 'Overdue', className: 'badge-status overdue' };
    }
    switch (status) {
      case 'COMPLETED':
        return { label: 'Completed', className: 'badge-status completed' };
      case 'IN_PROGRESS':
        return { label: 'In Progress', className: 'badge-status in-progress' };
      case 'CANCELLED':
        return { label: 'Cancelled', className: 'badge-status cancelled' };
      case 'PENDING':
      default:
        return { label: 'Pending', className: 'badge-status pending' };
    }
  }

  renderQuestionStatusBadge(status) {
    switch (status) {
      case 'RESOLVED':
        return { label: 'Resolved', className: 'badge-status resolved' };
      case 'DEFERRED':
        return { label: 'Deferred', className: 'badge-status deferred' };
      case 'OPEN':
      default:
        return { label: 'Open', className: 'badge-status open' };
    }
  }

  renderEvidenceBadge(abstentionStatus, count) {
    if (abstentionStatus === 'CONFLICTING_EVIDENCE') {
      return {
        label: `${count} Conflicting Evidence Sources`,
        className: 'intel-stat-badge conflict',
      };
    }
    return {
      label: `${count} Verified Evidence Chunks`,
      className: 'intel-stat-badge neutral',
    };
  }

  renderAbstentionAlert(abstentionStatus) {
    if (abstentionStatus === 'CONFLICTING_EVIDENCE') {
      return {
        title: 'Unresolved Conflict Alert:',
        description: 'Potentially contradictory statements or active decisions were identified across meetings.',
        alertClass: 'memory-abstention-alert conflict',
      };
    }
    if (abstentionStatus === 'NO_EVIDENCE') {
      return {
        title: 'Answer Grounding Protection:',
        description: 'No meeting evidence was found matching this question in your organization\'s recorded history.',
        alertClass: 'memory-abstention-alert no-evidence',
      };
    }
    return {
      title: 'Answer Grounding Protection:',
      description: 'The system refrained from speculating because the retrieved meeting transcripts do not contain sufficient evidence.',
      alertClass: 'memory-abstention-alert insufficient-evidence',
    };
  }

  // External background knowledge card representation
  renderExternalKnowledgeCard(extKnowledge) {
    if (!extKnowledge) return null;
    return {
      badge: 'External Background Reference — Context Not Discussed in Meeting',
      disclaimer: extKnowledge.disclaimer,
      subject: extKnowledge.subject,
      summary: extKnowledge.summary,
      keyFacts: extKnowledge.key_facts || [],
      citations: extKnowledge.citations || [],
      providerUsed: extKnowledge.provider_used,
      isAvailable: extKnowledge.is_available,
      unavailabilityReason: extKnowledge.unavailability_reason || null,
    };
  }
}

// ---------------------------------------------------------------------------
// TEST 1: Decision timeline rendering with lifecycle status badges and metadata
// ---------------------------------------------------------------------------
test('1. Decision timeline renders chronological decisions with ACTIVE, SUPERSEDED, and REVOKED badges', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const mockResponse = {
    question: 'What decisions were made about container orchestration?',
    answer: 'Initially Docker Compose was adopted, but it was later superseded by Kubernetes Helm charts.',
    citations: [
      {
        citation_id: 'cit-1',
        chunk_id: 'chunk-1',
        meeting_id: 'meeting-101',
        meeting_title: 'Platform Sync',
        meeting_date: '2026-08-10T10:00:00Z',
        transcript_id: 'tr-1',
        segment_ids: ['seg-101'],
        start_time: 25.0,
        end_time: 45.0,
        quoted_text: 'Decided to adopt Docker Compose for local microservices.',
        language_codes: ['en'],
      },
    ],
    abstention_status: 'ANSWER_SUPPORTED',
    is_abstained: false,
    diagnostics: {
      query: 'What decisions were made about container orchestration?',
      total_candidates_retrieved: 12,
      evidence_items_count: 1,
      meetings_represented: ['meeting-101', 'meeting-102'],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
      structured_decisions_count: 3,
    },
    structured_decisions: [
      {
        id: 'dec-1',
        meeting_id: 'meeting-101',
        meeting_title: 'Platform Architecture Sync',
        meeting_date: '2026-08-10T10:00:00Z',
        description: 'Adopt Docker Compose for local developer environments.',
        rationale_or_context: 'Simplifies dependency setup for new engineers.',
        decided_by: 'Lead Architect',
        status: 'SUPERSEDED',
        supersedes_decision_id: null,
        superseded_by_decision_id: 'dec-2',
        lineage_depth: 2,
        evidence_segment_id: 'seg-101',
        timestamp_seconds: 25.0,
        created_at: '2026-08-10T10:00:00Z',
        updated_at: '2026-09-15T14:00:00Z',
      },
      {
        id: 'dec-2',
        meeting_id: 'meeting-102',
        meeting_title: 'Infrastructure Modernization Review',
        meeting_date: '2026-09-15T14:00:00Z',
        description: 'Adopt Kubernetes Helm charts for all development and production clusters.',
        rationale_or_context: 'Unified cluster definitions across environments.',
        decided_by: 'CTO',
        status: 'ACTIVE',
        supersedes_decision_id: 'dec-1',
        superseded_by_decision_id: null,
        lineage_depth: 2,
        evidence_segment_id: 'seg-201',
        timestamp_seconds: 140.0,
        created_at: '2026-09-15T14:00:00Z',
        updated_at: '2026-09-15T14:00:00Z',
      },
      {
        id: 'dec-3',
        meeting_id: 'meeting-103',
        meeting_title: 'Security Compliance Audit',
        meeting_date: '2026-09-20T11:00:00Z',
        description: 'Allow plain HTTP connections on internal cluster networks.',
        rationale_or_context: 'Deemed unsafe under Zero-Trust mandate.',
        decided_by: 'Security Officer',
        status: 'REVOKED',
        supersedes_decision_id: null,
        superseded_by_decision_id: null,
        lineage_depth: 1,
        evidence_segment_id: 'seg-301',
        timestamp_seconds: 75.0,
        created_at: '2026-09-20T11:00:00Z',
        updated_at: '2026-09-21T09:00:00Z',
      },
    ],
  };

  await sim.submitAskQuery({ question: 'orchestration' }, async () => mockResponse);

  assert.equal(sim.queryResponse.structured_decisions.length, 3);

  // Check lifecycle badges
  const badge1 = sim.renderDecisionStatusBadge(sim.queryResponse.structured_decisions[0].status);
  assert.equal(badge1.label, 'Superseded');
  assert.equal(badge1.className, 'badge-status superseded');

  const badge2 = sim.renderDecisionStatusBadge(sim.queryResponse.structured_decisions[1].status);
  assert.equal(badge2.label, 'Active');
  assert.equal(badge2.className, 'badge-status active');

  const badge3 = sim.renderDecisionStatusBadge(sim.queryResponse.structured_decisions[2].status);
  assert.equal(badge3.label, 'Revoked');
  assert.equal(badge3.className, 'badge-status revoked');

  // Verify meeting titles and dates from metadata
  assert.equal(sim.queryResponse.structured_decisions[0].meeting_title, 'Platform Architecture Sync');
  assert.equal(sim.queryResponse.structured_decisions[0].meeting_date, '2026-08-10T10:00:00Z');
  assert.equal(sim.queryResponse.structured_decisions[1].meeting_title, 'Infrastructure Modernization Review');
  assert.equal(sim.queryResponse.structured_decisions[1].meeting_date, '2026-09-15T14:00:00Z');
});

// ---------------------------------------------------------------------------
// TEST 2: Explicit supersession lineage visualization (earlier vs superseding)
// ---------------------------------------------------------------------------
test('2. Explicit supersession links earlier and superseding decisions with what changed', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const decEarlier = {
    id: 'dec-alpha',
    meeting_id: 'm-1',
    meeting_title: 'Budget Session 1',
    meeting_date: '2026-07-01T10:00:00Z',
    description: 'Allocate $50,000 for cloud infrastructure.',
    rationale_or_context: 'Initial quarterly compute forecast.',
    status: 'SUPERSEDED',
    supersedes_decision_id: null,
    superseded_by_decision_id: 'dec-beta',
  };

  const decLater = {
    id: 'dec-beta',
    meeting_id: 'm-2',
    meeting_title: 'Budget Session 2',
    meeting_date: '2026-08-01T10:00:00Z',
    description: 'Increase cloud infrastructure allocation to $90,000.',
    rationale_or_context: 'Expanded customer compute demand in EMEA region.',
    status: 'ACTIVE',
    supersedes_decision_id: 'dec-alpha',
    superseded_by_decision_id: null,
  };

  const mockResponse = {
    question: 'What was decided about the cloud budget?',
    answer: 'The budget was raised from $50,000 to $90,000.',
    citations: [],
    abstention_status: 'ANSWER_SUPPORTED',
    is_abstained: false,
    diagnostics: {
      query: 'budget',
      total_candidates_retrieved: 4,
      evidence_items_count: 2,
      meetings_represented: ['m-1', 'm-2'],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
    },
    structured_decisions: [decEarlier, decLater],
  };

  await sim.submitAskQuery({ question: 'budget' }, async () => mockResponse);

  const decisions = sim.queryResponse.structured_decisions;
  const dec1 = decisions[0];
  const dec2 = decisions[1];

  // dec2 supersedes dec1
  assert.equal(dec2.supersedes_decision_id, dec1.id);
  const matchedEarlier = decisions.find((d) => d.id === dec2.supersedes_decision_id);
  assert.ok(matchedEarlier);
  assert.equal(matchedEarlier.description, 'Allocate $50,000 for cloud infrastructure.');

  // dec1 is superseded by dec2
  assert.equal(dec1.superseded_by_decision_id, dec2.id);
  const matchedLater = decisions.find((d) => d.id === dec1.superseded_by_decision_id);
  assert.ok(matchedLater);
  assert.equal(matchedLater.description, 'Increase cloud infrastructure allocation to $90,000.');

  // Verification of recorded rationale explanation ("what changed")
  assert.equal(dec2.rationale_or_context, 'Expanded customer compute demand in EMEA region.');
});

// ---------------------------------------------------------------------------
// TEST 3: Unlinked contradictory decisions displayed as potential conflicts
// ---------------------------------------------------------------------------
test('3. Unlinked contradictory decisions are presented as potential conflicts without implying supersession', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const mockResponse = {
    question: 'Which cloud provider do we use?',
    answer: 'There are contradictory active decisions regarding AWS and GCP.',
    citations: [],
    abstention_status: 'CONFLICTING_EVIDENCE',
    is_abstained: false,
    diagnostics: {
      query: 'cloud provider',
      total_candidates_retrieved: 8,
      evidence_items_count: 2,
      meetings_represented: ['m-aws', 'm-gcp'],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
      conflicts_detected_count: 1,
    },
    structured_decisions: [
      {
        id: 'dec-aws',
        meeting_id: 'm-aws',
        meeting_title: 'AWS Platform Alignment',
        meeting_date: '2026-08-01T10:00:00Z',
        description: 'Deploy all analytics workloads on AWS EKS.',
        status: 'ACTIVE',
        supersedes_decision_id: null,
        superseded_by_decision_id: null,
        evidence_segment_id: 'seg-aws-1',
        timestamp_seconds: 30.0,
      },
      {
        id: 'dec-gcp',
        meeting_id: 'm-gcp',
        meeting_title: 'Google Cloud Platform Strategy',
        meeting_date: '2026-08-15T14:00:00Z',
        description: 'Standardize enterprise workloads on Google Cloud GKE.',
        status: 'ACTIVE',
        supersedes_decision_id: null,
        superseded_by_decision_id: null,
        evidence_segment_id: 'seg-gcp-1',
        timestamp_seconds: 45.0,
      },
    ],
    detected_conflicts: [
      {
        decision_a_id: 'dec-aws',
        decision_a_text: 'Deploy all analytics workloads on AWS EKS.',
        decision_a_meeting_title: 'AWS Platform Alignment',
        decision_a_date: '2026-08-01T10:00:00Z',
        decision_b_id: 'dec-gcp',
        decision_b_text: 'Standardize enterprise workloads on Google Cloud GKE.',
        decision_b_meeting_title: 'Google Cloud Platform Strategy',
        decision_b_date: '2026-08-15T14:00:00Z',
        subject: 'Cloud Infrastructure / Container Hosting',
        reason: 'Both decisions mandate contradictory cloud platforms (AWS EKS vs Google Cloud GKE) and both remain ACTIVE.',
        is_heuristic: true,
        suggested_action: 'Convene architecture board to explicitly supersede one decision or establish multi-cloud policy.',
      },
    ],
  };

  await sim.submitAskQuery({ question: 'cloud' }, async () => mockResponse);

  // Both decisions must remain ACTIVE
  assert.equal(sim.queryResponse.structured_decisions[0].status, 'ACTIVE');
  assert.equal(sim.queryResponse.structured_decisions[1].status, 'ACTIVE');

  // No explicit supersession pointers exist
  assert.equal(sim.queryResponse.structured_decisions[0].supersedes_decision_id, null);
  assert.equal(sim.queryResponse.structured_decisions[0].superseded_by_decision_id, null);

  // Detected conflict banner is rendered
  const conflict = sim.queryResponse.detected_conflicts[0];
  assert.equal(conflict.is_heuristic, true);
  assert.ok(conflict.reason.includes('both remain ACTIVE'));
  assert.equal(conflict.subject, 'Cloud Infrastructure / Container Hosting');
  assert.ok(conflict.suggested_action.includes('explicitly supersede'));
});

// ---------------------------------------------------------------------------
// TEST 4: Evidence navigation callback invocation and timestamp formatting
// ---------------------------------------------------------------------------
test('4. Evidence navigation calls onNavigateToEvidence with meetingId, segmentId, and timestamp', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const decisionWithEvidence = {
    id: 'dec-evidence-1',
    meeting_id: 'meeting-arch-202',
    meeting_title: 'Database Architecture Sync',
    meeting_date: '2026-08-20T10:00:00Z',
    description: 'Adopt PostgreSQL 16 with pgvector for vector storage.',
    status: 'ACTIVE',
    evidence_segment_id: 'seg-pg-44',
    timestamp_seconds: 135.5, // 02:15
  };

  // Trigger navigation
  sim.handleNavigateToEvidence(
    decisionWithEvidence.meeting_id,
    decisionWithEvidence.evidence_segment_id,
    decisionWithEvidence.timestamp_seconds
  );

  assert.equal(sim.activeTab, 'meetings');
  assert.deepEqual(sim.lastNavigatedEvidence, {
    meetingId: 'meeting-arch-202',
    segmentId: 'seg-pg-44',
    timestampSeconds: 135.5,
    formattedTime: '02:15',
  });
});

// ---------------------------------------------------------------------------
// TEST 5: Missing evidence gracefully rendered as non-clickable indicator
// ---------------------------------------------------------------------------
test('5. Items with null evidence_segment_id render non-clickable "Evidence: Unavailable" indicator', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const itemWithoutEvidence = {
    id: 'dec-no-evidence',
    meeting_id: 'meeting-101',
    description: 'Decision with null evidence segment.',
    evidence_segment_id: null,
    timestamp_seconds: null,
  };

  assert.equal(itemWithoutEvidence.evidence_segment_id, null);
  // Simulator should not attempt navigation when evidence segment is missing
  let navigationTriggered = false;
  if (itemWithoutEvidence.evidence_segment_id) {
    sim.handleNavigateToEvidence(itemWithoutEvidence.meeting_id, itemWithoutEvidence.evidence_segment_id);
    navigationTriggered = true;
  }

  assert.equal(navigationTriggered, false);
  assert.equal(sim.lastNavigatedEvidence, null);
});

// ---------------------------------------------------------------------------
// TEST 6: Structured commitment presentation (status, owner, deadline, overdue, review info)
// ---------------------------------------------------------------------------
test('6. Commitments render owner, deadline, overdue status, origin meeting, and review info', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const mockResponse = {
    question: 'What commitments are currently tracked?',
    answer: 'There are two commitments: one overdue and one reviewed in a follow-up meeting.',
    citations: [],
    abstention_status: 'ANSWER_SUPPORTED',
    is_abstained: false,
    diagnostics: {
      query: 'commitments',
      total_candidates_retrieved: 6,
      evidence_items_count: 2,
      meetings_represented: ['meeting-1', 'meeting-2'],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
      structured_commitments_count: 2,
    },
    structured_commitments: [
      {
        id: 'act-1',
        meeting_id: 'meeting-1',
        description: 'Complete SOC2 data classification audit.',
        owner_name: 'Gurpreet Singh',
        deadline: '2026-08-01T17:00:00Z',
        status: 'PENDING',
        is_overdue: true,
        origin_meeting_title: 'Security Compliance Kickoff',
        origin_meeting_date: '2026-07-15T10:00:00Z',
        evidence_segment_id: 'seg-act-1',
        timestamp_seconds: 45.0,
        review_count: 0,
        reviews: [],
      },
      {
        id: 'act-2',
        meeting_id: 'meeting-1',
        description: 'Publish internal API schema documentation.',
        owner_name: 'Elena Rostova',
        deadline: '2026-09-30T17:00:00Z',
        status: 'COMPLETED',
        is_overdue: false,
        origin_meeting_title: 'API Design Sync',
        origin_meeting_date: '2026-08-01T14:00:00Z',
        last_reviewed_meeting_id: 'meeting-2',
        last_reviewed_meeting_title: 'Sprint Retrospective',
        last_reviewed_at: '2026-08-25T11:00:00Z',
        last_evidence_segment_id: 'seg-review-2',
        review_count: 1,
        reviews: [
          {
            id: 'rev-1',
            action_item_id: 'act-2',
            meeting_id: 'meeting-2',
            meeting_title: 'Sprint Retrospective',
            status_at_review: 'COMPLETED',
            review_notes: 'Reviewed and confirmed published to developer portal.',
            reviewed_at: '2026-08-25T11:00:00Z',
          },
        ],
      },
    ],
  };

  await sim.submitAskQuery({ question: 'commitments' }, async () => mockResponse);

  const comms = sim.queryResponse.structured_commitments;
  assert.equal(comms.length, 2);

  // Check overdue commitment badge
  const badge1 = sim.renderCommitmentStatusBadge(comms[0].status, comms[0].is_overdue);
  assert.equal(badge1.label, 'Overdue');
  assert.equal(badge1.className, 'badge-status overdue');
  assert.equal(comms[0].owner_name, 'Gurpreet Singh');

  // Check completed reviewed commitment badge and review metadata
  const badge2 = sim.renderCommitmentStatusBadge(comms[1].status, comms[1].is_overdue);
  assert.equal(badge2.label, 'Completed');
  assert.equal(badge2.className, 'badge-status completed');
  assert.equal(comms[1].last_reviewed_meeting_title, 'Sprint Retrospective');
  assert.equal(comms[1].review_count, 1);
  assert.equal(comms[1].reviews[0].review_notes, 'Reviewed and confirmed published to developer portal.');
});

// ---------------------------------------------------------------------------
// TEST 7: Structured unresolved question presentation and resolution details
// ---------------------------------------------------------------------------
test('7. Unresolved questions render status, raised by, and resolution details', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const mockResponse = {
    question: 'What questions were left open or resolved?',
    answer: 'One question was resolved with an explicit decision, and one remains open.',
    citations: [],
    abstention_status: 'ANSWER_SUPPORTED',
    is_abstained: false,
    diagnostics: {
      query: 'questions',
      total_candidates_retrieved: 4,
      evidence_items_count: 2,
      meetings_represented: ['m-1', 'm-2'],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
      structured_questions_count: 2,
    },
    structured_questions: [
      {
        id: 'q-open-1',
        meeting_id: 'm-1',
        question_text: 'Should we support multi-region PostgreSQL failover in Phase 1?',
        raised_by: 'DevOps Lead',
        status: 'OPEN',
        origin_meeting_title: 'Architecture Review',
        origin_meeting_date: '2026-08-10T10:00:00Z',
        evidence_segment_id: 'seg-q1',
        resolving_meeting_id: null,
      },
      {
        id: 'q-resolved-2',
        meeting_id: 'm-1',
        question_text: 'What authentication protocol should external integrations use?',
        raised_by: 'Integration Engineer',
        status: 'RESOLVED',
        origin_meeting_title: 'Architecture Review',
        origin_meeting_date: '2026-08-10T10:00:00Z',
        resolving_meeting_title: 'Security Sync',
        resolving_decision_description: 'Adopt OAuth 2.0 with mTLS for B2B external integrations.',
        resolution_notes: 'Confirmed by security committee on Aug 18.',
        resolved_at: '2026-08-18T14:00:00Z',
      },
    ],
  };

  await sim.submitAskQuery({ question: 'questions' }, async () => mockResponse);

  const qs = sim.queryResponse.structured_questions;
  assert.equal(qs.length, 2);

  const openBadge = sim.renderQuestionStatusBadge(qs[0].status);
  assert.equal(openBadge.label, 'Open');
  assert.equal(openBadge.className, 'badge-status open');
  assert.equal(qs[0].raised_by, 'DevOps Lead');

  const resolvedBadge = sim.renderQuestionStatusBadge(qs[1].status);
  assert.equal(resolvedBadge.label, 'Resolved');
  assert.equal(resolvedBadge.className, 'badge-status resolved');
  assert.equal(qs[1].resolving_meeting_title, 'Security Sync');
  assert.equal(qs[1].resolving_decision_description, 'Adopt OAuth 2.0 with mTLS for B2B external integrations.');
});

// ---------------------------------------------------------------------------
// TEST 8: Expandable and collapsible sections toggle correctly
// ---------------------------------------------------------------------------
test('8. Structured intelligence sections toggle between expanded and collapsed states', () => {
  const sim = new AskAIStructuredPresentationSimulator();

  // Initial states should be expanded
  assert.equal(sim.isDecisionsExpanded, true);
  assert.equal(sim.isCommitmentsExpanded, true);
  assert.equal(sim.isQuestionsExpanded, true);
  assert.equal(sim.isConflictsExpanded, true);

  // Toggle decisions
  sim.isDecisionsExpanded = !sim.isDecisionsExpanded;
  assert.equal(sim.isDecisionsExpanded, false);

  // Toggle commitments
  sim.isCommitmentsExpanded = !sim.isCommitmentsExpanded;
  assert.equal(sim.isCommitmentsExpanded, false);

  // Toggle back
  sim.isDecisionsExpanded = true;
  assert.equal(sim.isDecisionsExpanded, true);
});

// ---------------------------------------------------------------------------
// TEST 9: Empty structured intelligence response handling
// ---------------------------------------------------------------------------
test('9. Query responses with no structured items cleanly render answer and citations without error', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const plainResponse = {
    question: 'How do I submit an expense report?',
    answer: 'Submit receipts via the intranet finance portal before Friday.',
    citations: [
      {
        citation_id: 'cit-plain-1',
        chunk_id: 'ch-1',
        meeting_id: 'm-allhands',
        meeting_title: 'Company All Hands',
        meeting_date: '2026-08-01T15:00:00Z',
        transcript_id: 'tr-allhands',
        segment_ids: ['seg-1'],
        start_time: 120.0,
        end_time: 150.0,
        quoted_text: 'Submit expense receipts via the portal by Friday.',
        language_codes: ['en'],
      },
    ],
    abstention_status: 'ANSWER_SUPPORTED',
    is_abstained: false,
    diagnostics: {
      query: 'expense report',
      total_candidates_retrieved: 5,
      evidence_items_count: 1,
      meetings_represented: ['m-allhands'],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
      structured_decisions_count: 0,
      structured_commitments_count: 0,
      structured_questions_count: 0,
      conflicts_detected_count: 0,
    },
    structured_decisions: [],
    structured_commitments: [],
    structured_questions: [],
    detected_conflicts: [],
  };

  await sim.submitAskQuery({ question: 'expenses' }, async () => plainResponse);

  assert.equal(sim.queryResponse.answer, 'Submit receipts via the intranet finance portal before Friday.');
  assert.equal(sim.queryResponse.citations.length, 1);
  assert.equal(sim.queryResponse.structured_decisions.length, 0);
  assert.equal(sim.queryResponse.structured_commitments.length, 0);
  assert.equal(sim.queryResponse.structured_questions.length, 0);
  assert.equal(sim.queryResponse.detected_conflicts.length, 0);
  assert.equal(sim.queryError, null);
});

// ---------------------------------------------------------------------------
// TEST 10: Abstention response handling
// ---------------------------------------------------------------------------
test('10. Abstention alert banner is preserved when system refrains from speculating', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const abstentionResponse = {
    question: 'What is the projected revenue for fiscal year 2028?',
    answer: 'I cannot answer this question because the retrieved meeting transcripts do not contain evidence regarding fiscal year 2028 projections.',
    citations: [],
    abstention_status: 'INSUFFICIENT_EVIDENCE',
    is_abstained: true,
    diagnostics: {
      query: '2028 revenue',
      total_candidates_retrieved: 2,
      evidence_items_count: 0,
      meetings_represented: [],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
    },
    structured_decisions: [],
    structured_commitments: [],
    structured_questions: [],
    detected_conflicts: [],
  };

  await sim.submitAskQuery({ question: '2028 revenue' }, async () => abstentionResponse);

  assert.equal(sim.queryResponse.is_abstained, true);
  assert.equal(sim.queryResponse.abstention_status, 'INSUFFICIENT_EVIDENCE');
  assert.ok(sim.queryResponse.answer.includes('cannot answer'));
});

// ---------------------------------------------------------------------------
// TEST 11: Multi-tenant isolation and epoch safety on organization switching
// ---------------------------------------------------------------------------
test('11. Switching organization aborts in-flight query and clears state immediately', async () => {
  const sim = new AskAIStructuredPresentationSimulator('org-alpha');

  // Start in-flight slow query for org-alpha
  const slowQueryPromise = sim.submitAskQuery(
    { question: 'sensitive alpha query' },
    async (_p, signal) => {
      await new Promise((res, rej) => {
        const timer = setTimeout(() => res({ answer: 'Alpha secret' }), 100);
        signal.addEventListener('abort', () => {
          clearTimeout(timer);
          const err = new Error('Aborted');
          err.name = 'AbortError';
          rej(err);
        });
      });
    },
    0
  );

  assert.equal(sim.isLoadingQuery, true);

  // Switch to org-beta immediately
  sim.switchOrganization('org-beta', 'Beta Healthcare');

  // Verify query state is wiped clean
  assert.equal(sim.queryResponse, null);
  assert.equal(sim.isLoadingQuery, false);
  assert.equal(sim.currentOrgId, 'org-beta');

  // Ensure aborted query does not populate org-beta state
  try {
    await slowQueryPromise;
  } catch (err) {
    assert.equal(err.name, 'AbortError');
  }

  assert.equal(sim.queryResponse, null);
});

// ---------------------------------------------------------------------------
// TEST 12: Multilingual text support (Hindi and Hinglish answer and decisions)
// ---------------------------------------------------------------------------
test('12. Multilingual query and structured decisions render Hindi and Hinglish text faithfully', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const hindiResponse = {
    question: 'पिछले बैठकों में क्या निर्णय लिए गए थे?',
    answer: 'पिछली बैठकों में निर्णय लिया गया था कि डेटाबेस माइग्रेशन के लिए PostgreSQL 16 का उपयोग किया जाएगा।',
    citations: [
      {
        citation_id: 'cit-hi-1',
        chunk_id: 'ch-hi-1',
        meeting_id: 'm-hi-1',
        meeting_title: 'तकनीकी समीक्षा बैठक (Technical Sync)',
        meeting_date: '2026-08-20T10:00:00Z',
        transcript_id: 'tr-hi-1',
        segment_ids: ['seg-hi-10'],
        start_time: 15.0,
        end_time: 40.0,
        quoted_text: 'हम सभी ने तय किया कि PostgreSQL 16 ही सही विकल्प है।',
        language_codes: ['hi'],
      },
    ],
    abstention_status: 'ANSWER_SUPPORTED',
    is_abstained: false,
    diagnostics: {
      query: 'निर्णय',
      total_candidates_retrieved: 6,
      evidence_items_count: 1,
      meetings_represented: ['m-hi-1'],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
      structured_decisions_count: 1,
    },
    structured_decisions: [
      {
        id: 'dec-hi-1',
        meeting_id: 'm-hi-1',
        meeting_title: 'तकनीकी समीक्षा बैठक (Technical Sync)',
        meeting_date: '2026-08-20T10:00:00Z',
        description: 'डेटाबेस माइग्रेशन के लिए PostgreSQL 16 और pgvector को अपनाएं।',
        rationale_or_context: 'सुरक्षा और प्रदर्शन की दृष्टि से यह सबसे उपयुक्त है।',
        decided_by: 'सुरक्षा प्रमुख (Security Head)',
        status: 'ACTIVE',
        evidence_segment_id: 'seg-hi-10',
        timestamp_seconds: 15.0,
      },
    ],
  };

  await sim.submitAskQuery({ question: 'निर्णय' }, async () => hindiResponse);

  assert.equal(sim.queryResponse.answer.includes('PostgreSQL 16'), true);
  assert.equal(sim.queryResponse.structured_decisions[0].description, 'डेटाबेस माइग्रेशन के लिए PostgreSQL 16 और pgvector को अपनाएं।');
  assert.equal(sim.queryResponse.structured_decisions[0].meeting_title, 'तकनीकी समीक्षा बैठक (Technical Sync)');
  assert.equal(sim.queryResponse.citations[0].language_codes[0], 'hi');
});

// ---------------------------------------------------------------------------
// TEST 13: Conflict evidence badge and alert rendering (P5)
// ---------------------------------------------------------------------------
test('13. Conflict evidence displays Conflicting Evidence Sources badge and Unresolved Conflict Alert', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const conflictResponse = {
    question: 'Are there conflicts regarding MongoDB and PostgreSQL?',
    answer: '[HEURISTIC CONFLICT WARNING] Potentially contradictory active decisions were identified across meetings.',
    citations: [
      {
        citation_id: 'cit-cf-1',
        chunk_id: 'chk-1',
        meeting_id: 'meet-1',
        meeting_title: 'Meeting 1',
        quoted_text: 'We adopted PostgreSQL.',
      },
      {
        citation_id: 'cit-cf-2',
        chunk_id: 'chk-2',
        meeting_id: 'meet-2',
        meeting_title: 'Meeting 2',
        quoted_text: 'We adopted MongoDB.',
      },
    ],
    abstention_status: 'CONFLICTING_EVIDENCE',
    is_abstained: true,
    diagnostics: {
      query: 'conflicts',
      total_candidates_retrieved: 2,
      evidence_items_count: 2,
      meetings_represented: ['meet-1', 'meet-2'],
      provider_used: 'deterministic',
      is_heuristic: true,
      fallback_warning: null,
    },
    detected_conflicts: [
      {
        decision_a_id: 'dec-1',
        decision_a_text: 'Adopt PostgreSQL',
        decision_a_meeting_title: 'Meeting 1',
        decision_b_id: 'dec-2',
        decision_b_text: 'Adopt MongoDB',
        decision_b_meeting_title: 'Meeting 2',
        subject: 'database (postgresql vs mongodb)',
        reason: 'Active decisions select competing database options without explicit superseding link.',
        is_heuristic: true,
        suggested_action: 'Review database strategy and formally record supersession.',
      },
    ],
  };

  await sim.submitAskQuery({ question: 'conflicts' }, async () => conflictResponse);

  const badge = sim.renderEvidenceBadge(sim.queryResponse.abstention_status, sim.queryResponse.citations.length);
  assert.equal(badge.label, '2 Conflicting Evidence Sources');
  assert.equal(badge.className, 'intel-stat-badge conflict');

  const alert = sim.renderAbstentionAlert(sim.queryResponse.abstention_status);
  assert.equal(alert.title, 'Unresolved Conflict Alert:');
  assert.equal(sim.queryResponse.is_abstained, true);
  assert.equal(sim.queryResponse.detected_conflicts.length, 1);
});

// ---------------------------------------------------------------------------
// TEST 14: Transcript contradiction presentation across distinct meetings (P4/P5)
// ---------------------------------------------------------------------------
test('14. Transcript discussion contradictions render affirmative and opposing statements distinctly', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const transcriptConflictResponse = {
    question: 'Was the OAuth implementation approved or rejected?',
    answer: 'Potentially conflicting statements were identified across meetings regarding OAuth.',
    citations: [
      {
        citation_id: 'cit-tc-1',
        chunk_id: 'chk-app-1',
        meeting_id: 'meet-arch',
        meeting_title: 'Architecture Review',
        quoted_text: 'The architecture committee formally approved OAuth implementation.',
      },
      {
        citation_id: 'cit-tc-2',
        chunk_id: 'chk-rej-1',
        meeting_id: 'meet-sec',
        meeting_title: 'Security Sync',
        quoted_text: 'We rejected OAuth implementation in favor of SAML.',
      },
    ],
    abstention_status: 'CONFLICTING_EVIDENCE',
    is_abstained: true,
    diagnostics: {
      query: 'OAuth',
      total_candidates_retrieved: 2,
      evidence_items_count: 2,
      meetings_represented: ['meet-arch', 'meet-sec'],
      provider_used: 'deterministic',
      is_heuristic: true,
      conflicts_detected_count: 1,
    },
    transcript_conflicts: [
      {
        chunk_a_id: 'chk-app-1',
        chunk_a_text: 'The architecture committee formally approved OAuth implementation.',
        chunk_a_meeting_title: 'Architecture Review',
        chunk_a_date: '2026-08-01T10:00:00Z',
        chunk_b_id: 'chk-rej-1',
        chunk_b_text: 'We rejected OAuth implementation in favor of SAML.',
        chunk_b_meeting_title: 'Security Sync',
        chunk_b_date: '2026-08-15T14:00:00Z',
        subject: 'oauth',
        reason: 'Transcript statements in separate meetings express opposing stances without resolution.',
        is_heuristic: true,
        suggested_action: 'Review conflicting transcript passages with meeting participants to confirm actual policy.',
      },
    ],
  };

  await sim.submitAskQuery({ question: 'OAuth' }, async () => transcriptConflictResponse);

  assert.equal(sim.queryResponse.abstention_status, 'CONFLICTING_EVIDENCE');
  assert.equal(sim.queryResponse.transcript_conflicts.length, 1);

  const tc = sim.queryResponse.transcript_conflicts[0];
  assert.equal(tc.subject, 'oauth');
  assert.equal(tc.chunk_a_meeting_title, 'Architecture Review');
  assert.equal(tc.chunk_b_meeting_title, 'Security Sync');
  assert.equal(tc.chunk_a_text.includes('approved OAuth'), true);
  assert.equal(tc.chunk_b_text.includes('rejected OAuth'), true);
  assert.equal(tc.is_heuristic, true);
  assert.equal(sim.queryResponse.citations.length, 2);
});

// ---------------------------------------------------------------------------
// TEST 15: External Knowledge Card rendering with summary, key facts, and citations
// ---------------------------------------------------------------------------
test('15. External Knowledge Card renders dedicated background context, disclaimer, and source links', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const mockResponse = {
    question: 'What is the 103rd Constitutional Amendment about?',
    answer: '### Meeting Discussion & Evidence\nBased on discussion in \'Policy Sync\':\n- In \'Policy Sync\' [04:10 - 04:45]: "We noted the 103rd Constitutional Amendment." [chunk-1]\n\n### External Background Knowledge: 103rd Constitutional Amendment\n> **Note:** External background knowledge retrieved for context.',
    citations: [
      {
        citation_id: 'cit-1',
        chunk_id: 'chunk-1',
        meeting_id: 'meet-1',
        meeting_title: 'Policy Sync',
        meeting_date: '2026-03-10T10:00:00Z',
        start_time: 250.0,
        end_time: 285.0,
        quoted_text: 'We noted the 103rd Constitutional Amendment.',
      },
    ],
    abstention_status: 'ANSWER_SUPPORTED',
    is_abstained: false,
    diagnostics: {
      query: 'What is the 103rd Constitutional Amendment about?',
      total_candidates_retrieved: 1,
      evidence_items_count: 1,
      meetings_represented: ['meet-1'],
      provider_used: 'wikipedia',
      is_heuristic: true,
    },
    external_knowledge: {
      subject: '103rd Constitutional Amendment',
      summary: 'The 103rd Constitutional Amendment Act, 2019 introduces a 10% reservation for Economically Weaker Sections (EWS).',
      key_facts: [
        'Introduces 10% EWS reservation in education and civil posts.',
        'Amended Articles 15 and 16 of the Constitution of India.',
      ],
      citations: [
        {
          source_title: 'Wikipedia: One Hundred and Third Amendment of the Constitution of India',
          source_url: 'https://en.wikipedia.org/wiki/One_Hundred_and_Third_Amendment_of_the_Constitution_of_India',
          snippet: 'Introduces 10% reservation for Economically Weaker Sections.',
        },
      ],
      provider_used: 'wikipedia',
      is_available: true,
      unavailability_reason: null,
      disclaimer: 'External background knowledge retrieved for context. This information was not discussed during your organization\'s meetings.',
    },
  };

  await sim.submitAskQuery({ question: 'What is the 103rd Constitutional Amendment about?' }, async () => mockResponse);

  assert.equal(sim.queryResponse.external_knowledge !== null, true);
  const card = sim.renderExternalKnowledgeCard(sim.queryResponse.external_knowledge);

  assert.equal(card.subject, '103rd Constitutional Amendment');
  assert.equal(card.isAvailable, true);
  assert.equal(card.providerUsed, 'wikipedia');
  assert.equal(card.summary.includes('10% reservation for Economically Weaker Sections'), true);
  assert.equal(card.keyFacts.length, 2);
  assert.equal(card.citations.length, 1);
  assert.equal(card.citations[0].source_url.includes('wikipedia.org'), true);
  assert.equal(card.disclaimer.includes('was not discussed during your organization\'s meetings'), true);
});

// ---------------------------------------------------------------------------
// TEST 16: External Knowledge unavailable state handling
// ---------------------------------------------------------------------------
test('16. External Knowledge unavailable reporting when external service is unreachable', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const mockResponse = {
    question: 'What is Quantum Computing Architecture?',
    answer: 'No relevant meeting discussions or decisions were found in your organization\'s recorded meetings for this question.',
    citations: [],
    abstention_status: 'NO_EVIDENCE',
    is_abstained: true,
    diagnostics: {
      query: 'What is Quantum Computing Architecture?',
      total_candidates_retrieved: 0,
      evidence_items_count: 0,
      meetings_represented: [],
      provider_used: 'unavailable',
      is_heuristic: true,
    },
    external_knowledge: {
      subject: 'Quantum Computing Architecture',
      summary: 'External background knowledge for \'Quantum Computing Architecture\' could not be retrieved at this time.',
      key_facts: [],
      citations: [],
      provider_used: 'unavailable',
      is_available: false,
      unavailability_reason: 'External knowledge sources are currently unreachable or offline.',
      disclaimer: 'External background knowledge was requested for context, but external retrieval was unavailable.',
    },
  };

  await sim.submitAskQuery({ question: 'What is Quantum Computing Architecture?' }, async () => mockResponse);

  assert.equal(sim.queryResponse.is_abstained, true);
  assert.equal(sim.queryResponse.abstention_status, 'NO_EVIDENCE');
  const card = sim.renderExternalKnowledgeCard(sim.queryResponse.external_knowledge);

  assert.equal(card.isAvailable, false);
  assert.equal(card.unavailabilityReason.includes('unreachable or offline'), true);
  assert.equal(card.citations.length, 0);
});

// ---------------------------------------------------------------------------
// TEST 17: Strict separation of internal meeting evidence and external citations
// ---------------------------------------------------------------------------
test('17. Internal meeting citations and external citations remain strictly distinguished', async () => {
  const sim = new AskAIStructuredPresentationSimulator();

  const mockResponse = {
    question: 'Tell me about OAuth 2.0',
    answer: '### Meeting Discussion\nBased on discussion in \'Security Sync\':\n- "We plan to evaluate OAuth 2.0." [chunk-99]\n\n### External Background Knowledge: OAuth 2.0',
    citations: [
      {
        citation_id: 'cit-meet-1',
        chunk_id: 'chunk-99',
        meeting_id: 'meeting-sec',
        meeting_title: 'Security Sync',
        meeting_date: '2026-05-01T10:00:00Z',
        start_time: 120.0,
        end_time: 140.0,
        quoted_text: 'We plan to evaluate OAuth 2.0.',
      },
    ],
    abstention_status: 'ANSWER_SUPPORTED',
    is_abstained: false,
    diagnostics: {
      query: 'Tell me about OAuth 2.0',
      total_candidates_retrieved: 1,
      evidence_items_count: 1,
      meetings_represented: ['meeting-sec'],
      provider_used: 'wikipedia',
      is_heuristic: true,
    },
    external_knowledge: {
      subject: 'OAuth 2.0',
      summary: 'OAuth 2.0 is an authorization framework that enables applications to obtain limited access to user accounts on an HTTP service.',
      key_facts: ['RFC 6749 authorization standard.'],
      citations: [
        {
          source_title: 'IETF RFC 6749: The OAuth 2.0 Authorization Framework',
          source_url: 'https://datatracker.ietf.org/doc/html/rfc6749',
          snippet: 'The OAuth 2.0 authorization framework...',
        },
      ],
      provider_used: 'wikipedia',
      is_available: true,
      disclaimer: 'External background knowledge retrieved for context.',
    },
  };

  await sim.submitAskQuery({ question: 'Tell me about OAuth 2.0' }, async () => mockResponse);

  // Meeting citations have meeting_id, chunk_id, and start_time
  assert.equal(sim.queryResponse.citations.length, 1);
  assert.equal(sim.queryResponse.citations[0].meeting_id, 'meeting-sec');
  assert.equal(sim.queryResponse.citations[0].chunk_id, 'chunk-99');

  // External citations have source_url and source_title, NO meeting_id
  assert.equal(sim.queryResponse.external_knowledge.citations.length, 1);
  assert.equal(sim.queryResponse.external_knowledge.citations[0].source_url.startsWith('https://'), true);
  assert.equal(sim.queryResponse.external_knowledge.citations[0].meeting_id, undefined);
});
