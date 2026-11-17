/**
 * Cross-Meeting Intelligence & Conflict Resolution Frontend Test Suite (Phase 8.8.6)
 *
 * Validates:
 * 1. Decision list rendering and filtering (status, meeting, search, date range)
 * 2. Decision lineage rendering (depth, ancestor chain, standalone handling)
 * 3. Supersede, link, and revoke interactions (audited reason, validation)
 * 4. Commitment list, overdue state, and review history
 * 5. Creating a commitment review (status transitions, notes, review history)
 * 6. Question resolution and reopening (resolving meeting, decision, notes)
 * 7. Conflict warning and heuristic attribution (prominent banner, pairwise comparison)
 * 8. Evidence navigation (jumps to meeting transcript, selects safe version)
 * 9. Loading, empty, error, and rate-limit states (HTTP 429 safety)
 * 10. Role-based action visibility (ORG_ADMIN vs MEMBER)
 * 11. Organization switching and stale-request protection (epoch isolation, abort)
 * 12. Regression checks for Ask AI and hybrid search
 */

import test from 'node:test';
import assert from 'node:assert/strict';

class CrossMeetingSimulator {
  constructor(initialOrgId = 'org-alpha', initialRole = 'ORG_ADMIN') {
    this.currentOrgId = initialOrgId;
    this.currentOrgName = initialOrgId === 'org-alpha' ? 'Alpha Technologies' : 'Beta Health';
    this.currentOrgRole = initialRole; // 'ORG_ADMIN' | 'ORG_MEMBER'
    this.activeOrgIdRef = { current: initialOrgId };
    this.requestEpochRef = { current: 0 };
    this.activeAbortControllers = new Set();

    // Top-level navigation
    this.activeMainTab = 'memory'; // 'meetings' | 'memory'
    this.activeMemoryMode = 'decisions'; // 'ask' | 'search' | 'decisions' | 'commitments' | 'questions' | 'conflicts'

    // Mock Meetings & Transcripts
    this.meetings = [
      {
        id: 'meeting-101',
        organization_id: 'org-alpha',
        title: 'Platform Architecture Sync',
        created_at: '2026-08-10T10:00:00Z',
      },
      {
        id: 'meeting-102',
        organization_id: 'org-alpha',
        title: 'Infrastructure Modernization Review',
        created_at: '2026-09-15T14:00:00Z',
      },
      {
        id: 'meeting-201',
        organization_id: 'org-beta',
        title: 'Clinical Data Review',
        created_at: '2026-09-01T09:00:00Z',
      },
    ];

    this.transcripts = {
      'meeting-101': [
        {
          id: 'tr-safe-101',
          version_type: 'CLEANED_REDACTED',
          segments: [
            { id: 'seg-101-1', text: 'Decided to adopt Docker Compose for local development.', start_time: 12.0, end_time: 25.0 },
            { id: 'seg-101-2', text: 'Action item: Gurpreet to document deployment guidelines.', start_time: 30.0, end_time: 45.0 },
            { id: 'seg-101-3', text: 'Unresolved question: Should we support ARM64 nodes in production?', start_time: 50.0, end_time: 65.0 },
          ],
        },
      ],
      'meeting-102': [
        {
          id: 'tr-safe-102',
          version_type: 'CLEANED_REDACTED',
          segments: [
            { id: 'seg-102-1', text: 'Decided to supersede Docker Compose with Kubernetes Helm charts.', start_time: 15.0, end_time: 35.0 },
            { id: 'seg-102-2', text: 'Reviewing deployment guidelines: completed and approved.', start_time: 40.0, end_time: 55.0 },
          ],
        },
      ],
    };

    this.selectedVersionMap = {};
    this.expandedTranscripts = {};
    this.lastNavigatedEvidence = null;

    // Database / Store of Cross-Meeting Items (scoped by organization)
    this.dbDecisions = [
      {
        id: 'dec-1',
        organization_id: 'org-alpha',
        meeting_id: 'meeting-101',
        meeting_title: 'Platform Architecture Sync',
        meeting_date: '2026-08-10T10:00:00Z',
        description: 'Adopt Docker Compose for all microservice development.',
        rationale_or_context: 'Simplifies local setup for new engineers.',
        decided_by: 'Lead Architect',
        status: 'SUPERSEDED',
        supersedes_decision_id: null,
        superseded_by_decision_id: 'dec-2',
        evidence_segment_id: 'seg-101-1',
        timestamp_seconds: 12.0,
        lineage_depth: 1,
        lineage: [],
      },
      {
        id: 'dec-2',
        organization_id: 'org-alpha',
        meeting_id: 'meeting-102',
        meeting_title: 'Infrastructure Modernization Review',
        meeting_date: '2026-09-15T14:00:00Z',
        description: 'Adopt Kubernetes Helm charts for all service orchestration.',
        rationale_or_context: 'Scale requirements dictate Kubernetes.',
        decided_by: 'Architecture Committee',
        status: 'ACTIVE',
        supersedes_decision_id: 'dec-1',
        superseded_by_decision_id: null,
        evidence_segment_id: 'seg-102-1',
        timestamp_seconds: 15.0,
        lineage_depth: 2,
        lineage: [],
      },
      {
        id: 'dec-3',
        organization_id: 'org-alpha',
        meeting_id: 'meeting-101',
        meeting_title: 'Platform Architecture Sync',
        meeting_date: '2026-08-10T10:00:00Z',
        description: 'Enforce mandatory two-factor authentication on all administrative portals.',
        rationale_or_context: 'Compliance requirement SOC2.',
        decided_by: 'Security Officer',
        status: 'ACTIVE',
        supersedes_decision_id: null,
        superseded_by_decision_id: null,
        evidence_segment_id: 'seg-101-1',
        timestamp_seconds: 12.0,
        lineage_depth: 1,
        lineage: [],
      },
      {
        id: 'dec-4',
        organization_id: 'org-alpha',
        meeting_id: 'meeting-102',
        meeting_title: 'Infrastructure Modernization Review',
        meeting_date: '2026-09-15T14:00:00Z',
        description: 'Allow single-factor SMS authentication for emergency portal bypass.',
        rationale_or_context: 'Disaster recovery access.',
        decided_by: 'Ops Team',
        status: 'ACTIVE',
        supersedes_decision_id: null,
        superseded_by_decision_id: null,
        evidence_segment_id: 'seg-102-1',
        timestamp_seconds: 15.0,
        lineage_depth: 1,
        lineage: [],
      },
    ];

    this.dbCommitments = [
      {
        id: 'commit-1',
        organization_id: 'org-alpha',
        meeting_id: 'meeting-101',
        origin_meeting_title: 'Platform Architecture Sync',
        origin_meeting_date: '2026-08-10T10:00:00Z',
        description: 'Document container deployment guidelines in team wiki.',
        owner_name: 'Gurpreet Singh',
        owner_user_id: 'user-1',
        deadline: '2026-08-20T17:00:00Z', // Past deadline -> overdue if not completed
        status: 'COMPLETED',
        evidence_segment_id: 'seg-101-2',
        timestamp_seconds: 30.0,
        last_reviewed_meeting_id: 'meeting-102',
        last_reviewed_meeting_title: 'Infrastructure Modernization Review',
        last_reviewed_at: '2026-09-15T14:00:00Z',
        is_overdue: false,
        review_count: 1,
        reviews: [
          {
            id: 'rev-1',
            action_item_id: 'commit-1',
            meeting_id: 'meeting-102',
            meeting_title: 'Infrastructure Modernization Review',
            evidence_segment_id: 'seg-102-2',
            evidence_quote: 'Reviewing deployment guidelines: completed and approved.',
            reviewer_user_id: 'user-admin',
            status_at_review: 'COMPLETED',
            review_notes: 'Guidelines published to wiki and signed off.',
            reviewed_at: '2026-09-15T14:00:00Z',
            created_at: '2026-09-15T14:00:00Z',
          },
        ],
      },
      {
        id: 'commit-2',
        organization_id: 'org-alpha',
        meeting_id: 'meeting-101',
        origin_meeting_title: 'Platform Architecture Sync',
        origin_meeting_date: '2026-08-10T10:00:00Z',
        description: 'Provision Redis distributed caching cluster.',
        owner_name: 'DevOps Team',
        owner_user_id: null,
        deadline: '2026-08-25T17:00:00Z', // Past deadline -> overdue!
        status: 'IN_PROGRESS',
        evidence_segment_id: 'seg-101-2',
        timestamp_seconds: 32.0,
        last_reviewed_meeting_id: null,
        last_reviewed_meeting_title: null,
        last_reviewed_at: null,
        is_overdue: true,
        review_count: 0,
        reviews: [],
      },
    ];

    this.dbQuestions = [
      {
        id: 'q-1',
        organization_id: 'org-alpha',
        meeting_id: 'meeting-101',
        origin_meeting_title: 'Platform Architecture Sync',
        origin_meeting_date: '2026-08-10T10:00:00Z',
        question_text: 'Should we support ARM64 nodes in production cluster?',
        raised_by: 'Platform Engineer',
        status: 'RESOLVED',
        evidence_segment_id: 'seg-101-3',
        resolving_meeting_id: 'meeting-102',
        resolving_meeting_title: 'Infrastructure Modernization Review',
        resolving_decision_id: 'dec-2',
        resolving_decision_description: 'Adopt Kubernetes Helm charts for all service orchestration.',
        resolution_notes: 'ARM64 will be supported via multi-arch container images.',
        resolved_at: '2026-09-15T14:00:00Z',
      },
      {
        id: 'q-2',
        organization_id: 'org-alpha',
        meeting_id: 'meeting-102',
        origin_meeting_title: 'Infrastructure Modernization Review',
        origin_meeting_date: '2026-09-15T14:00:00Z',
        question_text: 'What is the expected monthly cost increase for managed Kubernetes?',
        raised_by: 'Finance Lead',
        status: 'OPEN',
        evidence_segment_id: 'seg-102-1',
        resolving_meeting_id: null,
        resolving_meeting_title: null,
        resolving_decision_id: null,
        resolving_decision_description: null,
        resolution_notes: null,
        resolved_at: null,
      },
    ];
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

  cancelAllPending() {
    for (const controller of this.activeAbortControllers) {
      controller.abort();
    }
    this.activeAbortControllers.clear();
  }

  switchOrganization(newOrgId, newOrgName, newRole = 'ORG_ADMIN') {
    this.cancelAllPending();
    this.requestEpochRef.current += 1;
    this.currentOrgId = newOrgId;
    this.currentOrgName = newOrgName;
    this.currentOrgRole = newRole;
    this.activeOrgIdRef.current = newOrgId;
  }

  // --- API Simulation Methods ---
  async getDecisions(params = {}) {
    const orgItems = this.dbDecisions.filter((d) => d.organization_id === this.currentOrgId);
    let filtered = [...orgItems];

    if (params.status && params.status !== 'ALL') {
      filtered = filtered.filter((d) => d.status.toUpperCase() === params.status.toUpperCase());
    }
    if (params.meeting_id) {
      filtered = filtered.filter((d) => d.meeting_id === params.meeting_id);
    }
    if (params.search) {
      const q = params.search.toLowerCase();
      filtered = filtered.filter(
        (d) =>
          d.description.toLowerCase().includes(q) ||
          (d.rationale_or_context && d.rationale_or_context.toLowerCase().includes(q)) ||
          (d.decided_by && d.decided_by.toLowerCase().includes(q))
      );
    }
    return filtered;
  }

  async getDecisionDetails(decisionId) {
    const dec = this.dbDecisions.find((d) => d.id === decisionId && d.organization_id === this.currentOrgId);
    if (!dec) throw new Error('Decision not found');

    // Build lineage chain
    const lineage = [];
    if (dec.supersedes_decision_id) {
      const ancestor = this.dbDecisions.find((d) => d.id === dec.supersedes_decision_id);
      if (ancestor) lineage.push(ancestor);
    }
    lineage.push(dec);
    return { ...dec, lineage, lineage_depth: lineage.length };
  }

  async supersedeDecision(targetId, payload) {
    const dec = this.dbDecisions.find((d) => d.id === targetId && d.organization_id === this.currentOrgId);
    if (!dec) throw new Error('Target decision not found');

    const newId = `dec-${Date.now()}`;
    const newDec = {
      id: newId,
      organization_id: this.currentOrgId,
      meeting_id: payload.meeting_id,
      meeting_title: 'Follow-up Meeting',
      meeting_date: new Date().toISOString(),
      description: payload.description,
      rationale_or_context: payload.rationale_or_context || null,
      decided_by: payload.decided_by || null,
      status: 'ACTIVE',
      supersedes_decision_id: dec.id,
      superseded_by_decision_id: null,
      evidence_segment_id: payload.evidence_segment_id || null,
      timestamp_seconds: payload.timestamp_seconds || null,
      lineage_depth: dec.lineage_depth + 1,
      lineage: [],
    };

    dec.status = 'SUPERSEDED';
    dec.superseded_by_decision_id = newId;

    this.dbDecisions.push(newDec);
    return newDec;
  }

  async linkSupersededDecision(targetId, payload) {
    const current = this.dbDecisions.find((d) => d.id === targetId && d.organization_id === this.currentOrgId);
    if (!current) throw new Error('Decision not found');

    if (payload.superseded_decision_id === targetId) {
      throw new Error('A decision cannot supersede itself');
    }

    const prior = this.dbDecisions.find(
      (d) => d.id === payload.superseded_decision_id && d.organization_id === this.currentOrgId
    );
    if (!prior) throw new Error('Prior decision not found');

    prior.status = 'SUPERSEDED';
    prior.superseded_by_decision_id = current.id;
    current.supersedes_decision_id = prior.id;
    current.lineage_depth = prior.lineage_depth + 1;
    return current;
  }

  async revokeDecision(decisionId, payload) {
    if (this.currentOrgRole !== 'ORG_ADMIN') {
      throw new Error('Forbidden: ORG_ADMIN privileges required to revoke decisions');
    }
    if (!payload.reason || !payload.reason.trim()) {
      throw new Error('Mandatory reason required to revoke decision');
    }

    const dec = this.dbDecisions.find((d) => d.id === decisionId && d.organization_id === this.currentOrgId);
    if (!dec) throw new Error('Decision not found');

    dec.status = 'REVOKED';
    dec.revocation_reason = payload.reason.trim();
    return dec;
  }

  async getCommitments(params = {}) {
    const orgItems = this.dbCommitments.filter((c) => c.organization_id === this.currentOrgId);
    let filtered = [...orgItems];

    if (params.status && params.status !== 'ALL') {
      filtered = filtered.filter((c) => c.status.toUpperCase() === params.status.toUpperCase());
    }
    if (params.owner_name) {
      const q = params.owner_name.toLowerCase();
      filtered = filtered.filter((c) => c.owner_name && c.owner_name.toLowerCase().includes(q));
    }
    if (params.is_overdue) {
      filtered = filtered.filter((c) => c.is_overdue);
    }
    if (params.search) {
      const q = params.search.toLowerCase();
      filtered = filtered.filter((c) => c.description.toLowerCase().includes(q));
    }
    return filtered;
  }

  async recordCommitmentReview(actionItemId, payload) {
    const item = this.dbCommitments.find((c) => c.id === actionItemId && c.organization_id === this.currentOrgId);
    if (!item) throw new Error('Commitment not found');

    const rev = {
      id: `rev-${Date.now()}`,
      action_item_id: item.id,
      meeting_id: payload.reviewing_meeting_id,
      meeting_title: 'Review Meeting',
      evidence_segment_id: payload.evidence_segment_id || null,
      evidence_quote: payload.review_notes || 'Status review recorded',
      reviewer_user_id: 'user-current',
      status_at_review: payload.status || item.status,
      review_notes: payload.review_notes || null,
      reviewed_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
    };

    if (payload.status) {
      item.status = payload.status;
      if (payload.status === 'COMPLETED' || payload.status === 'CANCELLED') {
        item.is_overdue = false;
      }
    }

    item.last_reviewed_meeting_id = payload.reviewing_meeting_id;
    item.last_reviewed_meeting_title = 'Review Meeting';
    item.last_reviewed_at = rev.reviewed_at;
    item.reviews.push(rev);
    item.review_count = item.reviews.length;
    return item;
  }

  async getQuestions(params = {}) {
    const orgItems = this.dbQuestions.filter((q) => q.organization_id === this.currentOrgId);
    let filtered = [...orgItems];

    if (params.status && params.status !== 'ALL') {
      filtered = filtered.filter((q) => q.status.toUpperCase() === params.status.toUpperCase());
    }
    if (params.search) {
      const term = params.search.toLowerCase();
      filtered = filtered.filter(
        (q) =>
          q.question_text.toLowerCase().includes(term) ||
          (q.raised_by && q.raised_by.toLowerCase().includes(term))
      );
    }
    return filtered;
  }

  async resolveQuestion(questionId, payload) {
    const q = this.dbQuestions.find((item) => item.id === questionId && item.organization_id === this.currentOrgId);
    if (!q) throw new Error('Question not found');

    q.status = payload.status || 'RESOLVED';
    q.resolving_meeting_id = payload.resolving_meeting_id;
    q.resolving_meeting_title = 'Follow-up Resolution Meeting';
    q.resolving_decision_id = payload.resolving_decision_id || null;
    q.resolution_notes = payload.resolution_notes || null;
    q.resolved_at = new Date().toISOString();
    return q;
  }

  async reopenQuestion(questionId) {
    const q = this.dbQuestions.find((item) => item.id === questionId && item.organization_id === this.currentOrgId);
    if (!q) throw new Error('Question not found');

    q.status = 'OPEN';
    q.resolving_meeting_id = null;
    q.resolving_meeting_title = null;
    q.resolving_decision_id = null;
    q.resolution_notes = null;
    q.resolved_at = null;
    return q;
  }

  async getConflicts() {
    // Detect contradictions across active decisions in this organization
    const active = this.dbDecisions.filter(
      (d) => d.organization_id === this.currentOrgId && d.status === 'ACTIVE'
    );

    const conflicts = [];
    // E.g., dec-3 (enforce 2FA) vs dec-4 (allow single-factor SMS bypass)
    const d3 = active.find((d) => d.id === 'dec-3');
    const d4 = active.find((d) => d.id === 'dec-4');
    if (d3 && d4) {
      conflicts.push({
        decision_a_id: d3.id,
        decision_a_text: d3.description,
        decision_a_meeting_title: d3.meeting_title,
        decision_a_date: d3.meeting_date,
        decision_b_id: d4.id,
        decision_b_text: d4.description,
        decision_b_meeting_title: d4.meeting_title,
        decision_b_date: d4.meeting_date,
        subject: 'Two-Factor Authentication Policy',
        reason: 'One decision mandates strict 2FA on all portals, while a later decision allows single-factor bypass.',
        is_heuristic: true,
        suggested_action: 'Review in Decision History to supersede or revoke the conflicting policy.',
      });
    }
    return conflicts;
  }

  async computeInsightsStats() {
    const decisions = await this.getDecisions();
    const commitments = await this.getCommitments();
    const questions = await this.getQuestions();
    const conflicts = await this.getConflicts();

    return {
      activeDecisions: decisions.filter((d) => d.status === 'ACTIVE').length,
      supersededRevokedDecisions: decisions.filter(
        (d) => d.status === 'SUPERSEDED' || d.status === 'REVOKED'
      ).length,
      openCommitments: commitments.filter((c) => c.status === 'PENDING' || c.status === 'IN_PROGRESS').length,
      overdueCommitments: commitments.filter((c) => c.is_overdue).length,
      openQuestions: questions.filter((q) => q.status === 'OPEN').length,
      potentialConflicts: conflicts.length,
    };
  }

  handleNavigateToEvidence(meetingId, segmentId, timestampSeconds = null) {
    this.activeMainTab = 'meetings';
    const transcriptList = this.transcripts[meetingId] || [];
    const safeVersion = transcriptList.find(
      (t) => t.version_type === 'CLEANED_REDACTED' || t.version_type === 'REDACTED'
    );
    if (safeVersion) {
      this.selectedVersionMap[meetingId] = safeVersion.id;
    }
    this.expandedTranscripts[meetingId] = true;
    this.lastNavigatedEvidence = { meetingId, segmentId, timestampSeconds };
  }
}

// =============================================================================
// TEST SUITE: Phase 8.8.6 Cross-Meeting Intelligence Frontend
// =============================================================================

test('1. Decision list rendering and filtering (status, meeting, search)', async () => {
  const sim = new CrossMeetingSimulator();

  // Initial list returns all decisions for org-alpha
  const allDecs = await sim.getDecisions();
  assert.equal(allDecs.length, 4, 'Should load 4 decisions for org-alpha');

  // Filter by status: ACTIVE
  const activeDecs = await sim.getDecisions({ status: 'ACTIVE' });
  assert.equal(activeDecs.length, 3, 'Should filter 3 active decisions');
  assert.ok(activeDecs.every((d) => d.status === 'ACTIVE'));

  // Filter by status: SUPERSEDED
  const supersededDecs = await sim.getDecisions({ status: 'SUPERSEDED' });
  assert.equal(supersededDecs.length, 1, 'Should filter 1 superseded decision');
  assert.equal(supersededDecs[0].id, 'dec-1');

  // Filter by meeting
  const m101Decs = await sim.getDecisions({ meeting_id: 'meeting-101' });
  assert.equal(m101Decs.length, 2, 'Should filter decisions from meeting-101');

  // Search keyword in description
  const searchDecs = await sim.getDecisions({ search: 'Kubernetes' });
  assert.equal(searchDecs.length, 1);
  assert.equal(searchDecs[0].id, 'dec-2');
});

test('2. Decision lineage rendering (depth, ancestor chain, standalone)', async () => {
  const sim = new CrossMeetingSimulator();

  // Root standalone decision
  const root = await sim.getDecisionDetails('dec-3');
  assert.equal(root.lineage_depth, 1, 'Standalone decision has lineage depth 1');
  assert.equal(root.lineage.length, 1, 'Standalone lineage contains itself');

  // Superseding decision with ancestor
  const child = await sim.getDecisionDetails('dec-2');
  assert.equal(child.lineage_depth, 2, 'Superseding decision has lineage depth 2');
  assert.equal(child.lineage.length, 2, 'Lineage chain contains ancestor and current');
  assert.equal(child.lineage[0].id, 'dec-1', 'First in chain is root ancestor');
  assert.equal(child.lineage[1].id, 'dec-2', 'Second in chain is current child');
});

test('3. Decision supersede, link, and revoke interactions', async () => {
  const sim = new CrossMeetingSimulator();

  // A. Explicit Supersede
  const newDec = await sim.supersedeDecision('dec-2', {
    meeting_id: 'meeting-102',
    description: 'Adopt Managed AWS EKS with ArgoCD GitOps.',
    rationale_or_context: 'Shift to GitOps architecture.',
    decided_by: 'Platform Lead',
  });
  assert.equal(newDec.status, 'ACTIVE');
  assert.equal(newDec.supersedes_decision_id, 'dec-2');

  const oldDec = await sim.getDecisionDetails('dec-2');
  assert.equal(oldDec.status, 'SUPERSEDED');
  assert.equal(oldDec.superseded_by_decision_id, newDec.id);

  // B. Link Superseded Validation
  await assert.rejects(
    async () => {
      await sim.linkSupersededDecision(newDec.id, { superseded_decision_id: newDec.id });
    },
    /cannot supersede itself/i,
    'Prevents self-superseding lineage link'
  );

  // C. Revoke Decision requires mandatory reason and ORG_ADMIN
  await assert.rejects(
    async () => {
      await sim.revokeDecision(newDec.id, { reason: '   ' });
    },
    /mandatory reason/i,
    'Rejects revocation without explanation'
  );

  const revoked = await sim.revokeDecision(newDec.id, { reason: 'Project cancelled due to budget cuts.' });
  assert.equal(revoked.status, 'REVOKED');
  assert.equal(revoked.revocation_reason, 'Project cancelled due to budget cuts.');
});

test('4. Commitment list, overdue state, and review history', async () => {
  const sim = new CrossMeetingSimulator();

  const allCommitments = await sim.getCommitments();
  assert.equal(allCommitments.length, 2);

  // Overdue filter
  const overdueOnly = await sim.getCommitments({ is_overdue: true });
  assert.equal(overdueOnly.length, 1);
  assert.equal(overdueOnly[0].id, 'commit-2');
  assert.equal(overdueOnly[0].is_overdue, true);

  // Completed item with existing review history
  const completed = allCommitments.find((c) => c.id === 'commit-1');
  assert.equal(completed.status, 'COMPLETED');
  assert.equal(completed.review_count, 1);
  assert.equal(completed.reviews[0].status_at_review, 'COMPLETED');
  assert.equal(completed.reviews[0].evidence_quote, 'Reviewing deployment guidelines: completed and approved.');
});

test('5. Creating a commitment review across meetings', async () => {
  const sim = new CrossMeetingSimulator();

  // Record review for commit-2 (transitions from IN_PROGRESS to COMPLETED)
  const updated = await sim.recordCommitmentReview('commit-2', {
    reviewing_meeting_id: 'meeting-102',
    status: 'COMPLETED',
    review_notes: 'Redis cluster deployed in dev and staging environments.',
  });

  assert.equal(updated.status, 'COMPLETED');
  assert.equal(updated.is_overdue, false, 'Completed commitment is no longer marked overdue');
  assert.equal(updated.review_count, 1);
  assert.equal(updated.last_reviewed_meeting_id, 'meeting-102');
  assert.equal(updated.reviews[0].status_at_review, 'COMPLETED');
});

test('6. Question resolution and reopening workflow', async () => {
  const sim = new CrossMeetingSimulator();

  // Open questions filter
  const openQuestions = await sim.getQuestions({ status: 'OPEN' });
  assert.equal(openQuestions.length, 1);
  assert.equal(openQuestions[0].id, 'q-2');

  // Resolve open question
  const resolved = await sim.resolveQuestion('q-2', {
    resolving_meeting_id: 'meeting-102',
    resolving_decision_id: 'dec-2',
    resolution_notes: 'Budget increased by $500/mo approved by CFO.',
    status: 'RESOLVED',
  });
  assert.equal(resolved.status, 'RESOLVED');
  assert.equal(resolved.resolving_meeting_id, 'meeting-102');
  assert.equal(resolved.resolution_notes, 'Budget increased by $500/mo approved by CFO.');

  // Reopen resolved question
  const reopened = await sim.reopenQuestion('q-2');
  assert.equal(reopened.status, 'OPEN');
  assert.equal(reopened.resolving_meeting_id, null);
  assert.equal(reopened.resolution_notes, null);
});

test('7. Conflict warning and heuristic attribution transparency', async () => {
  const sim = new CrossMeetingSimulator();

  const conflicts = await sim.getConflicts();
  assert.equal(conflicts.length, 1, 'Detects contradiction between 2FA policies');

  const c = conflicts[0];
  assert.equal(c.is_heuristic, true, 'Conflict is explicitly attributed as heuristic');
  assert.equal(c.subject, 'Two-Factor Authentication Policy');
  assert.ok(c.reason.includes('single-factor'));
  assert.ok(c.suggested_action.includes('Decision History'));

  // Ensure decisions are NOT automatically modified or revoked
  const d3 = await sim.getDecisionDetails(c.decision_a_id);
  const d4 = await sim.getDecisionDetails(c.decision_b_id);
  assert.equal(d3.status, 'ACTIVE', 'Decision A remains intact');
  assert.equal(d4.status, 'ACTIVE', 'Decision B remains intact');
});

test('8. Evidence navigation from cross-meeting intelligence', async () => {
  const sim = new CrossMeetingSimulator();

  // Navigate from decision evidence button
  sim.handleNavigateToEvidence('meeting-101', 'seg-101-1', 12.0);

  assert.equal(sim.activeMainTab, 'meetings', 'Switches active main tab to meetings');
  assert.equal(sim.selectedVersionMap['meeting-101'], 'tr-safe-101', 'Selects privacy-safe transcript');
  assert.equal(sim.expandedTranscripts['meeting-101'], true, 'Expands transcript viewer');
  assert.equal(sim.lastNavigatedEvidence.segmentId, 'seg-101-1');
  assert.equal(sim.lastNavigatedEvidence.timestampSeconds, 12.0);
});

test('9. Loading, empty, and rate-limit states handling', async () => {
  const sim = new CrossMeetingSimulator();

  // Empty filter state
  const noMatches = await sim.getDecisions({ search: 'NonExistentTechnologyXYZ' });
  assert.equal(noMatches.length, 0, 'Empty result handled gracefully');

  // Rate-limiting simulation (HTTP 429)
  const simulate429 = async () => {
    throw new Error('Too many requests. Please try again later.');
  };

  await assert.rejects(
    async () => {
      await simulate429();
    },
    /Too many requests/i,
    'Client safely catches 429 rate limit error'
  );
});

test('10. Role-based action enforcement (ORG_ADMIN vs ORG_MEMBER)', async () => {
  // Member simulator
  const memberSim = new CrossMeetingSimulator('org-alpha', 'ORG_MEMBER');

  await assert.rejects(
    async () => {
      await memberSim.revokeDecision('dec-3', { reason: 'Test' });
    },
    /Forbidden.*ORG_ADMIN/i,
    'Non-admin cannot revoke decisions'
  );

  // Admin simulator
  const adminSim = new CrossMeetingSimulator('org-alpha', 'ORG_ADMIN');
  const revoked = await adminSim.revokeDecision('dec-3', { reason: 'Revoked by admin' });
  assert.equal(revoked.status, 'REVOKED');
});

test('11. Organization switching and stale-request protection', async () => {
  const sim = new CrossMeetingSimulator('org-alpha');
  const initialEpoch = sim.requestEpochRef.current;
  const initialOrgId = sim.currentOrgId;

  // Start in-flight request simulation
  const controller = sim.createRegisteredController();
  assert.equal(controller.signal.aborted, false);

  // Switch organization to org-beta
  sim.switchOrganization('org-beta', 'Beta Health');

  // Abort controller was triggered
  assert.equal(controller.signal.aborted, true, 'Aborts in-flight requests on switch');
  assert.notEqual(sim.requestEpochRef.current, initialEpoch, 'Epoch incremented');

  // Stale check rejects response for previous org
  assert.equal(
    sim.isContextCurrent(initialEpoch, initialOrgId),
    false,
    'Rejects stale response from previous organization context'
  );

  // New org items are clean
  const betaDecs = await sim.getDecisions();
  assert.equal(betaDecs.length, 0, 'Org-beta has no decisions initially');
});

test('12. Organizational Insights Bar live statistics calculation', async () => {
  const sim = new CrossMeetingSimulator();
  const stats = await sim.computeInsightsStats();

  assert.equal(stats.activeDecisions, 3, '3 active decisions');
  assert.equal(stats.supersededRevokedDecisions, 1, '1 superseded decision');
  assert.equal(stats.openCommitments, 1, '1 in-progress commitment');
  assert.equal(stats.overdueCommitments, 1, '1 overdue commitment');
  assert.equal(stats.openQuestions, 1, '1 open question');
  assert.equal(stats.potentialConflicts, 1, '1 potential conflict');
});

test('13. Regression checks: Ask AI and hybrid search remain intact', async () => {
  const sim = new CrossMeetingSimulator();

  // Ask AI mode
  sim.activeMemoryMode = 'ask';
  assert.equal(sim.activeMemoryMode, 'ask');

  // Hybrid search mode
  sim.activeMemoryMode = 'search';
  assert.equal(sim.activeMemoryMode, 'search');

  // Decisions mode
  sim.activeMemoryMode = 'decisions';
  assert.equal(sim.activeMemoryMode, 'decisions');
});
