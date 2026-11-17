import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  HelpCircle,
  CheckCircle2,
  Clock,
  ArrowRight,
  RefreshCw,
  AlertOctagon,
  RotateCcw,
} from 'lucide-react';
import type {
  CrossMeetingQuestionItem,
  MeetingItem,
  QuestionFilterParams,
  QuestionResolutionPayload,
} from '../../types';
import {
  fetchCrossMeetingQuestions,
  resolveQuestion,
  reopenQuestion,
  isAbortError,
} from '../../services/api';

interface UnresolvedQuestionsViewProps {
  meetings: MeetingItem[];
  currentOrgRole?: string;
  onNavigateToEvidence: (meetingId: string, segmentId: string, timestampSeconds?: number | null) => void;
  onDataChanged?: () => void;
}

export const UnresolvedQuestionsView: React.FC<UnresolvedQuestionsViewProps> = ({
  meetings,
  currentOrgRole: _currentOrgRole,
  onNavigateToEvidence,
  onDataChanged,
}) => {
  const [questions, setQuestions] = useState<CrossMeetingQuestionItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [meetingFilter, setMeetingFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Resolve Modal State
  const [resolveTarget, setResolveTarget] = useState<CrossMeetingQuestionItem | null>(null);
  const [resolvingMeetingId, setResolvingMeetingId] = useState<string>('');
  const [resolvingDecisionId, setResolvingDecisionId] = useState<string>('');
  const [resolutionStatus, setResolutionStatus] = useState<string>('RESOLVED');
  const [resolutionNotes, setResolutionNotes] = useState<string>('');
  const [isSubmittingResolve, setIsSubmittingResolve] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);

  // Reopen Confirmation State
  const [reopenTarget, setReopenTarget] = useState<CrossMeetingQuestionItem | null>(null);
  const [isSubmittingReopen, setIsSubmittingReopen] = useState(false);
  const [reopenError, setReopenError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const requestEpochRef = useRef<number>(0);

  const loadQuestions = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const epoch = ++requestEpochRef.current;

    setIsLoading(true);
    setError(null);

    const params: QuestionFilterParams = {};
    if (statusFilter !== 'ALL') params.status = statusFilter;
    if (meetingFilter) params.meeting_id = meetingFilter;
    if (searchQuery.trim()) params.search = searchQuery.trim();

    try {
      const data = await fetchCrossMeetingQuestions(params, controller.signal);
      if (epoch === requestEpochRef.current) {
        setQuestions(data);
      }
    } catch (err: unknown) {
      if (!isAbortError(err) && epoch === requestEpochRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load questions');
      }
    } finally {
      if (epoch === requestEpochRef.current) {
        setIsLoading(false);
      }
    }
  }, [statusFilter, meetingFilter, searchQuery]);

  useEffect(() => {
    loadQuestions();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [loadQuestions]);

  // Submit Resolve
  const handleResolveSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resolveTarget) return;
    if (!resolvingMeetingId) {
      setResolveError('Please select the resolving meeting.');
      return;
    }

    setIsSubmittingResolve(true);
    setResolveError(null);

    const payload: QuestionResolutionPayload = {
      resolving_meeting_id: resolvingMeetingId,
      resolving_decision_id: resolvingDecisionId.trim() || undefined,
      resolution_notes: resolutionNotes.trim() || undefined,
      status: resolutionStatus,
    };

    try {
      await resolveQuestion(resolveTarget.id, payload);
      setResolveTarget(null);
      setResolvingDecisionId('');
      setResolutionNotes('');
      await loadQuestions();
      if (onDataChanged) onDataChanged();
    } catch (err: unknown) {
      setResolveError(err instanceof Error ? err.message : 'Failed to resolve question');
    } finally {
      setIsSubmittingResolve(false);
    }
  };

  // Submit Reopen
  const handleReopenSubmit = async () => {
    if (!reopenTarget) return;
    setIsSubmittingReopen(true);
    setReopenError(null);

    try {
      await reopenQuestion(reopenTarget.id);
      setReopenTarget(null);
      await loadQuestions();
      if (onDataChanged) onDataChanged();
    } catch (err: unknown) {
      setReopenError(err instanceof Error ? err.message : 'Failed to reopen question');
    } finally {
      setIsSubmittingReopen(false);
    }
  };

  const renderStatusBadge = (status: string) => {
    const s = status.toUpperCase();
    if (s === 'RESOLVED') {
      return (
        <span className="badge-status resolved">
          <CheckCircle2 size={12} />
          Resolved
        </span>
      );
    }
    if (s === 'DEFERRED') {
      return (
        <span className="badge-status deferred">
          <Clock size={12} />
          Deferred
        </span>
      );
    }
    return (
      <span className="badge-status open">
        <HelpCircle size={12} />
        Open
      </span>
    );
  };

  return (
    <div className="memory-card" data-testid="unresolved-questions-view">
      {/* Filter Row */}
      <div className="intel-filter-row">
        <input
          type="text"
          className="intel-search-input"
          placeholder="Search questions by text or raised-by..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          data-testid="question-search-input"
        />

        <select
          className="intel-select"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          data-testid="question-status-select"
        >
          <option value="ALL">All Statuses</option>
          <option value="OPEN">Open Questions</option>
          <option value="RESOLVED">Resolved Questions</option>
          <option value="DEFERRED">Deferred Questions</option>
        </select>

        <select
          className="intel-select"
          value={meetingFilter}
          onChange={(e) => setMeetingFilter(e.target.value)}
          data-testid="question-meeting-select"
        >
          <option value="">All Origin Meetings</option>
          {meetings.map((m) => (
            <option key={m.id} value={m.id}>
              {m.title || 'Untitled Meeting'}
            </option>
          ))}
        </select>

        <button
          type="button"
          className="intel-btn-sm"
          onClick={loadQuestions}
          disabled={isLoading}
          title="Refresh questions"
        >
          <RefreshCw size={14} className={isLoading ? 'spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="memory-error-box" data-testid="question-error-box">
          <AlertOctagon size={16} />
          <span>{error}</span>
          <button type="button" className="intel-btn-sm" onClick={loadQuestions}>
            Retry
          </button>
        </div>
      )}

      {/* Questions List */}
      {isLoading && questions.length === 0 ? (
        <div className="memory-loading-box">
          <RefreshCw size={24} className="spin" color="#6366f1" />
          <span>Loading cross-meeting questions...</span>
        </div>
      ) : questions.length === 0 ? (
        <div className="memory-empty-box" data-testid="question-empty-box">
          <HelpCircle size={32} color="#64748b" />
          <div className="memory-empty-title">No questions found</div>
          <div className="memory-empty-desc">
            No questions match the current filters. Open questions are tracked across meetings until explicitly resolved.
          </div>
        </div>
      ) : (
        <div className="intel-items-list" data-testid="questions-list">
          {questions.map((q) => {
            return (
              <div key={q.id} className="intel-item-card" data-testid={`question-card-${q.id}`}>
                <div className="intel-item-header">
                  <div className="intel-item-title">{q.question_text}</div>
                  {renderStatusBadge(q.status)}
                </div>

                <div className="intel-item-meta">
                  <span>
                    <strong>Origin Meeting:</strong> {q.origin_meeting_title || 'Meeting provenance unavailable'}
                  </span>
                  {q.origin_meeting_date && (
                    <>
                      <span>&bull;</span>
                      <span>{new Date(q.origin_meeting_date).toLocaleDateString()}</span>
                    </>
                  )}
                  {q.raised_by && (
                    <>
                      <span>&bull;</span>
                      <span>
                        <strong>Raised By:</strong> {q.raised_by}
                      </span>
                    </>
                  )}
                </div>

                {/* Resolution Details */}
                {q.status === 'RESOLVED' && (
                  <div className="intel-item-body" style={{ borderLeft: '3px solid #10b981' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600, color: '#10b981', marginBottom: '0.25rem' }}>
                      <CheckCircle2 size={13} />
                      <span>Resolution Information</span>
                    </div>
                    {q.resolving_meeting_title && (
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        Resolved in: <strong>{q.resolving_meeting_title}</strong>
                        {q.resolved_at && ` on ${new Date(q.resolved_at).toLocaleDateString()}`}
                      </div>
                    )}
                    {q.resolving_decision_description && (
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', marginTop: '0.2rem' }}>
                        Linked Decision: "{q.resolving_decision_description}"
                      </div>
                    )}
                    {q.resolution_notes && (
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', marginTop: '0.2rem' }}>
                        Notes: {q.resolution_notes}
                      </div>
                    )}
                  </div>
                )}

                {/* Actions */}
                <div className="intel-item-actions">
                  <div className="intel-action-group">
                    {q.evidence_segment_id && (
                      <button
                        type="button"
                        className="intel-btn-sm"
                        onClick={() => onNavigateToEvidence(q.meeting_id, q.evidence_segment_id!)}
                        data-testid={`evidence-btn-${q.id}`}
                      >
                        <span>View Evidence</span>
                        <ArrowRight size={12} />
                      </button>
                    )}
                  </div>

                  <div className="intel-action-group">
                    {q.status !== 'RESOLVED' ? (
                      <button
                        type="button"
                        className="intel-btn-sm primary"
                        onClick={() => {
                          setResolveTarget(q);
                          setResolvingMeetingId(meetings[0]?.id || q.meeting_id);
                          setResolvingDecisionId('');
                          setResolutionStatus('RESOLVED');
                          setResolutionNotes('');
                          setResolveError(null);
                        }}
                        data-testid={`resolve-btn-${q.id}`}
                        title="Record resolution of this question"
                      >
                        <CheckCircle2 size={13} />
                        <span>Resolve Question</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="intel-btn-sm"
                        onClick={() => {
                          setReopenTarget(q);
                          setReopenError(null);
                        }}
                        data-testid={`reopen-btn-${q.id}`}
                        title="Reopen this resolved question back to OPEN status"
                      >
                        <RotateCcw size={13} />
                        <span>Reopen Question</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* --- RESOLVE QUESTION MODAL --- */}
      {resolveTarget && (
        <div className="intel-modal-backdrop" data-testid="resolve-question-modal">
          <div className="intel-modal-window">
            <div className="intel-modal-title">
              <CheckCircle2 size={18} color="#10b981" />
              <span>Resolve Question</span>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Mark <strong>"{resolveTarget.question_text}"</strong> as resolved in a follow-up meeting.
            </div>

            {resolveError && (
              <div className="memory-error-box">
                <AlertOctagon size={14} />
                <span>{resolveError}</span>
              </div>
            )}

            <form onSubmit={handleResolveSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div className="memory-input-group">
                <label className="memory-input-label">Resolving Meeting *</label>
                <select
                  className="intel-select"
                  value={resolvingMeetingId}
                  onChange={(e) => setResolvingMeetingId(e.target.value)}
                  required
                  data-testid="resolving-meeting-select"
                >
                  <option value="">Select meeting where question was resolved...</option>
                  {meetings.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.title || 'Untitled Meeting'} ({new Date(m.created_at).toLocaleDateString()})
                    </option>
                  ))}
                </select>
              </div>

              <div className="memory-input-group">
                <label className="memory-input-label">Resolution Status *</label>
                <select
                  className="intel-select"
                  value={resolutionStatus}
                  onChange={(e) => setResolutionStatus(e.target.value)}
                  required
                  data-testid="resolving-status-select"
                >
                  <option value="RESOLVED">Resolved</option>
                  <option value="DEFERRED">Deferred</option>
                </select>
              </div>

              <div className="memory-input-group">
                <label className="memory-input-label">Resolution Notes</label>
                <textarea
                  className="intel-search-input"
                  style={{ minHeight: '80px', resize: 'vertical' }}
                  placeholder="Explain how this question was addressed or answered..."
                  value={resolutionNotes}
                  onChange={(e) => setResolutionNotes(e.target.value)}
                  data-testid="resolution-notes-input"
                />
              </div>

              <div className="intel-modal-footer">
                <button
                  type="button"
                  className="intel-btn-sm"
                  onClick={() => setResolveTarget(null)}
                  disabled={isSubmittingResolve}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="intel-btn-sm primary"
                  disabled={isSubmittingResolve}
                  data-testid="submit-resolve-btn"
                >
                  {isSubmittingResolve ? 'Resolving...' : 'Save Resolution'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* --- REOPEN CONFIRMATION MODAL --- */}
      {reopenTarget && (
        <div className="intel-modal-backdrop" data-testid="reopen-confirm-modal">
          <div className="intel-modal-window">
            <div className="intel-modal-title">
              <RotateCcw size={18} color="#6366f1" />
              <span>Reopen Question</span>
            </div>
            <div style={{ fontSize: '0.84rem', color: 'var(--text-primary)', lineHeight: 1.4 }}>
              Are you sure you want to reopen <strong>"{reopenTarget.question_text}"</strong>?
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              This will clear the previous resolution and mark this question back to OPEN status so it appears in active
              follow-up tracking.
            </div>

            {reopenError && (
              <div className="memory-error-box">
                <AlertOctagon size={14} />
                <span>{reopenError}</span>
              </div>
            )}

            <div className="intel-modal-footer">
              <button
                type="button"
                className="intel-btn-sm"
                onClick={() => setReopenTarget(null)}
                disabled={isSubmittingReopen}
              >
                Cancel
              </button>
              <button
                type="button"
                className="intel-btn-sm primary"
                onClick={handleReopenSubmit}
                disabled={isSubmittingReopen}
                data-testid="submit-reopen-btn"
              >
                {isSubmittingReopen ? 'Reopening...' : 'Confirm Reopen'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
