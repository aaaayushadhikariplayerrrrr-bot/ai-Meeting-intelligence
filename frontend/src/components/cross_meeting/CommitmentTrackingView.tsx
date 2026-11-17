import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Clock,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  PlayCircle,
  ArrowRight,
  RefreshCw,
  PlusCircle,
  History,
  AlertOctagon,
  Calendar,
  User,
} from 'lucide-react';
import type {
  CrossMeetingCommitmentItem,
  MeetingItem,
  CommitmentFilterParams,
  CommitmentReviewPayload,
  ActionItemReviewItem,
} from '../../types';
import {
  fetchCrossMeetingCommitments,
  fetchCommitmentReviews,
  recordCommitmentReview,
  isAbortError,
} from '../../services/api';

interface CommitmentTrackingViewProps {
  meetings: MeetingItem[];
  currentOrgRole?: string;
  onNavigateToEvidence: (meetingId: string, segmentId: string, timestampSeconds?: number | null) => void;
  onDataChanged?: () => void;
}

export const CommitmentTrackingView: React.FC<CommitmentTrackingViewProps> = ({
  meetings,
  currentOrgRole: _currentOrgRole,
  onNavigateToEvidence,
  onDataChanged,
}) => {
  const [commitments, setCommitments] = useState<CrossMeetingCommitmentItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [ownerFilter, setOwnerFilter] = useState<string>('');
  const [meetingFilter, setMeetingFilter] = useState<string>('');
  const [overdueOnly, setOverdueOnly] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Expanded Reviews
  const [expandedReviews, setExpandedReviews] = useState<Record<string, boolean>>({});
  const [reviewHistoryMap, setReviewHistoryMap] = useState<Record<string, ActionItemReviewItem[]>>({});
  const [loadingReviewId, setLoadingReviewId] = useState<string | null>(null);

  // Record Review Modal
  const [reviewTarget, setReviewTarget] = useState<CrossMeetingCommitmentItem | null>(null);
  const [reviewMeetingId, setReviewMeetingId] = useState<string>('');
  const [reviewNewStatus, setReviewNewStatus] = useState<string>('IN_PROGRESS');
  const [reviewNotes, setReviewNotes] = useState<string>('');
  const [isSubmittingReview, setIsSubmittingReview] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const requestEpochRef = useRef<number>(0);

  const loadCommitments = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const epoch = ++requestEpochRef.current;

    setIsLoading(true);
    setError(null);

    const params: CommitmentFilterParams = {};
    if (statusFilter !== 'ALL') params.status = statusFilter;
    if (ownerFilter.trim()) params.owner_name = ownerFilter.trim();
    if (meetingFilter) params.meeting_id = meetingFilter;
    if (overdueOnly) params.is_overdue = true;
    if (searchQuery.trim()) params.search = searchQuery.trim();

    try {
      const data = await fetchCrossMeetingCommitments(params, controller.signal);
      if (epoch === requestEpochRef.current) {
        setCommitments(data);
      }
    } catch (err: unknown) {
      if (!isAbortError(err) && epoch === requestEpochRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load commitments');
      }
    } finally {
      if (epoch === requestEpochRef.current) {
        setIsLoading(false);
      }
    }
  }, [statusFilter, ownerFilter, meetingFilter, overdueOnly, searchQuery]);

  useEffect(() => {
    loadCommitments();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [loadCommitments]);

  // Toggle and load review history
  const handleToggleReviews = async (actionItemId: string) => {
    const isCurrentlyExpanded = !!expandedReviews[actionItemId];
    if (isCurrentlyExpanded) {
      setExpandedReviews((prev) => ({ ...prev, [actionItemId]: false }));
      return;
    }

    if (!reviewHistoryMap[actionItemId]) {
      setLoadingReviewId(actionItemId);
      try {
        const reviews = await fetchCommitmentReviews(actionItemId);
        setReviewHistoryMap((prev) => ({ ...prev, [actionItemId]: reviews }));
      } catch (err: unknown) {
        console.error('Failed to load commitment reviews:', err);
      } finally {
        setLoadingReviewId(null);
      }
    }
    setExpandedReviews((prev) => ({ ...prev, [actionItemId]: true }));
  };

  // Submit Review Modal
  const handleReviewSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reviewTarget) return;
    if (!reviewMeetingId) {
      setReviewError('Please select the reviewing meeting.');
      return;
    }

    setIsSubmittingReview(true);
    setReviewError(null);

    const payload: CommitmentReviewPayload = {
      reviewing_meeting_id: reviewMeetingId,
      status: reviewNewStatus,
      review_notes: reviewNotes.trim() || undefined,
    };

    try {
      await recordCommitmentReview(reviewTarget.id, payload);
      setReviewTarget(null);
      setReviewNotes('');
      // Invalidate review cache for this item
      setReviewHistoryMap((prev) => {
        const next = { ...prev };
        delete next[reviewTarget.id];
        return next;
      });
      await loadCommitments();
      if (onDataChanged) onDataChanged();
    } catch (err: unknown) {
      setReviewError(err instanceof Error ? err.message : 'Failed to record commitment review');
    } finally {
      setIsSubmittingReview(false);
    }
  };

  const renderStatusBadge = (status: string, isOverdue: boolean) => {
    const s = status.toUpperCase();
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
        {isOverdue && (
          <span className="badge-status overdue" data-testid="overdue-badge">
            <AlertTriangle size={11} />
            Overdue
          </span>
        )}
        {s === 'COMPLETED' ? (
          <span className="badge-status completed">
            <CheckCircle2 size={11} />
            Completed
          </span>
        ) : s === 'IN_PROGRESS' ? (
          <span className="badge-status in-progress">
            <PlayCircle size={11} />
            In Progress
          </span>
        ) : s === 'CANCELLED' ? (
          <span className="badge-status cancelled">
            <XCircle size={11} />
            Cancelled
          </span>
        ) : (
          <span className="badge-status pending">
            <Clock size={11} />
            Pending
          </span>
        )}
      </div>
    );
  };

  return (
    <div className="memory-card" data-testid="commitment-tracking-view">
      {/* Filter Row */}
      <div className="intel-filter-row">
        <input
          type="text"
          className="intel-search-input"
          placeholder="Search commitments by description..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          data-testid="commitment-search-input"
        />

        <input
          type="text"
          className="intel-search-input"
          style={{ maxWidth: '180px' }}
          placeholder="Filter by owner..."
          value={ownerFilter}
          onChange={(e) => setOwnerFilter(e.target.value)}
          data-testid="commitment-owner-input"
        />

        <select
          className="intel-select"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          data-testid="commitment-status-select"
        >
          <option value="ALL">All Statuses</option>
          <option value="PENDING">Pending</option>
          <option value="IN_PROGRESS">In Progress</option>
          <option value="COMPLETED">Completed</option>
          <option value="CANCELLED">Cancelled</option>
        </select>

        <select
          className="intel-select"
          value={meetingFilter}
          onChange={(e) => setMeetingFilter(e.target.value)}
          data-testid="commitment-meeting-select"
        >
          <option value="">All Origin Meetings</option>
          {meetings.map((m) => (
            <option key={m.id} value={m.id}>
              {m.title || 'Untitled Meeting'}
            </option>
          ))}
        </select>

        <label className="intel-checkbox-label" data-testid="overdue-checkbox-label">
          <input
            type="checkbox"
            checked={overdueOnly}
            onChange={(e) => setOverdueOnly(e.target.checked)}
            data-testid="commitment-overdue-checkbox"
          />
          <span>Overdue Only</span>
        </label>

        <button
          type="button"
          className="intel-btn-sm"
          onClick={loadCommitments}
          disabled={isLoading}
          title="Refresh commitments"
        >
          <RefreshCw size={14} className={isLoading ? 'spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="memory-error-box" data-testid="commitment-error-box">
          <AlertOctagon size={16} />
          <span>{error}</span>
          <button type="button" className="intel-btn-sm" onClick={loadCommitments}>
            Retry
          </button>
        </div>
      )}

      {/* Commitments List */}
      {isLoading && commitments.length === 0 ? (
        <div className="memory-loading-box">
          <RefreshCw size={24} className="spin" color="#6366f1" />
          <span>Loading cross-meeting commitments...</span>
        </div>
      ) : commitments.length === 0 ? (
        <div className="memory-empty-box" data-testid="commitment-empty-box">
          <Clock size={32} color="#64748b" />
          <div className="memory-empty-title">No commitments found</div>
          <div className="memory-empty-desc">
            No action items match the selected status or filters. Commitments are tracked continuously across meetings.
          </div>
        </div>
      ) : (
        <div className="intel-items-list" data-testid="commitments-list">
          {commitments.map((item) => {
            const isExpanded = !!expandedReviews[item.id];
            const reviews = reviewHistoryMap[item.id] || item.reviews || [];

            return (
              <div
                key={item.id}
                className={`intel-item-card ${item.is_overdue ? 'overdue' : ''}`}
                data-testid={`commitment-card-${item.id}`}
              >
                <div className="intel-item-header">
                  <div className="intel-item-title">{item.description}</div>
                  {renderStatusBadge(item.status, item.is_overdue)}
                </div>

                <div className="intel-item-meta">
                  <span>
                    <User size={12} color="#94a3b8" />
                    <strong>Owner:</strong> {item.owner_name || 'Unassigned'}
                  </span>
                  <span>&bull;</span>
                  <span>
                    <Calendar size={12} color="#94a3b8" />
                    <strong>Deadline:</strong>{' '}
                    {item.deadline ? new Date(item.deadline).toLocaleDateString() : 'No deadline set'}
                  </span>
                  <span>&bull;</span>
                  <span>
                    <strong>Origin:</strong> {item.origin_meeting_title || 'Meeting origin unavailable'}
                  </span>
                  {item.last_reviewed_meeting_title && (
                    <>
                      <span>&bull;</span>
                      <span>
                        <strong>Last Reviewed:</strong> {item.last_reviewed_meeting_title}
                        {item.last_reviewed_at && ` (${new Date(item.last_reviewed_at).toLocaleDateString()})`}
                      </span>
                    </>
                  )}
                  {item.review_count > 0 && (
                    <>
                      <span>&bull;</span>
                      <span className="intel-stat-badge neutral">{item.review_count} Reviews</span>
                    </>
                  )}
                </div>

                {/* Actions */}
                <div className="intel-item-actions">
                  <div className="intel-action-group">
                    {item.evidence_segment_id && (
                      <button
                        type="button"
                        className="intel-btn-sm"
                        onClick={() =>
                          onNavigateToEvidence(
                            item.meeting_id,
                            item.evidence_segment_id!,
                            item.timestamp_seconds
                          )
                        }
                        data-testid={`evidence-btn-${item.id}`}
                      >
                        <span>View Evidence</span>
                        <ArrowRight size={12} />
                      </button>
                    )}

                    <button
                      type="button"
                      className="intel-btn-sm"
                      onClick={() => handleToggleReviews(item.id)}
                      data-testid={`review-history-btn-${item.id}`}
                    >
                      <History size={13} />
                      <span>{isExpanded ? 'Hide History' : `Reviews (${item.review_count})`}</span>
                      {loadingReviewId === item.id && <RefreshCw size={11} className="spin" />}
                    </button>
                  </div>

                  <div className="intel-action-group">
                    <button
                      type="button"
                      className="intel-btn-sm primary"
                      onClick={() => {
                        setReviewTarget(item);
                        setReviewMeetingId(meetings[0]?.id || item.meeting_id);
                        setReviewNewStatus(item.status);
                        setReviewNotes('');
                        setReviewError(null);
                      }}
                      data-testid={`record-review-btn-${item.id}`}
                      title="Record a review event in a follow-up meeting"
                    >
                      <PlusCircle size={13} />
                      <span>Record Review</span>
                    </button>
                  </div>
                </div>

                {/* Review History Accordion */}
                {isExpanded && (
                  <div className="intel-reviews-container" data-testid={`reviews-container-${item.id}`}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <History size={14} color="#6366f1" />
                      <span>Chronological Review History</span>
                    </div>

                    {reviews.length === 0 ? (
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        No review events have been recorded for this commitment yet.
                      </div>
                    ) : (
                      reviews.map((rev) => (
                        <div key={rev.id} className="intel-review-card" data-testid={`review-entry-${rev.id}`}>
                          <div className="intel-review-meta">
                            <span>
                              <strong>Reviewed In:</strong> {rev.meeting_title || 'Review Meeting'}
                            </span>
                            <span>{new Date(rev.reviewed_at).toLocaleDateString()}</span>
                          </div>
                          {rev.status_at_review && (
                            <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                              Status Transition: <strong>{rev.status_at_review}</strong>
                            </div>
                          )}
                          {rev.review_notes && (
                            <div style={{ color: 'var(--text-primary)', marginTop: '0.2rem' }}>
                              {rev.review_notes}
                            </div>
                          )}
                          {rev.evidence_quote && (
                            <div style={{ fontSize: '0.74rem', fontStyle: 'italic', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                              "{rev.evidence_quote}"
                            </div>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* --- RECORD REVIEW MODAL --- */}
      {reviewTarget && (
        <div className="intel-modal-backdrop" data-testid="record-review-modal">
          <div className="intel-modal-window">
            <div className="intel-modal-title">
              <PlusCircle size={18} color="#6366f1" />
              <span>Record Commitment Review</span>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Record progress or update the status of <strong>"{reviewTarget.description}"</strong> from a follow-up
              meeting.
            </div>

            {reviewError && (
              <div className="memory-error-box">
                <AlertOctagon size={14} />
                <span>{reviewError}</span>
              </div>
            )}

            <form onSubmit={handleReviewSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div className="memory-input-group">
                <label className="memory-input-label">Reviewing Meeting *</label>
                <select
                  className="intel-select"
                  value={reviewMeetingId}
                  onChange={(e) => setReviewMeetingId(e.target.value)}
                  required
                  data-testid="review-meeting-select"
                >
                  <option value="">Select meeting where review took place...</option>
                  {meetings.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.title || 'Untitled Meeting'} ({new Date(m.created_at).toLocaleDateString()})
                    </option>
                  ))}
                </select>
              </div>

              <div className="memory-input-group">
                <label className="memory-input-label">Updated Status *</label>
                <select
                  className="intel-select"
                  value={reviewNewStatus}
                  onChange={(e) => setReviewNewStatus(e.target.value)}
                  required
                  data-testid="review-status-select"
                >
                  <option value="PENDING">Pending</option>
                  <option value="IN_PROGRESS">In Progress</option>
                  <option value="COMPLETED">Completed</option>
                  <option value="CANCELLED">Cancelled</option>
                </select>
              </div>

              <div className="memory-input-group">
                <label className="memory-input-label">Review Discussion Notes</label>
                <textarea
                  className="intel-search-input"
                  style={{ minHeight: '80px', resize: 'vertical' }}
                  placeholder="Summary of status update, obstacles, or completion notes..."
                  value={reviewNotes}
                  onChange={(e) => setReviewNotes(e.target.value)}
                  data-testid="review-notes-input"
                />
              </div>

              <div className="intel-modal-footer">
                <button
                  type="button"
                  className="intel-btn-sm"
                  onClick={() => setReviewTarget(null)}
                  disabled={isSubmittingReview}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="intel-btn-sm primary"
                  disabled={isSubmittingReview}
                  data-testid="submit-review-btn"
                >
                  {isSubmittingReview ? 'Recording...' : 'Record Review'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
