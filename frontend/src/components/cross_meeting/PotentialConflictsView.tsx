import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  ShieldAlert,
  AlertTriangle,
  RefreshCw,
  AlertOctagon,
  FileQuestion,
  ExternalLink,
} from 'lucide-react';
import type {
  DecisionConflictItem,
  MeetingItem,
  ConflictFilterParams,
} from '../../types';
import {
  fetchDecisionConflicts,
  isAbortError,
} from '../../services/api';

interface PotentialConflictsViewProps {
  meetings: MeetingItem[];
  onNavigateToEvidence: (meetingId: string, segmentId: string, timestampSeconds?: number | null) => void;
  onNavigateToDecisions?: () => void;
}

export const PotentialConflictsView: React.FC<PotentialConflictsViewProps> = ({
  meetings,
  onNavigateToEvidence: _onNavigateToEvidence,
  onNavigateToDecisions,
}) => {
  const [conflicts, setConflicts] = useState<DecisionConflictItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [subjectFilter, setSubjectFilter] = useState<string>('');
  const [meetingFilter, setMeetingFilter] = useState<string>('');

  const abortControllerRef = useRef<AbortController | null>(null);
  const requestEpochRef = useRef<number>(0);

  const loadConflicts = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const epoch = ++requestEpochRef.current;

    setIsLoading(true);
    setError(null);

    const params: ConflictFilterParams = {};
    if (subjectFilter.trim()) params.subject = subjectFilter.trim();
    if (meetingFilter) params.meeting_id = meetingFilter;

    try {
      const data = await fetchDecisionConflicts(params, controller.signal);
      if (epoch === requestEpochRef.current) {
        setConflicts(data);
      }
    } catch (err: unknown) {
      if (!isAbortError(err) && epoch === requestEpochRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to detect potential conflicts');
      }
    } finally {
      if (epoch === requestEpochRef.current) {
        setIsLoading(false);
      }
    }
  }, [subjectFilter, meetingFilter]);

  useEffect(() => {
    loadConflicts();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [loadConflicts]);

  return (
    <div className="memory-card" data-testid="potential-conflicts-view">
      {/* Prominent Heuristic Warning Banner */}
      <div className="intel-conflict-banner" data-testid="conflict-heuristic-warning">
        <ShieldAlert size={22} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>HEURISTIC CONFLICT DETECTION:</strong> Potential contradictions between active decisions are identified
          using semantic and keyword analysis across meeting records.
          <div style={{ marginTop: '0.25rem', opacity: 0.9 }}>
            All detected conflicts are advisory recommendations. <strong>No decisions have been automatically modified or
            superseded.</strong> Review the conflicting statements and take explicit action in the Decision History tab if
            appropriate.
          </div>
        </div>
      </div>

      {/* Filter Row */}
      <div className="intel-filter-row">
        <input
          type="text"
          className="intel-search-input"
          placeholder="Filter by subject or topic..."
          value={subjectFilter}
          onChange={(e) => setSubjectFilter(e.target.value)}
          data-testid="conflict-subject-input"
        />

        <select
          className="intel-select"
          value={meetingFilter}
          onChange={(e) => setMeetingFilter(e.target.value)}
          data-testid="conflict-meeting-select"
        >
          <option value="">All Meetings</option>
          {meetings.map((m) => (
            <option key={m.id} value={m.id}>
              {m.title || 'Untitled Meeting'}
            </option>
          ))}
        </select>

        <button
          type="button"
          className="intel-btn-sm"
          onClick={loadConflicts}
          disabled={isLoading}
          title="Refresh detected conflicts"
        >
          <RefreshCw size={14} className={isLoading ? 'spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="memory-error-box" data-testid="conflict-error-box">
          <AlertOctagon size={16} />
          <span>{error}</span>
          <button type="button" className="intel-btn-sm" onClick={loadConflicts}>
            Retry
          </button>
        </div>
      )}

      {/* Conflict list */}
      {isLoading && conflicts.length === 0 ? (
        <div className="memory-loading-box">
          <RefreshCw size={24} className="spin" color="#6366f1" />
          <span>Analyzing cross-meeting decisions for potential conflicts...</span>
        </div>
      ) : conflicts.length === 0 ? (
        <div className="memory-empty-box" data-testid="conflict-empty-box">
          <CheckCircle2Icon size={32} color="#10b981" />
          <div className="memory-empty-title">No decision conflicts detected</div>
          <div className="memory-empty-desc">
            All active decisions across your organization appear consistent with no detected contradictions.
          </div>
        </div>
      ) : (
        <div className="intel-items-list" data-testid="conflicts-list">
          {conflicts.map((conflict, idx) => (
            <div
              key={`${conflict.decision_a_id}-${conflict.decision_b_id}-${idx}`}
              className="intel-item-card"
              style={{ borderLeft: '3px solid #f59e0b' }}
              data-testid={`conflict-card-${idx}`}
            >
              <div className="intel-item-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertTriangle size={16} color="#f59e0b" />
                  <span className="intel-item-title">Contradiction: {conflict.subject}</span>
                </div>
                <span className="badge-status superseded">Heuristic Alert</span>
              </div>

              {/* Side-by-side Decision Comparison */}
              <div className="intel-conflict-grid">
                {/* Decision A */}
                <div className="intel-conflict-decision-box">
                  <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#94a3b8', fontWeight: 600 }}>
                    Earlier Decision
                  </div>
                  <div style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                    "{conflict.decision_a_text}"
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                    Origin: <strong>{conflict.decision_a_meeting_title}</strong>
                    {conflict.decision_a_date && ` (${new Date(conflict.decision_a_date).toLocaleDateString()})`}
                  </div>
                </div>

                {/* Decision B */}
                <div className="intel-conflict-decision-box">
                  <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#94a3b8', fontWeight: 600 }}>
                    Conflicting Decision
                  </div>
                  <div style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                    "{conflict.decision_b_text}"
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                    Origin: <strong>{conflict.decision_b_meeting_title}</strong>
                    {conflict.decision_b_date && ` (${new Date(conflict.decision_b_date).toLocaleDateString()})`}
                  </div>
                </div>
              </div>

              {/* Reason / Analysis */}
              <div className="intel-item-body">
                <strong>Analysis:</strong> {conflict.reason}
              </div>

              {/* Suggested Action */}
              <div className="intel-item-actions">
                <div style={{ fontSize: '0.8rem', color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <FileQuestion size={14} />
                  <span>Recommendation: {conflict.suggested_action}</span>
                </div>

                {onNavigateToDecisions && (
                  <button
                    type="button"
                    className="intel-btn-sm primary"
                    onClick={onNavigateToDecisions}
                    data-testid="resolve-in-decisions-btn"
                  >
                    <span>Manage in Decisions</span>
                    <ExternalLink size={12} />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Helper component for empty check icon
function CheckCircle2Icon({ size, color }: { size: number; color: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}
