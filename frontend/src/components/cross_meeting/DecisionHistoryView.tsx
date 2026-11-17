import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  CheckCircle2,
  Archive,
  AlertOctagon,
  ArrowRight,
  GitBranch,
  RefreshCw,
  PlusCircle,
  Link as LinkIcon,
  XCircle,
  Shield,
  Layers,
} from 'lucide-react';
import type {
  CrossMeetingDecisionItem,
  MeetingItem,
  DecisionFilterParams,
  DecisionSupersedePayload,
  DecisionLinkSupersededPayload,
  DecisionRevokePayload,
} from '../../types';
import {
  fetchCrossMeetingDecisions,
  fetchDecisionDetails,
  supersedeDecision,
  linkSupersededDecision,
  revokeDecision,
  isAbortError,
} from '../../services/api';

interface DecisionHistoryViewProps {
  meetings: MeetingItem[];
  currentOrgRole?: string;
  onNavigateToEvidence: (meetingId: string, segmentId: string, timestampSeconds?: number | null) => void;
  onDataChanged?: () => void;
}

export const DecisionHistoryView: React.FC<DecisionHistoryViewProps> = ({
  meetings,
  currentOrgRole,
  onNavigateToEvidence,
  onDataChanged,
}) => {
  const [decisions, setDecisions] = useState<CrossMeetingDecisionItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [meetingFilter, setMeetingFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');

  // Expanded Lineage Map
  const [expandedLineage, setExpandedLineage] = useState<Record<string, boolean>>({});
  const [lineageDetails, setLineageDetails] = useState<Record<string, CrossMeetingDecisionItem>>({});
  const [loadingLineageId, setLoadingLineageId] = useState<string | null>(null);

  // Modals state
  const [supersedeModalTarget, setSupersedeModalTarget] = useState<CrossMeetingDecisionItem | null>(null);
  const [supersedeMeetingId, setSupersedeMeetingId] = useState<string>('');
  const [supersedeDescription, setSupersedeDescription] = useState<string>('');
  const [supersedeRationale, setSupersedeRationale] = useState<string>('');
  const [supersedeDecidedBy, setSupersedeDecidedBy] = useState<string>('');
  const [isSubmittingSupersede, setIsSubmittingSupersede] = useState(false);
  const [supersedeError, setSupersedeError] = useState<string | null>(null);

  const [linkModalTarget, setLinkModalTarget] = useState<CrossMeetingDecisionItem | null>(null);
  const [linkSelectedDecisionId, setLinkSelectedDecisionId] = useState<string>('');
  const [isSubmittingLink, setIsSubmittingLink] = useState(false);
  const [linkError, setLinkError] = useState<string | null>(null);

  const [revokeModalTarget, setRevokeModalTarget] = useState<CrossMeetingDecisionItem | null>(null);
  const [revokeReason, setRevokeReason] = useState<string>('');
  const [isSubmittingRevoke, setIsSubmittingRevoke] = useState(false);
  const [revokeError, setRevokeError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const requestEpochRef = useRef<number>(0);

  const loadDecisions = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const epoch = ++requestEpochRef.current;

    setIsLoading(true);
    setError(null);

    const params: DecisionFilterParams = {};
    if (statusFilter !== 'ALL') params.status = statusFilter;
    if (meetingFilter) params.meeting_id = meetingFilter;
    if (searchQuery.trim()) params.search = searchQuery.trim();
    if (startDate) params.start_date = new Date(startDate).toISOString();
    if (endDate) params.end_date = new Date(endDate).toISOString();

    try {
      const data = await fetchCrossMeetingDecisions(params, controller.signal);
      if (epoch === requestEpochRef.current) {
        setDecisions(data);
      }
    } catch (err: unknown) {
      if (!isAbortError(err) && epoch === requestEpochRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load decisions');
      }
    } finally {
      if (epoch === requestEpochRef.current) {
        setIsLoading(false);
      }
    }
  }, [statusFilter, meetingFilter, searchQuery, startDate, endDate]);

  useEffect(() => {
    loadDecisions();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [loadDecisions]);

  // Load decision lineage
  const handleToggleLineage = async (decisionId: string) => {
    const isCurrentlyExpanded = !!expandedLineage[decisionId];
    if (isCurrentlyExpanded) {
      setExpandedLineage((prev) => ({ ...prev, [decisionId]: false }));
      return;
    }

    if (!lineageDetails[decisionId]) {
      setLoadingLineageId(decisionId);
      try {
        const details = await fetchDecisionDetails(decisionId);
        setLineageDetails((prev) => ({ ...prev, [decisionId]: details }));
      } catch (err: unknown) {
        console.error('Failed to load decision lineage:', err);
      } finally {
        setLoadingLineageId(null);
      }
    }
    setExpandedLineage((prev) => ({ ...prev, [decisionId]: true }));
  };

  // Submit Supersede
  const handleSupersedeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!supersedeModalTarget) return;
    if (!supersedeMeetingId) {
      setSupersedeError('Please select a target meeting for the new decision.');
      return;
    }
    if (!supersedeDescription.trim()) {
      setSupersedeError('New decision description cannot be empty.');
      return;
    }

    setIsSubmittingSupersede(true);
    setSupersedeError(null);

    const payload: DecisionSupersedePayload = {
      meeting_id: supersedeMeetingId,
      description: supersedeDescription.trim(),
      rationale_or_context: supersedeRationale.trim() || undefined,
      decided_by: supersedeDecidedBy.trim() || undefined,
    };

    try {
      await supersedeDecision(supersedeModalTarget.id, payload);
      setSupersedeModalTarget(null);
      setSupersedeDescription('');
      setSupersedeRationale('');
      setSupersedeDecidedBy('');
      await loadDecisions();
      if (onDataChanged) onDataChanged();
    } catch (err: unknown) {
      setSupersedeError(err instanceof Error ? err.message : 'Failed to supersede decision');
    } finally {
      setIsSubmittingSupersede(false);
    }
  };

  // Submit Link Superseded
  const handleLinkSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!linkModalTarget) return;
    if (!linkSelectedDecisionId) {
      setLinkError('Please select a decision to mark as superseded.');
      return;
    }
    if (linkSelectedDecisionId === linkModalTarget.id) {
      setLinkError('A decision cannot supersede itself.');
      return;
    }

    setIsSubmittingLink(true);
    setLinkError(null);

    const payload: DecisionLinkSupersededPayload = {
      superseded_decision_id: linkSelectedDecisionId,
    };

    try {
      await linkSupersededDecision(linkModalTarget.id, payload);
      setLinkModalTarget(null);
      setLinkSelectedDecisionId('');
      await loadDecisions();
      if (onDataChanged) onDataChanged();
    } catch (err: unknown) {
      setLinkError(err instanceof Error ? err.message : 'Failed to link superseded decision');
    } finally {
      setIsSubmittingLink(false);
    }
  };

  // Submit Revoke
  const handleRevokeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!revokeModalTarget) return;
    if (!revokeReason.trim()) {
      setRevokeError('A mandatory reason is required to revoke a decision.');
      return;
    }

    setIsSubmittingRevoke(true);
    setRevokeError(null);

    const payload: DecisionRevokePayload = {
      reason: revokeReason.trim(),
    };

    try {
      await revokeDecision(revokeModalTarget.id, payload);
      setRevokeModalTarget(null);
      setRevokeReason('');
      await loadDecisions();
      if (onDataChanged) onDataChanged();
    } catch (err: unknown) {
      setRevokeError(err instanceof Error ? err.message : 'Failed to revoke decision');
    } finally {
      setIsSubmittingRevoke(false);
    }
  };

  const renderStatusBadge = (status: string) => {
    const s = status.toUpperCase();
    if (s === 'ACTIVE') {
      return (
        <span className="badge-status active">
          <CheckCircle2 size={12} />
          Active
        </span>
      );
    }
    if (s === 'SUPERSEDED') {
      return (
        <span className="badge-status superseded">
          <Archive size={12} />
          Superseded
        </span>
      );
    }
    if (s === 'REVOKED') {
      return (
        <span className="badge-status revoked">
          <AlertOctagon size={12} />
          Revoked
        </span>
      );
    }
    return <span className="badge-status pending">{status}</span>;
  };

  return (
    <div className="memory-card" data-testid="decision-history-view">
      {/* Header & Filter Row */}
      <div className="intel-filter-row">
        <input
          type="text"
          className="intel-search-input"
          placeholder="Search decisions by keyword, rationale, or decided-by..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          data-testid="decision-search-input"
        />

        <select
          className="intel-select"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          data-testid="decision-status-select"
        >
          <option value="ALL">All Statuses</option>
          <option value="ACTIVE">Active Only</option>
          <option value="SUPERSEDED">Superseded Only</option>
          <option value="REVOKED">Revoked Only</option>
        </select>

        <select
          className="intel-select"
          value={meetingFilter}
          onChange={(e) => setMeetingFilter(e.target.value)}
          data-testid="decision-meeting-select"
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
          onClick={loadDecisions}
          disabled={isLoading}
          title="Refresh decisions"
        >
          <RefreshCw size={14} className={isLoading ? 'spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Date Filter Row */}
      <div className="intel-filter-row" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        <span>Date Range:</span>
        <input
          type="date"
          className="intel-select"
          style={{ padding: '0.35rem 0.5rem' }}
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          data-testid="decision-start-date"
        />
        <span>to</span>
        <input
          type="date"
          className="intel-select"
          style={{ padding: '0.35rem 0.5rem' }}
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          data-testid="decision-end-date"
        />
        {(startDate || endDate) && (
          <button
            type="button"
            className="intel-btn-sm"
            onClick={() => {
              setStartDate('');
              setEndDate('');
            }}
          >
            Clear Dates
          </button>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="memory-error-box" data-testid="decision-error-box">
          <AlertOctagon size={16} />
          <span>{error}</span>
          <button type="button" className="intel-btn-sm" onClick={loadDecisions}>
            Retry
          </button>
        </div>
      )}

      {/* Decisions List */}
      {isLoading && decisions.length === 0 ? (
        <div className="memory-loading-box">
          <RefreshCw size={24} className="spin" color="#6366f1" />
          <span>Loading cross-meeting decisions...</span>
        </div>
      ) : decisions.length === 0 ? (
        <div className="memory-empty-box" data-testid="decision-empty-box">
          <Archive size={32} color="#64748b" />
          <div className="memory-empty-title">No decisions found</div>
          <div className="memory-empty-desc">
            No decisions match the selected status and search criteria. Decisions are extracted from transcribed meetings.
          </div>
        </div>
      ) : (
        <div className="intel-items-list" data-testid="decisions-list">
          {decisions.map((decision) => {
            const hasLineage = decision.lineage_depth > 1 || !!decision.supersedes_decision_id || !!decision.superseded_by_decision_id;
            const isExpanded = !!expandedLineage[decision.id];
            const details = lineageDetails[decision.id];
            const lineageChain = details?.lineage || decision.lineage || [];

            return (
              <div
                key={decision.id}
                className="intel-item-card"
                data-testid={`decision-card-${decision.id}`}
              >
                <div className="intel-item-header">
                  <div className="intel-item-title">
                    {decision.description}
                    {hasLineage && (
                      <span title="This decision has linked lineage" style={{ marginLeft: '0.4rem', color: '#6366f1' }}>
                        <GitBranch size={13} style={{ display: 'inline', verticalAlign: 'middle' }} />
                      </span>
                    )}
                  </div>
                  {renderStatusBadge(decision.status)}
                </div>

                {decision.rationale_or_context && (
                  <div className="intel-item-body">
                    <strong>Rationale:</strong> {decision.rationale_or_context}
                  </div>
                )}

                <div className="intel-item-meta">
                  <span>
                    <strong>Meeting:</strong> {decision.meeting_title || 'Meeting provenance unavailable'}
                  </span>
                  {decision.meeting_date && (
                    <>
                      <span>&bull;</span>
                      <span>{new Date(decision.meeting_date).toLocaleDateString()}</span>
                    </>
                  )}
                  {decision.decided_by && (
                    <>
                      <span>&bull;</span>
                      <span>
                        <strong>Decided By:</strong> {decision.decided_by}
                      </span>
                    </>
                  )}
                  {decision.lineage_depth > 1 && (
                    <>
                      <span>&bull;</span>
                      <span className="intel-stat-badge neutral">
                        Lineage Depth: {decision.lineage_depth}
                      </span>
                    </>
                  )}
                </div>

                {/* Lineage Info Banner */}
                {decision.supersedes_decision_id && (
                  <div style={{ fontSize: '0.78rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <GitBranch size={13} />
                    <span>Supersedes an earlier decision in organizational history</span>
                  </div>
                )}
                {decision.superseded_by_decision_id && (
                  <div style={{ fontSize: '0.78rem', color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <Archive size={13} />
                    <span>Superseded by a later active decision</span>
                  </div>
                )}

                {/* Actions & Lineage toggles */}
                <div className="intel-item-actions">
                  <div className="intel-action-group">
                    {decision.evidence_segment_id && (
                      <button
                        type="button"
                        className="intel-btn-sm"
                        onClick={() =>
                          onNavigateToEvidence(
                            decision.meeting_id,
                            decision.evidence_segment_id!,
                            decision.timestamp_seconds
                          )
                        }
                        data-testid={`evidence-btn-${decision.id}`}
                      >
                        <span>View Evidence</span>
                        <ArrowRight size={12} />
                      </button>
                    )}

                    <button
                      type="button"
                      className="intel-btn-sm"
                      onClick={() => handleToggleLineage(decision.id)}
                      data-testid={`lineage-btn-${decision.id}`}
                    >
                      <GitBranch size={13} />
                      <span>{isExpanded ? 'Hide Lineage' : 'View Lineage'}</span>
                      {loadingLineageId === decision.id && <RefreshCw size={11} className="spin" />}
                    </button>
                  </div>

                  <div className="intel-action-group">
                    {/* Supersede button: Available for any decision to supersede it with a new one */}
                    <button
                      type="button"
                      className="intel-btn-sm primary"
                      onClick={() => {
                        setSupersedeModalTarget(decision);
                        setSupersedeMeetingId(meetings[0]?.id || decision.meeting_id);
                        setSupersedeDescription('');
                        setSupersedeRationale('');
                        setSupersedeDecidedBy('');
                        setSupersedeError(null);
                      }}
                      data-testid={`supersede-btn-${decision.id}`}
                      title="Supersede this decision with a newer decision"
                    >
                      <PlusCircle size={13} />
                      <span>Supersede</span>
                    </button>

                    {/* Link existing decision */}
                    <button
                      type="button"
                      className="intel-btn-sm"
                      onClick={() => {
                        setLinkModalTarget(decision);
                        setLinkSelectedDecisionId('');
                        setLinkError(null);
                      }}
                      data-testid={`link-btn-${decision.id}`}
                      title="Link an existing decision as superseded by this one"
                    >
                      <LinkIcon size={13} />
                      <span>Link Superseded</span>
                    </button>

                    {/* Revoke button: Only active decisions can be revoked; requires ORG_ADMIN if role specified */}
                    {decision.status === 'ACTIVE' && (
                      <button
                        type="button"
                        className="intel-btn-sm danger"
                        disabled={currentOrgRole !== undefined && currentOrgRole !== 'ORG_ADMIN'}
                        onClick={() => {
                          setRevokeModalTarget(decision);
                          setRevokeReason('');
                          setRevokeError(null);
                        }}
                        data-testid={`revoke-btn-${decision.id}`}
                        title={
                          currentOrgRole !== undefined && currentOrgRole !== 'ORG_ADMIN'
                            ? 'Admin privileges required to revoke organizational decisions'
                            : 'Revoke this decision with an audited reason'
                        }
                      >
                        <XCircle size={13} />
                        <span>Revoke</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Lineage Chain Accordion */}
                {isExpanded && (
                  <div className="intel-lineage-container" data-testid={`lineage-container-${decision.id}`}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <Layers size={14} color="#6366f1" />
                      <span>Historical Lineage Chain</span>
                    </div>

                    {lineageChain.length <= 1 ? (
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        This is an original standalone decision with no prior ancestry.
                      </div>
                    ) : (
                      lineageChain.map((step, idx) => (
                        <div key={step.id} className="intel-lineage-step">
                          <span style={{ fontWeight: 600, color: '#6366f1' }}>#{idx + 1}</span>
                          <span style={{ flex: 1, color: step.id === decision.id ? '#ffffff' : 'var(--text-muted)' }}>
                            "{step.description}"
                          </span>
                          {renderStatusBadge(step.status || (step.id === decision.id ? decision.status : 'SUPERSEDED'))}
                          {idx < lineageChain.length - 1 && <span className="intel-lineage-arrow">&darr;</span>}
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

      {/* --- MODAL 1: SUPERSEDE DECISION --- */}
      {supersedeModalTarget && (
        <div className="intel-modal-backdrop" data-testid="supersede-modal">
          <div className="intel-modal-window">
            <div className="intel-modal-title">
              <PlusCircle size={18} color="#6366f1" />
              <span>Supersede Decision</span>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Superseding <strong>"{supersedeModalTarget.description}"</strong> will mark it as{' '}
              <span className="badge-status superseded">SUPERSEDED</span> and create a new active decision linked to its
              history.
            </div>

            {supersedeError && (
              <div className="memory-error-box">
                <AlertOctagon size={14} />
                <span>{supersedeError}</span>
              </div>
            )}

            <form onSubmit={handleSupersedeSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div className="memory-input-group">
                <label className="memory-input-label">Origin Meeting for New Decision *</label>
                <select
                  className="intel-select"
                  value={supersedeMeetingId}
                  onChange={(e) => setSupersedeMeetingId(e.target.value)}
                  required
                  data-testid="supersede-meeting-select"
                >
                  <option value="">Select meeting...</option>
                  {meetings.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.title || 'Untitled Meeting'} ({new Date(m.created_at).toLocaleDateString()})
                    </option>
                  ))}
                </select>
              </div>

              <div className="memory-input-group">
                <label className="memory-input-label">New Decision Description *</label>
                <textarea
                  className="intel-search-input"
                  style={{ minHeight: '80px', resize: 'vertical' }}
                  placeholder="Describe the new active decision that supersedes the previous one..."
                  value={supersedeDescription}
                  onChange={(e) => setSupersedeDescription(e.target.value)}
                  required
                  data-testid="supersede-desc-input"
                />
              </div>

              <div className="memory-input-group">
                <label className="memory-input-label">Rationale or Context</label>
                <textarea
                  className="intel-search-input"
                  style={{ minHeight: '60px', resize: 'vertical' }}
                  placeholder="Reason for changing or replacing this decision..."
                  value={supersedeRationale}
                  onChange={(e) => setSupersedeRationale(e.target.value)}
                  data-testid="supersede-rationale-input"
                />
              </div>

              <div className="memory-input-group">
                <label className="memory-input-label">Decided By</label>
                <input
                  type="text"
                  className="intel-search-input"
                  placeholder="e.g. Architecture Committee, Tech Lead"
                  value={supersedeDecidedBy}
                  onChange={(e) => setSupersedeDecidedBy(e.target.value)}
                  data-testid="supersede-decidedby-input"
                />
              </div>

              <div className="intel-modal-footer">
                <button
                  type="button"
                  className="intel-btn-sm"
                  onClick={() => setSupersedeModalTarget(null)}
                  disabled={isSubmittingSupersede}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="intel-btn-sm primary"
                  disabled={isSubmittingSupersede}
                  data-testid="submit-supersede-btn"
                >
                  {isSubmittingSupersede ? 'Superseding...' : 'Supersede Decision'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* --- MODAL 2: LINK SUPERSEDED DECISION --- */}
      {linkModalTarget && (
        <div className="intel-modal-backdrop" data-testid="link-modal">
          <div className="intel-modal-window">
            <div className="intel-modal-title">
              <LinkIcon size={18} color="#6366f1" />
              <span>Link Existing Superseded Decision</span>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Mark an existing earlier decision as superseded by <strong>"{linkModalTarget.description}"</strong>.
            </div>

            {linkError && (
              <div className="memory-error-box">
                <AlertOctagon size={14} />
                <span>{linkError}</span>
              </div>
            )}

            <form onSubmit={handleLinkSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div className="memory-input-group">
                <label className="memory-input-label">Select Prior Decision to Mark as Superseded *</label>
                <select
                  className="intel-select"
                  value={linkSelectedDecisionId}
                  onChange={(e) => setLinkSelectedDecisionId(e.target.value)}
                  required
                  data-testid="link-decision-select"
                >
                  <option value="">Select prior decision...</option>
                  {decisions
                    .filter((d) => d.id !== linkModalTarget.id && d.status !== 'SUPERSEDED')
                    .map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.description.slice(0, 70)}... ({d.meeting_title || 'Meeting'})
                      </option>
                    ))}
                </select>
              </div>

              <div className="intel-modal-footer">
                <button
                  type="button"
                  className="intel-btn-sm"
                  onClick={() => setLinkModalTarget(null)}
                  disabled={isSubmittingLink}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="intel-btn-sm primary"
                  disabled={isSubmittingLink}
                  data-testid="submit-link-btn"
                >
                  {isSubmittingLink ? 'Linking...' : 'Link Decision'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* --- MODAL 3: REVOKE DECISION --- */}
      {revokeModalTarget && (
        <div className="intel-modal-backdrop" data-testid="revoke-modal">
          <div className="intel-modal-window">
            <div className="intel-modal-title" style={{ color: '#ef4444' }}>
              <Shield size={18} color="#ef4444" />
              <span>Revoke Decision</span>
            </div>
            <div style={{ fontSize: '0.84rem', color: 'var(--text-primary)', lineHeight: 1.4 }}>
              Are you sure you want to revoke <strong>"{revokeModalTarget.description}"</strong>?
            </div>
            <div style={{ fontSize: '0.8rem', color: '#ef4444', background: 'rgba(239, 68, 68, 0.1)', padding: '0.5rem', borderRadius: '4px' }}>
              Revocation marks the decision as permanently revoked. An audited explanation is mandatory and will be
              recorded in the organization's intelligence log.
            </div>

            {revokeError && (
              <div className="memory-error-box">
                <AlertOctagon size={14} />
                <span>{revokeError}</span>
              </div>
            )}

            <form onSubmit={handleRevokeSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div className="memory-input-group">
                <label className="memory-input-label">Mandatory Reason for Revocation *</label>
                <textarea
                  className="intel-search-input"
                  style={{ minHeight: '80px', resize: 'vertical' }}
                  placeholder="Explain why this decision is being revoked..."
                  value={revokeReason}
                  onChange={(e) => setRevokeReason(e.target.value)}
                  required
                  data-testid="revoke-reason-input"
                />
              </div>

              <div className="intel-modal-footer">
                <button
                  type="button"
                  className="intel-btn-sm"
                  onClick={() => setRevokeModalTarget(null)}
                  disabled={isSubmittingRevoke}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="intel-btn-sm danger"
                  disabled={isSubmittingRevoke}
                  data-testid="submit-revoke-btn"
                >
                  {isSubmittingRevoke ? 'Revoking...' : 'Confirm Revocation'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
