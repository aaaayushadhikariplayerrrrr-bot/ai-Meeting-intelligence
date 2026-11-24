import React, { useState } from 'react';
import {
  GitCommit,
  ArrowRight,
  ExternalLink,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Calendar,
  Layers,
  X,
} from 'lucide-react';
import type { CrossMeetingDecisionItem } from '../../types';

export interface DecisionLineageGraphProps {
  decisions: CrossMeetingDecisionItem[];
  selectedDecisionId?: string | null;
  onSelectDecision?: (decision: CrossMeetingDecisionItem) => void;
  onNavigateToEvidence?: (meetingId: string, segmentId: string, timestampSeconds?: number) => void;
}

export const DecisionLineageGraph: React.FC<DecisionLineageGraphProps> = ({
  decisions,
  selectedDecisionId,
  onSelectDecision,
  onNavigateToEvidence,
}) => {
  const [activeModalDecision, setActiveModalDecision] = useState<CrossMeetingDecisionItem | null>(null);

  // Group into lineage chains or trees
  // Find roots (decisions that do not supersede any other decision, or earliest in chain)
  const decisionMap = new Map<string, CrossMeetingDecisionItem>();
  decisions.forEach((d) => decisionMap.set(d.id, d));

  // Build pairs of (parent -> child)
  const edges: { from: CrossMeetingDecisionItem; to: CrossMeetingDecisionItem }[] = [];
  decisions.forEach((child) => {
    if (child.supersedes_decision_id) {
      const parent = decisionMap.get(child.supersedes_decision_id);
      if (parent) {
        edges.push({ from: parent, to: child });
      }
    }
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'ACTIVE':
        return (
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.2rem',
            padding: '0.15rem 0.4rem',
            borderRadius: '4px',
            fontSize: '0.65rem',
            fontWeight: 700,
            background: 'rgba(16, 185, 129, 0.15)',
            color: '#34d399',
            border: '1px solid rgba(16, 185, 129, 0.3)',
          }}>
            <CheckCircle2 size={10} /> Active
          </span>
        );
      case 'SUPERSEDED':
        return (
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.2rem',
            padding: '0.15rem 0.4rem',
            borderRadius: '4px',
            fontSize: '0.65rem',
            fontWeight: 700,
            background: 'rgba(245, 158, 11, 0.15)',
            color: '#fbbf24',
            border: '1px solid rgba(245, 158, 11, 0.3)',
          }}>
            <AlertTriangle size={10} /> Superseded
          </span>
        );
      case 'REVOKED':
        return (
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.2rem',
            padding: '0.15rem 0.4rem',
            borderRadius: '4px',
            fontSize: '0.65rem',
            fontWeight: 700,
            background: 'rgba(239, 68, 68, 0.15)',
            color: '#f87171',
            border: '1px solid rgba(239, 68, 68, 0.3)',
          }}>
            <XCircle size={10} /> Revoked
          </span>
        );
      default:
        return null;
    }
  };

  const formatDate = (isoStr?: string | null) => {
    if (!isoStr) return 'Date unknown';
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
      return isoStr;
    }
  };

  return (
    <div style={{
      background: 'rgba(15, 23, 42, 0.7)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      borderRadius: '8px',
      padding: '1.25rem',
      marginBottom: '1.5rem',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={18} color="#818cf8" />
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#f1f5f9' }}>
            Decision Lineage & Evolution Graph
          </h3>
          <span style={{
            fontSize: '0.7rem',
            padding: '0.1rem 0.45rem',
            background: 'rgba(99, 102, 241, 0.15)',
            border: '1px solid rgba(99, 102, 241, 0.3)',
            borderRadius: '4px',
            color: '#a5b4fc',
          }}>
            {edges.length} Lineage Link{edges.length === 1 ? '' : 's'} Detected
          </span>
        </div>
      </div>

      {edges.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '2rem',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: '6px',
          color: '#64748b',
          fontSize: '0.85rem',
        }}>
          <GitCommit size={28} style={{ opacity: 0.5, marginBottom: '0.5rem' }} />
          <p>No multi-meeting decision lineage connections established yet.</p>
          <p style={{ fontSize: '0.75rem', color: '#475569' }}>
            Use the "Supersede" button on any decision card to formally link an earlier policy to its updated decision.
          </p>
        </div>
      ) : (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}>
          {edges.map((edge) => (
            <div
              key={`${edge.from.id}->${edge.to.id}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                borderRadius: '8px',
                padding: '1rem',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              {/* Prior Decision Node */}
              <div
                onClick={() => {
                  setActiveModalDecision(edge.from);
                  if (onSelectDecision) onSelectDecision(edge.from);
                }}
                style={{
                  flex: 1,
                  background: 'rgba(30, 41, 59, 0.8)',
                  border: selectedDecisionId === edge.from.id ? '2px solid #818cf8' : '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '6px',
                  padding: '0.75rem',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  {getStatusBadge(edge.from.status)}
                  <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                    <Calendar size={11} /> {formatDate(edge.from.meeting_date)}
                  </span>
                </div>
                <div style={{ fontSize: '0.825rem', color: '#f8fafc', fontWeight: 500, lineHeight: 1.4 }}>
                  {edge.from.description}
                </div>
                <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '0.4rem' }}>
                  Meeting: {edge.from.meeting_title || 'Untitled Meeting'}
                </div>
              </div>

              {/* Connecting Evolution Indicator */}
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.25rem',
                color: '#818cf8',
              }}>
                <span style={{ fontSize: '0.65rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Superseded By
                </span>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: 'rgba(99, 102, 241, 0.15)',
                  border: '1px solid rgba(99, 102, 241, 0.4)',
                }}>
                  <ArrowRight size={16} />
                </div>
              </div>

              {/* Superseding Decision Node */}
              <div
                onClick={() => {
                  setActiveModalDecision(edge.to);
                  if (onSelectDecision) onSelectDecision(edge.to);
                }}
                style={{
                  flex: 1,
                  background: 'rgba(30, 41, 59, 0.8)',
                  border: selectedDecisionId === edge.to.id ? '2px solid #818cf8' : '1px solid rgba(99, 102, 241, 0.3)',
                  borderRadius: '6px',
                  padding: '0.75rem',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  {getStatusBadge(edge.to.status)}
                  <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                    <Calendar size={11} /> {formatDate(edge.to.meeting_date)}
                  </span>
                </div>
                <div style={{ fontSize: '0.825rem', color: '#f8fafc', fontWeight: 500, lineHeight: 1.4 }}>
                  {edge.to.description}
                </div>
                <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '0.4rem' }}>
                  Meeting: {edge.to.meeting_title || 'Untitled Meeting'}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Decision Detail Modal */}
      {activeModalDecision && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '1rem',
        }}>
          <div style={{
            background: '#1e293b',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            borderRadius: '10px',
            maxWidth: '560px',
            width: '100%',
            padding: '1.5rem',
            position: 'relative',
          }}>
            <button
              type="button"
              onClick={() => setActiveModalDecision(null)}
              style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                background: 'none',
                border: 'none',
                color: '#94a3b8',
                cursor: 'pointer',
              }}
            >
              <X size={18} />
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              {getStatusBadge(activeModalDecision.status)}
              <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                Origin: {activeModalDecision.meeting_title || 'Meeting'} &bull; {formatDate(activeModalDecision.meeting_date)}
              </span>
            </div>

            <h4 style={{ color: '#f8fafc', fontSize: '1rem', fontWeight: 600, margin: '0.5rem 0', lineHeight: 1.4 }}>
              {activeModalDecision.description}
            </h4>

            {activeModalDecision.rationale_or_context && (
              <div style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                borderRadius: '6px',
                padding: '0.75rem',
                margin: '0.75rem 0',
                fontSize: '0.8rem',
                color: '#cbd5e1',
                lineHeight: 1.5,
              }}>
                <div style={{ fontSize: '0.7rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.25rem' }}>
                  Rationale & Context:
                </div>
                {activeModalDecision.rationale_or_context}
              </div>
            )}

            {activeModalDecision.decided_by && (
              <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '1rem' }}>
                Decided By: <strong style={{ color: '#e2e8f0' }}>{activeModalDecision.decided_by}</strong>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '1rem' }}>
              {activeModalDecision.evidence_segment_id && onNavigateToEvidence && (
                <button
                  type="button"
                  onClick={() => {
                    const segId = activeModalDecision.evidence_segment_id!;
                    const mId = activeModalDecision.meeting_id;
                    const ts = activeModalDecision.timestamp_seconds || undefined;
                    setActiveModalDecision(null);
                    onNavigateToEvidence(mId, segId, ts);
                  }}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    padding: '0.45rem 0.85rem',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    background: '#6366f1',
                    border: 'none',
                    color: '#ffffff',
                    cursor: 'pointer',
                  }}
                >
                  <ExternalLink size={13} /> Jump to Transcript Evidence
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
