import React from 'react';
import {
  CheckCircle2,
  Archive,
  Clock,
  AlertTriangle,
  HelpCircle,
  ShieldAlert,
  RefreshCw,
} from 'lucide-react';
import type { OrganizationalInsightsStats } from '../../types';

interface OrganizationalInsightsBarProps {
  stats: OrganizationalInsightsStats | null;
  isLoading: boolean;
  activeMode: string;
  onSelectMode: (mode: 'ask' | 'search' | 'decisions' | 'commitments' | 'questions' | 'conflicts') => void;
  onRefresh?: () => void;
}

export const OrganizationalInsightsBar: React.FC<OrganizationalInsightsBarProps> = ({
  stats,
  isLoading,
  activeMode,
  onSelectMode,
  onRefresh,
}) => {
  return (
    <div className="intel-insights-bar" data-testid="organizational-insights-bar">
      {/* 1. Active Decisions */}
      <button
        type="button"
        className={`intel-stat-card ${activeMode === 'decisions' ? 'active' : ''}`}
        onClick={() => onSelectMode('decisions')}
        title="View active organizational decisions"
      >
        <div className="intel-stat-header">
          <span>Active Decisions</span>
          <CheckCircle2 size={15} color="#10b981" />
        </div>
        <div className="intel-stat-value">
          {isLoading && !stats ? '...' : (stats?.activeDecisions ?? 0)}
          <span className="intel-stat-badge neutral">Live</span>
        </div>
      </button>

      {/* 2. Superseded & Revoked */}
      <button
        type="button"
        className={`intel-stat-card ${activeMode === 'decisions' ? 'active' : ''}`}
        onClick={() => onSelectMode('decisions')}
        title="View superseded and revoked decisions"
      >
        <div className="intel-stat-header">
          <span>Superseded / Revoked</span>
          <Archive size={15} color="#94a3b8" />
        </div>
        <div className="intel-stat-value">
          {isLoading && !stats ? '...' : (stats?.supersededRevokedDecisions ?? 0)}
          <span className="intel-stat-badge neutral">Historical</span>
        </div>
      </button>

      {/* 3. Open Commitments */}
      <button
        type="button"
        className={`intel-stat-card ${activeMode === 'commitments' ? 'active' : ''}`}
        onClick={() => onSelectMode('commitments')}
        title="View open cross-meeting commitments"
      >
        <div className="intel-stat-header">
          <span>Open Commitments</span>
          <Clock size={15} color="#3b82f6" />
        </div>
        <div className="intel-stat-value">
          {isLoading && !stats ? '...' : (stats?.openCommitments ?? 0)}
          <span className="intel-stat-badge neutral">Tracking</span>
        </div>
      </button>

      {/* 4. Overdue Commitments */}
      <button
        type="button"
        className={`intel-stat-card ${activeMode === 'commitments' ? 'active' : ''}`}
        onClick={() => onSelectMode('commitments')}
        title="View overdue commitments across meetings"
      >
        <div className="intel-stat-header">
          <span>Overdue Commitments</span>
          <AlertTriangle size={15} color="#ef4444" />
        </div>
        <div className="intel-stat-value">
          {isLoading && !stats ? '...' : (stats?.overdueCommitments ?? 0)}
          {(stats?.overdueCommitments ?? 0) > 0 ? (
            <span className="intel-stat-badge alert">Requires Review</span>
          ) : (
            <span className="intel-stat-badge neutral">On Track</span>
          )}
        </div>
      </button>

      {/* 5. Open Questions */}
      <button
        type="button"
        className={`intel-stat-card ${activeMode === 'questions' ? 'active' : ''}`}
        onClick={() => onSelectMode('questions')}
        title="View unresolved questions raised in meetings"
      >
        <div className="intel-stat-header">
          <span>Open Questions</span>
          <HelpCircle size={15} color="#8b5cf6" />
        </div>
        <div className="intel-stat-value">
          {isLoading && !stats ? '...' : (stats?.openQuestions ?? 0)}
          <span className="intel-stat-badge neutral">Unresolved</span>
        </div>
      </button>

      {/* 6. Potential Conflicts */}
      <button
        type="button"
        className={`intel-stat-card ${activeMode === 'conflicts' ? 'active' : ''}`}
        onClick={() => onSelectMode('conflicts')}
        title="View heuristic decision conflicts across meetings"
      >
        <div className="intel-stat-header">
          <span>Potential Conflicts</span>
          <ShieldAlert size={15} color="#f59e0b" />
        </div>
        <div className="intel-stat-value">
          {isLoading && !stats ? '...' : (stats?.potentialConflicts ?? 0)}
          {(stats?.potentialConflicts ?? 0) > 0 ? (
            <span className="intel-stat-badge warning">Heuristic</span>
          ) : (
            <span className="intel-stat-badge neutral">Clear</span>
          )}
        </div>
      </button>

      {/* Optional Refresh control */}
      {onRefresh && (
        <button
          type="button"
          className="intel-btn-sm"
          style={{ alignSelf: 'center', height: 'fit-content', padding: '0.5rem' }}
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh cross-meeting metrics"
        >
          <RefreshCw size={14} className={isLoading ? 'spin' : ''} />
          <span>Refresh</span>
        </button>
      )}
    </div>
  );
};
