import React from 'react';
import { Sparkles } from 'lucide-react';
import './FloatingAskMemora.css';

export interface FloatingAskMemoraProps {
  onClick: () => void;
  activeMeetingTitle?: string | null;
  activeProjectName?: string | null;
  isVisible?: boolean;
}

export const FloatingAskMemora: React.FC<FloatingAskMemoraProps> = ({
  onClick,
  activeMeetingTitle,
  activeProjectName,
  isVisible = true,
}) => {
  if (!isVisible) return null;

  const contextHint = activeMeetingTitle
    ? `Meeting: ${activeMeetingTitle}`
    : activeProjectName
    ? `Project: ${activeProjectName}`
    : 'Organizational Memory';

  return (
    <div className="floating-ask-memora-wrapper" role="region" aria-label="AI Assistant Shortcut">
      <button
        type="button"
        className="floating-ask-memora-btn"
        onClick={onClick}
        aria-label={`Ask MEMORA (${contextHint})`}
        title={`Ask MEMORA anything about ${contextHint} (Shortcut)`}
        data-testid="floating-ask-memora-btn"
      >
        <div className="floating-btn-glow" />
        <div className="floating-btn-icon-pulse">
          <Sparkles size={18} className="floating-btn-icon" />
        </div>
        <span className="floating-btn-text">Ask MEMORA</span>
        {activeMeetingTitle && (
          <span className="floating-btn-context-pill" title={activeMeetingTitle}>
            Meeting
          </span>
        )}
      </button>
    </div>
  );
};
