import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import {
  Sparkles,
  Search,
  MessageSquare,
  Layers,
  AlertTriangle,
  Info,
  CheckCircle2,
  RefreshCw,
  Cpu,
  ArrowRight,
  Shield,
  Hash,
  GitBranch,
  CheckSquare,
  HelpCircle,
  ShieldAlert,
  Archive,
  AlertOctagon,
  ChevronDown,
  ChevronUp,
  Clock,
  ExternalLink,
  Globe,
  BookOpen,
  FileAudio,
  AlertCircle,
  Users,
} from 'lucide-react';
import type {
  MeetingItem,
  ProjectItem,
  MemoryQueryResponse,
  MemorySearchResponse,
  AbstentionStatus,
  OrganizationalInsightsStats,
} from '../types';
import {
  queryOrganizationalMemory,
  searchOrganizationalMemory,
  fetchCrossMeetingDecisions,
  fetchCrossMeetingCommitments,
  fetchCrossMeetingQuestions,
  fetchDecisionConflicts,
  isAbortError,
} from '../services/api';
import './OrganizationalMemoryView.css';
import './cross_meeting/CrossMeetingIntelligence.css';

import { OrganizationalInsightsBar } from './cross_meeting/OrganizationalInsightsBar';
import { DecisionHistoryView } from './cross_meeting/DecisionHistoryView';
import { CommitmentTrackingView } from './cross_meeting/CommitmentTrackingView';
import { UnresolvedQuestionsView } from './cross_meeting/UnresolvedQuestionsView';
import { PotentialConflictsView } from './cross_meeting/PotentialConflictsView';

interface OrganizationalMemoryViewProps {
  meetings: MeetingItem[];
  projects?: ProjectItem[];
  onNavigateToEvidence: (meetingId: string, segmentId: string, timestampSeconds?: number | null) => void;
  currentOrgName?: string;
  currentOrgRole?: string;
  currentOrgId?: string;
  initialMeetingId?: string;
  initialProjectId?: string;
  focusInputTrigger?: number;
}

const renderInlineMarkdown = (text: string): React.ReactNode => {
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*.*?\*\*|\*.*?\*|\[.*?\]\(.*?\))/g;
  let lastIdx = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.slice(lastIdx, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(<strong key={match.index}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(<em key={match.index}>{token.slice(1, -1)}</em>);
    } else if (token.startsWith('[') && token.includes('](') && token.endsWith(')')) {
      const splitIdx = token.indexOf('](');
      const label = token.slice(1, splitIdx);
      const url = token.slice(splitIdx + 2, -1);
      parts.push(
        <a key={match.index} href={url} target="_blank" rel="noopener noreferrer" className="memory-ext-link">
          {label} <ExternalLink size={11} style={{ display: 'inline', verticalAlign: 'middle', marginLeft: '2px' }} />
        </a>
      );
    }
    lastIdx = regex.lastIndex;
  }
  if (lastIdx < text.length) {
    parts.push(text.slice(lastIdx));
  }
  return parts.length > 0 ? parts : text;
};

const renderFormattedAnswer = (text: string) => {
  if (!text) return null;

  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let listItems: string[] = [];

  const flushList = (keyPrefix: string) => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`${keyPrefix}-ul`} className="memory-answer-list">
          {listItems.map((li, idx) => (
            <li key={idx}>{renderInlineMarkdown(li)}</li>
          ))}
        </ul>
      );
      listItems = [];
    }
  };

  lines.forEach((line, i) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList(`flush-${i}`);
      return;
    }

    if (trimmed.startsWith('### ')) {
      flushList(`h3-${i}`);
      elements.push(
        <h4 key={`h3-${i}`} className="memory-answer-heading">
          {trimmed.slice(4)}
        </h4>
      );
    } else if (trimmed.startsWith('## ')) {
      flushList(`h2-${i}`);
      elements.push(
        <h3 key={`h2-${i}`} className="memory-answer-heading-lg">
          {trimmed.slice(3)}
        </h3>
      );
    } else if (trimmed.startsWith('> ')) {
      flushList(`quote-${i}`);
      elements.push(
        <blockquote key={`quote-${i}`} className="memory-answer-blockquote">
          {renderInlineMarkdown(trimmed.slice(2))}
        </blockquote>
      );
    } else if (trimmed.startsWith('- ') || trimmed.startsWith('• ') || trimmed.startsWith('* ')) {
      listItems.push(trimmed.slice(2));
    } else {
      flushList(`p-${i}`);
      elements.push(
        <p key={`p-${i}`} className="memory-answer-paragraph">
          {renderInlineMarkdown(trimmed)}
        </p>
      );
    }
  });

  flushList('final-flush');
  return <div className="memory-formatted-content">{elements}</div>;
};

export const OrganizationalMemoryView: React.FC<OrganizationalMemoryViewProps> = ({
  meetings,
  projects = [],
  onNavigateToEvidence,
  currentOrgName,
  currentOrgRole,
  currentOrgId,
  initialMeetingId,
  initialProjectId,
  focusInputTrigger,
}) => {
  const [activeMode, setActiveMode] = useState<'ask' | 'search' | 'decisions' | 'commitments' | 'questions' | 'conflicts'>('ask');
  const [selectedProjectId, setSelectedProjectId] = useState<string>(initialProjectId || '');
  const [savedHistory, setSavedHistory] = useState<string[]>(() => {
    try {
      const stored = localStorage.getItem('memora_ask_ai_history');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  // --- Insights Stats State ---
  const [stats, setStats] = useState<OrganizationalInsightsStats | null>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const statsAbortControllerRef = useRef<AbortController | null>(null);
  const statsEpochRef = useRef<number>(0);

  // --- Ask AI State ---
  const [question, setQuestion] = useState('');
  const [askMeetingId, setAskMeetingId] = useState<string>(initialMeetingId || '');

  // Synchronize initial meeting / project / focus trigger changes
  useEffect(() => {
    if (initialMeetingId) {
      setAskMeetingId(initialMeetingId);
    }
  }, [initialMeetingId]);

  useEffect(() => {
    if (initialProjectId) {
      setSelectedProjectId(initialProjectId);
    }
  }, [initialProjectId]);

  useEffect(() => {
    if (focusInputTrigger) {
      setActiveMode('ask');
      const timer = setTimeout(() => {
        const inputEl = document.getElementById('memory-question-input') as HTMLTextAreaElement | null;
        if (inputEl) {
          inputEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
          inputEl.focus();
        }
      }, 60);
      return () => clearTimeout(timer);
    }
  }, [focusInputTrigger]);
  const [askStartDate, setAskStartDate] = useState<string>('');
  const [askEndDate, setAskEndDate] = useState<string>('');
  const [maxEvidenceChunks, setMaxEvidenceChunks] = useState<number>(5);
  const [isLoadingQuery, setIsLoadingQuery] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryResponse, setQueryResponse] = useState<MemoryQueryResponse | null>(null);
  const queryAbortControllerRef = useRef<AbortController | null>(null);
  const queryRequestEpochRef = useRef<number>(0);

  // Expandable sections for structured intelligence in Ask AI
  const [isDecisionsExpanded, setIsDecisionsExpanded] = useState<boolean>(true);
  const [isCommitmentsExpanded, setIsCommitmentsExpanded] = useState<boolean>(true);
  const [isQuestionsExpanded, setIsQuestionsExpanded] = useState<boolean>(true);
  const [isConflictsExpanded, setIsConflictsExpanded] = useState<boolean>(true);

  // --- Search Memory State ---
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMeetingId, setSearchMeetingId] = useState<string>('');
  const [searchStartDate, setSearchStartDate] = useState<string>('');
  const [searchEndDate, setSearchEndDate] = useState<string>('');
  const [isLoadingSearch, setIsLoadingSearch] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchResponse, setSearchResponse] = useState<MemorySearchResponse | null>(null);
  const searchAbortControllerRef = useRef<AbortController | null>(null);
  const searchRequestEpochRef = useRef<number>(0);

  // Format seconds to mm:ss
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Context-aware suggested questions for Ask AI
  const suggestedQuestions = useMemo(() => {
    const selectedMeeting = askMeetingId ? meetings.find((m) => m.id === askMeetingId) : null;
    const selectedProject = selectedProjectId ? projects.find((p) => p.id === selectedProjectId) : null;

    if (selectedMeeting) {
      return [
        {
          category: 'Decisions',
          text: `What were the most important decisions in "${selectedMeeting.title}"?`,
          icon: <CheckCircle2 size={12} color="#34d399" />,
        },
        {
          category: 'Action Items',
          text: `What are the pending action items from this meeting?`,
          icon: <Clock size={12} color="#60a5fa" />,
        },
        {
          category: 'Responsibilities',
          text: `What responsibilities were assigned to each participant in this meeting?`,
          icon: <Users size={12} color="#a78bfa" />,
        },
        {
          category: 'Unresolved',
          text: `What unresolved questions or open risks remain from this meeting?`,
          icon: <HelpCircle size={12} color="#f59e0b" />,
        },
        {
          category: 'External & Legal',
          text: `Explain any technical terms or legal amendments mentioned in this meeting.`,
          icon: <Globe size={12} color="#38bdf8" />,
        },
        {
          category: 'Topics',
          text: `What was discussed regarding key agenda topics in "${selectedMeeting.title}"?`,
          icon: <BookOpen size={12} color="#ec4899" />,
        },
      ];
    }

    if (selectedProject) {
      return [
        {
          category: 'Decisions',
          text: `What key decisions were made across ${selectedProject.name}?`,
          icon: <CheckCircle2 size={12} color="#34d399" />,
        },
        {
          category: 'Evolution',
          text: `What changed between the previous and latest meetings in ${selectedProject.name}?`,
          icon: <GitBranch size={12} color="#818cf8" />,
        },
        {
          category: 'Action Items',
          text: `What are the pending action items and commitments for ${selectedProject.name}?`,
          icon: <Clock size={12} color="#60a5fa" />,
        },
        {
          category: 'Responsibilities',
          text: `What responsibilities were assigned to each participant in ${selectedProject.name}?`,
          icon: <Users size={12} color="#a78bfa" />,
        },
        {
          category: 'Unresolved',
          text: `What contradictory decisions or unresolved questions exist in ${selectedProject.name}?`,
          icon: <HelpCircle size={12} color="#f59e0b" />,
        },
        {
          category: 'Technical',
          text: `Explain the technical specifications or architecture adopted in ${selectedProject.name}.`,
          icon: <Globe size={12} color="#38bdf8" />,
        },
      ];
    }

    return [
      {
        category: 'Decisions',
        text: 'What were the most important decisions across recent meetings?',
        icon: <CheckCircle2 size={12} color="#34d399" />,
      },
      {
        category: 'Action Items',
        text: 'What are the pending action items and who is responsible for each?',
        icon: <Clock size={12} color="#60a5fa" />,
      },
      {
        category: 'Evolution',
        text: 'What changed between the previous and latest meetings?',
        icon: <GitBranch size={12} color="#818cf8" />,
      },
      {
        category: 'Responsibilities',
        text: 'What responsibilities were assigned to each participant?',
        icon: <Users size={12} color="#a78bfa" />,
      },
      {
        category: 'Unresolved',
        text: 'What unresolved questions or open issues remain across meetings?',
        icon: <HelpCircle size={12} color="#f59e0b" />,
      },
      {
        category: 'External & Legal',
        text: 'Explain the 103rd Constitutional Amendment or other technical terms mentioned.',
        icon: <Globe size={12} color="#38bdf8" />,
      },
      {
        category: 'Topics',
        text: 'What was discussed about infrastructure, deployment, and security?',
        icon: <BookOpen size={12} color="#ec4899" />,
      },
    ];
  }, [askMeetingId, selectedProjectId, meetings, projects]);

  // --- Load Live Insights Stats ---
  const loadStats = useCallback(async () => {
    if (statsAbortControllerRef.current) {
      statsAbortControllerRef.current.abort();
    }
    const controller = new AbortController();
    statsAbortControllerRef.current = controller;
    const epoch = ++statsEpochRef.current;

    setIsLoadingStats(true);
    try {
      const [decisionsRes, commitmentsRes, questionsRes, conflictsRes] = await Promise.allSettled([
        fetchCrossMeetingDecisions({}, controller.signal),
        fetchCrossMeetingCommitments({}, controller.signal),
        fetchCrossMeetingQuestions({}, controller.signal),
        fetchDecisionConflicts({}, controller.signal),
      ]);

      if (epoch === statsEpochRef.current) {
        let activeDecisions = 0;
        let supersededRevokedDecisions = 0;
        if (decisionsRes.status === 'fulfilled') {
          const decs = decisionsRes.value;
          activeDecisions = decs.filter((d) => d.status === 'ACTIVE').length;
          supersededRevokedDecisions = decs.filter(
            (d) => d.status === 'SUPERSEDED' || d.status === 'REVOKED'
          ).length;
        }

        let openCommitments = 0;
        let overdueCommitments = 0;
        if (commitmentsRes.status === 'fulfilled') {
          const comms = commitmentsRes.value;
          openCommitments = comms.filter(
            (c) => c.status === 'PENDING' || c.status === 'IN_PROGRESS'
          ).length;
          overdueCommitments = comms.filter((c) => c.is_overdue).length;
        }

        let openQuestions = 0;
        if (questionsRes.status === 'fulfilled') {
          const qs = questionsRes.value;
          openQuestions = qs.filter((q) => q.status === 'OPEN').length;
        }

        let potentialConflicts = 0;
        if (conflictsRes.status === 'fulfilled') {
          potentialConflicts = conflictsRes.value.length;
        }

        setStats({
          activeDecisions,
          supersededRevokedDecisions,
          openCommitments,
          overdueCommitments,
          openQuestions,
          potentialConflicts,
        });
      }
    } catch (err: unknown) {
      if (!isAbortError(err)) {
        console.error('Failed to load organizational stats:', err);
      }
    } finally {
      if (epoch === statsEpochRef.current) {
        setIsLoadingStats(false);
      }
    }
  }, []);

  useEffect(() => {
    // When organization boundary changes, abort any active queries and reset results
    if (queryAbortControllerRef.current) {
      queryAbortControllerRef.current.abort();
    }
    if (searchAbortControllerRef.current) {
      searchAbortControllerRef.current.abort();
    }
    ++queryRequestEpochRef.current;
    ++searchRequestEpochRef.current;
    setQueryResponse(null);
    setQueryError(null);
    setIsLoadingQuery(false);
    setSearchResponse(null);
    setSearchError(null);
    setIsLoadingSearch(false);

    loadStats();
    return () => {
      if (statsAbortControllerRef.current) {
        statsAbortControllerRef.current.abort();
      }
      if (queryAbortControllerRef.current) {
        queryAbortControllerRef.current.abort();
      }
      if (searchAbortControllerRef.current) {
        searchAbortControllerRef.current.abort();
      }
    };
  }, [loadStats, currentOrgId]);

  // --- Handle Ask AI Submission ---
  const handleAskSubmit = useCallback(async (e?: React.FormEvent, overrideQuestion?: string) => {
    if (e) e.preventDefault();
    const cleanQ = (overrideQuestion !== undefined ? overrideQuestion : question).trim();
    if (!cleanQ) {
      setQueryError('Please enter a question to ask organizational memory.');
      return;
    }
    setQuestion(cleanQ);

    if (askStartDate && askEndDate && askStartDate > askEndDate) {
      setQueryError('Start date cannot be later than end date.');
      return;
    }

    // Update saved history
    setSavedHistory((prev) => {
      const next = [cleanQ, ...prev.filter((q) => q !== cleanQ)].slice(0, 8);
      try {
        localStorage.setItem('memora_ask_ai_history', JSON.stringify(next));
      } catch {}
      return next;
    });

    // Cancel in-flight query request if any
    if (queryAbortControllerRef.current) {
      queryAbortControllerRef.current.abort();
    }
    const controller = new AbortController();
    queryAbortControllerRef.current = controller;
    const epoch = ++queryRequestEpochRef.current;

    setIsLoadingQuery(true);
    setQueryError(null);
    setQueryResponse(null);

    try {
      const resp = await queryOrganizationalMemory(
        {
          question: cleanQ,
          meeting_id: askMeetingId || undefined,
          project_id: selectedProjectId || undefined,
          start_date: askStartDate ? new Date(askStartDate).toISOString() : undefined,
          end_date: askEndDate ? new Date(askEndDate).toISOString() : undefined,
          max_evidence_chunks: maxEvidenceChunks,
        },
        controller.signal
      );
      if (epoch === queryRequestEpochRef.current) {
        setQueryResponse(resp);
      }
    } catch (err: unknown) {
      if (!isAbortError(err) && epoch === queryRequestEpochRef.current) {
        setQueryError(err instanceof Error ? err.message : 'Failed to query organizational memory');
      }
    } finally {
      if (epoch === queryRequestEpochRef.current) {
        setIsLoadingQuery(false);
      }
    }
  }, [question, askMeetingId, selectedProjectId, askStartDate, askEndDate, maxEvidenceChunks]);

  // --- Handle Search Submission ---
  const handleSearchSubmit = useCallback(async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const cleanQ = searchQuery.trim();
    if (!cleanQ) {
      setSearchError('Please enter search terms to search memory passages.');
      return;
    }

    if (searchStartDate && searchEndDate && searchStartDate > searchEndDate) {
      setSearchError('Start date cannot be later than end date.');
      return;
    }

    // Cancel in-flight search request if any
    if (searchAbortControllerRef.current) {
      searchAbortControllerRef.current.abort();
    }
    const controller = new AbortController();
    searchAbortControllerRef.current = controller;
    const epoch = ++searchRequestEpochRef.current;

    setIsLoadingSearch(true);
    setSearchError(null);
    setSearchResponse(null);

    try {
      const resp = await searchOrganizationalMemory(
        {
          query: cleanQ,
          meeting_id: searchMeetingId || undefined,
          project_id: selectedProjectId || undefined,
          start_date: searchStartDate ? new Date(searchStartDate).toISOString() : undefined,
          end_date: searchEndDate ? new Date(searchEndDate).toISOString() : undefined,
          limit: 20,
        },
        controller.signal
      );
      if (epoch === searchRequestEpochRef.current) {
        setSearchResponse(resp);
      }
    } catch (err: unknown) {
      if (!isAbortError(err) && epoch === searchRequestEpochRef.current) {
        setSearchError(err instanceof Error ? err.message : 'Failed to search memory passages');
      }
    } finally {
      if (epoch === searchRequestEpochRef.current) {
        setIsLoadingSearch(false);
      }
    }
  }, [searchQuery, searchMeetingId, selectedProjectId, searchStartDate, searchEndDate]);

  // Render provider badge
  const renderProviderBadge = (providerUsed: string, isHeuristic: boolean) => {
    if (isHeuristic) {
      return (
        <span className="memory-provider-badge heuristic" title="Generated using deterministic rule extraction.">
          <Layers size={12} />
          Deterministic Fallback
        </span>
      );
    }
    if (providerUsed === 'gemini') {
      return (
        <span className="memory-provider-badge gemini" title="Generated via Google Gemini cloud execution.">
          <Sparkles size={12} />
          Google Gemini (Generative)
        </span>
      );
    }
    if (providerUsed === 'ollama') {
      return (
        <span className="memory-provider-badge ollama" title="Generated via local Ollama LLM execution.">
          <Sparkles size={12} />
          Ollama LLM (Generative)
        </span>
      );
    }
    return (
      <span className="memory-provider-badge deterministic">
        <Cpu size={12} />
        {providerUsed}
      </span>
    );
  };

  // Render abstention badge
  const renderAbstentionBadge = (status: AbstentionStatus | null, isAbstained: boolean) => {
    if (!isAbstained && status === 'ANSWER_SUPPORTED') {
      return (
        <span className="memory-status-badge supported">
          <CheckCircle2 size={12} />
          Supported by Evidence
        </span>
      );
    }
    switch (status) {
      case 'NO_EVIDENCE':
        return (
          <span className="memory-status-badge no-evidence">
            <Info size={12} />
            No Meeting Evidence Found
          </span>
        );
      case 'INSUFFICIENT_EVIDENCE':
        return (
          <span className="memory-status-badge insufficient">
            <AlertTriangle size={12} />
            Insufficient Meeting Evidence
          </span>
        );
      case 'CONFLICTING_EVIDENCE':
        return (
          <span className="memory-status-badge conflicting">
            <AlertTriangle size={12} />
            Conflicting Evidence Across Meetings
          </span>
        );
      case 'MODEL_UNAVAILABLE':
        return (
          <span className="memory-status-badge model-unavailable">
            <Cpu size={12} />
            Embedding Model Unavailable
          </span>
        );
      case 'RETRIEVAL_UNAVAILABLE':
        return (
          <span className="memory-status-badge conflicting">
            <AlertTriangle size={12} />
            Retrieval Subsystem Unavailable
          </span>
        );
      default:
        return null;
    }
  };

  // Render decision status badge
  const renderDecisionStatusBadge = (status: string) => {
    switch (status) {
      case 'ACTIVE':
        return (
          <span className="badge-status active" data-testid="decision-badge-active">
            <CheckCircle2 size={11} />
            Active
          </span>
        );
      case 'SUPERSEDED':
        return (
          <span className="badge-status superseded" data-testid="decision-badge-superseded">
            <Archive size={11} />
            Superseded
          </span>
        );
      case 'REVOKED':
        return (
          <span className="badge-status revoked" data-testid="decision-badge-revoked">
            <AlertOctagon size={11} />
            Revoked
          </span>
        );
      default:
        return <span className="badge-status pending">{status}</span>;
    }
  };

  // Render commitment status badge
  const renderCommitmentStatusBadge = (status: string, isOverdue: boolean) => {
    if (isOverdue) {
      return (
        <span className="badge-status overdue" data-testid="commitment-badge-overdue">
          <AlertTriangle size={11} />
          Overdue
        </span>
      );
    }
    switch (status) {
      case 'COMPLETED':
        return (
          <span className="badge-status completed" data-testid="commitment-badge-completed">
            <CheckCircle2 size={11} />
            Completed
          </span>
        );
      case 'IN_PROGRESS':
        return (
          <span className="badge-status in-progress" data-testid="commitment-badge-in-progress">
            In Progress
          </span>
        );
      case 'CANCELLED':
        return (
          <span className="badge-status cancelled" data-testid="commitment-badge-cancelled">
            <AlertOctagon size={11} />
            Cancelled
          </span>
        );
      case 'PENDING':
      default:
        return (
          <span className="badge-status pending" data-testid="commitment-badge-pending">
            Pending
          </span>
        );
    }
  };

  // Render question status badge
  const renderQuestionStatusBadge = (status: string) => {
    switch (status) {
      case 'RESOLVED':
        return (
          <span className="badge-status resolved" data-testid="question-badge-resolved">
            <CheckCircle2 size={11} />
            Resolved
          </span>
        );
      case 'DEFERRED':
        return (
          <span className="badge-status deferred" data-testid="question-badge-deferred">
            <Clock size={11} />
            Deferred
          </span>
        );
      case 'OPEN':
      default:
        return (
          <span className="badge-status open" data-testid="question-badge-open">
            <HelpCircle size={11} />
            Open
          </span>
        );
    }
  };

  return (
    <div className="memory-view-container" data-testid="organizational-memory-view">
      {/* Top Header Card */}
      <div className="memory-view-header">
        <div className="memory-header-info">
          <div className="memory-title-row">
            <Sparkles size={22} color="#6366f1" />
            Organizational Memory & Cross-Meeting Intelligence
          </div>
          <div className="memory-subtitle">
            Cross-meeting semantic intelligence, decision lineage, commitments, and conflict resolution for{' '}
            <strong>{currentOrgName || 'your organization'}</strong>. Grounded strictly in privacy-safe transcripts.
          </div>
        </div>

        {/* Mode Selector */}
        <div className="memory-mode-toggle" role="tablist" aria-label="Memory view modes">
          <button
            role="tab"
            aria-selected={activeMode === 'ask'}
            className={`memory-mode-btn ${activeMode === 'ask' ? 'active' : ''}`}
            onClick={() => setActiveMode('ask')}
            data-testid="mode-tab-ask"
          >
            <MessageSquare size={15} />
            Ask AI
          </button>
          <button
            role="tab"
            aria-selected={activeMode === 'search'}
            className={`memory-mode-btn ${activeMode === 'search' ? 'active' : ''}`}
            onClick={() => setActiveMode('search')}
            data-testid="mode-tab-search"
          >
            <Search size={15} />
            Search Memory
          </button>
          <button
            role="tab"
            aria-selected={activeMode === 'decisions'}
            className={`memory-mode-btn ${activeMode === 'decisions' ? 'active' : ''}`}
            onClick={() => setActiveMode('decisions')}
            data-testid="mode-tab-decisions"
          >
            <GitBranch size={15} />
            Decisions
          </button>
          <button
            role="tab"
            aria-selected={activeMode === 'commitments'}
            className={`memory-mode-btn ${activeMode === 'commitments' ? 'active' : ''}`}
            onClick={() => setActiveMode('commitments')}
            data-testid="mode-tab-commitments"
          >
            <CheckSquare size={15} />
            Commitments
          </button>
          <button
            role="tab"
            aria-selected={activeMode === 'questions'}
            className={`memory-mode-btn ${activeMode === 'questions' ? 'active' : ''}`}
            onClick={() => setActiveMode('questions')}
            data-testid="mode-tab-questions"
          >
            <HelpCircle size={15} />
            Questions
          </button>
          <button
            role="tab"
            aria-selected={activeMode === 'conflicts'}
            className={`memory-mode-btn ${activeMode === 'conflicts' ? 'active' : ''}`}
            onClick={() => setActiveMode('conflicts')}
            data-testid="mode-tab-conflicts"
          >
            <ShieldAlert size={15} />
            Conflicts
          </button>
        </div>
      </div>

      {/* Organizational Insights Metrics Bar */}
      <OrganizationalInsightsBar
        stats={stats}
        isLoading={isLoadingStats}
        activeMode={activeMode}
        onSelectMode={(mode) => setActiveMode(mode as typeof activeMode)}
        onRefresh={loadStats}
      />

      {/* ========================================================================= */}
      {/* MODE 1: ASK ORGANIZATIONAL AI                                             */}
      {/* ========================================================================= */}
      {activeMode === 'ask' && (
        <div className="memory-card">
          <form onSubmit={handleAskSubmit} className="memory-input-group">
            <label htmlFor="memory-question-input" className="memory-input-label">
              Ask a question across your organization's recorded meetings:
            </label>
            <textarea
              id="memory-question-input"
              className="memory-textarea"
              placeholder="e.g., What was finalized regarding container orchestration or database migration?"
              value={question}
              onChange={(e) => {
                setQuestion(e.target.value);
                if (queryError) setQueryError(null);
              }}
              rows={3}
              data-testid="memory-question-input"
            />

            {/* Suggested Questions Section */}
            <div className="memory-suggested-section" data-testid="suggested-questions-section">
              <div className="memory-suggested-header">
                <span className="memory-examples-label" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                  <Sparkles size={12} color="#818cf8" /> Suggested Questions:
                </span>
                <span className="memory-suggested-hint">
                  {askMeetingId ? 'Adapted to selected meeting' : selectedProjectId ? 'Adapted to selected project' : 'Click to populate or run'}
                </span>
              </div>
              <div className="memory-suggested-pills-grid">
                {suggestedQuestions.map((sug, i) => (
                  <div key={i} className="memory-suggested-chip">
                    <button
                      type="button"
                      className="memory-example-btn memory-suggested-btn"
                      onClick={() => {
                        setQuestion(sug.text);
                        if (queryError) setQueryError(null);
                      }}
                      title="Click to populate question into input"
                    >
                      <span className="suggested-chip-icon">{sug.icon}</span>
                      <span className="suggested-chip-category">{sug.category}:</span>
                      <span className="suggested-chip-text">"{sug.text}"</span>
                    </button>
                    <button
                      type="button"
                      className="memory-suggested-quick-run"
                      onClick={() => handleAskSubmit(undefined, sug.text)}
                      title="Directly Ask AI this question"
                      aria-label={`Directly ask: ${sug.text}`}
                    >
                      <ArrowRight size={11} />
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Saved Recent Questions */}
            {savedHistory.length > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap', margin: '0.4rem 0' }}>
                <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                  <Clock size={11} /> Recent:
                </span>
                {savedHistory.map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setQuestion(item);
                      if (queryError) setQueryError(null);
                    }}
                    style={{
                      background: 'rgba(255, 255, 255, 0.04)',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      borderRadius: '12px',
                      padding: '0.15rem 0.55rem',
                      fontSize: '0.7rem',
                      color: '#cbd5e1',
                      cursor: 'pointer',
                    }}
                  >
                    {item.length > 40 ? item.slice(0, 38) + '...' : item}
                  </button>
                ))}
              </div>
            )}

            {/* Scope & Date Filters */}
            <div className="memory-filters-row">
              {projects.length > 0 && (
                <div className="memory-filter-item">
                  <label className="memory-filter-label" htmlFor="ask-project-filter">
                    Scope to Project:
                  </label>
                  <select
                    id="ask-project-filter"
                    className="memory-select"
                    value={selectedProjectId}
                    onChange={(e) => setSelectedProjectId(e.target.value)}
                    data-testid="ask-project-filter"
                  >
                    <option value="">All Projects Across Organization</option>
                    {projects.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div className="memory-filter-item">
                <label className="memory-filter-label" htmlFor="ask-meeting-filter">
                  Scope to Meeting:
                </label>
                <select
                  id="ask-meeting-filter"
                  className="memory-select"
                  value={askMeetingId}
                  onChange={(e) => setAskMeetingId(e.target.value)}
                  data-testid="ask-meeting-filter"
                >
                  <option value="">All Meetings Across Organization</option>
                  {meetings.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.title || 'Untitled Meeting'} ({new Date(m.created_at).toLocaleDateString()})
                    </option>
                  ))}
                </select>
              </div>

              <div className="memory-filter-item">
                <label className="memory-filter-label" htmlFor="ask-start-date">
                  Start Date:
                </label>
                <input
                  type="date"
                  id="ask-start-date"
                  className="memory-date-input"
                  value={askStartDate}
                  onChange={(e) => setAskStartDate(e.target.value)}
                  data-testid="ask-start-date"
                />
              </div>

              <div className="memory-filter-item">
                <label className="memory-filter-label" htmlFor="ask-end-date">
                  End Date:
                </label>
                <input
                  type="date"
                  id="ask-end-date"
                  className="memory-date-input"
                  value={askEndDate}
                  onChange={(e) => setAskEndDate(e.target.value)}
                  data-testid="ask-end-date"
                />
              </div>

              <div className="memory-filter-item">
                <label className="memory-filter-label" htmlFor="ask-chunks-limit">
                  Max Evidence:
                </label>
                <select
                  id="ask-chunks-limit"
                  className="memory-select"
                  value={maxEvidenceChunks}
                  onChange={(e) => setMaxEvidenceChunks(Number(e.target.value))}
                  data-testid="ask-chunks-limit"
                >
                  <option value={3}>3 chunks</option>
                  <option value={5}>5 chunks (default)</option>
                  <option value={10}>10 chunks</option>
                </select>
              </div>

              <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'flex-end' }}>
                <button
                  type="submit"
                  className="memory-submit-btn"
                  disabled={isLoadingQuery || !question.trim()}
                  data-testid="ask-submit-btn"
                >
                  {isLoadingQuery ? (
                    <>
                      <RefreshCw size={15} className="spin" />
                      <span>Synthesizing Answer...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={15} />
                      <span>Ask AI</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>

          {/* Error Message */}
          {queryError && (
            <div className="memory-error-box" data-testid="query-error-box">
              <AlertTriangle size={18} color="#ef4444" />
              <span>{queryError}</span>
            </div>
          )}

          {/* Loading State */}
          {isLoadingQuery && (
            <div className="memory-loading-box">
              <RefreshCw size={28} className="spin" color="#6366f1" />
              <div>
                <strong>Synthesizing organizational memory answer...</strong>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.25rem' }}>
                  Retrieving semantically matched passages from privacy-redacted transcripts and verifying facts.
                </div>
              </div>
            </div>
          )}

          {/* Query Response Section */}
          {queryResponse && !isLoadingQuery && (
            <div className="memory-answer-card" data-testid="query-response-card">
              {/* Answer Source Indicator Banner */}
              {(() => {
                const sourceType = queryResponse.answer_source_type || (() => {
                  const hasExt = Boolean(queryResponse.external_knowledge && queryResponse.external_knowledge.is_available);
                  const hasEv = Boolean(
                    queryResponse.citations.length > 0 ||
                    (queryResponse.structured_decisions && queryResponse.structured_decisions.length > 0) ||
                    (queryResponse.structured_commitments && queryResponse.structured_commitments.length > 0)
                  );
                  if (queryResponse.is_abstained || (!hasEv && !hasExt)) return 'INSUFFICIENT_EVIDENCE';
                  if (hasEv && hasExt) return 'COMBINED';
                  if (hasExt && !hasEv) return 'EXTERNAL_KNOWLEDGE';
                  return 'MEETING_EVIDENCE';
                })();

                return (
                  <div
                    className={`memory-source-indicator-banner source-${sourceType.toLowerCase()}`}
                    data-testid="answer-source-indicator"
                  >
                    <div className="source-indicator-header">
                      <div className="source-indicator-badge">
                        {sourceType === 'MEETING_EVIDENCE' && (
                          <>
                            <FileAudio size={14} color="#34d399" />
                            <span className="source-label">Source: Meeting Evidence</span>
                          </>
                        )}
                        {sourceType === 'EXTERNAL_KNOWLEDGE' && (
                          <>
                            <Globe size={14} color="#38bdf8" />
                            <span className="source-label">Source: External Knowledge</span>
                          </>
                        )}
                        {sourceType === 'COMBINED' && (
                          <>
                            <Sparkles size={14} color="#c084fc" />
                            <span className="source-label">Source: Combined Answer</span>
                          </>
                        )}
                        {sourceType === 'INSUFFICIENT_EVIDENCE' && (
                          <>
                            <AlertCircle size={14} color="#fbbf24" />
                            <span className="source-label">Source: Insufficient Evidence</span>
                          </>
                        )}
                      </div>
                      <span className="source-indicator-type-tag">
                        {sourceType.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="source-indicator-desc">
                      {sourceType === 'MEETING_EVIDENCE' &&
                        'Information retrieved and verified directly from organization meeting transcripts.'}
                      {sourceType === 'EXTERNAL_KNOWLEDGE' &&
                        'Information obtained from external knowledge sources to explain subjects mentioned in meetings.'}
                      {sourceType === 'COMBINED' &&
                        'Both meeting evidence and external knowledge were used to synthesize this answer.'}
                      {sourceType === 'INSUFFICIENT_EVIDENCE' &&
                        'Available meeting records and external knowledge do not provide sufficient verified evidence.'}
                    </div>
                  </div>
                );
              })()}

              {/* Answer Header Bar */}
              <div className="memory-answer-header">
                <div className="memory-answer-badges">
                  {renderAbstentionBadge(queryResponse.abstention_status, queryResponse.is_abstained)}
                  {renderProviderBadge(
                    queryResponse.diagnostics.provider_used,
                    queryResponse.diagnostics.is_heuristic
                  )}
                  {queryResponse.citations.length > 0 && (
                    <span
                      className={`intel-stat-badge ${queryResponse.abstention_status === 'CONFLICTING_EVIDENCE' ? 'conflict' : 'neutral'}`}
                      data-testid="evidence-count-badge"
                    >
                      <Hash size={11} />
                      {queryResponse.abstention_status === 'CONFLICTING_EVIDENCE'
                        ? `${queryResponse.citations.length} Conflicting Evidence Sources`
                        : `${queryResponse.citations.length} Verified Evidence Chunks`}
                    </span>
                  )}
                </div>

                <div className="memory-answer-provenance">
                  {queryResponse.diagnostics.meetings_represented.length > 0 && (
                    <span>
                      Synthesized from {queryResponse.diagnostics.meetings_represented.length} meeting
                      {queryResponse.diagnostics.meetings_represented.length === 1 ? '' : 's'}
                    </span>
                  )}
                </div>
              </div>

              {/* Abstention Notice if is_abstained is true */}
              {queryResponse.is_abstained && (
                <div className="memory-abstention-alert" data-testid="abstention-alert">
                  <AlertTriangle size={18} color="#f59e0b" />
                  <div>
                    {queryResponse.abstention_status === 'CONFLICTING_EVIDENCE' ? (
                      <>
                        <strong>Unresolved Conflict Alert:</strong> Potentially contradictory statements or active decisions
                        were identified across meetings. The system abstained from picking a winning decision because no explicit
                        superseding link or revocation was recorded.
                      </>
                    ) : queryResponse.abstention_status === 'NO_EVIDENCE' ? (
                      <>
                        <strong>Answer Grounding Protection:</strong> No meeting evidence was found matching this question in your
                        organization's recorded history.
                      </>
                    ) : (
                      <>
                        <strong>Answer Grounding Protection:</strong> The system refrained from speculating because the
                        retrieved meeting transcripts do not contain sufficient evidence to answer this question reliably.
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Answer Text */}
              <div className="memory-answer-text" data-testid="query-answer-text">
                {renderFormattedAnswer(queryResponse.answer)}
              </div>

              {/* Dedicated External Background Knowledge Card */}
              {queryResponse.external_knowledge && (
                <div className="memory-external-knowledge-card" data-testid="external-knowledge-card">
                  <div className="memory-external-knowledge-header">
                    <div className="memory-external-knowledge-badge">
                      <Globe size={15} color="#3b82f6" />
                      <span>External Background Reference — Context Not Discussed in Meeting</span>
                    </div>
                    <span className="intel-version-badge" style={{ textTransform: 'uppercase' }}>
                      {queryResponse.external_knowledge.provider_used}
                    </span>
                  </div>

                  {/* Non-alarmist honest disclaimer */}
                  <div className="memory-external-disclaimer" data-testid="external-knowledge-disclaimer">
                    <Info size={14} color="#3b82f6" style={{ flexShrink: 0, marginTop: '2px' }} />
                    <span>{queryResponse.external_knowledge.disclaimer}</span>
                  </div>

                  {queryResponse.external_knowledge.is_available ? (
                    <>
                      <div className="memory-external-subject-title">
                        Subject: <strong>{queryResponse.external_knowledge.subject}</strong>
                      </div>

                      <div className="memory-external-summary" data-testid="external-knowledge-summary">
                        {queryResponse.external_knowledge.summary}
                      </div>

                      {queryResponse.external_knowledge.key_facts && queryResponse.external_knowledge.key_facts.length > 0 && (
                        <div className="memory-external-facts-section">
                          <div className="memory-external-facts-title">Key Facts & Background:</div>
                          <ul className="memory-external-facts-list" data-testid="external-knowledge-facts">
                            {queryResponse.external_knowledge.key_facts.map((fact, fIdx) => (
                              <li key={fIdx}>{fact}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {queryResponse.external_knowledge.citations && queryResponse.external_knowledge.citations.length > 0 && (
                        <div className="memory-external-citations-section">
                          <div className="memory-external-facts-title">Reliable External Sources:</div>
                          <div className="memory-external-sources-list" data-testid="external-knowledge-sources">
                            {queryResponse.external_knowledge.citations.map((c, cIdx) => (
                              <a
                                key={cIdx}
                                href={c.source_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="memory-external-source-pill"
                                data-testid={`external-source-${cIdx}`}
                              >
                                <BookOpen size={12} />
                                <span>{c.source_title}</span>
                                <ExternalLink size={11} />
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="memory-external-unavailable-box" data-testid="external-knowledge-unavailable">
                      <AlertTriangle size={15} color="#f59e0b" />
                      <span>
                        External background retrieval for <strong>'{queryResponse.external_knowledge.subject}'</strong> was unavailable
                        {queryResponse.external_knowledge.unavailability_reason ? ` (${queryResponse.external_knowledge.unavailability_reason})` : ''}.
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Fallback Warning if heuristic fallback used */}
              {queryResponse.diagnostics.fallback_warning && (
                <div
                  style={{
                    fontSize: '0.78rem',
                    color: '#94a3b8',
                    fontStyle: 'italic',
                    padding: '0.25rem 0.5rem',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: '4px',
                  }}
                >
                  <Info size={12} style={{ display: 'inline', marginRight: '4px' }} />
                  {queryResponse.diagnostics.fallback_warning}
                </div>
              )}

              {/* 1. Structured Decision Timeline & Lifecycle */}
              {queryResponse.structured_decisions && queryResponse.structured_decisions.length > 0 && (
                <div className="memory-structured-section" data-testid="ask-decisions-timeline">
                  <div className="memory-structured-header">
                    <div className="memory-structured-title">
                      <GitBranch size={16} color="#6366f1" />
                      <span>Decision Timeline & Lifecycle ({queryResponse.structured_decisions.length})</span>
                    </div>
                    <button
                      type="button"
                      className="memory-structured-toggle-btn"
                      onClick={() => setIsDecisionsExpanded(!isDecisionsExpanded)}
                      aria-expanded={isDecisionsExpanded}
                      data-testid="toggle-decisions-btn"
                    >
                      <span>{isDecisionsExpanded ? 'Collapse' : 'Expand'}</span>
                      {isDecisionsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                  </div>

                  {isDecisionsExpanded && (
                    <div className="intel-items-list" data-testid="ask-decisions-list">
                      {queryResponse.structured_decisions.map((decision, dIdx) => {
                        const earlierDec = decision.supersedes_decision_id
                          ? queryResponse.structured_decisions?.find((d) => d.id === decision.supersedes_decision_id)
                          : null;
                        const supersedingDec = decision.superseded_by_decision_id
                          ? queryResponse.structured_decisions?.find((d) => d.id === decision.superseded_by_decision_id)
                          : null;

                        return (
                          <div
                            key={decision.id || dIdx}
                            className={`intel-item-card ${decision.status === 'SUPERSEDED' ? 'superseded' : ''}`}
                            data-testid={`ask-decision-card-${decision.id}`}
                          >
                            <div className="intel-item-header">
                              <div className="intel-item-title">
                                {decision.description}
                                {decision.lineage_depth > 1 && (
                                  <span
                                    title="This decision has linked lineage"
                                    style={{ marginLeft: '0.4rem', color: '#6366f1' }}
                                  >
                                    <GitBranch size={13} style={{ display: 'inline', verticalAlign: 'middle' }} />
                                  </span>
                                )}
                              </div>
                              {renderDecisionStatusBadge(decision.status)}
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
                            </div>

                            {/* Explicit Supersession Lineage Connections */}
                            {decision.supersedes_decision_id && (
                              <div className="ask-supersession-banner supersedes" data-testid={`supersedes-banner-${decision.id}`}>
                                <GitBranch size={13} style={{ flexShrink: 0, marginTop: '2px' }} />
                                <span>
                                  <strong>Supersedes earlier decision:</strong>{' '}
                                  {earlierDec ? `"${earlierDec.description}" (${earlierDec.meeting_title || 'earlier meeting'})` : 'earlier decision in organizational history'}.
                                  {decision.rationale_or_context && (
                                    <> What changed: {decision.rationale_or_context}</>
                                  )}
                                </span>
                              </div>
                            )}

                            {decision.superseded_by_decision_id && (
                              <div className="ask-supersession-banner superseded-by" data-testid={`superseded-by-banner-${decision.id}`}>
                                <Archive size={13} style={{ flexShrink: 0, marginTop: '2px' }} />
                                <span>
                                  <strong>Superseded by later decision:</strong>{' '}
                                  {supersedingDec ? `"${supersedingDec.description}" (${supersedingDec.meeting_title || 'later meeting'})` : 'a later active decision'}.
                                </span>
                              </div>
                            )}

                            {/* Action / Evidence Button */}
                            <div className="intel-item-actions">
                              <div className="intel-action-group">
                                {decision.evidence_segment_id ? (
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
                                    data-testid={`ask-decision-evidence-${decision.id}`}
                                  >
                                    <span>View Evidence</span>
                                    {decision.timestamp_seconds != null && (
                                      <span style={{ fontFamily: 'monospace', opacity: 0.85 }}>
                                        ({formatTime(decision.timestamp_seconds)})
                                      </span>
                                    )}
                                    <ArrowRight size={12} />
                                  </button>
                                ) : (
                                  <span className="intel-evidence-unavailable" data-testid={`ask-decision-no-evidence-${decision.id}`}>
                                    <Info size={11} />
                                    Evidence: Unavailable
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* 2. Structured Commitments & Action Items */}
              {queryResponse.structured_commitments && queryResponse.structured_commitments.length > 0 && (
                <div className="memory-structured-section" data-testid="ask-commitments-section">
                  <div className="memory-structured-header">
                    <div className="memory-structured-title">
                      <CheckSquare size={16} color="#6366f1" />
                      <span>Commitments & Action Items ({queryResponse.structured_commitments.length})</span>
                    </div>
                    <button
                      type="button"
                      className="memory-structured-toggle-btn"
                      onClick={() => setIsCommitmentsExpanded(!isCommitmentsExpanded)}
                      aria-expanded={isCommitmentsExpanded}
                      data-testid="toggle-commitments-btn"
                    >
                      <span>{isCommitmentsExpanded ? 'Collapse' : 'Expand'}</span>
                      {isCommitmentsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                  </div>

                  {isCommitmentsExpanded && (
                    <div className="intel-items-list" data-testid="ask-commitments-list">
                      {queryResponse.structured_commitments.map((commitment, cIdx) => (
                        <div
                          key={commitment.id || cIdx}
                          className={`intel-item-card ${commitment.is_overdue ? 'overdue' : ''}`}
                          data-testid={`ask-commitment-card-${commitment.id}`}
                        >
                          <div className="intel-item-header">
                            <div className="intel-item-title">{commitment.description}</div>
                            {renderCommitmentStatusBadge(commitment.status, commitment.is_overdue)}
                          </div>

                          <div className="intel-item-meta">
                            <span>
                              <strong>Owner:</strong> {commitment.owner_name || 'Unassigned'}
                            </span>
                            {commitment.deadline && (
                              <>
                                <span>&bull;</span>
                                <span>
                                  <strong>Deadline:</strong> {new Date(commitment.deadline).toLocaleDateString()}
                                </span>
                              </>
                            )}
                            <span>&bull;</span>
                            <span>
                              <strong>Origin:</strong> {commitment.origin_meeting_title || 'Meeting provenance unavailable'}
                            </span>
                            {commitment.origin_meeting_date && (
                              <span>({new Date(commitment.origin_meeting_date).toLocaleDateString()})</span>
                            )}
                          </div>

                          {/* Subsequent Cross-Meeting Review Info */}
                          {(commitment.review_count > 0 || commitment.last_reviewed_meeting_title) && (
                            <div className="intel-item-body" style={{ fontSize: '0.8rem', background: 'rgba(99, 102, 241, 0.08)' }}>
                              <div>
                                <strong>Cross-Meeting Review:</strong> Reviewed in{' '}
                                <strong>{commitment.last_reviewed_meeting_title || 'follow-up meeting'}</strong>
                                {commitment.last_reviewed_at && ` on ${new Date(commitment.last_reviewed_at).toLocaleDateString()}`}
                                {commitment.review_count > 1 && ` (${commitment.review_count} total reviews)`}.
                              </div>
                              {commitment.reviews && commitment.reviews.length > 0 && commitment.reviews[0].review_notes && (
                                <div style={{ marginTop: '0.25rem', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                                  "{commitment.reviews[0].review_notes}"
                                </div>
                              )}
                            </div>
                          )}

                          {/* Action / Evidence Button */}
                          <div className="intel-item-actions">
                            <div className="intel-action-group">
                              {commitment.last_evidence_segment_id && commitment.last_reviewed_meeting_id ? (
                                <button
                                  type="button"
                                  className="intel-btn-sm"
                                  onClick={() =>
                                    onNavigateToEvidence(
                                      commitment.last_reviewed_meeting_id!,
                                      commitment.last_evidence_segment_id!,
                                      null
                                    )
                                  }
                                  data-testid={`ask-commitment-review-evidence-${commitment.id}`}
                                >
                                  <span>View Review Evidence</span>
                                  <ArrowRight size={12} />
                                </button>
                              ) : commitment.evidence_segment_id ? (
                                <button
                                  type="button"
                                  className="intel-btn-sm"
                                  onClick={() =>
                                    onNavigateToEvidence(
                                      commitment.meeting_id,
                                      commitment.evidence_segment_id!,
                                      commitment.timestamp_seconds
                                    )
                                  }
                                  data-testid={`ask-commitment-evidence-${commitment.id}`}
                                >
                                  <span>View Origin Evidence</span>
                                  {commitment.timestamp_seconds != null && (
                                    <span style={{ fontFamily: 'monospace', opacity: 0.85 }}>
                                      ({formatTime(commitment.timestamp_seconds)})
                                    </span>
                                  )}
                                  <ArrowRight size={12} />
                                </button>
                              ) : (
                                <span className="intel-evidence-unavailable" data-testid={`ask-commitment-no-evidence-${commitment.id}`}>
                                  <Info size={11} />
                                  Evidence: Unavailable
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* 3. Structured Unresolved Questions */}
              {queryResponse.structured_questions && queryResponse.structured_questions.length > 0 && (
                <div className="memory-structured-section" data-testid="ask-questions-section">
                  <div className="memory-structured-header">
                    <div className="memory-structured-title">
                      <HelpCircle size={16} color="#6366f1" />
                      <span>Unresolved & Cross-Meeting Questions ({queryResponse.structured_questions.length})</span>
                    </div>
                    <button
                      type="button"
                      className="memory-structured-toggle-btn"
                      onClick={() => setIsQuestionsExpanded(!isQuestionsExpanded)}
                      aria-expanded={isQuestionsExpanded}
                      data-testid="toggle-questions-btn"
                    >
                      <span>{isQuestionsExpanded ? 'Collapse' : 'Expand'}</span>
                      {isQuestionsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                  </div>

                  {isQuestionsExpanded && (
                    <div className="intel-items-list" data-testid="ask-questions-list">
                      {queryResponse.structured_questions.map((questionItem, qIdx) => (
                        <div
                          key={questionItem.id || qIdx}
                          className="intel-item-card"
                          data-testid={`ask-question-card-${questionItem.id}`}
                        >
                          <div className="intel-item-header">
                            <div className="intel-item-title">{questionItem.question_text}</div>
                            {renderQuestionStatusBadge(questionItem.status)}
                          </div>

                          <div className="intel-item-meta">
                            {questionItem.raised_by && (
                              <span>
                                <strong>Raised By:</strong> {questionItem.raised_by}
                              </span>
                            )}
                            {questionItem.raised_by && <span>&bull;</span>}
                            <span>
                              <strong>Origin:</strong> {questionItem.origin_meeting_title || 'Meeting provenance unavailable'}
                            </span>
                            {questionItem.origin_meeting_date && (
                              <span>({new Date(questionItem.origin_meeting_date).toLocaleDateString()})</span>
                            )}
                          </div>

                          {/* Resolution Details */}
                          {questionItem.status === 'RESOLVED' && (
                            <div className="intel-item-body" style={{ fontSize: '0.8rem', background: 'rgba(16, 185, 129, 0.08)' }}>
                              <div>
                                <strong>Resolved:</strong>{' '}
                                {questionItem.resolving_meeting_title ? `in "${questionItem.resolving_meeting_title}"` : 'in subsequent meeting'}
                                {questionItem.resolved_at && ` on ${new Date(questionItem.resolved_at).toLocaleDateString()}`}.
                              </div>
                              {questionItem.resolving_decision_description && (
                                <div style={{ marginTop: '0.2rem' }}>
                                  <strong>Resolving Decision:</strong> "{questionItem.resolving_decision_description}"
                                </div>
                              )}
                              {questionItem.resolution_notes && (
                                <div style={{ marginTop: '0.2rem', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                                  Notes: {questionItem.resolution_notes}
                                </div>
                              )}
                            </div>
                          )}

                          {/* Action / Evidence Button */}
                          <div className="intel-item-actions">
                            <div className="intel-action-group">
                              {questionItem.evidence_segment_id ? (
                                <button
                                  type="button"
                                  className="intel-btn-sm"
                                  onClick={() =>
                                    onNavigateToEvidence(
                                      questionItem.meeting_id,
                                      questionItem.evidence_segment_id!,
                                      null
                                    )
                                  }
                                  data-testid={`ask-question-evidence-${questionItem.id}`}
                                >
                                  <span>View Evidence</span>
                                  <ArrowRight size={12} />
                                </button>
                              ) : (
                                <span className="intel-evidence-unavailable" data-testid={`ask-question-no-evidence-${questionItem.id}`}>
                                  <Info size={11} />
                                  Evidence: Unavailable
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* 4. Potential Decision Conflicts */}
              {queryResponse.detected_conflicts && queryResponse.detected_conflicts.length > 0 && (
                <div className="memory-structured-section" data-testid="ask-conflicts-section">
                  <div className="memory-structured-header">
                    <div className="memory-structured-title" style={{ color: '#f59e0b' }}>
                      <ShieldAlert size={16} color="#f59e0b" />
                      <span>Potential Decision Conflicts Detected ({queryResponse.detected_conflicts.length})</span>
                    </div>
                    <button
                      type="button"
                      className="memory-structured-toggle-btn"
                      onClick={() => setIsConflictsExpanded(!isConflictsExpanded)}
                      aria-expanded={isConflictsExpanded}
                      data-testid="toggle-conflicts-btn"
                    >
                      <span>{isConflictsExpanded ? 'Collapse' : 'Expand'}</span>
                      {isConflictsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                  </div>

                  {isConflictsExpanded && (
                    <div className="intel-items-list" data-testid="ask-conflicts-list">
                      {/* Prominent Non-Alarmist Warning Banner */}
                      <div className="intel-conflict-banner" data-testid="ask-conflict-warning-banner">
                        <ShieldAlert size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
                        <div>
                          <strong>ADVISORY: Potential Contradiction Detected (Heuristic)</strong>
                          <div style={{ marginTop: '0.25rem', opacity: 0.9 }}>
                            The system detected semantically contradictory decisions across meetings on this topic.
                            <strong> Both decisions currently remain ACTIVE</strong> in organizational memory until formally superseded or revoked.
                          </div>
                        </div>
                      </div>

                      {queryResponse.detected_conflicts.map((conflict, cIdx) => {
                        const decA = queryResponse.structured_decisions?.find((d) => d.id === conflict.decision_a_id);
                        const decB = queryResponse.structured_decisions?.find((d) => d.id === conflict.decision_b_id);

                        return (
                          <div
                            key={`${conflict.decision_a_id}-${conflict.decision_b_id}-${cIdx}`}
                            className="intel-item-card intel-conflict-card"
                            style={{ borderLeft: '3px solid #f59e0b' }}
                            data-testid={`ask-conflict-card-${cIdx}`}
                          >
                            <div className="intel-item-header">
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <AlertTriangle size={15} color="#f59e0b" />
                                <span className="intel-item-title">Contradiction Topic: {conflict.subject}</span>
                              </div>
                              <span className="badge-status superseded">Heuristic Alert</span>
                            </div>

                            {/* Side-by-side or stacked decision comparison */}
                            <div className="intel-conflict-grid">
                              {/* Decision A */}
                              <div className="intel-conflict-decision-box">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                  <span style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#94a3b8', fontWeight: 600 }}>
                                    Decision A
                                  </span>
                                  <span className="badge-status active" style={{ fontSize: '0.68rem', padding: '0.1rem 0.4rem' }}>
                                    Active
                                  </span>
                                </div>
                                <div style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                                  "{conflict.decision_a_text}"
                                </div>
                                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                                  Origin: <strong>{conflict.decision_a_meeting_title}</strong>
                                  {conflict.decision_a_date && ` (${new Date(conflict.decision_a_date).toLocaleDateString()})`}
                                </div>
                                {decA?.rationale_or_context && (
                                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                                    Rationale: {decA.rationale_or_context}
                                  </div>
                                )}
                                <div style={{ marginTop: '0.35rem' }}>
                                  {decA?.evidence_segment_id ? (
                                    <button
                                      type="button"
                                      className="intel-btn-sm"
                                      onClick={() =>
                                        onNavigateToEvidence(decA.meeting_id, decA.evidence_segment_id!, decA.timestamp_seconds)
                                      }
                                      data-testid={`ask-conflict-evidence-a-${cIdx}`}
                                    >
                                      <span>View Evidence A</span>
                                      <ArrowRight size={12} />
                                    </button>
                                  ) : (
                                    <span className="intel-evidence-unavailable">
                                      <Info size={11} /> Evidence: Unavailable
                                    </span>
                                  )}
                                </div>
                              </div>

                              {/* Decision B */}
                              <div className="intel-conflict-decision-box">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                  <span style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#94a3b8', fontWeight: 600 }}>
                                    Decision B
                                  </span>
                                  <span className="badge-status active" style={{ fontSize: '0.68rem', padding: '0.1rem 0.4rem' }}>
                                    Active
                                  </span>
                                </div>
                                <div style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                                  "{conflict.decision_b_text}"
                                </div>
                                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                                  Origin: <strong>{conflict.decision_b_meeting_title}</strong>
                                  {conflict.decision_b_date && ` (${new Date(conflict.decision_b_date).toLocaleDateString()})`}
                                </div>
                                {decB?.rationale_or_context && (
                                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                                    Rationale: {decB.rationale_or_context}
                                  </div>
                                )}
                                <div style={{ marginTop: '0.35rem' }}>
                                  {decB?.evidence_segment_id ? (
                                    <button
                                      type="button"
                                      className="intel-btn-sm"
                                      onClick={() =>
                                        onNavigateToEvidence(decB.meeting_id, decB.evidence_segment_id!, decB.timestamp_seconds)
                                      }
                                      data-testid={`ask-conflict-evidence-b-${cIdx}`}
                                    >
                                      <span>View Evidence B</span>
                                      <ArrowRight size={12} />
                                    </button>
                                  ) : (
                                    <span className="intel-evidence-unavailable">
                                      <Info size={11} /> Evidence: Unavailable
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>

                            {/* Contradiction Reason & Suggested Action */}
                            <div className="intel-item-body" style={{ fontSize: '0.8rem', background: 'rgba(245, 158, 11, 0.08)' }}>
                              <div>
                                <strong>Contradiction Analysis:</strong> {conflict.reason}
                              </div>
                              <div style={{ marginTop: '0.3rem', color: '#fbbf24' }}>
                                <strong>Recommended Action:</strong> {conflict.suggested_action}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* 5. Potential Transcript Contradictions Across Meetings (P4/P5) */}
              {queryResponse.transcript_conflicts && queryResponse.transcript_conflicts.length > 0 && (
                <div className="memory-structured-section" data-testid="ask-transcript-conflicts-section">
                  <div className="memory-structured-header">
                    <div className="memory-structured-title" style={{ color: '#f59e0b' }}>
                      <ShieldAlert size={16} color="#f59e0b" />
                      <span>Transcript Discussion Contradictions ({queryResponse.transcript_conflicts.length})</span>
                    </div>
                  </div>

                  <div className="intel-items-list" data-testid="ask-transcript-conflicts-list">
                    <div className="intel-conflict-banner" data-testid="ask-transcript-conflict-banner">
                      <AlertTriangle size={16} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
                      <div>
                        <strong>Heuristic Transcript Contradiction Warning:</strong> Potentially opposing statements were
                        identified across meetings regarding this topic. Note that these are conversational discussion points from transcripts,
                        not formally recorded decision conflicts.
                      </div>
                    </div>

                    {queryResponse.transcript_conflicts.map((tc, tcIdx) => (
                      <div key={tcIdx} className="intel-conflict-card" data-testid={`ask-transcript-conflict-card-${tcIdx}`}>
                        <div className="intel-conflict-header">
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span className="intel-conflict-subject">Topic: {tc.subject}</span>
                            <span className="intel-heuristic-badge">Heuristic Transcript Match</span>
                          </div>
                        </div>

                        <div className="intel-conflict-decisions-grid">
                          {/* Chunk A */}
                          <div className="intel-conflict-decision-box">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#94a3b8', fontWeight: 600 }}>
                                Affirmative Statement (Meeting A)
                              </span>
                            </div>
                            <div style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 500, fontStyle: 'italic' }}>
                              "{tc.chunk_a_text}"
                            </div>
                            <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                              Meeting: <strong>{tc.chunk_a_meeting_title}</strong>
                              {tc.chunk_a_date && ` (${new Date(tc.chunk_a_date).toLocaleDateString()})`}
                            </div>
                          </div>

                          {/* Chunk B */}
                          <div className="intel-conflict-decision-box">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#94a3b8', fontWeight: 600 }}>
                                Opposing Statement (Meeting B)
                              </span>
                            </div>
                            <div style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 500, fontStyle: 'italic' }}>
                              "{tc.chunk_b_text}"
                            </div>
                            <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                              Meeting: <strong>{tc.chunk_b_meeting_title}</strong>
                              {tc.chunk_b_date && ` (${new Date(tc.chunk_b_date).toLocaleDateString()})`}
                            </div>
                          </div>
                        </div>

                        <div className="intel-item-body" style={{ fontSize: '0.8rem', background: 'rgba(245, 158, 11, 0.08)' }}>
                          <div>
                            <strong>Contradiction Analysis:</strong> {tc.reason}
                          </div>
                          <div style={{ marginTop: '0.3rem', color: '#fbbf24' }}>
                            <strong>Recommended Action:</strong> {tc.suggested_action}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Citations & Verified Evidence */}
              <div className="memory-citations-section">
                <div className="memory-citations-header">
                  <div className="memory-citations-title">
                    <Shield size={14} color="#6366f1" />
                    <span>
                      {queryResponse.abstention_status === 'CONFLICTING_EVIDENCE'
                        ? `Conflicting Evidence Citations (${queryResponse.citations.length} Sources)`
                        : `Evidence Grounding (${queryResponse.citations.length} Sources)`}
                    </span>
                  </div>
                  <span className="memory-citations-count">
                    {queryResponse.diagnostics.total_candidates_retrieved} candidates scanned
                  </span>
                </div>

                {queryResponse.citations.length === 0 ? (
                  <div className="memory-no-citations">
                    No matching transcript evidence chunks were found for this query in the current organization.
                  </div>
                ) : (
                  <div className="memory-citations-list">
                    {queryResponse.citations.map((cit, idx) => (
                      <div
                        key={cit.citation_id || idx}
                        className="memory-citation-card"
                        data-testid={`citation-card-${idx}`}
                      >
                        <div className="memory-citation-meta">
                          <span className="memory-citation-meeting">{cit.meeting_title}</span>
                          {cit.meeting_date && (
                            <span className="memory-citation-date">
                              {new Date(cit.meeting_date).toLocaleDateString()}
                            </span>
                          )}
                          <span className="memory-citation-time">
                            {formatTime(cit.start_time)} - {formatTime(cit.end_time)}
                          </span>
                          {cit.language_codes?.length > 0 && (
                            <span className="intel-version-badge" style={{ fontSize: '0.7rem' }}>
                              {cit.language_codes.join(', ').toUpperCase()}
                            </span>
                          )}
                        </div>

                        <div className="memory-citation-quote">"{cit.quoted_text}"</div>

                        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.4rem' }}>
                          <button
                            type="button"
                            className="memory-jump-btn"
                            onClick={() => {
                              const segId = cit.segment_ids?.[0] || '';
                              onNavigateToEvidence(cit.meeting_id, segId, cit.start_time);
                            }}
                            data-testid={`jump-evidence-btn-${idx}`}
                          >
                            <span>View in Meeting Transcript</span>
                            <ArrowRight size={12} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODE 2: HYBRID SEARCH MEMORY                                              */}
      {/* ========================================================================= */}
      {activeMode === 'search' && (
        <div className="memory-card">
          <form onSubmit={handleSearchSubmit} className="memory-input-group">
            <label htmlFor="memory-search-input" className="memory-input-label">
              Search transcripts using hybrid keyword and multilingual semantic matching:
            </label>
            <div style={{ display: 'flex', gap: '0.5rem', width: '100%' }}>
              <input
                id="memory-search-input"
                type="text"
                className="memory-search-input"
                placeholder="e.g., PostgreSQL replication, token refresh, OAuth scopes"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  if (searchError) setSearchError(null);
                }}
                data-testid="memory-search-input"
              />
              <button
                type="submit"
                className="memory-submit-btn"
                disabled={isLoadingSearch || !searchQuery.trim()}
                data-testid="memory-search-submit-btn"
              >
                {isLoadingSearch ? (
                  <>
                    <RefreshCw size={15} className="spin" />
                    <span>Searching...</span>
                  </>
                ) : (
                  <>
                    <Search size={15} />
                    <span>Search</span>
                  </>
                )}
              </button>
            </div>

            {/* Scope & Date Filters */}
            <div className="memory-filters-row">
              {projects.length > 0 && (
                <div className="memory-filter-item">
                  <label className="memory-filter-label" htmlFor="search-project-filter">
                    Project Scope:
                  </label>
                  <select
                    id="search-project-filter"
                    className="memory-select"
                    value={selectedProjectId}
                    onChange={(e) => setSelectedProjectId(e.target.value)}
                    data-testid="search-project-filter"
                  >
                    <option value="">All Projects Across Organization</option>
                    {projects.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div className="memory-filter-item">
                <label className="memory-filter-label" htmlFor="search-meeting-filter">
                  Meeting Scope:
                </label>
                <select
                  id="search-meeting-filter"
                  className="memory-select"
                  value={searchMeetingId}
                  onChange={(e) => setSearchMeetingId(e.target.value)}
                  data-testid="search-meeting-filter"
                >
                  <option value="">All Meetings Across Organization</option>
                  {meetings.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.title || 'Untitled Meeting'} ({new Date(m.created_at).toLocaleDateString()})
                    </option>
                  ))}
                </select>
              </div>

              <div className="memory-filter-item">
                <label className="memory-filter-label" htmlFor="search-start-date">
                  Start Date:
                </label>
                <input
                  type="date"
                  id="search-start-date"
                  className="memory-date-input"
                  value={searchStartDate}
                  onChange={(e) => setSearchStartDate(e.target.value)}
                  data-testid="search-start-date"
                />
              </div>

              <div className="memory-filter-item">
                <label className="memory-filter-label" htmlFor="search-end-date">
                  End Date:
                </label>
                <input
                  type="date"
                  id="search-end-date"
                  className="memory-date-input"
                  value={searchEndDate}
                  onChange={(e) => setSearchEndDate(e.target.value)}
                  data-testid="search-end-date"
                />
              </div>

              {(searchMeetingId || searchStartDate || searchEndDate) && (
                <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'flex-end' }}>
                  <button
                    type="button"
                    className="memory-clear-filters-btn"
                    onClick={() => {
                      setSearchMeetingId('');
                      setSearchStartDate('');
                      setSearchEndDate('');
                    }}
                  >
                    Clear Filters
                  </button>
                </div>
              )}
            </div>
          </form>

          {/* Search Error */}
          {searchError && (
            <div className="memory-error-box" data-testid="search-error-box">
              <AlertTriangle size={18} color="#ef4444" />
              <span>{searchError}</span>
            </div>
          )}

          {/* Search Loading */}
          {isLoadingSearch && (
            <div className="memory-loading-box">
              <RefreshCw size={28} className="spin" color="#6366f1" />
              <div>
                <strong>Searching organizational memory passages...</strong>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.25rem' }}>
                  Computing Reciprocal Rank Fusion (RRF) across SQLite FTS5 keyword and E5 vector retrieval.
                </div>
              </div>
            </div>
          )}

          {/* Search Results */}
          {searchResponse && !isLoadingSearch && (
            <div className="memory-results-container" data-testid="search-results-container">
              <div className="memory-results-summary">
                <span className="memory-results-count">
                  {searchResponse.total_results} matching passage{searchResponse.total_results === 1 ? '' : 's'} found
                </span>
                <span className="memory-results-note">
                  Ranked by hybrid keyword + vector semantic score (RRF)
                </span>
              </div>

              {searchResponse.results.length === 0 ? (
                <div className="memory-empty-box">
                  <Search size={32} color="#64748b" />
                  <div className="memory-empty-title">No matching memory passages found</div>
                  <div className="memory-empty-desc">
                    No transcript chunks matched your query in the current organization. Memory passages become available
                    after meetings are transcribed and privacy-redacted.
                  </div>
                </div>
              ) : (
                searchResponse.results.map((res) => (
                  <div key={res.chunk_id} className="memory-result-card" data-testid={`search-result-${res.chunk_id}`}>
                    <div className="memory-result-header">
                      <span className="memory-result-title">{res.meeting_title}</span>
                      <div className="memory-result-meta">
                        <span>
                          {res.meeting_date ? new Date(res.meeting_date).toLocaleDateString() : 'Recorded meeting'}
                        </span>
                        <span>&bull;</span>
                        <span style={{ fontFamily: 'monospace' }}>
                          {formatTime(res.start_time)} - {formatTime(res.end_time)}
                        </span>
                        {res.dominant_language && (
                          <span className="intel-version-badge" style={{ fontSize: '0.7rem' }}>
                            {res.dominant_language.toUpperCase()}
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="memory-result-text">"{res.text}"</div>

                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.25rem' }}>
                      <button
                        type="button"
                        className="memory-jump-btn"
                        onClick={() => {
                          const segId = res.segment_ids?.[0] || '';
                          onNavigateToEvidence(res.meeting_id, segId, res.start_time);
                        }}
                      >
                        <span>Open in Transcript</span>
                        <ArrowRight size={12} />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODE 3: DECISION HISTORY & LINEAGE                                        */}
      {/* ========================================================================= */}
      {activeMode === 'decisions' && (
        <DecisionHistoryView
          meetings={meetings}
          currentOrgRole={currentOrgRole}
          onNavigateToEvidence={onNavigateToEvidence}
          onDataChanged={loadStats}
        />
      )}

      {/* ========================================================================= */}
      {/* MODE 4: COMMITMENT TRACKING & REVIEWS                                     */}
      {/* ========================================================================= */}
      {activeMode === 'commitments' && (
        <CommitmentTrackingView
          meetings={meetings}
          currentOrgRole={currentOrgRole}
          onNavigateToEvidence={onNavigateToEvidence}
          onDataChanged={loadStats}
        />
      )}

      {/* ========================================================================= */}
      {/* MODE 5: UNRESOLVED QUESTIONS & RESOLUTION                                 */}
      {/* ========================================================================= */}
      {activeMode === 'questions' && (
        <UnresolvedQuestionsView
          meetings={meetings}
          currentOrgRole={currentOrgRole}
          onNavigateToEvidence={onNavigateToEvidence}
          onDataChanged={loadStats}
        />
      )}

      {/* ========================================================================= */}
      {/* MODE 6: POTENTIAL CONFLICTS & HEURISTICS                                  */}
      {/* ========================================================================= */}
      {activeMode === 'conflicts' && (
        <PotentialConflictsView
          meetings={meetings}
          onNavigateToEvidence={onNavigateToEvidence}
          onNavigateToDecisions={() => setActiveMode('decisions')}
        />
      )}
    </div>
  );
};
