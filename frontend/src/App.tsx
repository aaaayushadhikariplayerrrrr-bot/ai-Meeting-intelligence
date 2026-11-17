import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Activity,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Server,
  Database,
  Cpu,
  Layers,
  Sparkles,
  ExternalLink,
  Radio,
  FileCode2,
  Upload,
  FileAudio,
  FileVideo,
  Trash2,
  Clock,
  AlertTriangle,
  Play,
  PlayCircle,
  Mic,
  Copy,
  Check,
  FileText,
  Shield,
  Wand2,
  Columns,
  User,
  Building,
  LogOut
} from 'lucide-react';
import {
  checkBackendHealth,
  fetchRootInfo,
  fetchMeetings,
  fetchJobStatus,
  uploadMediaFile,
  ingestYouTube,
  deleteRawMedia,
  startTranscription,
  fetchTranscripts,
  fetchHardwareDiagnostics,
  cleanTranscript,
  redactTranscript,
  processTranscript,
  triggerExtractionJob,
  fetchMeetingIntelligence,
  updateActionItemStatus,
  isAbortError
} from './services/api';
import type { HealthCheckResult } from './services/api';
import type {
  HardwareSpec,
  PhaseItem,
  MeetingItem,
  ProcessingJob,
  TranscriptItem,
  HardwareDiagnostics,
  MeetingIntelligence,
} from './types';
import { useAuth } from './context/AuthContext';
import { AuthScreen } from './components/AuthScreen';
import { OrgSelectionScreen } from './components/OrgSelectionScreen';
import { MeetingIntelligenceView } from './components/MeetingIntelligenceView';
import { OrganizationalMemoryView } from './components/OrganizationalMemoryView';
import './App.css';

const HARDWARE_SPECS: HardwareSpec[] = [
  {
    category: 'Dedicated GPU',
    label: 'GPU & VRAM',
    value: 'NVIDIA GeForce RTX 5060 Laptop GPU',
    status: 'verified',
    detail: '8,151 MiB VRAM (CUDA UMD 13.5 / Driver 621.01)',
  },
  {
    category: 'AI Acceleration',
    label: 'PyTorch & CUDA',
    value: 'PyTorch 2.11.0+cu128',
    status: 'ready',
    detail: 'CUDA 12.8 acceleration active on device:0',
  },
  {
    category: 'Media Engine',
    label: 'FFmpeg & FFprobe',
    value: 'FFmpeg 9.0.2 Full Build',
    status: 'verified',
    detail: 'Used for media inspection, audio extraction, and 16kHz mono normalization',
  },
  {
    category: 'Host CPU',
    label: 'Processor',
    value: 'AMD Ryzen 7 260 w/ Radeon 780M',
    status: 'verified',
    detail: '8 physical cores, 16 logical threads',
  },
  {
    category: 'System Memory',
    label: 'Host RAM',
    value: '16 GB RAM',
    status: 'verified',
    detail: '15.31 GiB physical memory available',
  },
  {
    category: 'Storage',
    label: 'Disk Space',
    value: '812+ GB Free Space',
    status: 'verified',
    detail: 'Target drive C: for media storage and future models',
  },
];

const ROADMAP_PHASES: PhaseItem[] = [
  {
    id: 1,
    title: 'Phase 1: Foundation & Diagnostics',
    status: 'completed',
    description: 'Hardware inspection, FastAPI backend, SQLite connection, React+TS frontend dashboard.',
    technologies: ['FastAPI', 'React', 'TypeScript', 'Vite', 'SQLite'],
  },
  {
    id: 2,
    title: 'Phase 2: Core Database Architecture',
    status: 'completed',
    description: 'Domain entities, multilingual segment modeling, transcript versions, Alembic migrations.',
    technologies: ['SQLAlchemy 2.0', 'Alembic', 'SQLite Foreign Keys', 'Pytest'],
  },
  {
    id: 3,
    title: 'Phase 3: Media Ingestion & Lifecycle',
    status: 'completed',
    description: 'Audio/Video upload, YouTube ingestion with yt-dlp, FFmpeg 16kHz mono normalization, safe deletion.',
    technologies: ['FFmpeg/FFprobe', 'yt-dlp', 'Async Background Tasks', 'SHA-256'],
  },
  {
    id: 4,
    title: 'Phase 4: Multilingual ASR Pipeline',
    status: 'completed',
    description: 'Faster-Whisper on RTX 5060, CTranslate2, long-audio chunking, Hindi, Gujarati, English, Hinglish.',
    technologies: ['Whisper Large-v3 Turbo', 'CUDA 12.8 / RTX 5060', 'CTranslate2', 'Long-Audio Chunking'],
  },
  {
    id: 5,
    title: 'Phase 5: Cleaning & Privacy Redaction',
    status: 'completed',
    description: 'Readability cleaning, filler & stutter reduction, Indian PII redaction, derived versioning, audit logging.',
    technologies: ['Transcript Cleaner', 'Privacy Redactor', 'PII Detection', 'Alembic Versioning', 'Audit Trails'],
  },
  {
    id: 6,
    title: 'Phase 6: Intelligence & Extraction',
    status: 'planned',
    description: 'Automated executive summaries, key decisions, commitments, action items with owners and deadlines.',
    technologies: ['Ollama', 'Local LLM', 'Structured Output'],
  },
  {
    id: 6,
    title: 'Phase 6: Dual-Source RAG & Cross-Meeting Memory',
    status: 'planned',
    description: 'Cross-meeting memory search, external reference grounding, intelligent router, timestamp citations.',
    technologies: ['ChromaDB', 'Multilingual Embeddings', 'Hybrid Retrieval'],
  },
  {
    id: 7,
    title: 'Phase 7: Privacy, PII Redaction & Audit Logs',
    status: 'planned',
    description: 'Role-based access control (RBAC), PII/secret masking, raw audio deletion policies, audit logs.',
    technologies: ['JWT Auth', 'PII Masking', 'Audit Trails'],
  },
];

export const App: React.FC = () => {
  const { user, isAuthenticated, isLoading, currentOrg, organizations, switchOrganization, logout } = useAuth();
  const [isSwitchingOrg, setIsSwitchingOrg] = useState<boolean>(false);
  const activeOrgIdRef = useRef<string | null>(currentOrg?.organization_id || null);

  // Phase 6C.4: Request lifecycle management
  const requestEpochRef = useRef<number>(0);
  const activeAbortControllersRef = useRef<Set<AbortController>>(new Set());
  const activeUploadAbortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    activeOrgIdRef.current = currentOrg?.organization_id || null;
  }, [currentOrg]);

  /**
   * Helper to check whether a captured request epoch and organization ID are still current.
   */
  const isContextCurrent = useCallback(
    (capturedEpoch: number, capturedOrgId: string | null): boolean => {
      return (
        capturedEpoch === requestEpochRef.current &&
        capturedOrgId !== null &&
        capturedOrgId === activeOrgIdRef.current
      );
    },
    []
  );

  /**
   * Register a new AbortController for an in-flight operation.
   */
  const createRegisteredController = useCallback((): AbortController => {
    const controller = new AbortController();
    activeAbortControllersRef.current.add(controller);
    return controller;
  }, []);

  /**
   * Unregister an AbortController when its operation settles.
   */
  const unregisterController = useCallback((controller: AbortController): void => {
    activeAbortControllersRef.current.delete(controller);
  }, []);

  /**
   * Abort and clean up all active tenant-scoped operations (fetch requests and uploads).
   */
  const cancelAllActiveOperations = useCallback((): void => {
    // 1. Abort all registered fetch controllers
    activeAbortControllersRef.current.forEach((controller) => {
      try {
        controller.abort();
      } catch {
        // Ignore abort errors
      }
    });
    activeAbortControllersRef.current.clear();

    // 2. Abort any active upload
    if (activeUploadAbortControllerRef.current) {
      try {
        activeUploadAbortControllerRef.current.abort();
      } catch {
        // Ignore
      }
      activeUploadAbortControllerRef.current = null;
    }
  }, []);

  // Cancel all operations on unmount
  useEffect(() => {
    return () => {
      cancelAllActiveOperations();
    };
  }, [cancelAllActiveOperations]);

  const [healthResult, setHealthResult] = useState<HealthCheckResult | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [activeInspectorTab, setActiveInspectorTab] = useState<'health' | 'meetings' | 'asr'>('health');
  const [activeMainTab, setActiveMainTab] = useState<'meetings' | 'memory' | 'system'>('meetings');

  // Phase 3 Ingestion State
  const [ingestionTab, setIngestionTab] = useState<'file' | 'youtube'>('file');
  const [meetings, setMeetings] = useState<MeetingItem[]>([]);
  const [activeJob, setActiveJob] = useState<ProcessingJob | null>(null);
  const [isDeletingAssetId, setIsDeletingAssetId] = useState<string | null>(null);

  // File Upload Form State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileTitle, setFileTitle] = useState<string>('');
  const [fileDesc, setFileDesc] = useState<string>('');
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // YouTube Ingestion Form State
  const [youtubeUrl, setYoutubeUrl] = useState<string>('');
  const [ytTitle, setYtTitle] = useState<string>('');
  const [ytDesc, setYtDesc] = useState<string>('');
  const [confirmedAuth, setConfirmedAuth] = useState<boolean>(false);
  const [isSubmittingYt, setIsSubmittingYt] = useState<boolean>(false);
  const [ytError, setYtError] = useState<string | null>(null);

  // Phase 4 & Phase 5 Transcription State
  const [meetingTranscripts, setMeetingTranscripts] = useState<Record<string, TranscriptItem[]>>({});
  const [isTranscribingMap, setIsTranscribingMap] = useState<Record<string, boolean>>({});
  const [selectedLangMap, setSelectedLangMap] = useState<Record<string, string>>({});
  const [expandedTranscripts, setExpandedTranscripts] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [hardwareDiagnostics, setHardwareDiagnostics] = useState<HardwareDiagnostics | null>(null);

  // Phase 5 Multi-version and Comparison State
  const [selectedVersionMap, setSelectedVersionMap] = useState<Record<string, string>>({});
  const [compareModeMap, setCompareModeMap] = useState<Record<string, boolean>>({});
  const [isProcessingActionMap, setIsProcessingActionMap] = useState<Record<string, string | null>>({});

  // Phase 7 Structured Meeting Intelligence State
  const [meetingIntelligence, setMeetingIntelligence] = useState<Record<string, MeetingIntelligence | null>>({});
  const [isLoadingIntelligence, setIsLoadingIntelligence] = useState<Record<string, boolean>>({});
  const [intelligenceError, setIntelligenceError] = useState<Record<string, string | null>>({});
  const [expandedIntelligence, setExpandedIntelligence] = useState<Record<string, boolean>>({});
  const [isExtractingMap, setIsExtractingMap] = useState<Record<string, boolean>>({});
  const [evidenceNavErrorMap, setEvidenceNavErrorMap] = useState<Record<string, string | null>>({});

  const clearTenantDashboardState = useCallback((): void => {
    setMeetings([]);
    setMeetingTranscripts({});
    setMeetingIntelligence({});
    setActiveJob(null);
    setSelectedVersionMap({});
    setCompareModeMap({});
    setIsTranscribingMap({});
    setSelectedLangMap({});
    setExpandedTranscripts({});
    setIsLoadingIntelligence({});
    setIntelligenceError({});
    setExpandedIntelligence({});
    setIsExtractingMap({});
    setEvidenceNavErrorMap({});
    setIsUploading(false);
    setUploadProgress(0);
    setSelectedFile(null);
    setFileError(null);
    setIsSubmittingYt(false);
    setYtError(null);
    setIsDeletingAssetId(null);
    if (!user?.is_super_admin) {
      setHardwareDiagnostics(null);
      setActiveInspectorTab((prev) => (prev === 'asr' ? 'health' : prev));
    }
  }, [user?.is_super_admin]);

  const loadMeetings = useCallback(async () => {
    if (!currentOrg) return;
    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    const controller = createRegisteredController();
    try {
      const data = await fetchMeetings(controller.signal);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setMeetings(data);
      }
    } catch (err: unknown) {
      if (!isAbortError(err)) {
        // Backend may be offline or no meetings yet
      }
    } finally {
      unregisterController(controller);
    }
  }, [currentOrg, createRegisteredController, isContextCurrent, unregisterController]);

  const handleSwitchOrganization = async (targetOrgId: string) => {
    if (!targetOrgId || targetOrgId === currentOrg?.organization_id || isSwitchingOrg) {
      return;
    }

    const targetMembership = organizations.find((o) => o.organization_id === targetOrgId);
    if (!targetMembership) {
      alert(`Unauthorized organization selection: user does not hold membership in organization '${targetOrgId}'.`);
      return;
    }

    setIsSwitchingOrg(true);

    // 1. Invalidate the old request epoch synchronously
    requestEpochRef.current += 1;
    const newEpoch = requestEpochRef.current;

    // 2. Abort all registered tenant-scoped fetch requests & 3. Abort any active XHR upload
    cancelAllActiveOperations();

    // 4. Clear old tenant-specific dashboard state
    clearTenantDashboardState();
    setActiveMainTab('meetings');

    try {
      // 5. Update the active organization context
      switchOrganization(targetOrgId);
      activeOrgIdRef.current = targetOrgId;

      // 6. Load data for the newly selected organization
      const controller = createRegisteredController();
      try {
        const freshMeetings = await fetchMeetings(controller.signal);
        if (isContextCurrent(newEpoch, targetOrgId)) {
          setMeetings(freshMeetings);
        }
      } finally {
        unregisterController(controller);
      }
    } catch (err: unknown) {
      if (!isAbortError(err)) {
        alert(err instanceof Error ? err.message : 'Failed to switch organization');
      }
    } finally {
      setIsSwitchingOrg(false);
    }
  };

  const handleLogout = async () => {
    // 1. Invalidate the request epoch immediately
    requestEpochRef.current += 1;
    activeOrgIdRef.current = null;

    // 2. Abort all active tenant-scoped fetch requests & 3. Abort active XHR uploads
    cancelAllActiveOperations();

    // 4 & 5. Clear tenant-specific in-memory state & stop polling
    clearTenantDashboardState();
    setHardwareDiagnostics(null);
    setActiveInspectorTab('health');
    setActiveMainTab('meetings');

    // 6. Complete the existing AuthContext logout behavior
    await logout();
  };

  const loadTranscriptsForMeeting = useCallback(async (meetingId: string) => {
    if (!currentOrg) return;
    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    const controller = createRegisteredController();
    try {
      const data = await fetchTranscripts(meetingId, controller.signal);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setMeetingTranscripts((prev) => ({ ...prev, [meetingId]: data }));
      }
    } catch (e) {
      if (!isAbortError(e)) {
        console.error('Failed to load transcripts:', e);
      }
    } finally {
      unregisterController(controller);
    }
  }, [currentOrg, createRegisteredController, isContextCurrent, unregisterController]);

  const loadHardwareDiagnostics = useCallback(async () => {
    if (!user?.is_super_admin) {
      setHardwareDiagnostics(null);
      return;
    }
    try {
      const data = await fetchHardwareDiagnostics();
      setHardwareDiagnostics(data);
    } catch {
      // Diagnostics offline
      setHardwareDiagnostics(null);
    }
  }, [user?.is_super_admin]);

  const handleStartMeetingTranscription = async (meetingId: string) => {
    if (!currentOrg) return;
    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    const rawChoice = selectedLangMap[meetingId] || '';
    let lang: string | undefined = undefined;
    let mode: string = 'standard';

    if (rawChoice === 'hinglish' || rawChoice === 'hi-en') {
      mode = 'hinglish';
      lang = undefined;
    } else if (rawChoice) {
      lang = rawChoice;
      mode = 'standard';
    }

    setIsTranscribingMap((prev) => ({ ...prev, [meetingId]: true }));
    const controller = createRegisteredController();
    try {
      const resp = await startTranscription(meetingId, lang, mode, controller.signal);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setActiveJob({
          id: resp.job_id,
          meeting_id: resp.meeting_id,
          job_type: 'TRANSCRIPTION',
          status: 'IN_PROGRESS',
          progress: resp.progress || 10,
          error_message: null,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        setExpandedTranscripts((prev) => ({ ...prev, [meetingId]: true }));
      }
    } catch (err: unknown) {
      if (!isAbortError(err)) {
        if (isContextCurrent(capturedEpoch, capturedOrgId)) {
          alert(err instanceof Error ? err.message : 'Failed to start transcription');
        }
      }
    } finally {
      unregisterController(controller);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsTranscribingMap((prev) => ({ ...prev, [meetingId]: false }));
      }
    }
  };

  const handleCopyTranscriptText = (transcript: TranscriptItem) => {
    const text = transcript.segments
      .map((s) => `[${formatTime(s.start_time)} - ${formatTime(s.end_time)}] ${s.text}`)
      .join('\n');
    navigator.clipboard.writeText(text);
    setCopiedId(transcript.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleCleanTranscript = async (meetingId: string) => {
    if (!currentOrg) return;
    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    setIsProcessingActionMap((prev) => ({ ...prev, [meetingId]: 'clean' }));
    const controller = createRegisteredController();
    try {
      const derived = await cleanTranscript(meetingId, undefined, controller.signal);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        await loadTranscriptsForMeeting(meetingId);
        if (isContextCurrent(capturedEpoch, capturedOrgId)) {
          setSelectedVersionMap((prev) => ({ ...prev, [meetingId]: derived.id }));
        }
      }
    } catch (err: unknown) {
      if (!isAbortError(err)) {
        if (isContextCurrent(capturedEpoch, capturedOrgId)) {
          alert(err instanceof Error ? err.message : 'Failed to clean transcript');
        }
      }
    } finally {
      unregisterController(controller);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsProcessingActionMap((prev) => ({ ...prev, [meetingId]: null }));
      }
    }
  };

  const handleRedactTranscript = async (meetingId: string) => {
    if (!currentOrg) return;
    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    setIsProcessingActionMap((prev) => ({ ...prev, [meetingId]: 'redact' }));
    const controller = createRegisteredController();
    try {
      const derived = await redactTranscript(meetingId, undefined, controller.signal);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        await loadTranscriptsForMeeting(meetingId);
        if (isContextCurrent(capturedEpoch, capturedOrgId)) {
          setSelectedVersionMap((prev) => ({ ...prev, [meetingId]: derived.id }));
        }
      }
    } catch (err: unknown) {
      if (!isAbortError(err)) {
        if (isContextCurrent(capturedEpoch, capturedOrgId)) {
          alert(err instanceof Error ? err.message : 'Failed to redact transcript');
        }
      }
    } finally {
      unregisterController(controller);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsProcessingActionMap((prev) => ({ ...prev, [meetingId]: null }));
      }
    }
  };

  const handleProcessBothTranscript = async (meetingId: string) => {
    if (!currentOrg) return;
    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    setIsProcessingActionMap((prev) => ({ ...prev, [meetingId]: 'both' }));
    const controller = createRegisteredController();
    try {
      const derived = await processTranscript(meetingId, true, true, undefined, controller.signal);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        await loadTranscriptsForMeeting(meetingId);
        if (isContextCurrent(capturedEpoch, capturedOrgId)) {
          setSelectedVersionMap((prev) => ({ ...prev, [meetingId]: derived.id }));
        }
      }
    } catch (err: unknown) {
      if (!isAbortError(err)) {
        if (isContextCurrent(capturedEpoch, capturedOrgId)) {
          alert(err instanceof Error ? err.message : 'Failed to process transcript');
        }
      }
    } finally {
      unregisterController(controller);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsProcessingActionMap((prev) => ({ ...prev, [meetingId]: null }));
      }
    }
  };

  const loadIntelligenceForMeeting = useCallback(async (meetingId: string) => {
    if (!currentOrg) return;
    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    const controller = createRegisteredController();
    setIsLoadingIntelligence((prev) => ({ ...prev, [meetingId]: true }));
    setIntelligenceError((prev) => ({ ...prev, [meetingId]: null }));

    try {
      const data = await fetchMeetingIntelligence(meetingId, controller.signal);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setMeetingIntelligence((prev) => ({ ...prev, [meetingId]: data }));
      }
    } catch (e: unknown) {
      if (isContextCurrent(capturedEpoch, capturedOrgId) && !isAbortError(e)) {
        const msg = e instanceof Error ? e.message : 'Failed to load intelligence';
        setIntelligenceError((prev) => ({ ...prev, [meetingId]: msg }));
      }
    } finally {
      unregisterController(controller);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsLoadingIntelligence((prev) => ({ ...prev, [meetingId]: false }));
      }
    }
  }, [currentOrg, createRegisteredController, isContextCurrent, unregisterController]);

  const handleTriggerExtraction = useCallback(async (
    meetingId: string,
    options: { outputLanguage: string; providerOverride?: string }
  ) => {
    if (!currentOrg) return;
    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    const controller = createRegisteredController();
    setIsExtractingMap((prev) => ({ ...prev, [meetingId]: true }));
    setIntelligenceError((prev) => ({ ...prev, [meetingId]: null }));

    try {
      const res = await triggerExtractionJob(
        meetingId,
        {
          output_language: options.outputLanguage,
          provider_override: options.providerOverride,
        },
        controller.signal
      );

      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setActiveJob({
          id: res.job_id,
          meeting_id: res.meeting_id,
          job_type: res.job_type,
          status: res.status,
          progress: 0,
          error_message: null,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        setExpandedIntelligence((prev) => ({ ...prev, [meetingId]: true }));
      }
    } catch (e: unknown) {
      if (isContextCurrent(capturedEpoch, capturedOrgId) && !isAbortError(e)) {
        const msg = e instanceof Error ? e.message : 'Extraction request failed';
        setIntelligenceError((prev) => ({ ...prev, [meetingId]: msg }));
      }
    } finally {
      unregisterController(controller);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsExtractingMap((prev) => ({ ...prev, [meetingId]: false }));
      }
    }
  }, [currentOrg, createRegisteredController, isContextCurrent, unregisterController]);

  const handleUpdateActionStatus = useCallback(async (
    meetingId: string,
    itemId: string,
    newStatus: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED'
  ) => {
    const controller = createRegisteredController();
    try {
      const updated = await updateActionItemStatus(itemId, newStatus, controller.signal);
      setMeetingIntelligence((prev) => {
        const current = prev[meetingId];
        if (!current) return prev;
        return {
          ...prev,
          [meetingId]: {
            ...current,
            action_items: current.action_items.map((item) =>
              item.id === itemId ? { ...item, status: updated.status } : item
            ),
          },
        };
      });
    } finally {
      unregisterController(controller);
    }
  }, [createRegisteredController, unregisterController]);

  const handleNavigateToEvidence = useCallback((
    meetingId: string,
    segmentId: string,
    _timestampSeconds?: number | null
  ) => {
    // Clear any prior navigation error
    setEvidenceNavErrorMap((prev) => ({ ...prev, [meetingId]: null }));
    setExpandedTranscripts((prev) => ({ ...prev, [meetingId]: true }));

    const transcripts = meetingTranscripts[meetingId] || [];
    if (!meetingTranscripts[meetingId] || transcripts.length === 0) {
      loadTranscriptsForMeeting(meetingId);
    }

    // Identify which loaded transcript contains the requested segment ID
    const targetTranscript = transcripts.find((t) =>
      t.segments && t.segments.some((seg) => seg.id === segmentId)
    );

    if (transcripts.length > 0 && !targetTranscript) {
      setEvidenceNavErrorMap((prev) => ({
        ...prev,
        [meetingId]: 'Referenced evidence segment is not available in current transcripts.',
      }));
      return;
    }

    // Select that transcript version before attempting navigation
    if (targetTranscript) {
      setSelectedVersionMap((prev) => {
        if (prev[meetingId] === targetTranscript.id) return prev;
        return { ...prev, [meetingId]: targetTranscript.id };
      });
    }

    // Wait for the correct transcript DOM elements to render before scrolling
    let attempts = 0;
    const maxAttempts = 25;
    const checkIntervalMs = 50;

    const tryScrollToSegment = () => {
      const elem = document.getElementById(`segment-${segmentId}`);
      if (elem) {
        elem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        elem.classList.add('evidence-highlighted');
        setTimeout(() => elem.classList.remove('evidence-highlighted'), 2500);
      } else if (attempts < maxAttempts) {
        attempts += 1;
        setTimeout(tryScrollToSegment, checkIntervalMs);
      } else {
        const meetingElem = document.getElementById(`meeting-${meetingId}`);
        if (meetingElem) {
          meetingElem.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        setEvidenceNavErrorMap((prev) => ({
          ...prev,
          [meetingId]: 'Referenced evidence segment could not be located in view.',
        }));
      }
    };

    setTimeout(tryScrollToSegment, 40);
  }, [meetingTranscripts, loadTranscriptsForMeeting]);

  const handleMemoryNavigateToEvidence = useCallback((
    meetingId: string,
    segmentId: string,
    timestampSeconds?: number | null
  ) => {
    setActiveMainTab('meetings');
    handleNavigateToEvidence(meetingId, segmentId, timestampSeconds);
  }, [handleNavigateToEvidence]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const renderSegmentText = (text: string) => {
    const parts = text.split(/(\[[A-Z_]+_REDACTED\])/g);
    return parts.map((part, i) => {
      if (part.startsWith('[') && part.endsWith('_REDACTED]')) {
        return (
          <span key={i} className="redacted-tag-inline">
            <Shield size={11} style={{ display: 'inline', marginRight: '3px', verticalAlign: '-1px' }} />
            {part}
          </span>
        );
      }
      return part;
    });
  };

  const runHealthCheck = useCallback(async () => {
    setIsRefreshing(true);
    const result = await checkBackendHealth();
    setHealthResult(result);

    await fetchRootInfo();
    if (isAuthenticated && currentOrg) {
      await loadMeetings();
    }
    if (isAuthenticated && user?.is_super_admin) {
      await loadHardwareDiagnostics();
    } else {
      setHardwareDiagnostics(null);
    }
    setIsRefreshing(false);
  }, [isAuthenticated, user?.is_super_admin, currentOrg, loadMeetings, loadHardwareDiagnostics]);

  useEffect(() => {
    runHealthCheck();
    const interval = setInterval(runHealthCheck, 12000);
    return () => clearInterval(interval);
  }, [runHealthCheck]);

  useEffect(() => {
    let isMounted = true;
    const fetchTenantData = async () => {
      await Promise.resolve();
      if (!isMounted) return;
      if (isAuthenticated && currentOrg) {
        await loadMeetings();
      }
      if (isAuthenticated && user?.is_super_admin) {
        await loadHardwareDiagnostics();
      } else {
        setHardwareDiagnostics(null);
        setActiveInspectorTab((prev) => (prev === 'asr' ? 'health' : prev));
      }
    };
    fetchTenantData();
    return () => {
      isMounted = false;
    };
  }, [isAuthenticated, user?.is_super_admin, currentOrg, loadMeetings, loadHardwareDiagnostics]);

  // Poll active processing job
  useEffect(() => {
    if (!activeJob || activeJob.status === 'COMPLETED' || activeJob.status === 'FAILED' || !currentOrg) {
      return;
    }

    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;
    let inFlightController: AbortController | null = null;
    let isCancelled = false;

    const pollTick = async () => {
      if (isCancelled || !isContextCurrent(capturedEpoch, capturedOrgId)) {
        return;
      }

      // Abort previous in-flight poll tick if still running
      if (inFlightController) {
        try {
          inFlightController.abort();
        } catch {
          // Ignore
        }
      }

      inFlightController = createRegisteredController();
      try {
        const job = await fetchJobStatus(activeJob.id, inFlightController.signal);
        if (isCancelled || !isContextCurrent(capturedEpoch, capturedOrgId)) {
          return;
        }

        setActiveJob(job);
        if (job.status === 'COMPLETED' || job.status === 'FAILED') {
          await loadMeetings();
          if (job.job_type === 'TRANSCRIPTION') {
            await loadTranscriptsForMeeting(activeJob.meeting_id);
          }
          if (job.job_type === 'EXTRACTION') {
            await loadIntelligenceForMeeting(activeJob.meeting_id);
          }
        }
      } catch (err) {
        if (!isAbortError(err)) {
          // Ignore temporary polling glitches
        }
      } finally {
        if (inFlightController) {
          unregisterController(inFlightController);
          inFlightController = null;
        }
      }
    };

    const timer = setInterval(pollTick, 1500);

    return () => {
      isCancelled = true;
      clearInterval(timer);
      if (inFlightController) {
        try {
          inFlightController.abort();
        } catch {
          // Ignore
        }
        unregisterController(inFlightController);
        inFlightController = null;
      }
    };
  }, [
    activeJob,
    currentOrg,
    createRegisteredController,
    isContextCurrent,
    loadMeetings,
    loadTranscriptsForMeeting,
    loadIntelligenceForMeeting,
    unregisterController,
  ]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setFileError(null);
      if (!fileTitle) {
        setFileTitle(file.name.replace(/\.[^/.]+$/, ''));
      }
    }
  };

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setFileError('Please select an audio or video file to upload.');
      return;
    }
    if (!currentOrg) return;

    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    setFileError(null);
    setIsUploading(true);
    setUploadProgress(0);

    const uploadController = new AbortController();
    activeUploadAbortControllerRef.current = uploadController;

    try {
      const response = await uploadMediaFile(
        selectedFile,
        fileTitle || undefined,
        fileDesc || undefined,
        (progress) => {
          if (isContextCurrent(capturedEpoch, capturedOrgId)) {
            setUploadProgress(progress);
          }
        },
        uploadController.signal
      );

      if (!isContextCurrent(capturedEpoch, capturedOrgId)) {
        // Discard response from aborted/stale context
        return;
      }

      // Reset form
      setSelectedFile(null);
      setFileTitle('');
      setFileDesc('');
      if (fileInputRef.current) fileInputRef.current.value = '';

      // Track active background job
      setActiveJob({
        id: response.job_id,
        meeting_id: response.meeting_id,
        job_type: 'INGESTION',
        status: 'IN_PROGRESS',
        progress: 10,
        error_message: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

      await loadMeetings();
    } catch (err: unknown) {
      if (isAbortError(err)) {
        return;
      }
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        const msg = err instanceof Error ? err.message : 'Upload failed';
        setFileError(msg);
      }
    } finally {
      if (activeUploadAbortControllerRef.current === uploadController) {
        activeUploadAbortControllerRef.current = null;
      }
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsUploading(false);
      }
    }
  };

  const handleYouTubeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!youtubeUrl) {
      setYtError('Please enter a YouTube video URL.');
      return;
    }
    if (!confirmedAuth) {
      setYtError('You must confirm you have authorization to process this YouTube content.');
      return;
    }
    if (!currentOrg) return;

    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    setYtError(null);
    setIsSubmittingYt(true);

    const controller = createRegisteredController();
    try {
      const response = await ingestYouTube(
        youtubeUrl,
        ytTitle || undefined,
        ytDesc || undefined,
        confirmedAuth,
        controller.signal
      );

      if (!isContextCurrent(capturedEpoch, capturedOrgId)) {
        return;
      }

      // Reset form
      setYoutubeUrl('');
      setYtTitle('');
      setYtDesc('');
      setConfirmedAuth(false);

      // Track active background job
      setActiveJob({
        id: response.job_id,
        meeting_id: response.meeting_id,
        job_type: 'INGESTION',
        status: 'IN_PROGRESS',
        progress: 15,
        error_message: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

      await loadMeetings();
    } catch (err: unknown) {
      if (isAbortError(err)) {
        return;
      }
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        const msg = err instanceof Error ? err.message : 'YouTube ingestion failed';
        setYtError(msg);
      }
    } finally {
      unregisterController(controller);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsSubmittingYt(false);
      }
    }
  };

  const handleDeleteRawMedia = async (assetId: string) => {
    if (!window.confirm('Are you sure you want to securely delete this raw media file? This action is logged.')) {
      return;
    }
    if (!currentOrg) return;

    const capturedEpoch = requestEpochRef.current;
    const capturedOrgId = currentOrg.organization_id;

    setIsDeletingAssetId(assetId);
    const controller = createRegisteredController();
    try {
      await deleteRawMedia(assetId, controller.signal);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        await loadMeetings();
      }
    } catch (err: unknown) {
      if (!isAbortError(err)) {
        if (isContextCurrent(capturedEpoch, capturedOrgId)) {
          alert(err instanceof Error ? err.message : 'Failed to delete raw media');
        }
      }
    } finally {
      unregisterController(controller);
      if (isContextCurrent(capturedEpoch, capturedOrgId)) {
        setIsDeletingAssetId(null);
      }
    }
  };

  const isConnected = healthResult?.data?.status === 'ok';

  // 1. Loading state during session restoration
  if (isLoading) {
    return (
      <div className="auth-loading-container">
        <RefreshCw size={28} className="spinning" color="#6366f1" />
        <span>Verifying secure session...</span>
      </div>
    );
  }

  // 2. Unauthenticated state: display authentication screen
  if (!isAuthenticated) {
    return <AuthScreen />;
  }

  // 3. Organization selection required: multiple organizations (or 0) without active selection
  if (!currentOrg || organizations.length === 0) {
    return <OrgSelectionScreen />;
  }

  return (
    <div className="app-container">
      {/* Navigation Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="brand-section">
            <div className="brand-icon-wrapper">
              <Sparkles size={22} />
            </div>
            <div>
              <h1 className="brand-title">Multilingual Meeting Intelligence</h1>
              <div className="brand-subtitle">Organizational Memory & Dual-Source RAG Platform</div>
            </div>
          </div>

          <div className="header-actions">
            <div
              className={`status-pill ${
                isRefreshing && !healthResult
                  ? 'checking'
                  : isConnected
                  ? 'connected'
                  : 'disconnected'
              }`}
            >
              <div className="status-dot" />
              <span>
                {isRefreshing && !healthResult
                  ? 'Checking...'
                  : isConnected
                  ? 'Backend Connected'
                  : 'Backend Disconnected'}
              </span>
            </div>

            <button
              className="refresh-button"
              onClick={runHealthCheck}
              disabled={isRefreshing || isSwitchingOrg}
              title="Refresh connection status"
            >
              <RefreshCw size={14} className={isRefreshing ? 'spinning' : ''} />
              <span>{isRefreshing ? 'Checking...' : 'Refresh'}</span>
            </button>

            <div className="header-divider" />

            {/* Organization Display / Switcher */}
            {organizations.length > 1 ? (
              <div className="org-switcher-wrapper">
                <Building size={14} style={{ color: '#818cf8', flexShrink: 0 }} />
                <select
                  className="org-switcher-select"
                  value={currentOrg.organization_id}
                  onChange={(e) => handleSwitchOrganization(e.target.value)}
                  disabled={isSwitchingOrg}
                  title="Switch active organization workspace"
                  aria-label="Active Organization Workspace"
                >
                  {organizations.map((org) => (
                    <option key={org.organization_id} value={org.organization_id}>
                      {org.organization_name} ({org.role})
                    </option>
                  ))}
                </select>
                <span className={`header-role-badge ${currentOrg.role === 'ORG_ADMIN' ? 'admin' : 'member'}`}>
                  <Shield size={10} />
                  {currentOrg.role}
                </span>
                {isSwitchingOrg && <RefreshCw size={12} className="spinning" color="#818cf8" />}
              </div>
            ) : (
              <div className="header-single-org">
                <Building size={14} style={{ color: '#818cf8', flexShrink: 0 }} />
                <span>{currentOrg.organization_name}</span>
                <span className={`header-role-badge ${currentOrg.role === 'ORG_ADMIN' ? 'admin' : 'member'}`}>
                  <Shield size={10} />
                  {currentOrg.role}
                </span>
              </div>
            )}

            {/* Authenticated User Badge */}
            <div className="header-user-badge" title={`Signed in as ${user?.email}`}>
              <User size={14} style={{ color: '#94a3b8' }} />
              <span>{user?.full_name || user?.email}</span>
            </div>

            {/* Logout Action */}
            <button
              className="header-logout-btn"
              onClick={handleLogout}
              title="Sign out of workspace"
              aria-label="Sign out"
            >
              <LogOut size={14} />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </header>

      {/* Top Navigation Bar */}
      <nav className="app-top-nav" aria-label="Main Navigation">
        <div className="app-top-nav-inner">
          <button
            type="button"
            className={`app-nav-tab ${activeMainTab === 'meetings' ? 'active' : ''}`}
            onClick={() => setActiveMainTab('meetings')}
            data-testid="nav-tab-meetings"
          >
            <Layers size={16} />
            <span>Meetings & Media</span>
            {meetings.length > 0 && <span className="app-nav-badge">{meetings.length}</span>}
          </button>
          <button
            type="button"
            className={`app-nav-tab ${activeMainTab === 'memory' ? 'active' : ''}`}
            onClick={() => setActiveMainTab('memory')}
            data-testid="nav-tab-memory"
          >
            <Sparkles size={16} />
            <span>Organizational Memory & Ask AI</span>
          </button>
          <button
            type="button"
            className={`app-nav-tab ${activeMainTab === 'system' ? 'active' : ''}`}
            onClick={() => setActiveMainTab('system')}
            data-testid="nav-tab-system"
          >
            <Server size={16} />
            <span>System & Diagnostics</span>
          </button>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="app-main">
        {/* ORGANIZATIONAL MEMORY & CROSS-MEETING INTELLIGENCE (PHASE 8.6 & 8.8.6) */}
        {activeMainTab === 'memory' && (
          <OrganizationalMemoryView
            meetings={meetings}
            onNavigateToEvidence={handleMemoryNavigateToEvidence}
            currentOrgName={currentOrg?.organization_name}
            currentOrgRole={currentOrg?.role}
            currentOrgId={currentOrg?.organization_id}
          />
        )}

        {/* MEETINGS & MEDIA INGESTION WORKSPACE */}
        {activeMainTab === 'meetings' && (
          <>
            {/* MEDIA INGESTION HUB (PHASE 3) */}
            <section>
          <div className="section-title">
            <Upload size={22} color="#6366f1" />
            Media Ingestion Hub
          </div>

          <div className="ingestion-card">
            {/* Ingestion Tabs */}
            <div className="ingestion-tabs">
              <button
                className={`ingestion-tab-btn ${ingestionTab === 'file' ? 'active' : ''}`}
                onClick={() => setIngestionTab('file')}
              >
                <FileAudio size={16} /> Audio / Video File Upload
              </button>
              <button
                className={`ingestion-tab-btn ${ingestionTab === 'youtube' ? 'active' : ''}`}
                onClick={() => setIngestionTab('youtube')}
              >
                <PlayCircle size={16} color="#ef4444" /> YouTube Stream Ingestion
              </button>
            </div>

            <div className="ingestion-body">
              {/* Tab 1: File Upload */}
              {ingestionTab === 'file' && (
                <form onSubmit={handleFileUpload} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <div
                    className="dropzone"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      type="file"
                      ref={fileInputRef}
                      style={{ display: 'none' }}
                      accept=".wav,.mp3,.m4a,.flac,.mp4,.mkv,.mov,.webm"
                      onChange={handleFileChange}
                    />
                    <div className="dropzone-icon">
                      <Upload size={26} />
                    </div>
                    <div className="dropzone-title">
                      {selectedFile ? selectedFile.name : 'Click or drop meeting audio/video file here'}
                    </div>
                    <div className="dropzone-hint">
                      Supported: WAV, MP3, M4A, FLAC, MP4, MKV, MOV, WebM (up to 500 MB)
                    </div>

                    {selectedFile && (
                      <div className="file-selected-badge">
                        <CheckCircle2 size={14} />
                        {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB &bull; Ready for upload
                      </div>
                    )}
                  </div>

                  {fileError && (
                    <div style={{ color: '#ef4444', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <AlertTriangle size={14} /> {fileError}
                    </div>
                  )}

                  <div className="form-grid">
                    <div className="form-group">
                      <label className="form-label">Meeting Title (Optional)</label>
                      <input
                        type="text"
                        className="form-input"
                        placeholder="e.g. Q4 Multilingual Sprint Retrospective"
                        value={fileTitle}
                        onChange={(e) => setFileTitle(e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Description (Optional)</label>
                      <input
                        type="text"
                        className="form-input"
                        placeholder="e.g. Weekly review discussing architecture and deadlines"
                        value={fileDesc}
                        onChange={(e) => setFileDesc(e.target.value)}
                      />
                    </div>
                  </div>

                  {isUploading && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#a5b4fc' }}>
                        <span>Uploading file to secure storage...</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <div className="progress-bar-track">
                        <div className="progress-bar-fill" style={{ width: `${uploadProgress}%` }} />
                      </div>
                    </div>
                  )}

                  <div className="submit-action-row">
                    <button
                      type="submit"
                      className="btn-primary"
                      disabled={!selectedFile || isUploading}
                    >
                      {isUploading ? (
                        <>
                          <RefreshCw size={16} className="spinning" /> Uploading ({uploadProgress}%)...
                        </>
                      ) : (
                        <>
                          <Upload size={16} /> Ingest Media File
                        </>
                      )}
                    </button>
                  </div>
                </form>
              )}

              {/* Tab 2: YouTube Ingestion */}
              {ingestionTab === 'youtube' && (
                <form onSubmit={handleYouTubeSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <div className="form-group full-width">
                    <label className="form-label">YouTube Video URL</label>
                    <input
                      type="url"
                      className="form-input"
                      placeholder="https://www.youtube.com/watch?v=..."
                      value={youtubeUrl}
                      onChange={(e) => setYoutubeUrl(e.target.value)}
                      required
                    />
                  </div>

                  <div className="form-grid">
                    <div className="form-group">
                      <label className="form-label">Custom Meeting Title (Optional)</label>
                      <input
                        type="text"
                        className="form-input"
                        placeholder="Leave blank to use YouTube video title"
                        value={ytTitle}
                        onChange={(e) => setYtTitle(e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Description (Optional)</label>
                      <input
                        type="text"
                        className="form-input"
                        placeholder="Notes or context for this video"
                        value={ytDesc}
                        onChange={(e) => setYtDesc(e.target.value)}
                      />
                    </div>
                  </div>

                  <label className="checkbox-container">
                    <input
                      type="checkbox"
                      checked={confirmedAuth}
                      onChange={(e) => setConfirmedAuth(e.target.checked)}
                      required
                    />
                    <span className="checkbox-text">
                      <strong>Legal Authorization Confirmation:</strong> I confirm that I have legal authorization
                      to access and process the submitted external YouTube content. The system does not bypass DRM or
                      paywalls.
                    </span>
                  </label>

                  {ytError && (
                    <div style={{ color: '#ef4444', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <AlertTriangle size={14} /> {ytError}
                    </div>
                  )}

                  <div className="submit-action-row">
                    <button
                      type="submit"
                      className="btn-primary"
                      disabled={!youtubeUrl || !confirmedAuth || isSubmittingYt}
                    >
                      {isSubmittingYt ? (
                        <>
                          <RefreshCw size={16} className="spinning" /> Ingesting YouTube Stream...
                        </>
                      ) : (
                        <>
                          <Play size={16} /> Ingest YouTube Meeting
                        </>
                      )}
                    </button>
                  </div>
                </form>
              )}

              {/* Active Background Job Monitor */}
              {activeJob && (
                <div className="active-job-banner">
                  <div className="active-job-header">
                    <div className="job-title">
                      <Clock size={16} color="#818cf8" />
                      Job: {activeJob.job_type} &bull; Status: {activeJob.status}
                    </div>
                    <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#34d399' }}>
                      {Math.round(activeJob.progress)}%
                    </span>
                  </div>

                  <div className="progress-bar-track">
                    <div className="progress-bar-fill" style={{ width: `${activeJob.progress}%` }} />
                  </div>

                  <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                    {activeJob.status === 'IN_PROGRESS' &&
                      (activeJob.job_type === 'TRANSCRIPTION'
                        ? 'Transcribing audio with Faster-Whisper on RTX 5060 GPU...'
                        : activeJob.job_type === 'EXTRACTION'
                        ? 'Extracting structured meeting intelligence (summary, decisions, actions, risks)...'
                        : 'Processing media: Validating streams and normalizing audio to 16kHz mono WAV...')}
                    {activeJob.status === 'COMPLETED' &&
                      (activeJob.job_type === 'TRANSCRIPTION'
                        ? 'Transcription completed successfully! Timestamps and segments generated.'
                        : activeJob.job_type === 'EXTRACTION'
                        ? 'Meeting intelligence extracted successfully! Summary, decisions, and action items ready.'
                        : 'Media ingestion completed! Normalized 16kHz mono WAV asset generated.')}
                    {activeJob.status === 'FAILED' && `Error: ${activeJob.error_message || 'Processing failed'}`}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* INGESTED MEETINGS HISTORY */}
        <section className="history-section">
          <div className="section-title">
            <Layers size={22} color="#10b981" />
            Ingested Meetings & Media History ({meetings.length})
          </div>

          {meetings.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#64748b', background: '#161b2a', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)' }}>
              No meetings ingested yet. Upload an audio/video file or YouTube link above to start.
            </div>
          ) : (
            <div className="history-grid">
              {meetings.map((meeting) => (
                <div key={meeting.id} id={`meeting-${meeting.id}`} className="meeting-item-card">
                  <div className="meeting-card-top">
                    <div className="meeting-title-row">
                      <span className="meeting-title">{meeting.title}</span>
                      <span className={`source-badge ${meeting.input_source_type.toLowerCase()}`}>
                        {meeting.input_source_type === 'YOUTUBE' ? <PlayCircle size={12} /> : meeting.input_source_type === 'VIDEO' ? <FileVideo size={12} /> : <FileAudio size={12} />}
                        {meeting.input_source_type}
                      </span>
                    </div>
                    <span className="meeting-date">
                      {new Date(meeting.created_at).toLocaleDateString()} {new Date(meeting.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>

                  {meeting.description && (
                    <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>{meeting.description}</div>
                  )}

                  {/* Associated Media Assets */}
                  <div className="assets-box">
                    <div className="assets-title">Media Assets ({meeting.media_assets?.length || 0})</div>
                    {meeting.media_assets && meeting.media_assets.length > 0 ? (
                      meeting.media_assets.map((asset) => (
                        <div key={asset.id} className="asset-row">
                          <div className="asset-info">
                            <span className={`asset-tag ${asset.media_type === 'NORMALIZED_AUDIO' ? 'normalized' : ''}`}>
                              {asset.media_type}
                            </span>
                            <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{asset.file_name}</span>
                            {asset.duration_seconds && (
                              <span style={{ color: '#94a3b8' }}>({Math.round(asset.duration_seconds)}s)</span>
                            )}
                            {asset.file_size_bytes && (
                              <span style={{ color: '#64748b' }}>{(asset.file_size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
                            )}
                          </div>

                          <div>
                            {asset.is_raw_deleted ? (
                              <span className="deleted-tag">Raw Deleted</span>
                            ) : (
                              currentOrg?.role === 'ORG_ADMIN' && asset.media_type !== 'NORMALIZED_AUDIO' && (
                                <button
                                  className="btn-delete-raw"
                                  onClick={() => handleDeleteRawMedia(asset.id)}
                                  disabled={isDeletingAssetId === asset.id}
                                  title="Safely delete raw media file from disk"
                                >
                                  <Trash2 size={12} style={{ display: 'inline', marginRight: '4px' }} />
                                  {isDeletingAssetId === asset.id ? 'Deleting...' : 'Delete Raw'}
                                </button>
                              )
                            )}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div style={{ fontSize: '0.8rem', color: '#64748b' }}>No media assets registered</div>
                    )}
                  </div>

                  {/* Phase 4: Multilingual ASR Transcription Controls & Viewer */}
                  <div className="transcription-panel">
                    <div className="transcription-actions">
                      <select
                        className="language-select"
                        value={selectedLangMap[meeting.id] || ''}
                        onChange={(e) =>
                          setSelectedLangMap((prev) => ({ ...prev, [meeting.id]: e.target.value }))
                        }
                      >
                        <option value="">Auto-Detect Spoken Language</option>
                        <option value="en">English (en)</option>
                        <option value="hi">Hindi (hi)</option>
                        <option value="gu">Gujarati (gu)</option>
                        <option value="hinglish">Hinglish Code-Switching (Hindi-English)</option>
                      </select>

                      <button
                        className="btn-transcribe"
                        onClick={() => handleStartMeetingTranscription(meeting.id)}
                        disabled={
                          isTranscribingMap[meeting.id] ||
                          (activeJob?.status === 'IN_PROGRESS' &&
                            activeJob.meeting_id === meeting.id &&
                            activeJob.job_type === 'TRANSCRIPTION')
                        }
                      >
                        <Mic size={14} />
                        {isTranscribingMap[meeting.id] ||
                        (activeJob?.status === 'IN_PROGRESS' &&
                          activeJob.meeting_id === meeting.id &&
                          activeJob.job_type === 'TRANSCRIPTION')
                          ? 'Transcribing on GPU...'
                          : 'Transcribe with Whisper'}
                      </button>

                      <button
                        className="btn-copy"
                        onClick={() => {
                          const isExp = !!expandedTranscripts[meeting.id];
                          setExpandedTranscripts((prev) => ({ ...prev, [meeting.id]: !isExp }));
                          if (!isExp) loadTranscriptsForMeeting(meeting.id);
                        }}
                      >
                        <FileText size={14} />
                        {expandedTranscripts[meeting.id] ? 'Hide Transcript' : 'View Transcript'}
                      </button>

                      <button
                        className={`btn-copy ${expandedIntelligence[meeting.id] ? 'active' : ''}`}
                        onClick={() => {
                          const isExp = !!expandedIntelligence[meeting.id];
                          setExpandedIntelligence((prev) => ({ ...prev, [meeting.id]: !isExp }));
                          if (!isExp) {
                            if (!meetingIntelligence[meeting.id]) {
                              loadIntelligenceForMeeting(meeting.id);
                            }
                            if (!meetingTranscripts[meeting.id]) {
                              loadTranscriptsForMeeting(meeting.id);
                            }
                          }
                        }}
                        title="View structured meeting intelligence (summary, decisions, actions, risks, questions)"
                      >
                        <Sparkles size={14} color="#818cf8" />
                        {expandedIntelligence[meeting.id] ? 'Hide Intelligence' : 'Meeting Intelligence'}
                      </button>
                    </div>

                    {/* Transcripts List & Phase 5 Multi-Version Intelligence Panel */}
                    {expandedTranscripts[meeting.id] && (
                      <div className="transcript-container">
                        {meetingTranscripts[meeting.id] && meetingTranscripts[meeting.id].length > 0 ? (
                          (() => {
                            const transcripts = meetingTranscripts[meeting.id];
                            const origTranscript = transcripts.find((t) => t.transcript_type === 'ORIGINAL') || transcripts[0];

                            // Strict Privacy Check: Identify genuinely privacy-safe transcripts (CLEANED_REDACTED or REDACTED only)
                            const privacySafeTranscripts = transcripts.filter(
                              (t) => t.transcript_type === 'CLEANED_REDACTED' || t.transcript_type === 'REDACTED'
                            );

                            // Privacy-safe priority: CLEANED_REDACTED (latest) > REDACTED (latest)
                            const preferredPrivacySafe = privacySafeTranscripts.length > 0
                              ? [...privacySafeTranscripts].sort((a, b) => {
                                  if (a.transcript_type !== b.transcript_type) {
                                    return a.transcript_type === 'CLEANED_REDACTED' ? -1 : 1;
                                  }
                                  return b.version - a.version;
                                })[0]
                              : null;

                            // Default selection:
                            // Prefer explicit user selection if set, else preferred privacy-safe version.
                            // If NO privacy-safe version exists, fallback to transcripts[0] with explicit unredacted warning banners.
                            const hasPrivacySafe = !!preferredPrivacySafe;
                            const defaultTranscript = preferredPrivacySafe || transcripts[0];
                            const activeId = selectedVersionMap[meeting.id] || defaultTranscript.id;
                            const activeTranscript = transcripts.find((t) => t.id === activeId) || defaultTranscript;
                            const isActivePrivacySafe = activeTranscript.transcript_type === 'CLEANED_REDACTED' || activeTranscript.transcript_type === 'REDACTED';
                            const isComparing = !!compareModeMap[meeting.id] && origTranscript && activeTranscript.id !== origTranscript.id;
                            const isBusyAction = isProcessingActionMap[meeting.id];

                            return (
                              <div className="transcript-version-box">
                                {/* Version Selector Tabs & Intelligence Action Bar */}
                                <div className="transcript-toolbar">
                                  <div className="version-tabs-group">
                                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', marginRight: '0.3rem' }}>
                                      Versions ({transcripts.length}):
                                    </span>
                                    {transcripts.map((t) => {
                                      const isRedactedType = t.transcript_type === 'CLEANED_REDACTED' || t.transcript_type === 'REDACTED';
                                      return (
                                        <button
                                          key={t.id}
                                          className={`version-tab-btn ${t.id === activeTranscript.id ? 'active' : ''} ${t.transcript_type.toLowerCase()}`}
                                          onClick={() => setSelectedVersionMap((prev) => ({ ...prev, [meeting.id]: t.id }))}
                                          title={isRedactedType ? 'Privacy-Safe Redacted Version' : 'Contains Unredacted Sensitive Data'}
                                        >
                                          <span className="v-tag">{t.transcript_type}</span>
                                          <span className="v-num">v{t.version}</span>
                                          {t.is_primary && <span className="v-primary-dot" title="Primary ASR Record" />}
                                        </button>
                                      );
                                    })}
                                  </div>

                                  {/* Intelligence & Privacy Derivation Actions */}
                                  <div className="derivation-actions-row">
                                    {currentOrg?.role === 'ORG_ADMIN' && (
                                      <button
                                        className="btn-intel-action clean"
                                        onClick={() => handleCleanTranscript(meeting.id)}
                                        disabled={!!isBusyAction}
                                        title="Generate Cleaned transcript (removes fillers/stutters; unredacted)"
                                      >
                                        <Wand2 size={13} />
                                        {isBusyAction === 'clean' ? 'Cleaning...' : 'Clean'}
                                      </button>
                                    )}

                                    <button
                                      className="btn-intel-action redact"
                                      onClick={() => handleRedactTranscript(meeting.id)}
                                      disabled={!!isBusyAction}
                                      title="Generate Redacted transcript (redacts emails, phones, Aadhaar, PAN, secrets)"
                                    >
                                      <Shield size={13} />
                                      {isBusyAction === 'redact' ? 'Redacting...' : 'Redact PII'}
                                    </button>

                                    <button
                                      className="btn-intel-action process"
                                      onClick={() => handleProcessBothTranscript(meeting.id)}
                                      disabled={!!isBusyAction}
                                      title="Generate Cleaned + Redacted derived transcript (Recommended)"
                                    >
                                      <Sparkles size={13} />
                                      {isBusyAction === 'both' ? 'Processing...' : 'Clean & Redact'}
                                    </button>

                                    {origTranscript && transcripts.length > 1 && (
                                      <button
                                        className={`btn-intel-action compare ${isComparing ? 'active' : ''}`}
                                        onClick={() =>
                                          setCompareModeMap((prev) => ({ ...prev, [meeting.id]: !prev[meeting.id] }))
                                        }
                                        title="Toggle side-by-side comparison with original ASR transcript"
                                      >
                                        <Columns size={13} />
                                        {isComparing ? 'Exit Comparison' : 'Compare with Original'}
                                      </button>
                                    )}

                                    <button
                                      className="btn-copy"
                                      onClick={() => handleCopyTranscriptText(activeTranscript)}
                                      title={isActivePrivacySafe ? 'Copy privacy-safe transcript text' : 'Warning: Copying unredacted transcript with raw sensitive data'}
                                    >
                                      {copiedId === activeTranscript.id ? (
                                        <>
                                          <Check size={12} color="#34d399" /> Copied
                                        </>
                                      ) : (
                                        <>
                                          <Copy size={12} /> {isActivePrivacySafe ? 'Copy' : 'Copy Unredacted'}
                                        </>
                                      )}
                                    </button>
                                  </div>
                                </div>

                                {/* Privacy Alert Banner: When NO Privacy-Safe Version Exists */}
                                {!hasPrivacySafe && (
                                  <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    gap: '0.6rem',
                                    background: 'rgba(239, 68, 68, 0.12)',
                                    border: '1px solid rgba(239, 68, 68, 0.4)',
                                    borderRadius: '6px',
                                    padding: '0.55rem 0.8rem',
                                    margin: '0.6rem 0.8rem 0 0.8rem',
                                    fontSize: '0.8rem',
                                    color: '#fca5a5',
                                  }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                      <AlertTriangle size={15} color="#ef4444" style={{ flexShrink: 0 }} />
                                      <span>
                                        <strong>No Privacy-Safe Transcript Available:</strong> All current versions are unredacted and may contain sensitive PII or credentials.
                                      </span>
                                    </div>
                                    <button
                                      className="btn-intel-action process"
                                      style={{ padding: '0.25rem 0.55rem', fontSize: '0.75rem', flexShrink: 0 }}
                                      onClick={() => handleProcessBothTranscript(meeting.id)}
                                      disabled={!!isBusyAction}
                                    >
                                      <Sparkles size={11} /> Clean & Redact Now
                                    </button>
                                  </div>
                                )}

                                {/* Active Transcript Details Header */}
                                <div className="transcript-metadata-bar">
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                                    <span className="transcript-badge">
                                      {activeTranscript.transcript_type} v{activeTranscript.version}
                                    </span>
                                    {isActivePrivacySafe ? (
                                      <span style={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: '0.3rem',
                                        padding: '0.15rem 0.5rem',
                                        borderRadius: '4px',
                                        fontSize: '0.72rem',
                                        fontWeight: 600,
                                        background: 'rgba(16, 185, 129, 0.15)',
                                        color: '#34d399',
                                        border: '1px solid rgba(16, 185, 129, 0.3)'
                                      }}>
                                        <Shield size={11} /> Privacy-Safe (Redacted)
                                      </span>
                                    ) : (
                                      <span style={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: '0.3rem',
                                        padding: '0.15rem 0.5rem',
                                        borderRadius: '4px',
                                        fontSize: '0.72rem',
                                        fontWeight: 600,
                                        background: 'rgba(239, 68, 68, 0.15)',
                                        color: '#f87171',
                                        border: '1px solid rgba(239, 68, 68, 0.3)'
                                      }}>
                                        <AlertTriangle size={11} /> Unredacted Version
                                      </span>
                                    )}
                                    {activeTranscript.parent_transcript_id && (
                                      <span className="parent-link-badge">
                                        Derived from Source
                                      </span>
                                    )}
                                    <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                                      Language:{' '}
                                      <strong style={{ color: '#e2e8f0' }}>
                                        {activeTranscript.language_detected || 'Unknown'}
                                      </strong>
                                    </span>
                                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                                      ({activeTranscript.segments.length} segments)
                                    </span>
                                    {activeTranscript.segments.some((s) => s.is_redacted) && (
                                      <span className="has-redactions-badge">
                                        <Shield size={11} style={{ display: 'inline', marginRight: '3px' }} />
                                        Contains Protected PII Placeholders
                                      </span>
                                    )}
                                  </div>
                                </div>

                                {/* Privacy Alert Banner for Raw Original ASR Record */}
                                {activeTranscript.transcript_type === 'ORIGINAL' && (
                                  <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.6rem',
                                    background: 'rgba(245, 158, 11, 0.12)',
                                    border: '1px solid rgba(245, 158, 11, 0.4)',
                                    borderRadius: '6px',
                                    padding: '0.5rem 0.8rem',
                                    margin: '0.6rem 0.8rem',
                                    fontSize: '0.8rem',
                                    color: '#fbbf24',
                                  }}>
                                    <AlertTriangle size={15} color="#f59e0b" style={{ flexShrink: 0 }} />
                                    <span>
                                      <strong>Unredacted ASR Evidence Record:</strong> You are viewing raw transcription data. Sensitive details (emails, phone numbers, identity numbers, credentials) are unredacted.
                                    </span>
                                  </div>
                                )}

                                {/* Privacy Alert Banner for Cleaned Unredacted Version */}
                                {activeTranscript.transcript_type === 'CLEANED' && (
                                  <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.6rem',
                                    background: 'rgba(245, 158, 11, 0.12)',
                                    border: '1px solid rgba(245, 158, 11, 0.4)',
                                    borderRadius: '6px',
                                    padding: '0.5rem 0.8rem',
                                    margin: '0.6rem 0.8rem',
                                    fontSize: '0.8rem',
                                    color: '#fbbf24',
                                  }}>
                                    <AlertTriangle size={15} color="#f59e0b" style={{ flexShrink: 0 }} />
                                    <span>
                                      <strong>Unredacted Cleaned Version:</strong> Disfluencies and fillers have been removed, but sensitive PII (emails, phones, government IDs, credentials) has <em>NOT</em> been redacted.
                                    </span>
                                  </div>
                                )}

                                {/* Segment Display / Comparison Layout */}
                                {isComparing ? (
                                  <div className="transcript-comparison-grid">
                                    {/* Left: Original Evidence */}
                                    <div className="comparison-col orig-col">
                                      <div className="col-header">
                                        <span className="col-title">Original ASR Evidence Record (v{origTranscript.version})</span>
                                        <span className="col-subtitle">Raw acoustic output with verbatim stutters & sensitive values</span>
                                      </div>
                                      <div className="transcript-segments">
                                        {origTranscript.segments.map((seg) => (
                                          <div key={seg.id} id={`orig-segment-${seg.id}`} className="segment-item original-verbatim">
                                            <div className="segment-meta">
                                              <span className="timestamp-pill">
                                                [{formatTime(seg.start_time)} - {formatTime(seg.end_time)}]
                                              </span>
                                              <span className={`lang-pill ${seg.language_code.toLowerCase()}`}>
                                                {seg.language_code}
                                              </span>
                                              {seg.speaker_label && (
                                                <span className="speaker-pill">{seg.speaker_label}</span>
                                              )}
                                            </div>
                                            <div className="segment-text">{seg.text}</div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>

                                    {/* Right: Selected Processed Version */}
                                    <div className="comparison-col proc-col">
                                      <div className="col-header">
                                        <span className="col-title">
                                          {activeTranscript.transcript_type} (v{activeTranscript.version})
                                        </span>
                                        <span className="col-subtitle">
                                          Cleaned readability & protected PII redactions
                                        </span>
                                      </div>
                                      <div className="transcript-segments">
                                        {activeTranscript.segments.map((seg) => (
                                          <div
                                            key={seg.id}
                                            id={`segment-${seg.id}`}
                                            className={`segment-item ${seg.is_redacted ? 'redacted-item' : ''}`}
                                          >
                                            <div className="segment-meta">
                                              <span className="timestamp-pill">
                                                [{formatTime(seg.start_time)} - {formatTime(seg.end_time)}]
                                              </span>
                                              <span className={`lang-pill ${seg.language_code.toLowerCase()}`}>
                                                {seg.language_code}
                                              </span>
                                              {seg.is_redacted && (
                                                <span className="redacted-pill">
                                                  <Shield size={10} />
                                                  {seg.redaction_metadata?.join(', ') || 'REDACTED'}
                                                </span>
                                              )}
                                              {seg.source_segment_id && (
                                                <span className="evidence-link-pill" title="Mapped to original evidence segment">
                                                  Evidence Linked
                                                </span>
                                              )}
                                            </div>
                                            <div className="segment-text">{renderSegmentText(seg.text)}</div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  </div>
                                ) : (
                                  /* Standard Single Version View */
                                  <div className="transcript-segments">
                                    {activeTranscript.segments.map((seg) => (
                                      <div
                                        key={seg.id}
                                        id={`segment-${seg.id}`}
                                        className={`segment-item ${seg.is_redacted ? 'redacted-item' : ''}`}
                                      >
                                        <div className="segment-meta">
                                          <span className="timestamp-pill">
                                            [{formatTime(seg.start_time)} - {formatTime(seg.end_time)}]
                                          </span>
                                          <span className={`lang-pill ${seg.language_code.toLowerCase()}`}>
                                            {seg.language_code}
                                          </span>
                                          {seg.speaker_label && (
                                            <span className="speaker-pill">{seg.speaker_label}</span>
                                          )}
                                          {seg.is_redacted && (
                                            <span className="redacted-pill">
                                              <Shield size={10} />
                                              {seg.redaction_metadata?.join(', ') || 'REDACTED'}
                                            </span>
                                          )}
                                          {seg.source_segment_id && (
                                            <span className="evidence-link-pill" title="Mapped to original evidence segment">
                                              Evidence Linked
                                            </span>
                                          )}
                                        </div>
                                        <div className="segment-text">{renderSegmentText(seg.text)}</div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          })()
                        ) : (
                          <div style={{ fontSize: '0.825rem', color: '#64748b', padding: '0.5rem 0' }}>
                            No transcripts generated yet for this meeting. Click "Transcribe with Whisper" to start GPU-accelerated speech recognition.
                          </div>
                        )}
                      </div>
                    )}

                    {/* Phase 7: Structured Meeting Intelligence Viewer */}
                    {expandedIntelligence[meeting.id] && (
                      <MeetingIntelligenceView
                        meetingId={meeting.id}
                        intelligence={meetingIntelligence[meeting.id] || null}
                        isLoading={!!isLoadingIntelligence[meeting.id]}
                        error={intelligenceError[meeting.id] || null}
                        hasPrivacySafeTranscript={
                          (meetingTranscripts[meeting.id] || []).some(
                            (t) => t.transcript_type === 'CLEANED_REDACTED' || t.transcript_type === 'REDACTED'
                          )
                        }
                        onExtract={(opts) => handleTriggerExtraction(meeting.id, opts)}
                        isExtracting={
                          !!isExtractingMap[meeting.id] ||
                          (activeJob?.status === 'IN_PROGRESS' &&
                            activeJob.meeting_id === meeting.id &&
                            activeJob.job_type === 'EXTRACTION')
                        }
                        onUpdateActionStatus={(itemId, newStatus) =>
                          handleUpdateActionStatus(meeting.id, itemId, newStatus)
                        }
                        onNavigateToEvidence={(segmentId, ts) =>
                          handleNavigateToEvidence(meeting.id, segmentId, ts)
                        }
                        evidenceNavigationMessage={evidenceNavErrorMap[meeting.id] || null}
                        activeJobMessage={
                          activeJob?.meeting_id === meeting.id && activeJob?.job_type === 'EXTRACTION'
                            ? activeJob.status === 'IN_PROGRESS'
                              ? `Extraction in progress (${Math.round(activeJob.progress)}%)...`
                              : activeJob.status === 'PENDING'
                              ? 'Extraction job queued in worker...'
                              : null
                            : null
                        }
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
        </>
        )}

        {/* SYSTEM & DIAGNOSTICS VIEW */}
        {activeMainTab === 'system' && (
          <>
            {/* Hero Section */}
            <section className="hero-banner">
              <div className="hero-badge">
                <Radio size={12} />
                Phase 8: Multilingual Organizational Memory Active
              </div>
              <h2 className="hero-title">Multilingual Intelligence Platform</h2>
              <p className="hero-description">
                Privacy-aware cross-meeting organizational intelligence system for English, Hindi, Gujarati, and Hinglish.
                Submit meetings via audio, video, or YouTube, extract structured decisions and action items, and ask cross-meeting questions grounded in verified evidence.
              </p>
            </section>

            {/* Live System Metrics */}
            <section className="metrics-grid">
              <div className="metric-card">
                <div className="metric-header">
                  <span>Backend API Status</span>
                  <Server size={18} />
                </div>
                <div
                  className="metric-value"
                  style={{ color: isConnected ? '#10b981' : '#ef4444' }}
                >
                  {isConnected ? (
                    <>
                      <CheckCircle2 size={24} /> Online
                    </>
                  ) : (
                    <>
                      <XCircle size={24} /> Offline
                    </>
                  )}
                </div>
                <div className="metric-sub">
                  {healthResult?.data?.environment
                    ? `Environment: ${healthResult.data.environment}`
                    : 'Target: http://127.0.0.1:8000'}
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-header">
                  <span>API Roundtrip Latency</span>
                  <Activity size={18} />
                </div>
                <div className="metric-value">
                  {healthResult ? `${healthResult.latencyMs} ms` : '--'}
                </div>
                <div className="metric-sub">FastAPI async endpoint ping</div>
              </div>

              <div className="metric-card">
                <div className="metric-header">
                  <span>Database Connection</span>
                  <Database size={18} />
                </div>
                <div
                  className="metric-value"
                  style={{
                    color:
                      healthResult?.data?.database === 'connected' ? '#10b981' : '#ef4444',
                  }}
                >
                  {healthResult?.data?.database === 'connected' ? 'Connected' : 'Unavailable'}
                </div>
                <div className="metric-sub">Engine: SQLite with FK Enforcement</div>
              </div>

              <div className="metric-card">
                <div className="metric-header">
                  <span>API & App Version</span>
                  <FileCode2 size={18} />
                </div>
                <div className="metric-value">
                  {healthResult?.data?.version ? `v${healthResult.data.version}` : 'v0.8.6'}
                </div>
                <div className="metric-sub">Phase 8: Organizational Memory & Ask AI</div>
              </div>
            </section>

            {/* Verified Development & Hardware Environment */}
            <section>
              <div className="section-title">
                <Cpu size={22} color="#6366f1" />
                Verified Hardware & Runtime Environment
              </div>
              <div className="hardware-grid">
                {HARDWARE_SPECS.map((spec, index) => (
                  <div key={index} className="hardware-card">
                    <div className="hardware-top">
                      <span className="hardware-category">{spec.category}</span>
                      <span className="tag-verified">
                        <CheckCircle2 size={12} /> {spec.status}
                      </span>
                    </div>
                    <div className="hardware-name">{spec.value}</div>
                    <div className="hardware-detail">{spec.detail}</div>
                  </div>
                ))}
              </div>
            </section>

            {/* Phase Roadmap */}
            <section>
              <div className="section-title">
                <Layers size={22} color="#8b5cf6" />
                Incremental Implementation Roadmap
              </div>
              <div className="roadmap-grid">
                {ROADMAP_PHASES.map((phase) => (
                  <div
                    key={phase.id}
                    className={`phase-card ${phase.status === 'completed' ? 'active' : ''}`}
                  >
                    <div className="phase-header">
                      <span className="phase-number">Phase {phase.id}</span>
                      <span className={`phase-status-badge ${phase.status}`}>
                        {phase.status === 'completed' ? 'Completed' : 'Planned'}
                      </span>
                    </div>
                    <h3 className="phase-title">{phase.title}</h3>
                    <p className="phase-desc">{phase.description}</p>
                    <div className="tech-tags">
                      {phase.technologies.map((t, idx) => (
                        <span key={idx} className="tech-tag">
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Live API Inspector */}
            <section className="inspector-card">
              <div className="inspector-header">
                <div className="inspector-tabs">
                  <button
                    className={`tab-btn ${activeInspectorTab === 'health' ? 'active' : ''}`}
                    onClick={() => setActiveInspectorTab('health')}
                  >
                    GET /health
                  </button>
                  <button
                    className={`tab-btn ${activeInspectorTab === 'meetings' ? 'active' : ''}`}
                    onClick={() => setActiveInspectorTab('meetings')}
                  >
                    GET /api/v1/media/meetings
                  </button>
                  {user?.is_super_admin && (
                    <button
                      className={`tab-btn ${activeInspectorTab === 'asr' ? 'active' : ''}`}
                      onClick={() => setActiveInspectorTab('asr')}
                    >
                      GET /api/v1/transcription/hardware/diagnostics
                    </button>
                  )}
                </div>
                <a
                  href="http://127.0.0.1:8000/docs"
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    fontSize: '0.8rem',
                    color: '#6366f1',
                    textDecoration: 'none',
                  }}
                >
                  Open Swagger Docs <ExternalLink size={12} />
                </a>
              </div>

              <div className="inspector-content">
                <pre className="json-viewer">
                  {activeInspectorTab === 'health'
                    ? JSON.stringify(healthResult?.data ?? { error: healthResult?.error ?? 'Waiting for backend...' }, null, 2)
                    : activeInspectorTab === 'meetings'
                    ? JSON.stringify(meetings.length > 0 ? meetings : { message: 'No meetings ingested yet or backend offline.' }, null, 2)
                    : user?.is_super_admin
                    ? JSON.stringify(hardwareDiagnostics ?? { message: 'Fetching ASR hardware diagnostics...' }, null, 2)
                    : JSON.stringify({ message: 'Hardware diagnostics are restricted to Super Administrators.' }, null, 2)}
                </pre>
              </div>
            </section>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Multilingual Meeting Intelligence System &bull; Phase 4: Multilingual ASR Pipeline Active &bull; Local GPU Execution (RTX 5060)
        </p>
      </footer>
    </div>
  );
};

export default App;
