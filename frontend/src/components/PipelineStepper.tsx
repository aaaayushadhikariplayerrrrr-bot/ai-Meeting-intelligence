import React from 'react';
import {
  Check,
  AlertTriangle,
  RefreshCw,
  FileAudio,
  Languages,
  ShieldCheck,
  Brain,
  Database,
} from 'lucide-react';
import type { ProcessingJob, TranscriptItem, MeetingIntelligence, MeetingItem } from '../types';
import './PipelineStepper.css';

export interface PipelineStepperProps {
  meeting?: MeetingItem;
  meetingId?: string;
  meetingStatus?: string;
  jobs?: ProcessingJob[];
  transcripts?: TranscriptItem[];
  intelligence?: MeetingIntelligence | null;
  onRetryJob?: (jobId: string) => Promise<void>;
  onRetry?: (jobId: string) => Promise<void>;
  isRetryingJobId?: string | null;
  isRetrying?: boolean;
  isDeriving?: boolean;
}

interface StageInfo {
  id: string;
  label: string;
  sublabel: string;
  icon: React.ReactNode;
  status: 'COMPLETED' | 'IN_PROGRESS' | 'PENDING' | 'FAILED' | 'READY' | 'IDLE';
  progress?: number;
  job?: ProcessingJob;
  errorMessage?: string;
}

export const PipelineStepper: React.FC<PipelineStepperProps> = ({
  meeting,
  meetingId,
  meetingStatus,
  jobs = [],
  transcripts = [],
  intelligence,
  onRetryJob,
  onRetry,
  isRetryingJobId,
  isRetrying,
  isDeriving,
}) => {
  const mId = meeting?.id || meetingId || '';
  const mStatus = meeting?.status || meetingStatus || 'INGESTED';
  const mJobs = meeting?.processing_jobs || jobs;
  const retryHandler = onRetryJob || onRetry;
  const isRetryingCurrent = isRetrying || isRetryingJobId === mId;

  // 1. Identify jobs for each stage
  const ingestionJob = mJobs.find((j) => j.job_type === 'INGESTION');
  const transcriptionJob = mJobs.find((j) => j.job_type === 'TRANSCRIPTION');
  const extractionJob = mJobs.find((j) => j.job_type === 'EXTRACTION');
  const indexingJob = mJobs.find((j) => j.job_type === 'EMBEDDING');

  // Check original & derived transcripts across both transcript objects and meeting metadata
  const meetingTranscriptTypes = meeting?.transcript_types || [];
  const allTranscriptTypes = [
    ...transcripts.map((t) => t.transcript_type),
    ...meetingTranscriptTypes,
  ].filter(Boolean);

  const hasOriginalTranscript = allTranscriptTypes.some((t) => {
    const norm = String(t).toUpperCase().trim();
    return norm === 'ORIGINAL' || norm === 'VERBATIM';
  });

  const hasDerivedTranscript = allTranscriptTypes.some((t) => {
    const norm = String(t).toUpperCase().trim();
    return (
      norm === 'CLEANED' ||
      norm === 'REDACTED' ||
      norm === 'CLEANED_REDACTED' ||
      norm.includes('CLEAN') ||
      norm.includes('REDACT')
    );
  });

  // Determine stage 1: Ingestion
  let s1Status: StageInfo['status'] = 'IDLE';
  if (ingestionJob) {
    if (ingestionJob.status === 'COMPLETED') s1Status = 'COMPLETED';
    else if (ingestionJob.status === 'FAILED') s1Status = 'FAILED';
    else if (ingestionJob.status === 'IN_PROGRESS') s1Status = 'IN_PROGRESS';
    else s1Status = 'PENDING';
  } else if (mStatus !== 'SCHEDULED') {
    s1Status = 'COMPLETED';
  }

  // Determine stage 2: ASR
  let s2Status: StageInfo['status'] = 'IDLE';
  if (transcriptionJob) {
    if (transcriptionJob.status === 'COMPLETED' || hasOriginalTranscript) s2Status = 'COMPLETED';
    else if (transcriptionJob.status === 'FAILED') s2Status = 'FAILED';
    else if (transcriptionJob.status === 'IN_PROGRESS') s2Status = 'IN_PROGRESS';
    else s2Status = 'PENDING';
  } else if (hasOriginalTranscript) {
    s2Status = 'COMPLETED';
  } else if (s1Status === 'COMPLETED') {
    s2Status = 'READY';
  }

  // Determine stage 4: Memory Index (Hybrid vector & FTS5)
  let s4Status: StageInfo['status'] = 'IDLE';
  if (indexingJob) {
    if (indexingJob.status === 'COMPLETED') s4Status = 'COMPLETED';
    else if (indexingJob.status === 'FAILED') s4Status = 'FAILED';
    else if (indexingJob.status === 'IN_PROGRESS') s4Status = 'IN_PROGRESS';
    else s4Status = 'PENDING';
  } else if (s2Status === 'COMPLETED') {
    s4Status = 'READY';
  }

  // Determine stage 5: Intelligence Extraction (Summary, decisions, actions)
  let s5Status: StageInfo['status'] = 'IDLE';
  if (intelligence && (intelligence.summary || intelligence.decisions?.length || intelligence.action_items?.length)) {
    s5Status = 'COMPLETED';
  } else if (extractionJob) {
    if (extractionJob.status === 'COMPLETED') s5Status = 'COMPLETED';
    else if (extractionJob.status === 'FAILED') s5Status = 'FAILED';
    else if (extractionJob.status === 'IN_PROGRESS') s5Status = 'IN_PROGRESS';
    else s5Status = 'PENDING';
  } else if (s4Status === 'COMPLETED' || s2Status === 'COMPLETED') {
    s5Status = 'READY';
  }

  // Determine stage 3: Derivation (Clean & PII Redaction)
  let s3Status: StageInfo['status'] = 'IDLE';
  if (isDeriving) {
    s3Status = 'IN_PROGRESS';
  } else if (hasDerivedTranscript || s4Status === 'COMPLETED' || s5Status === 'COMPLETED') {
    s3Status = 'COMPLETED';
  } else if (s2Status === 'COMPLETED') {
    s3Status = 'READY';
  }

  const stages: StageInfo[] = [
    {
      id: 'ingestion',
      label: '1. Ingestion',
      sublabel: 'Media upload / stream',
      icon: <FileAudio size={14} />,
      status: s1Status,
      progress: ingestionJob?.progress,
      job: ingestionJob,
      errorMessage: ingestionJob?.error_message || undefined,
    },
    {
      id: 'asr',
      label: '2. Multilingual ASR',
      sublabel: 'Whisper local ASR',
      icon: <Languages size={14} />,
      status: s2Status,
      progress: transcriptionJob?.progress,
      job: transcriptionJob,
      errorMessage: transcriptionJob?.error_message || undefined,
    },
    {
      id: 'derivation',
      label: '3. Derivation',
      sublabel: 'Clean & PII Redaction',
      icon: <ShieldCheck size={14} />,
      status: s3Status,
    },
    {
      id: 'memory',
      label: '4. Memory Index',
      sublabel: 'Hybrid vector & FTS5',
      icon: <Database size={14} />,
      status: s4Status,
      progress: indexingJob?.progress,
      job: indexingJob,
      errorMessage: indexingJob?.error_message || undefined,
    },
    {
      id: 'extraction',
      label: '5. Intelligence',
      sublabel: 'Summary, decisions, actions',
      icon: <Brain size={14} />,
      status: s5Status,
      progress: extractionJob?.progress,
      job: extractionJob,
      errorMessage: extractionJob?.error_message || undefined,
    },
  ];

  return (
    <div className="pipeline-stepper" role="region" aria-label="Processing Pipeline">
      <div className="pipeline-track">
        {stages.map((stage, idx) => {
          const isFailed = stage.status === 'FAILED';
          const isInProgress = stage.status === 'IN_PROGRESS';
          const isCompleted = stage.status === 'COMPLETED';
          const isLast = idx === stages.length - 1;
          const isConnectorCompleted = isCompleted && !isLast;

          return (
            <div
              key={stage.id}
              className={`pipeline-stage-item ${stage.status.toLowerCase()}`}
            >
              <div className="stage-node-row">
                <div
                  className={`stage-node ${stage.status.toLowerCase()}`}
                  title={`${stage.label}: ${stage.status}`}
                >
                  {stage.icon}
                </div>

                {!isLast && (
                  <div
                    className={`stage-connector ${
                      isConnectorCompleted ? 'completed' : ''
                    }`}
                  />
                )}
              </div>

              <div className="stage-content">
                <div className="stage-label-row">
                  <span className="stage-title">{stage.label}</span>
                  {isCompleted && (
                    <span className="stage-status-indicator completed" title="Stage Completed">
                      <Check size={11} strokeWidth={2.5} />
                    </span>
                  )}
                  {isInProgress && (
                    <span className="stage-status-indicator in_progress" title="Processing in progress">
                      <RefreshCw size={10} className="spin" />
                    </span>
                  )}
                  {isFailed && (
                    <span className="stage-status-indicator failed" title="Stage Failed">
                      <AlertTriangle size={11} />
                    </span>
                  )}
                </div>

                <div className="stage-sublabel" title={stage.sublabel}>
                  {stage.sublabel}
                </div>

                {isInProgress && stage.progress !== undefined && (
                  <div className="stage-progress-track" title={`${Math.round(stage.progress)}%`}>
                    <div
                      className="stage-progress-bar"
                      style={{ width: `${Math.round(stage.progress)}%` }}
                    />
                  </div>
                )}

                {isFailed && (
                  <div className="stage-retry-container">
                    {stage.errorMessage && (
                      <div className="stage-error-message" title={stage.errorMessage}>
                        {stage.errorMessage.length > 35
                          ? `${stage.errorMessage.slice(0, 35)}...`
                          : stage.errorMessage}
                      </div>
                    )}
                    {stage.job && retryHandler && (
                      <button
                        type="button"
                        className="stage-retry-button"
                        disabled={isRetryingCurrent || isRetryingJobId === stage.job.id}
                        onClick={() => retryHandler(stage.job!.id)}
                        title="Safely retry failed job"
                      >
                        <RefreshCw
                          size={10}
                          className={isRetryingCurrent || isRetryingJobId === stage.job.id ? 'spin' : ''}
                        />
                        <span>{isRetryingCurrent || isRetryingJobId === stage.job.id ? 'Retrying...' : 'Retry'}</span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
