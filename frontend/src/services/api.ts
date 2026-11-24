import type {
  HealthResponse,
  MeetingItem,
  ProcessingJob,
  IngestResponse,
  MediaAsset,
  TranscriptItem,
  HardwareDiagnostics,
  LoginRequest,
  RegisterRequest,
  TokenResponse,
  AuthMeResponse,
  OrgMembershipResponse,
  MeetingIntelligence,
  ExtractIntelligenceRequest,
  ExtractIntelligenceJobResponse,
  ActionItem,
  MemoryQueryRequest,
  MemoryQueryResponse,
  MemorySearchRequest,
  MemorySearchResponse,
  CrossMeetingDecisionItem,
  ActionItemReviewItem,
  CrossMeetingCommitmentItem,
  CrossMeetingQuestionItem,
  DecisionConflictItem,
  DecisionSupersedePayload,
  DecisionLinkSupersededPayload,
  DecisionRevokePayload,
  CommitmentReviewPayload,
  QuestionResolutionPayload,
  DecisionFilterParams,
  CommitmentFilterParams,
  QuestionFilterParams,
  ConflictFilterParams,
  ProjectItem,
  CreateProjectPayload,
  MeetingParticipantItem,
  PaginatedMeetingsResponse,
  MeetingFilterParams,
  DomainLookupResponse,
  VerifyDomainRequest,
  OrgMemberItem,
} from '../types';

const API_BASE_URL = import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000';

// --- Centralized Session Storage Keys & Configuration ---
const TOKEN_STORAGE_KEY = 'meet_intel_token';
const ORG_ID_STORAGE_KEY = 'meet_intel_org_id';

export function getAuthToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setAuthToken(token: string | null): void {
  try {
    if (token) {
      sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      sessionStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  } catch {
    // Storage access restricted or quota exceeded
  }
}

export function getSelectedOrgId(): string | null {
  try {
    return sessionStorage.getItem(ORG_ID_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setSelectedOrgId(orgId: string | null): void {
  try {
    if (orgId) {
      sessionStorage.setItem(ORG_ID_STORAGE_KEY, orgId);
    } else {
      sessionStorage.removeItem(ORG_ID_STORAGE_KEY);
    }
  } catch {
    // Storage access restricted or quota exceeded
  }
}

export function clearAuthSession(): void {
  try {
    sessionStorage.removeItem(TOKEN_STORAGE_KEY);
    sessionStorage.removeItem(ORG_ID_STORAGE_KEY);
  } catch {
    // Storage access restricted
  }
}

export function getAuthHeaders(includeOrg: boolean = true): Record<string, string> {
  const headers: Record<string, string> = {
    'Accept': 'application/json',
  };
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (includeOrg) {
    const orgId = getSelectedOrgId();
    if (orgId) {
      headers['X-Organization-Id'] = orgId;
    }
  }
  return headers;
}

// --- Centralized Unauthorized (401) Handler ---
export type UnauthorizedHandler = () => void;
let onUnauthorizedCallback: UnauthorizedHandler | null = null;

export function setUnauthorizedHandler(handler: UnauthorizedHandler | null): void {
  onUnauthorizedCallback = handler;
}

function handleUnauthorized(): void {
  clearAuthSession();
  if (onUnauthorizedCallback) {
    try {
      onUnauthorizedCallback();
    } catch (err) {
      console.error('Error executing onUnauthorized callback:', err);
    }
  }
}

// --- Error Extraction Helper ---
async function extractErrorMessage(response: Response, fallbackMessage: string): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') {
      return data.detail;
    }
    if (Array.isArray(data.detail) && data.detail.length > 0) {
      // Pydantic validation error array
      return data.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join('; ');
    }
    if (data.message && typeof data.message === 'string') {
      return data.message;
    }
  } catch {
    // Non-JSON response
  }
  return `${fallbackMessage} (${response.status})`;
}

/**
 * Check whether an error was caused by an intentional abort/cancellation.
 */
export function isAbortError(err: unknown): boolean {
  if (!err) return false;
  if (typeof DOMException !== 'undefined' && err instanceof DOMException && err.name === 'AbortError') {
    return true;
  }
  if (err instanceof Error && (err.name === 'AbortError' || err.message.toLowerCase().includes('aborted'))) {
    return true;
  }
  return false;
}

// --- Centralized Authenticated Fetch Wrapper ---
async function authFetch(
  url: string,
  options: RequestInit = {},
  includeOrg: boolean = true
): Promise<Response> {
  const authHeaders = getAuthHeaders(includeOrg);
  const mergedHeaders: Record<string, string> = {
    ...authHeaders,
    ...(options.headers as Record<string, string> || {}),
  };

  const response = await fetch(url, {
    ...options,
    headers: mergedHeaders,
  });

  if (response.status === 401) {
    handleUnauthorized();
    const message = await extractErrorMessage(response, 'Session expired or not authenticated');
    throw new Error(message);
  }

  return response;
}

// --- Public Endpoints (No Auth Required) ---

export interface HealthCheckResult {
  data: HealthResponse | null;
  latencyMs: number;
  error: string | null;
  timestamp: Date;
}

export async function checkBackendHealth(): Promise<HealthCheckResult> {
  const startTime = performance.now();
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    const latencyMs = Math.round(performance.now() - startTime);

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
    }

    const data: HealthResponse = await response.json();
    return {
      data,
      latencyMs,
      error: null,
      timestamp: new Date(),
    };
  } catch (err: unknown) {
    const latencyMs = Math.round(performance.now() - startTime);
    const errorMessage = err instanceof Error ? err.message : 'Unknown network error';
    return {
      data: null,
      latencyMs,
      error: errorMessage,
      timestamp: new Date(),
    };
  }
}

export async function fetchRootInfo(): Promise<Record<string, unknown> | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

// --- Authentication Endpoints ---

export async function loginUser(payload: LoginRequest): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Login failed');
    // Do NOT trigger global logout on login failures
    throw new Error(message);
  }

  const data: TokenResponse = await response.json();
  return data;
}

export async function registerUser(payload: RegisterRequest): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Registration failed');
    throw new Error(message);
  }

  const data: TokenResponse = await response.json();
  return data;
}

export async function logoutUser(signal?: AbortSignal): Promise<void> {
  const token = getAuthToken();
  if (!token) {
    return;
  }

  try {
    await fetch(`${API_BASE_URL}/api/v1/auth/logout`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      signal,
    });
  } catch (err: unknown) {
    if (isAbortError(err)) {
      throw err;
    }
    // Network errors or offline failures are caught gracefully to guarantee client-side cleanup
  }
}

export async function fetchCurrentUser(): Promise<AuthMeResponse> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/auth/me`, {
    method: 'GET',
  }, false);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch user profile');
    throw new Error(message);
  }

  return await response.json();
}

export async function fetchUserOrganizations(): Promise<OrgMembershipResponse[]> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/auth/organizations`, {
    method: 'GET',
  }, false);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch organizations');
    throw new Error(message);
  }

  return await response.json();
}

export async function lookupDomain(domain: string, signal?: AbortSignal): Promise<DomainLookupResponse> {
  const params = new URLSearchParams({ domain: domain.trim() });
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/organizations/by-domain?${params.toString()}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
    signal,
  });
  if (!response.ok) {
    return { domain, exists: false, is_verified: false };
  }
  return await response.json();
}

export async function verifyDomain(
  data: VerifyDomainRequest,
  signal?: AbortSignal
): Promise<{ status: string; message: string }> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/auth/verify-domain`, {
    method: 'POST',
    body: JSON.stringify(data),
    signal,
  });
  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Domain verification failed');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchOrgMembers(signal?: AbortSignal): Promise<OrgMemberItem[]> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/auth/organization/members`, {
    method: 'GET',
    signal,
  });
  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch organization members');
    throw new Error(message);
  }
  return await response.json();
}

export async function updateMemberRole(
  membershipId: string,
  role: string,
  signal?: AbortSignal
): Promise<OrgMemberItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/auth/organization/members/${membershipId}/role`, {
    method: 'PATCH',
    body: JSON.stringify({ role }),
    signal,
  });
  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to update member role');
    throw new Error(message);
  }
  return await response.json();
}

// --- Protected Media Ingestion Endpoints ---

export async function fetchMeetings(
  params?: MeetingFilterParams | AbortSignal,
  signal?: AbortSignal
): Promise<MeetingItem[] | PaginatedMeetingsResponse> {
  let filterParams: MeetingFilterParams | undefined = undefined;
  let abortSignal: AbortSignal | undefined = signal;

  if (params) {
    if (params instanceof AbortSignal || ('aborted' in (params as object) && !('page' in (params as object) || 'search' in (params as object)))) {
      abortSignal = params as AbortSignal;
    } else {
      filterParams = params as MeetingFilterParams;
    }
  }

  const query = new URLSearchParams();
  if (filterParams) {
    if (filterParams.page !== undefined) query.set('page', filterParams.page.toString());
    if (filterParams.limit !== undefined) query.set('limit', filterParams.limit.toString());
    if (filterParams.skip !== undefined) query.set('skip', filterParams.skip.toString());
    if (filterParams.search) query.set('search', filterParams.search);
    if (filterParams.project_id) query.set('project_id', filterParams.project_id);
    if (filterParams.category) query.set('category', filterParams.category);
    if (filterParams.status) query.set('status', filterParams.status);
    if (filterParams.start_date) query.set('start_date', filterParams.start_date);
    if (filterParams.end_date) query.set('end_date', filterParams.end_date);
    if (filterParams.participant) query.set('participant', filterParams.participant);
    if (filterParams.paginate) query.set('paginate', 'true');
  }
  const queryString = query.toString();
  const url = `${API_BASE_URL}/api/v1/media/meetings${queryString ? `?${queryString}` : ''}`;
  const response = await authFetch(url, {
    method: 'GET',
    signal: abortSignal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to load meetings');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchJobStatus(jobId: string, signal?: AbortSignal): Promise<ProcessingJob> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/media/jobs/${jobId}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to load job status');
    throw new Error(message);
  }
  return await response.json();
}

export async function uploadMediaFile(
  file: File,
  title?: string,
  description?: string,
  onUploadProgress?: (progressPercent: number) => void,
  signal?: AbortSignal,
  extraOptions?: {
    meetingDate?: string;
    category?: string;
    projectId?: string;
    participants?: { display_name: string; email?: string; role?: string }[];
  }
): Promise<IngestResponse> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      return reject(new DOMException('Upload aborted', 'AbortError'));
    }

    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);
    if (title) formData.append('title', title);
    if (description) formData.append('description', description);
    if (extraOptions?.meetingDate) formData.append('meeting_date', extraOptions.meetingDate);
    if (extraOptions?.category) formData.append('category', extraOptions.category);
    if (extraOptions?.projectId) formData.append('project_id', extraOptions.projectId);
    if (extraOptions?.participants && extraOptions.participants.length > 0) {
      formData.append('participants', JSON.stringify(extraOptions.participants));
    }

    let isAborted = false;
    const onAbort = () => {
      isAborted = true;
      try {
        xhr.abort();
      } catch {
        // Ignore
      }
      reject(new DOMException('Upload aborted', 'AbortError'));
    };

    if (signal) {
      signal.addEventListener('abort', onAbort, { once: true });
    }

    xhr.open('POST', `${API_BASE_URL}/api/v1/media/upload`);

    // Attach centralized authentication and tenant headers
    const authHeaders = getAuthHeaders(true);
    for (const [key, val] of Object.entries(authHeaders)) {
      xhr.setRequestHeader(key, val);
    }
    // IMPORTANT: Do NOT set Content-Type header on FormData; browser automatically generates boundary!

    if (xhr.upload && onUploadProgress) {
      xhr.upload.onprogress = (event) => {
        if (isAborted || signal?.aborted) return;
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onUploadProgress(percent);
        }
      };
    }

    xhr.onload = () => {
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
      if (isAborted || signal?.aborted) {
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data: IngestResponse = JSON.parse(xhr.responseText);
          resolve(data);
        } catch {
          reject(new Error('Invalid JSON response from server'));
        }
      } else {
        if (xhr.status === 401) {
          handleUnauthorized();
        }
        try {
          const errData = JSON.parse(xhr.responseText);
          reject(new Error(errData.detail || `Upload failed with status ${xhr.status}`));
        } catch {
          reject(new Error(`Upload failed with status ${xhr.status}`));
        }
      }
    };

    xhr.onerror = () => {
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
      if (isAborted || signal?.aborted) return;
      reject(new Error('Network error during media upload'));
    };

    xhr.onabort = () => {
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
      if (!isAborted) {
        reject(new DOMException('Upload aborted', 'AbortError'));
      }
    };

    xhr.send(formData);
  });
}

export async function ingestYouTube(
  url: string,
  title?: string,
  description?: string,
  confirmedAuthorization: boolean = false,
  signal?: AbortSignal,
  extraOptions?: {
    meetingDate?: string;
    category?: string;
    projectId?: string;
    participants?: { display_name: string; email?: string; role?: string }[];
  }
): Promise<IngestResponse> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/media/youtube`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      url,
      title: title || undefined,
      description: description || undefined,
      meeting_date: extraOptions?.meetingDate || undefined,
      category: extraOptions?.category || undefined,
      project_id: extraOptions?.projectId || undefined,
      participants: extraOptions?.participants || undefined,
      confirmed_authorization: confirmedAuthorization,
    }),
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `YouTube ingestion failed (${response.status})`);
  }
  return data;
}

export async function deleteRawMedia(assetId: string, signal?: AbortSignal): Promise<MediaAsset> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/media/assets/${assetId}`, {
    method: 'DELETE',
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to delete media asset (${response.status})`);
  }
  return data;
}

export async function deleteMeeting(meetingId: string, signal?: AbortSignal): Promise<{ message: string; meeting_id: string }> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/media/meetings/${meetingId}`, {
    method: 'DELETE',
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to delete meeting (${response.status})`);
  }
  return data;
}

export async function retryProcessingJob(jobId: string, signal?: AbortSignal): Promise<ProcessingJob> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/media/jobs/${jobId}/retry`, {
    method: 'POST',
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to retry job (${response.status})`);
  }
  return data;
}

export function getMeetingAudioStreamUrl(meetingId: string): string {
  return `${API_BASE_URL}/api/v1/media/meetings/${meetingId}/audio`;
}

// --- Project API Endpoints ---

export async function fetchProjects(signal?: AbortSignal): Promise<ProjectItem[]> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/projects`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to load projects');
    throw new Error(message);
  }
  return await response.json();
}

export async function createProject(payload: CreateProjectPayload, signal?: AbortSignal): Promise<ProjectItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to create project');
    throw new Error(message);
  }
  return await response.json();
}

export async function deleteProject(projectId: string, signal?: AbortSignal): Promise<void> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/projects/${projectId}`, {
    method: 'DELETE',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to delete project');
    throw new Error(message);
  }
}

// --- Meeting Participants API Endpoints ---

export async function fetchMeetingParticipants(meetingId: string, signal?: AbortSignal): Promise<MeetingParticipantItem[]> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/media/meetings/${meetingId}/participants`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to load meeting participants');
    throw new Error(message);
  }
  return await response.json();
}

export async function addMeetingParticipant(
  meetingId: string,
  payload: { display_name: string; email?: string; role?: string },
  signal?: AbortSignal
): Promise<MeetingParticipantItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/media/meetings/${meetingId}/participants`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to add participant');
    throw new Error(message);
  }
  return await response.json();
}

export async function deleteMeetingParticipant(
  meetingId: string,
  participantId: string,
  signal?: AbortSignal
): Promise<void> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/media/meetings/${meetingId}/participants/${participantId}`, {
    method: 'DELETE',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to remove participant');
    throw new Error(message);
  }
}

// --- Protected Transcription Endpoints ---

export async function startTranscription(
  meetingId: string,
  language?: string,
  transcriptionMode?: string,
  signal?: AbortSignal
): Promise<{ job_id: string; meeting_id: string; job_type: string; status: string; progress: number }> {
  const payload: Record<string, string> = {};
  if (language && language !== 'auto') {
    payload.language = language;
  }
  if (transcriptionMode) {
    payload.transcription_mode = transcriptionMode;
  }

  const response = await authFetch(`${API_BASE_URL}/api/v1/transcription/start/${meetingId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to start transcription (${response.status})`);
  }
  return data;
}

export async function fetchTranscripts(meetingId: string, signal?: AbortSignal): Promise<TranscriptItem[]> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/transcription/${meetingId}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to load transcripts');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchDefaultTranscript(meetingId: string, signal?: AbortSignal): Promise<TranscriptItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/transcription/${meetingId}/default`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'No privacy-safe transcript available');
    throw new Error(message);
  }
  return await response.json();
}

export async function cleanTranscript(
  meetingId: string,
  sourceTranscriptId?: string,
  signal?: AbortSignal
): Promise<TranscriptItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/transcription/${meetingId}/clean`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      source_transcript_id: sourceTranscriptId || undefined,
    }),
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to clean transcript (${response.status})`);
  }
  return data;
}

export async function redactTranscript(
  meetingId: string,
  sourceTranscriptId?: string,
  signal?: AbortSignal
): Promise<TranscriptItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/transcription/${meetingId}/redact`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      source_transcript_id: sourceTranscriptId || undefined,
    }),
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to redact transcript (${response.status})`);
  }
  return data;
}

export async function processTranscript(
  meetingId: string,
  clean: boolean = true,
  redact: boolean = true,
  sourceTranscriptId?: string,
  signal?: AbortSignal
): Promise<TranscriptItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/transcription/${meetingId}/process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      clean,
      redact,
      source_transcript_id: sourceTranscriptId || undefined,
    }),
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to process transcript (${response.status})`);
  }
  return data;
}

export async function fetchTranscriptionJob(jobId: string, signal?: AbortSignal): Promise<ProcessingJob> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/transcription/jobs/${jobId}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch job status');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchHardwareDiagnostics(): Promise<HardwareDiagnostics> {
  // Diagnostics is platform-scoped (requires Super Admin) and does not require an organization header
  const response = await authFetch(`${API_BASE_URL}/api/v1/transcription/hardware/diagnostics`, {
    method: 'GET',
  }, false);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to load hardware diagnostics');
    throw new Error(message);
  }
  return await response.json();
}

// --- Phase 7: Structured Meeting Intelligence Endpoints ---

export async function triggerExtractionJob(
  meetingId: string,
  payload?: ExtractIntelligenceRequest,
  signal?: AbortSignal
): Promise<ExtractIntelligenceJobResponse> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/meetings/${meetingId}/intelligence/extract`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload || {}),
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Extraction request failed (${response.status})`);
  }
  return data;
}

export async function fetchMeetingIntelligence(
  meetingId: string,
  signal?: AbortSignal
): Promise<MeetingIntelligence> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/meetings/${meetingId}/intelligence`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch meeting intelligence');
    throw new Error(message);
  }
  return await response.json();
}

export async function updateActionItemStatus(
  itemId: string,
  newStatus: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED',
  signal?: AbortSignal
): Promise<ActionItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/action-items/${itemId}/status`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ status: newStatus }),
    signal,
  }, true);

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to update action item status (${response.status})`);
  }
  return data;
}

// --- Phase 8: Organizational Memory Endpoints ---

export async function queryOrganizationalMemory(
  payload: MemoryQueryRequest,
  signal?: AbortSignal
): Promise<MemoryQueryResponse> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/memory/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to query organizational memory');
    throw new Error(message);
  }
  return await response.json();
}

export async function searchOrganizationalMemory(
  payload: MemorySearchRequest,
  signal?: AbortSignal
): Promise<MemorySearchResponse> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/memory/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to search organizational memory');
    throw new Error(message);
  }
  return await response.json();
}

// --- Phase 8.8: Cross-Meeting Intelligence & Conflict Detection Endpoints ---

function buildQueryString(params: Record<string, string | number | boolean | null | undefined>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      query.append(key, String(value));
    }
  }
  const qs = query.toString();
  return qs ? `?${qs}` : '';
}

export async function fetchCrossMeetingDecisions(
  params: DecisionFilterParams = {},
  signal?: AbortSignal
): Promise<CrossMeetingDecisionItem[]> {
  const qs = buildQueryString(params as Record<string, string | number | boolean | null | undefined>);
  const response = await authFetch(`${API_BASE_URL}/api/v1/decisions${qs}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch decisions');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchDecisionDetails(
  decisionId: string,
  signal?: AbortSignal
): Promise<CrossMeetingDecisionItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/decisions/${encodeURIComponent(decisionId)}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch decision details');
    throw new Error(message);
  }
  return await response.json();
}

export async function supersedeDecision(
  decisionId: string,
  payload: DecisionSupersedePayload,
  signal?: AbortSignal
): Promise<CrossMeetingDecisionItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/decisions/${encodeURIComponent(decisionId)}/supersede`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to supersede decision');
    throw new Error(message);
  }
  return await response.json();
}

export async function linkSupersededDecision(
  decisionId: string,
  payload: DecisionLinkSupersededPayload,
  signal?: AbortSignal
): Promise<CrossMeetingDecisionItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/decisions/${encodeURIComponent(decisionId)}/link-superseded`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to link superseded decision');
    throw new Error(message);
  }
  return await response.json();
}

export async function revokeDecision(
  decisionId: string,
  payload: DecisionRevokePayload,
  signal?: AbortSignal
): Promise<CrossMeetingDecisionItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/decisions/${encodeURIComponent(decisionId)}/revoke`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to revoke decision');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchCrossMeetingCommitments(
  params: CommitmentFilterParams = {},
  signal?: AbortSignal
): Promise<CrossMeetingCommitmentItem[]> {
  const qs = buildQueryString(params as Record<string, string | number | boolean | null | undefined>);
  const response = await authFetch(`${API_BASE_URL}/api/v1/commitments${qs}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch commitments');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchCommitmentDetails(
  actionItemId: string,
  signal?: AbortSignal
): Promise<CrossMeetingCommitmentItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/commitments/${encodeURIComponent(actionItemId)}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch commitment details');
    throw new Error(message);
  }
  return await response.json();
}

export async function recordCommitmentReview(
  actionItemId: string,
  payload: CommitmentReviewPayload,
  signal?: AbortSignal
): Promise<CrossMeetingCommitmentItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/commitments/${encodeURIComponent(actionItemId)}/reviews`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to record commitment review');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchCommitmentReviews(
  actionItemId: string,
  signal?: AbortSignal
): Promise<ActionItemReviewItem[]> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/commitments/${encodeURIComponent(actionItemId)}/reviews`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch commitment reviews');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchCrossMeetingQuestions(
  params: QuestionFilterParams = {},
  signal?: AbortSignal
): Promise<CrossMeetingQuestionItem[]> {
  const qs = buildQueryString(params as Record<string, string | number | boolean | null | undefined>);
  const response = await authFetch(`${API_BASE_URL}/api/v1/questions${qs}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch questions');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchQuestionDetails(
  questionId: string,
  signal?: AbortSignal
): Promise<CrossMeetingQuestionItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/questions/${encodeURIComponent(questionId)}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to fetch question details');
    throw new Error(message);
  }
  return await response.json();
}

export async function resolveQuestion(
  questionId: string,
  payload: QuestionResolutionPayload,
  signal?: AbortSignal
): Promise<CrossMeetingQuestionItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/questions/${encodeURIComponent(questionId)}/resolve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to resolve question');
    throw new Error(message);
  }
  return await response.json();
}

export async function reopenQuestion(
  questionId: string,
  signal?: AbortSignal
): Promise<CrossMeetingQuestionItem> {
  const response = await authFetch(`${API_BASE_URL}/api/v1/questions/${encodeURIComponent(questionId)}/reopen`, {
    method: 'POST',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to reopen question');
    throw new Error(message);
  }
  return await response.json();
}

export async function fetchDecisionConflicts(
  params: ConflictFilterParams = {},
  signal?: AbortSignal
): Promise<DecisionConflictItem[]> {
  const qs = buildQueryString(params as Record<string, string | number | boolean | null | undefined>);
  const response = await authFetch(`${API_BASE_URL}/api/v1/conflicts${qs}`, {
    method: 'GET',
    signal,
  }, true);

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to detect conflicts');
    throw new Error(message);
  }
  return await response.json();
}

export async function downloadMeetingReportPdf(
  meetingId: string,
  meetingTitle?: string,
  signal?: AbortSignal
): Promise<void> {
  const response = await authFetch(
    `${API_BASE_URL}/api/v1/media/meetings/${encodeURIComponent(meetingId)}/export/pdf`,
    {
      method: 'GET',
      signal,
    },
    true
  );

  if (!response.ok) {
    const message = await extractErrorMessage(response, 'Failed to export meeting report PDF');
    throw new Error(message);
  }

  const blob = await response.blob();
  const cleanTitle = (meetingTitle || 'Meeting')
    .replace(/[^a-zA-Z0-9_\- ]/g, '')
    .trim()
    .replace(/\s+/g, '_')
    .slice(0, 40);
  const defaultFilename = `MEMORA_Report_${cleanTitle || 'Meeting'}.pdf`;

  let filename = defaultFilename;
  const disposition = response.headers.get('content-disposition');
  if (disposition && disposition.includes('filename=')) {
    const match = disposition.match(/filename="?([^"]+)"?/);
    if (match && match[1]) {
      filename = match[1];
    }
  }

  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.URL.revokeObjectURL(url);
}


