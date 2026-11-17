export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  database: string;
  timestamp: string;
}

export interface HardwareSpec {
  category: string;
  label: string;
  value: string;
  status: 'verified' | 'ready' | 'pending';
  detail: string;
}

export interface PhaseItem {
  id: number;
  title: string;
  status: 'completed' | 'in-progress' | 'planned';
  description: string;
  technologies: string[];
}

export interface MediaAsset {
  id: string;
  meeting_id: string;
  file_name: string;
  storage_path: string;
  media_type: string;
  mime_type: string;
  file_size_bytes: number | null;
  duration_seconds: number | null;
  checksum_sha256: string | null;
  is_raw_deleted: boolean;
  created_at: string;
}

export interface ProcessingJob {
  id: string;
  meeting_id: string;
  job_type: string;
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED' | 'CANCELLED' | string;
  progress: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface MeetingItem {
  id: string;
  organization_id: string;
  title: string;
  description: string | null;
  input_source_type: string;
  source_url_or_reference: string | null;
  status: string;
  created_at: string;
  media_assets: MediaAsset[];
  processing_jobs: ProcessingJob[];
}

export interface IngestResponse {
  meeting_id: string;
  job_id: string;
  status: string;
  message: string;
  media_asset_id?: string;
}

export interface TranscriptSegmentItem {
  id: string;
  sequence_order: number;
  start_time: number;
  end_time: number;
  text: string;
  language_code: string;
  confidence: number | null;
  speaker_label: string | null;
  is_redacted?: boolean;
  source_segment_id?: string | null;
  redaction_metadata?: string[] | null;
}

export interface TranscriptItem {
  id: string;
  meeting_id: string;
  transcript_type: string;
  version: number;
  language_detected: string | null;
  is_primary: boolean;
  parent_transcript_id?: string | null;
  created_at: string;
  segments: TranscriptSegmentItem[];
}

export interface HardwareDiagnostics {
  gpu_info: {
    torch_cuda_available: boolean;
    device_count: number;
    device_name: string | null;
    compute_capability: [number, number] | null;
    total_vram_mb: number;
    free_vram_mb: number;
    allocated_vram_mb: number;
    cudnn_available: boolean;
    cudnn_version: string | null;
  };
  ctranslate2_cuda_support: boolean;
  ctranslate2_message: string;
  ctranslate2_supported_compute_types: string[];
  active_engine: {
    engine_name: string;
    model_name: string;
    status: string;
    device: string;
    compute_type: string;
    is_gpu_available: boolean;
    supported_languages: string[];
  };
}

export interface UserResponse {
  id: string;
  email: string;
  full_name: string;
  is_super_admin: boolean;
  is_active: boolean;
}

export interface OrgMembershipResponse {
  organization_id: string;
  organization_name: string;
  organization_slug: string;
  role: string;
}

export interface AuthMeResponse {
  user: UserResponse;
  organizations: OrgMembershipResponse[];
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  full_name: string;
  password: string;
  organization_name: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

// --- Phase 7: Structured Meeting Intelligence Schemas ---

export interface TopicItem {
  id: string;
  meeting_id: string;
  title: string;
  description: string | null;
  start_time: number | null;
  end_time: number | null;
  created_at: string;
  updated_at: string;
}

export interface DecisionItem {
  id: string;
  meeting_id: string;
  description: string;
  rationale_or_context: string | null;
  decided_by: string | null;
  evidence_segment_id: string | null;
  timestamp_seconds: number | null;
  status?: 'ACTIVE' | 'SUPERSEDED' | 'REVOKED' | string;
  supersedes_decision_id?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ActionItem {
  id: string;
  meeting_id: string;
  description: string;
  owner_name: string | null;
  owner_user_id: string | null;
  deadline: string | null;
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED';
  evidence_segment_id: string | null;
  timestamp_seconds: number | null;
  created_at: string;
  updated_at: string;
}

export interface RiskItem {
  id: string;
  meeting_id: string;
  description: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | string;
  mitigation_suggestion: string | null;
  evidence_segment_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface UnresolvedQuestionItem {
  id: string;
  meeting_id: string;
  question_text: string;
  raised_by: string | null;
  status: 'OPEN' | 'RESOLVED' | 'DEFERRED' | string;
  evidence_segment_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface MeetingSummaryItem {
  id: string;
  meeting_id: string;
  transcript_id: string | null;
  executive_summary: string;
  detailed_summary: string | null;
  key_takeaways: string | null;
  version: number;
  provider_used?: string | null;
  is_heuristic?: boolean | null;
  fallback_warning?: string | null;
  created_at: string;
  updated_at: string;
}

export interface MeetingIntelligence {
  meeting_id: string;
  summary: MeetingSummaryItem | null;
  topics: TopicItem[];
  decisions: DecisionItem[];
  action_items: ActionItem[];
  risks: RiskItem[];
  unresolved_questions: UnresolvedQuestionItem[];
  has_intelligence: boolean;
  provider_used?: string | null;
  is_heuristic?: boolean | null;
  fallback_warning?: string | null;
}

export interface ExtractIntelligenceRequest {
  source_transcript_id?: string;
  output_language?: string;
  provider_override?: 'deterministic' | 'ollama' | string;
}

export interface ExtractIntelligenceJobResponse {
  job_id: string;
  meeting_id: string;
  job_type: string;
  status: string;
  message: string;
}

export interface ActionItemStatusUpdate {
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED';
}

// --- Phase 8: Organizational Memory & RAG Schemas ---

export type AbstentionStatus =
  | 'NO_EVIDENCE'
  | 'INSUFFICIENT_EVIDENCE'
  | 'CONFLICTING_EVIDENCE'
  | 'MODEL_UNAVAILABLE'
  | 'RETRIEVAL_UNAVAILABLE'
  | 'ANSWER_SUPPORTED';

export interface MemoryCitation {
  citation_id: string;
  chunk_id: string;
  meeting_id: string;
  meeting_title: string;
  meeting_date: string | null;
  transcript_id: string;
  segment_ids: string[];
  start_time: number;
  end_time: number;
  quoted_text: string;
  language_codes: string[];
}

export interface MemoryDiagnostics {
  query: string;
  total_candidates_retrieved: number;
  evidence_items_count: number;
  meetings_represented: string[];
  provider_used: string;
  is_heuristic: boolean;
  fallback_warning: string | null;
  structured_decisions_count?: number;
  structured_commitments_count?: number;
  structured_questions_count?: number;
  conflicts_detected_count?: number;
}

export interface MemoryQueryRequest {
  question: string;
  meeting_id?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  max_evidence_chunks?: number;
  output_language?: string;
  organization_id?: string | null;
}

export interface MemoryQueryResponse {
  question: string;
  answer: string;
  citations: MemoryCitation[];
  abstention_status: AbstentionStatus | null;
  is_abstained: boolean;
  diagnostics: MemoryDiagnostics;
  structured_decisions?: CrossMeetingDecisionItem[];
  structured_commitments?: CrossMeetingCommitmentItem[];
  structured_questions?: CrossMeetingQuestionItem[];
  detected_conflicts?: DecisionConflictItem[];
  transcript_conflicts?: TranscriptConflictItem[];
}

export interface MemorySearchRequest {
  query: string;
  meeting_id?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  limit?: number;
  organization_id?: string | null;
}

export interface MemorySearchResultItem {
  chunk_id: string;
  meeting_id: string;
  meeting_title: string;
  meeting_date: string | null;
  transcript_id: string;
  segment_ids: string[];
  start_time: number;
  end_time: number;
  text: string;
  language_codes: string[];
  dominant_language: string;
  rrf_score: number;
  keyword_rank: number | null;
  semantic_rank: number | null;
}

export interface MemorySearchResponse {
  query: string;
  total_results: number;
  results: MemorySearchResultItem[];
}

// --- Phase 8.8: Cross-Meeting Intelligence & Conflict Resolution Schemas ---

export interface CrossMeetingDecisionItem extends DecisionItem {
  status: 'ACTIVE' | 'SUPERSEDED' | 'REVOKED' | string;
  supersedes_decision_id: string | null;
  meeting_title: string | null;
  meeting_date: string | null;
  superseded_by_decision_id: string | null;
  lineage_depth: number;
  lineage: DecisionItem[];
}

export interface ActionItemReviewItem {
  id: string;
  action_item_id: string;
  meeting_id: string;
  meeting_title: string | null;
  evidence_segment_id: string | null;
  evidence_quote: string | null;
  reviewer_user_id: string | null;
  status_at_review: string | null;
  review_notes: string | null;
  reviewed_at: string;
  created_at: string;
}

export interface CrossMeetingCommitmentItem {
  id: string;
  meeting_id: string;
  description: string;
  owner_name: string | null;
  owner_user_id: string | null;
  deadline: string | null;
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED' | string;
  evidence_segment_id: string | null;
  timestamp_seconds: number | null;
  last_reviewed_meeting_id: string | null;
  last_evidence_segment_id: string | null;
  last_reviewed_at: string | null;
  created_at: string;
  updated_at: string;
  origin_meeting_title: string | null;
  origin_meeting_date: string | null;
  last_reviewed_meeting_title: string | null;
  is_overdue: boolean;
  review_count: number;
  reviews: ActionItemReviewItem[];
}

export interface CrossMeetingQuestionItem {
  id: string;
  meeting_id: string;
  question_text: string;
  raised_by: string | null;
  status: 'OPEN' | 'RESOLVED' | 'DEFERRED' | string;
  evidence_segment_id: string | null;
  resolved_at: string | null;
  resolving_meeting_id: string | null;
  resolving_decision_id: string | null;
  resolution_notes: string | null;
  created_at: string;
  updated_at: string;
  origin_meeting_title: string | null;
  origin_meeting_date: string | null;
  resolving_meeting_title: string | null;
  resolving_decision_description: string | null;
}

export interface DecisionConflictItem {
  decision_a_id: string;
  decision_a_text: string;
  decision_a_meeting_title: string;
  decision_a_date: string | null;
  decision_b_id: string;
  decision_b_text: string;
  decision_b_meeting_title: string;
  decision_b_date: string | null;
  subject: string;
  reason: string;
  is_heuristic: boolean;
  suggested_action: string;
}

export interface TranscriptConflictItem {
  chunk_a_id: string;
  chunk_a_text: string;
  chunk_a_meeting_title: string;
  chunk_a_date: string | null;
  chunk_b_id: string;
  chunk_b_text: string;
  chunk_b_meeting_title: string;
  chunk_b_date: string | null;
  subject: string;
  reason: string;
  is_heuristic: boolean;
  suggested_action: string;
}

// Request Payload Types
export interface DecisionSupersedePayload {
  meeting_id: string;
  description: string;
  rationale_or_context?: string | null;
  decided_by?: string | null;
  evidence_segment_id?: string | null;
  timestamp_seconds?: number | null;
}

export interface DecisionLinkSupersededPayload {
  superseded_decision_id: string;
}

export interface DecisionRevokePayload {
  reason: string;
}

export interface CommitmentReviewPayload {
  reviewing_meeting_id: string;
  evidence_segment_id?: string | null;
  status?: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED' | string;
  review_notes?: string | null;
}

export interface QuestionResolutionPayload {
  resolving_meeting_id: string;
  resolving_decision_id?: string | null;
  resolution_notes?: string | null;
  status?: 'OPEN' | 'RESOLVED' | 'DEFERRED' | string;
}

export interface DecisionFilterParams {
  status?: string;
  meeting_id?: string;
  start_date?: string;
  end_date?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export interface CommitmentFilterParams {
  status?: string;
  owner_name?: string;
  owner_user_id?: string;
  is_overdue?: boolean;
  due_before?: string;
  due_after?: string;
  meeting_id?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export interface QuestionFilterParams {
  status?: string;
  meeting_id?: string;
  start_date?: string;
  end_date?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export interface ConflictFilterParams {
  subject?: string;
  meeting_id?: string;
}

export interface OrganizationalInsightsStats {
  activeDecisions: number;
  supersededRevokedDecisions: number;
  openCommitments: number;
  overdueCommitments: number;
  openQuestions: number;
  potentialConflicts: number;
}


