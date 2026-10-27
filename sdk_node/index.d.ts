export interface ClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
}

export interface PIICandidate {
  entity_type: string;
  start: number;
  end: number;
  text: string;
  confidence: number;
  source: string;
  detector_name: string;
  is_validated: boolean;
  metadata?: Record<string, any>;
}

export interface RedactionSpan {
  entity_type: string;
  original_start: number;
  original_end: number;
  redacted_start: number;
  redacted_end: number;
  placeholder: string;
  confidence: number;
  source: string;
  is_validated: boolean;
}

export interface RedactResponse {
  redacted_text: string;
  spans: RedactionSpan[];
  detected_count: number;
  redacted_count: number;
  processing_time_ms: number;
  mode: string;
  session_id?: string;
  degraded_mode: boolean;
}

export interface DetectResponse {
  entities: PIICandidate[];
  total_detected: number;
  processing_time_ms: number;
  degraded_mode: boolean;
}

export interface BatchRedactResponse {
  results: RedactResponse[];
  total_processed: number;
  total_processing_time_ms: number;
}

export interface AnalyzeResponse {
  total_candidates: number;
  risk_score: number;
  category_breakdown: Record<string, { count: number; confidence_avg: number; validated_count: number }>;
  candidates: PIICandidate[];
  processing_time_ms: number;
}

export class LocalShieldClient {
  constructor(options?: ClientOptions);
  health(): Promise<{ status: string; model_loaded: boolean; version: string; offline_mode: boolean }>;
  ready(): Promise<{ ready: boolean; model_loaded: boolean; degraded_mode: boolean; backend: string }>;
  getTaxonomy(): Promise<{ version: string; total_categories: number; categories: Record<string, any> }>;
  detect(text: string, options?: { categories?: string[]; minConfidence?: number }): Promise<DetectResponse>;
  redact(
    text: string,
    options?: {
      categories?: string[];
      mode?: "mask" | "pseudonymize" | "type_preserving" | "selective";
      enableReversible?: boolean;
      sessionId?: string;
    }
  ): Promise<RedactResponse>;
  redactBatch(
    texts: string[],
    options?: {
      categories?: string[];
      mode?: "mask" | "pseudonymize" | "type_preserving" | "selective";
    }
  ): Promise<BatchRedactResponse>;
  analyze(text: string): Promise<AnalyzeResponse>;
}
