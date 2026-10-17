export interface HealthResponse {
  status: string;
  version: string;
  qdrant_connected: boolean;
  qdrant_collection_exists: boolean;
}

export interface AuthStatusResponse {
  authenticated: boolean;
  user_id?: string;
  email?: string;
  scopes: string[];
  message: string;
}

export interface AuthURLResponse {
  authorization_url: string;
  state: string;
}

export interface DriveFileMetadata {
  id: string;
  name: string;
  mime_type: string;
  size?: number;
  modified_time?: string;
  web_view_link?: string;
  icon_link?: string;
}

export interface DriveFileListResponse {
  files: DriveFileMetadata[];
  total_count: number;
}

export interface SyncDriveRequest {
  user_id?: string;
  folder_id?: string;
  file_ids?: string[];
  chunk_size?: number;
  chunk_overlap?: number;
  force_resync?: boolean;
}

export interface SyncDriveResult {
  user_id: string;
  files_processed: number;
  chunks_created: number;
  skipped_files: number;
  errors: string[];
}

export interface LLMConfig {
  llm_provider: 'openrouter' | 'openai' | 'fallback';
  openai_api_key?: string;
  openai_chat_model: string;
  openrouter_api_key?: string;
  openrouter_chat_model: string;
  embedding_provider: 'openai' | 'openrouter' | 'mock';
  openai_embedding_model: string;
  openrouter_embedding_model: string;
}

export interface LLMConfigStatusResponse {
  llm_provider: string;
  embedding_provider: string;
  openai_chat_model: string;
  openai_embedding_model: string;
  openrouter_chat_model: string;
  openrouter_embedding_model: string;
  has_openai_key: boolean;
  has_openrouter_key: boolean;
  status: string;
}

export interface FileUploadResult {
  filename: string;
  file_id: string;
  size: number;
  chunks_created: number;
  status: string;
  error?: string;
}

export interface BatchUploadResponse {
  total_files: number;
  total_chunks: number;
  results: FileUploadResult[];
}

export interface SearchQuery {
  query: string;
  top_k?: number;
  file_ids?: string[];
  score_threshold?: number;
}

export interface ChunkMetadata {
  file_id: string;
  file_name: string;
  mime_type: string;
  chunk_index: number;
  total_chunks: number;
  char_count: number;
  web_view_link?: string;
  extra?: Record<string, any>;
}

export interface SearchResultItem {
  id: string;
  text: string;
  score: number;
  metadata: ChunkMetadata;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  total_results: number;
}

export interface SourceCitation {
  file_id: string;
  file_name: string;
  chunk_index: number;
  score: number;
  text_snippet: string;
  web_view_link?: string;
}

export interface AskQuestionRequest {
  question: string;
  top_k?: number;
  file_ids?: string[];
  score_threshold?: number;
}

export interface AskQuestionResponse {
  question: string;
  answer: string;
  citations: SourceCitation[];
  confidence_score: number;
  model_used: string;
}
