import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  AskQuestionRequest,
  AskQuestionResponse,
  AuthStatusResponse,
  AuthURLResponse,
  BatchUploadResponse,
  DriveFileListResponse,
  HealthResponse,
  LLMConfig,
  LLMConfigStatusResponse,
  SearchQuery,
  SearchResponse,
  SyncDriveRequest,
  SyncDriveResult,
} from '../models/rag.models';

@Injectable({
  providedIn: 'root',
})
export class RagApiService {
  private http = inject(HttpClient);
  // Default to relative or standard port 8000
  private baseUrl = 'http://localhost:8000';

  setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/+$/, '');
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  // System & Health
  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  // LLM Configuration
  getLlmConfig(): Observable<LLMConfigStatusResponse> {
    return this.http.get<LLMConfigStatusResponse>(`${this.baseUrl}/api/v1/config/llm`);
  }

  updateLlmConfig(config: LLMConfig): Observable<LLMConfigStatusResponse> {
    return this.http.post<LLMConfigStatusResponse>(`${this.baseUrl}/api/v1/config/llm`, config);
  }

  // File & ZIP Upload
  uploadFiles(files: File[], chunkSize?: number, chunkOverlap?: number): Observable<BatchUploadResponse> {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file, file.name);
    }
    if (chunkSize) {
      formData.append('chunk_size', chunkSize.toString());
    }
    if (chunkOverlap) {
      formData.append('chunk_overlap', chunkOverlap.toString());
    }
    return this.http.post<BatchUploadResponse>(`${this.baseUrl}/api/v1/config/upload/files`, formData);
  }

  // Google OAuth2
  getAuthStatus(userId: string = 'default_user'): Observable<AuthStatusResponse> {
    return this.http.get<AuthStatusResponse>(`${this.baseUrl}/api/v1/auth/status?user_id=${encodeURIComponent(userId)}`);
  }

  getAuthLoginUrl(userId: string = 'default_user'): Observable<AuthURLResponse> {
    return this.http.get<AuthURLResponse>(`${this.baseUrl}/api/v1/auth/login?user_id=${encodeURIComponent(userId)}`);
  }

  revokeAuth(userId: string = 'default_user'): Observable<{ message: string; user_id: string }> {
    return this.http.delete<{ message: string; user_id: string }>(`${this.baseUrl}/api/v1/auth/token?user_id=${encodeURIComponent(userId)}`);
  }

  // Google Drive Files & Sync
  listDriveFiles(userId: string = 'default_user', folderId?: string, query?: string): Observable<DriveFileListResponse> {
    let url = `${this.baseUrl}/api/v1/drive/files?user_id=${encodeURIComponent(userId)}`;
    if (folderId) {
      url += `&folder_id=${encodeURIComponent(folderId)}`;
    }
    if (query) {
      url += `&query=${encodeURIComponent(query)}`;
    }
    return this.http.get<DriveFileListResponse>(url);
  }

  syncDrive(request: SyncDriveRequest): Observable<SyncDriveResult> {
    return this.http.post<SyncDriveResult>(`${this.baseUrl}/api/v1/drive/sync`, request);
  }

  // RAG Search & QA
  searchVectors(query: SearchQuery): Observable<SearchResponse> {
    return this.http.post<SearchResponse>(`${this.baseUrl}/api/v1/rag/search`, query);
  }

  askQuestion(req: AskQuestionRequest): Observable<AskQuestionResponse> {
    return this.http.post<AskQuestionResponse>(`${this.baseUrl}/api/v1/rag/ask`, req);
  }
}
