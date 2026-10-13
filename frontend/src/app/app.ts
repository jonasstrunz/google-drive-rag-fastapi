import { Component, inject, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RagApiService } from './services/rag-api.service';
import {
  AskQuestionResponse,
  AuthStatusResponse,
  BatchUploadResponse,
  DriveFileMetadata,
  HealthResponse,
  LLMConfig,
  LLMConfigStatusResponse,
  SearchResponse,
  SyncDriveResult,
} from './models/rag.models';

type TabType = 'chat' | 'drive' | 'upload' | 'settings';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit {
  private api = inject(RagApiService);

  // Active Tab
  activeTab = signal<TabType>('chat');

  // Backend Health
  health = signal<HealthResponse | null>(null);
  backendUrl = signal<string>('http://localhost:8000');

  // Google OAuth & Drive State
  userId = signal<string>('default_user');
  authStatus = signal<AuthStatusResponse | null>(null);
  driveFiles = signal<DriveFileMetadata[]>([]);
  selectedDriveFileIds = signal<Set<string>>(new Set<string>());
  isLoadingDrive = signal<boolean>(false);
  driveSyncResult = signal<SyncDriveResult | null>(null);
  isSyncingDrive = signal<boolean>(false);

  // LLM Config State
  llmConfigStatus = signal<LLMConfigStatusResponse | null>(null);
  llmForm: LLMConfig = {
    llm_provider: 'openrouter',
    openrouter_api_key: '',
    openrouter_chat_model: 'anthropic/claude-3.5-sonnet',
    openai_api_key: '',
    openai_chat_model: 'gpt-4o-mini',
    embedding_provider: 'openai',
    openai_embedding_model: 'text-embedding-3-small',
    openrouter_embedding_model: 'openai/text-embedding-3-small',
  };
  isSavingConfig = signal<boolean>(false);
  configSuccessMessage = signal<string | null>(null);

  // File Upload State
  selectedFiles: File[] = [];
  isUploading = signal<boolean>(false);
  uploadResult = signal<BatchUploadResponse | null>(null);
  uploadChunkSize = signal<number>(500);
  uploadChunkOverlap = signal<number>(50);

  // Chat & Search State
  questionInput = signal<string>('');
  isSearching = signal<boolean>(false);
  isAnswering = signal<boolean>(false);
  activeAnswer = signal<AskQuestionResponse | null>(null);
  activeSearchResults = signal<SearchResponse | null>(null);
  chatHistory = signal<
    Array<{
      type: 'user' | 'assistant';
      content: string;
      citations?: any[];
      model?: string;
      timestamp: Date;
    }>
  >([]);

  // Toast / Notification
  toastMessage = signal<{ text: string; type: 'success' | 'error' | 'info' } | null>(null);

  ngOnInit(): void {
    this.refreshAll();
  }

  showToast(text: string, type: 'success' | 'error' | 'info' = 'info') {
    this.toastMessage.set({ text, type });
    setTimeout(() => {
      this.toastMessage.set(null);
    }, 4500);
  }

  refreshAll() {
    this.checkHealth();
    this.fetchLlmConfig();
    this.fetchAuthStatus();
  }

  checkHealth() {
    this.api.getHealth().subscribe({
      next: (h) => this.health.set(h),
      error: () => this.health.set(null),
    });
  }

  // --- LLM Settings Management ---
  fetchLlmConfig() {
    this.api.getLlmConfig().subscribe({
      next: (cfg) => {
        this.llmConfigStatus.set(cfg);
        this.llmForm.llm_provider = cfg.llm_provider as any;
        this.llmForm.embedding_provider = cfg.embedding_provider as any;
        this.llmForm.openai_chat_model = cfg.openai_chat_model;
        this.llmForm.openai_embedding_model = cfg.openai_embedding_model;
        this.llmForm.openrouter_chat_model = cfg.openrouter_chat_model;
        this.llmForm.openrouter_embedding_model = cfg.openrouter_embedding_model;
      },
      error: (err) => console.error('Failed fetching LLM config', err),
    });
  }

  saveLlmConfig() {
    this.isSavingConfig.set(true);
    this.configSuccessMessage.set(null);

    this.api.updateLlmConfig(this.llmForm).subscribe({
      next: (res) => {
        this.llmConfigStatus.set(res);
        this.isSavingConfig.set(false);
        this.configSuccessMessage.set('Configuration successfully saved and activated in runtime.');
        this.showToast('LLM and Embedding settings updated!', 'success');
      },
      error: (err) => {
        this.isSavingConfig.set(false);
        this.showToast('Failed to update LLM configuration.', 'error');
      },
    });
  }

  // --- Google Drive & OAuth Management ---
  fetchAuthStatus() {
    this.api.getAuthStatus(this.userId()).subscribe({
      next: (status) => {
        this.authStatus.set(status);
        if (status.authenticated) {
          this.loadDriveFiles();
        }
      },
      error: (err) => console.error('Failed checking auth status', err),
    });
  }

  connectGoogleDrive() {
    this.api.getAuthLoginUrl(this.userId()).subscribe({
      next: (res) => {
        if (res.authorization_url) {
          window.open(res.authorization_url, '_blank', 'width=600,height=700');
          this.showToast('Google OAuth window opened. Complete login and return here.', 'info');
        }
      },
      error: (err) => {
        this.showToast('Could not generate Google Auth URL. Check client credentials.', 'error');
      },
    });
  }

  disconnectGoogleDrive() {
    this.api.revokeAuth(this.userId()).subscribe({
      next: () => {
        this.authStatus.set(null);
        this.driveFiles.set([]);
        this.showToast('Google Drive disconnected.', 'info');
        this.fetchAuthStatus();
      },
      error: () => this.showToast('Failed to disconnect Google Drive.', 'error'),
    });
  }

  loadDriveFiles() {
    this.isLoadingDrive.set(true);
    this.api.listDriveFiles(this.userId()).subscribe({
      next: (data) => {
        this.driveFiles.set(data.files);
        this.isLoadingDrive.set(false);
      },
      error: (err) => {
        this.isLoadingDrive.set(false);
        console.error('Error fetching drive files', err);
      },
    });
  }

  toggleDriveFileSelection(fileId: string) {
    const current = new Set(this.selectedDriveFileIds());
    if (current.has(fileId)) {
      current.delete(fileId);
    } else {
      current.add(fileId);
    }
    this.selectedDriveFileIds.set(current);
  }

  selectAllDriveFiles() {
    const allIds = new Set(this.driveFiles().map((f) => f.id));
    this.selectedDriveFileIds.set(allIds);
  }

  deselectAllDriveFiles() {
    this.selectedDriveFileIds.set(new Set());
  }

  syncDriveFiles() {
    this.isSyncingDrive.set(true);
    this.driveSyncResult.set(null);

    const selectedList = Array.from(this.selectedDriveFileIds());
    const payload = {
      user_id: this.userId(),
      file_ids: selectedList.length > 0 ? selectedList : undefined,
    };

    this.api.syncDrive(payload).subscribe({
      next: (res) => {
        this.driveSyncResult.set(res);
        this.isSyncingDrive.set(false);
        this.showToast(`Drive Synced! Processed ${res.files_processed} files (${res.chunks_created} chunks).`, 'success');
      },
      error: (err) => {
        this.isSyncingDrive.set(false);
        this.showToast('Failed syncing files from Google Drive.', 'error');
      },
    });
  }

  // --- Document & ZIP Upload ---
  onFilesSelected(event: any) {
    const files = event.target.files;
    if (files && files.length > 0) {
      this.selectedFiles = Array.from(files);
    }
  }

  onFileDrop(event: DragEvent) {
    event.preventDefault();
    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      this.selectedFiles = Array.from(event.dataTransfer.files);
    }
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
  }

  uploadSelectedFiles() {
    if (!this.selectedFiles || this.selectedFiles.length === 0) return;

    this.isUploading.set(true);
    this.uploadResult.set(null);

    this.api.uploadFiles(this.selectedFiles, this.uploadChunkSize(), this.uploadChunkOverlap()).subscribe({
      next: (res) => {
        this.uploadResult.set(res);
        this.isUploading.set(false);
        this.selectedFiles = [];
        this.showToast(`Upload complete: ${res.total_files} files indexed into ${res.total_chunks} chunks.`, 'success');
      },
      error: (err) => {
        this.isUploading.set(false);
        this.showToast('Upload and indexing failed.', 'error');
      },
    });
  }

  // --- RAG Chat & Search ---
  askQuestion() {
    const q = this.questionInput().trim();
    if (!q) return;

    // Add user message to history
    this.chatHistory.update((hist) => [
      ...hist,
      { type: 'user', content: q, timestamp: new Date() },
    ]);

    this.isAnswering.set(true);
    this.questionInput.set('');

    this.api.askQuestion({ question: q, top_k: 5 }).subscribe({
      next: (res) => {
        this.activeAnswer.set(res);
        this.chatHistory.update((hist) => [
          ...hist,
          {
            type: 'assistant',
            content: res.answer,
            citations: res.citations,
            model: res.model_used,
            timestamp: new Date(),
          },
        ]);
        this.isAnswering.set(false);
      },
      error: (err) => {
        this.isAnswering.set(false);
        this.chatHistory.update((hist) => [
          ...hist,
          {
            type: 'assistant',
            content: 'Sorry, an error occurred while processing your question.',
            timestamp: new Date(),
          },
        ]);
        this.showToast('Failed to retrieve answer from RAG pipeline.', 'error');
      },
    });
  }

  quickSearch() {
    const q = this.questionInput().trim();
    if (!q) return;

    this.isSearching.set(true);
    this.api.searchVectors({ query: q, top_k: 6 }).subscribe({
      next: (res) => {
        this.activeSearchResults.set(res);
        this.isSearching.set(false);
      },
      error: () => {
        this.isSearching.set(false);
        this.showToast('Search query failed.', 'error');
      },
    });
  }

  formatBytes(bytes?: number): string {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}
