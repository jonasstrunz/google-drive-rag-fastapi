import io
from datetime import datetime
from typing import List, Optional

from docx import Document
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

from src.app.schemas.rag import DriveFileMetadata
from src.app.services.google_auth import GoogleAuthService

# Supported Google Workspace export mime types
GOOGLE_DOC_MIMETYPES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
}

# Binary files we parse
SUPPORTED_BINARY_MIMETYPES = [
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
]


class GoogleDriveService:
    def __init__(self, auth_service: Optional[GoogleAuthService] = None, drive_resource: Optional[Resource] = None):
        self.auth_service = auth_service or GoogleAuthService()
        self._drive_resource = drive_resource

    def get_drive_client(self, user_id: str = "default_user") -> Resource:
        """Returns Google Drive API v3 client with valid user credentials."""
        if self._drive_resource:
            return self._drive_resource

        credentials = self.auth_service.get_credentials(user_id)
        if not credentials:
            raise PermissionError(f"User {user_id} is not authenticated with Google Drive.")

        return build("drive", "v3", credentials=credentials)

    def list_files(
        self,
        user_id: str = "default_user",
        folder_id: Optional[str] = None,
        page_size: int = 100,
    ) -> List[DriveFileMetadata]:
        """Lists files in Google Drive, optionally filtered by folder."""
        service = self.get_drive_client(user_id)

        query_parts = ["trashed = false", "mimeType != 'application/vnd.google-apps.folder'"]
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")

        query = " and ".join(query_parts)

        fields = "nextPageToken, files(id, name, mimeType, size, modifiedTime, webViewLink, iconLink)"
        results = (
            service.files()
            .list(q=query, pageSize=page_size, fields=fields)
            .execute()
        )

        items = results.get("files", [])
        files = []
        for item in items:
            mod_time = None
            if item.get("modifiedTime"):
                try:
                    mod_time = datetime.fromisoformat(item["modifiedTime"].replace("Z", "+00:00"))
                except Exception:
                    pass

            files.append(
                DriveFileMetadata(
                    id=item["id"],
                    name=item["name"],
                    mime_type=item["mimeType"],
                    size=int(item["size"]) if item.get("size") else None,
                    modified_time=mod_time,
                    web_view_link=item.get("webViewLink"),
                    icon_link=item.get("iconLink"),
                )
            )
        return files

    def get_file_metadata(self, file_id: str, user_id: str = "default_user") -> DriveFileMetadata:
        """Fetches metadata for a single file."""
        service = self.get_drive_client(user_id)
        item = (
            service.files()
            .get(
                file_id=file_id,
                fields="id, name, mimeType, size, modifiedTime, webViewLink, iconLink",
            )
            .execute()
        )

        mod_time = None
        if item.get("modifiedTime"):
            try:
                mod_time = datetime.fromisoformat(item["modifiedTime"].replace("Z", "+00:00"))
            except Exception:
                pass

        return DriveFileMetadata(
            id=item["id"],
            name=item["name"],
            mime_type=item["mimeType"],
            size=int(item["size"]) if item.get("size") else None,
            modified_time=mod_time,
            web_view_link=item.get("webViewLink"),
            icon_link=item.get("iconLink"),
        )

    def extract_file_content(self, file_id: str, user_id: str = "default_user") -> tuple[str, DriveFileMetadata]:
        """Downloads/exports file from Google Drive and extracts clean text."""
        service = self.get_drive_client(user_id)
        metadata = self.get_file_metadata(file_id, user_id)
        mime_type = metadata.mime_type

        text_content = ""

        # Google Docs / Sheets / Slides export
        if mime_type in GOOGLE_DOC_MIMETYPES:
            export_mime = GOOGLE_DOC_MIMETYPES[mime_type]
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            text_content = fh.getvalue().decode("utf-8", errors="ignore")

        # Binary or plain files (PDF, DOCX, TXT, MD, CSV, JSON)
        elif (
            mime_type in SUPPORTED_BINARY_MIMETYPES
            or mime_type.startswith("text/")
            or metadata.name.endswith((".txt", ".md", ".csv", ".json", ".pdf", ".docx"))
        ):
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            bytes_data = fh.getvalue()

            if mime_type == "application/pdf" or metadata.name.endswith(".pdf"):
                text_content = self._extract_pdf_text(bytes_data)
            elif (
                mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                or metadata.name.endswith(".docx")
            ):
                text_content = self._extract_docx_text(bytes_data)
            else:
                text_content = bytes_data.decode("utf-8", errors="ignore")
        else:
            raise ValueError(
                f"Unsupported file format: {mime_type} ({metadata.name}). "
                "Supported formats: Google Docs/Sheets/Slides, PDF, DOCX, TXT, Markdown, CSV, JSON."
            )

        return text_content.strip(), metadata

    @staticmethod
    def _extract_pdf_text(pdf_bytes: bytes) -> str:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_docx_text(docx_bytes: bytes) -> str:
        docx_file = io.BytesIO(docx_bytes)
        doc = Document(docx_file)
        text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(text_parts)
