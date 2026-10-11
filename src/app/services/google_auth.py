import json
from pathlib import Path
from typing import Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from src.app.core.config import settings
from src.app.core.logging import logger
from src.app.schemas.rag import TokenInfo


class GoogleAuthService:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        token_storage_path: Optional[str] = None,
    ):
        self.client_id = client_id or settings.GOOGLE_CLIENT_ID
        self.client_secret = client_secret or settings.GOOGLE_CLIENT_SECRET
        self.redirect_uri = redirect_uri or settings.GOOGLE_REDIRECT_URI
        self.scopes = scopes or settings.google_scopes_list
        self.token_storage_path = Path(token_storage_path or settings.TOKEN_STORAGE_PATH)
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        self.token_storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_client_config(self) -> dict:
        return {
            "web": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.redirect_uri],
            }
        }

    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """Generates the Google OAuth2 authorization URL and state token."""
        flow = Flow.from_client_config(
            self._get_client_config(),
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
        )
        if state:
            flow.state = state

        authorization_url, generated_state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return authorization_url, generated_state

    def exchange_code_for_tokens(self, code: str, state: Optional[str] = None) -> TokenInfo:
        """Exchanges authorization code for access and refresh tokens."""
        flow = Flow.from_client_config(
            self._get_client_config(),
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
            state=state,
        )
        flow.fetch_token(code=code)
        credentials = flow.credentials

        # Get user email using userinfo api
        email = None
        user_id = "default_user"
        try:
            oauth2_client = build("oauth2", "v2", credentials=credentials)
            user_info = oauth2_client.userinfo().get().execute()
            email = user_info.get("email")
            user_id = user_info.get("id") or email or "default_user"
        except Exception as e:
            logger.warning(f"Could not fetch user profile info: {e}")

        token_info = TokenInfo(
            user_id=user_id,
            email=email,
            access_token=credentials.token,
            refresh_token=credentials.refresh_token,
            token_uri=credentials.token_uri,
            client_id=credentials.client_id or self.client_id,
            client_secret=credentials.client_secret or self.client_secret,
            scopes=credentials.scopes or self.scopes,
            expiry=credentials.expiry,
        )
        self.save_tokens(token_info)
        return token_info

    def save_tokens(self, token_info: TokenInfo) -> None:
        """Saves tokens for a user to JSON file storage."""
        data: Dict[str, dict] = {}
        if self.token_storage_path.exists():
            try:
                with open(self.token_storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        token_dict = token_info.model_dump(mode="json")
        data[token_info.user_id] = token_dict
        # Also store under default_user if only one user
        if len(data) == 1:
            data["default_user"] = token_dict

        with open(self.token_storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved OAuth credentials for user {token_info.user_id}")

    def load_tokens(self, user_id: str = "default_user") -> Optional[TokenInfo]:
        """Loads tokens for a user from storage."""
        if not self.token_storage_path.exists():
            return None

        try:
            with open(self.token_storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            user_data = data.get(user_id) or data.get("default_user")
            if not user_data:
                # Return first user found if any
                if data:
                    user_data = next(iter(data.values()))
                else:
                    return None
            return TokenInfo(**user_data)
        except Exception as e:
            logger.error(f"Error loading tokens for user {user_id}: {e}")
            return None

    def get_credentials(self, user_id: str = "default_user") -> Optional[Credentials]:
        """Returns valid Google OAuth2 Credentials, refreshing expired tokens if necessary."""
        token_info = self.load_tokens(user_id)
        if not token_info:
            return None

        credentials = Credentials(
            token=token_info.access_token,
            refresh_token=token_info.refresh_token,
            token_uri=token_info.token_uri,
            client_id=token_info.client_id or self.client_id,
            client_secret=token_info.client_secret or self.client_secret,
            scopes=token_info.scopes or self.scopes,
        )

        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                token_info.access_token = credentials.token
                token_info.expiry = credentials.expiry
                self.save_tokens(token_info)
                logger.info(f"Refreshed token for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to refresh token for {user_id}: {e}")
                return None

        return credentials

    def is_authenticated(self, user_id: str = "default_user") -> bool:
        creds = self.get_credentials(user_id)
        return creds is not None and creds.valid
