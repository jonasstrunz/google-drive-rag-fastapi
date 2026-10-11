from unittest.mock import MagicMock, patch

from src.app.services.google_auth import GoogleAuthService


def test_auth_service_storage(tmp_path, mock_token_info):
    storage_path = tmp_path / "tokens.json"
    service = GoogleAuthService(
        client_id="test_id",
        client_secret="test_secret",
        token_storage_path=str(storage_path),
    )

    # Save token
    service.save_tokens(mock_token_info)
    assert storage_path.exists()

    # Load token by user_id
    loaded = service.load_tokens(mock_token_info.user_id)
    assert loaded is not None
    assert loaded.user_id == mock_token_info.user_id
    assert loaded.access_token == mock_token_info.access_token
    assert loaded.email == mock_token_info.email


def test_auth_service_load_nonexistent(tmp_path):
    storage_path = tmp_path / "nonexistent.json"
    service = GoogleAuthService(token_storage_path=str(storage_path))
    assert service.load_tokens("unknown_user") is None


def test_get_authorization_url(tmp_path):
    service = GoogleAuthService(
        client_id="test_client_id",
        client_secret="test_secret",
        redirect_uri="http://localhost:8000/callback",
        token_storage_path=str(tmp_path / "tokens.json"),
    )

    with patch("src.app.services.google_auth.Flow.from_client_config") as mock_flow_cls:
        mock_flow = MagicMock()
        mock_flow.authorization_url.return_value = ("https://accounts.google.com/o/oauth2/auth?client_id=test", "test_state_123")
        mock_flow_cls.return_value = mock_flow

        url, state = service.get_authorization_url(state="custom_state")
        assert "accounts.google.com" in url
        assert state == "test_state_123"
        mock_flow_cls.assert_called_once()


def test_exchange_code_for_tokens(tmp_path):
    service = GoogleAuthService(
        client_id="test_client_id",
        client_secret="test_secret",
        redirect_uri="http://localhost:8000/callback",
        token_storage_path=str(tmp_path / "tokens.json"),
    )

    mock_credentials = MagicMock()
    mock_credentials.token = "new_access_token"
    mock_credentials.refresh_token = "new_refresh_token"
    mock_credentials.token_uri = "https://oauth2.googleapis.com/token"
    mock_credentials.client_id = "test_client_id"
    mock_credentials.client_secret = "test_secret"
    mock_credentials.scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    mock_credentials.expiry = None

    with patch("src.app.services.google_auth.Flow.from_client_config") as mock_flow_cls, \
         patch("src.app.services.google_auth.build") as mock_build:

        mock_flow = MagicMock()
        mock_flow.credentials = mock_credentials
        mock_flow_cls.return_value = mock_flow

        mock_userinfo_client = MagicMock()
        mock_userinfo_client.userinfo().get().execute.return_value = {
            "id": "google_sub_123",
            "email": "user@gmail.com",
        }
        mock_build.return_value = mock_userinfo_client

        token_info = service.exchange_code_for_tokens(code="auth_code_123", state="state_xyz")

        assert token_info.user_id == "google_sub_123"
        assert token_info.email == "user@gmail.com"
        assert token_info.access_token == "new_access_token"
        assert token_info.refresh_token == "new_refresh_token"

        # Check that token was persisted
        loaded = service.load_tokens("google_sub_123")
        assert loaded is not None
        assert loaded.access_token == "new_access_token"


def test_get_credentials_refresh(tmp_path, mock_token_info):
    service = GoogleAuthService(token_storage_path=str(tmp_path / "tokens.json"))
    service.save_tokens(mock_token_info)

    with patch("src.app.services.google_auth.Credentials") as mock_creds_cls:
        mock_creds = MagicMock()
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_tok"
        mock_creds.token = "refreshed_access_token"
        mock_creds.expiry = None
        mock_creds.valid = True
        mock_creds_cls.return_value = mock_creds

        creds = service.get_credentials(mock_token_info.user_id)
        assert creds is not None
        mock_creds.refresh.assert_called_once()
