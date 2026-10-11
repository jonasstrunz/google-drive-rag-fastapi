from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from src.app.api.deps import get_auth_service
from src.app.core.logging import logger
from src.app.schemas.rag import (
    AuthCallbackRequest,
    AuthStatusResponse,
    AuthURLResponse,
    TokenInfo,
)
from src.app.services.google_auth import GoogleAuthService

router = APIRouter(prefix="/auth", tags=["Google OAuth2"])


@router.get("/login", response_model=AuthURLResponse)
def get_google_auth_url(
    state: Optional[str] = Query(None, description="Optional CSRF state parameter"),
    redirect: bool = Query(False, description="If true, 307 redirects user directly to Google OAuth consent"),
    auth_service: GoogleAuthService = Depends(get_auth_service),
):
    """Initiates Google OAuth2 authorization flow."""
    try:
        url, generated_state = auth_service.get_authorization_url(state=state)
        if redirect:
            return RedirectResponse(url=url)
        return AuthURLResponse(authorization_url=url, state=generated_state)
    except Exception as e:
        logger.error(f"Failed to generate Google auth URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize OAuth flow: {str(e)}",
        )


@router.get("/callback", response_model=TokenInfo)
def google_auth_callback_get(
    code: str = Query(..., description="Authorization code returned by Google"),
    state: Optional[str] = Query(None, description="State token"),
    auth_service: GoogleAuthService = Depends(get_auth_service),
):
    """OAuth2 callback handler (GET request from Google redirect)."""
    try:
        token_info = auth_service.exchange_code_for_tokens(code=code, state=state)
        return token_info
    except Exception as e:
        logger.error(f"OAuth code exchange failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to exchange authorization code: {str(e)}",
        )


@router.post("/callback", response_model=TokenInfo)
def google_auth_callback_post(
    body: AuthCallbackRequest,
    auth_service: GoogleAuthService = Depends(get_auth_service),
):
    """OAuth2 callback handler (POST request for programmatic / SPA clients)."""
    try:
        token_info = auth_service.exchange_code_for_tokens(code=body.code, state=body.state)
        return token_info
    except Exception as e:
        logger.error(f"OAuth code exchange failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to exchange authorization code: {str(e)}",
        )


@router.get("/status", response_model=AuthStatusResponse)
def get_auth_status(
    user_id: str = Query("default_user", description="User identifier"),
    auth_service: GoogleAuthService = Depends(get_auth_service),
):
    """Checks whether the user has valid Google OAuth2 credentials stored."""
    token_info = auth_service.load_tokens(user_id)
    if not token_info:
        return AuthStatusResponse(
            authenticated=False,
            user_id=user_id,
            scopes=[],
            message="No credentials found for user. Please authenticate at /api/v1/auth/login.",
        )

    is_auth = auth_service.is_authenticated(user_id)
    return AuthStatusResponse(
        authenticated=is_auth,
        user_id=token_info.user_id,
        email=token_info.email,
        scopes=token_info.scopes,
        message="User authenticated successfully." if is_auth else "Token expired or invalid.",
    )
