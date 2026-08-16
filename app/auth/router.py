from fastapi import APIRouter, HTTPException, status

from app.auth.models import LoginRequest, TokenResponse
from app.auth.repository import UserRepository
from app.auth.security import JwtTokenService


def create_auth_router(users: UserRepository, tokens: JwtTokenService) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["认证"])

    @router.post("/login", response_model=TokenResponse)
    async def login(payload: LoginRequest) -> TokenResponse:
        user = users.verify_credentials(payload.username, payload.password)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
            )
        return TokenResponse(
            access_token=tokens.issue(user.id),
            expires_in=tokens.ttl_seconds,
        )

    return router
