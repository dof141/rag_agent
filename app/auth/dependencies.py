from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.repository import UserRepository
from app.auth.security import InvalidTokenError, JwtTokenService


def build_current_user_dependency(users: UserRepository, tokens: JwtTokenService):
    bearer = HTTPBearer(auto_error=False)

    async def get_current_user(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ):
        unauthorized = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证信息无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise unauthorized
        try:
            user_id = tokens.verify(credentials.credentials)
        except InvalidTokenError:
            raise unauthorized
        user = users.get_by_id(user_id)
        if user is None:
            raise unauthorized
        return user

    return get_current_user
