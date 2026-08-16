from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class User:
    id: str
    username: str
    role: str


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
