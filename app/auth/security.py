from base64 import urlsafe_b64decode, urlsafe_b64encode
import hashlib
import hmac
import json
import os
import time


class InvalidTokenError(ValueError):
    pass


def _b64encode(value: bytes) -> str:
    return urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return urlsafe_b64decode(value + "=" * (-len(value) % 4))


class PasswordHasher:
    def hash(self, password: str) -> str:
        salt = os.urandom(16)
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, dklen=32
        )
        return f"scrypt$16384$8$1${_b64encode(salt)}${_b64encode(digest)}"

    def verify(self, password: str, encoded: str) -> bool:
        try:
            algorithm, n, r, p, salt, expected = encoded.split("$")
            if algorithm != "scrypt":
                return False
            actual = hashlib.scrypt(
                password.encode("utf-8"),
                salt=_b64decode(salt),
                n=int(n),
                r=int(r),
                p=int(p),
                dklen=32,
            )
            return hmac.compare_digest(actual, _b64decode(expected))
        except (ValueError, TypeError):
            return False


class JwtTokenService:
    def __init__(self, secret: str, ttl_seconds: int):
        if len(secret) < 16 or ttl_seconds <= 0:
            raise ValueError("JWT 配置无效")
        self._secret = secret.encode("utf-8")
        self._ttl_seconds = ttl_seconds

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def issue(self, user_id: str, *, now: int | None = None) -> str:
        issued_at = int(time.time()) if now is None else now
        header = _b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
        )
        payload = _b64encode(
            json.dumps(
                {"sub": user_id, "iat": issued_at, "exp": issued_at + self._ttl_seconds},
                separators=(",", ":"),
            ).encode()
        )
        signing_input = f"{header}.{payload}"
        signature = _b64encode(
            hmac.new(self._secret, signing_input.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{signing_input}.{signature}"

    def verify(self, token: str, *, now: int | None = None) -> str:
        try:
            header_part, payload_part, signature_part = token.split(".")
            header = json.loads(_b64decode(header_part))
            payload = json.loads(_b64decode(payload_part))
            signing_input = f"{header_part}.{payload_part}"
            expected = hmac.new(
                self._secret, signing_input.encode("ascii"), hashlib.sha256
            ).digest()
            if header != {"alg": "HS256", "typ": "JWT"}:
                raise InvalidTokenError("token header 无效")
            if not hmac.compare_digest(expected, _b64decode(signature_part)):
                raise InvalidTokenError("token 签名无效")
            current = int(time.time()) if now is None else now
            if not isinstance(payload.get("sub"), str) or int(payload["exp"]) <= current:
                raise InvalidTokenError("token 已过期或缺少 subject")
            return payload["sub"]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, InvalidTokenError):
                raise
            raise InvalidTokenError("token 无效") from exc
