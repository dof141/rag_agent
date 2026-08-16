from cryptography.fernet import Fernet


class SecretCipher:
    def __init__(self, master_key: str):
        self._fernet = Fernet(master_key.encode("ascii"))

    def encrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")

    def mask(self, value: str | None) -> str | None:
        if value is None:
            return None
        suffix = value[-4:] if len(value) >= 4 else value
        return f"****{suffix}"
