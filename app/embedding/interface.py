from abc import ABC, abstractmethod
from typing import TypedDict


class EmbeddingResult(TypedDict):
    dense: list[list[float]]
    sparse: list[dict[int, float]]


class EmbeddingError(RuntimeError):
    """Base error raised by embedding adapters."""


class EmbeddingConfigurationError(EmbeddingError):
    """The selected adapter is not configured correctly."""


class EmbeddingRequestError(EmbeddingError):
    """The remote embedding request failed."""


class EmbeddingResponseError(EmbeddingError):
    """The embedding provider returned an invalid response."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Generate dense and sparse vectors in input order."""


def validate_texts(texts: list[str]) -> None:
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts 必须是包含文本的非空列表")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("texts 中的每一项都必须是字符串")
