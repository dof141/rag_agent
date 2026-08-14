from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RerankItem:
    index: int
    score: float | None


@dataclass(frozen=True)
class RerankOutcome:
    items: list[RerankItem]
    degraded: bool = False
    warning_code: str | None = None
    warning_message: str | None = None


class RerankerError(RuntimeError):
    """Base error raised by reranker adapters."""


class RerankerConfigurationError(RerankerError):
    """The selected reranker adapter is not configured correctly."""


class RerankerRequestError(RerankerError):
    """The remote reranker request failed."""


class RerankerResponseError(RerankerError):
    """The reranker provider returned an invalid response."""


class RerankerProvider(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[RerankItem]:
        ...
