from dataclasses import dataclass


@dataclass(frozen=True)
class SearchQuery:
    text: str
    dense: tuple[float, ...]
    sparse: dict[int, float] | None = None


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    content: str
    item_name: str
    file_title: str
    parent_title: str
    source: str = "knowledge_base"
