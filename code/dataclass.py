from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Document:
    path: str
    text: str

@dataclass(frozen=True)
class ArticlePolicy:
    number: int
    title: str
    priority: str  # "p1"|"p2"|"p3"|"p4"|"skip"
    hil: str  # "hil-yes"|"hil-cond"|"hil-no"
    action: str
    chapter: str | None = None
    section : str | None = None

@dataclass(frozen=True)
class KeywordNode:
    node: int
    keywords: list[str]
    chapter: int | None = None
    section: int | None = None


@dataclass(frozen=True)
class Prompts:
    scope_gate_system: str
    scope_gate_user: str
    article_check_system: str
    article_check_user: str
