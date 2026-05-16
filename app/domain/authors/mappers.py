"""Author mapping helpers between provider/internal/API contracts."""

from __future__ import annotations

from app.domain.authors.types import AuthorDTO


def author_dto_from_mapping(data: dict) -> AuthorDTO:
    """Map an author dictionary to the internal author transfer type."""

    return AuthorDTO(**data)
