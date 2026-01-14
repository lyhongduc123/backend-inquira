"""
Database models for the Exegent application.

All models inherit from DatabaseBase which itself inherits from SQLAlchemy's Base.
Import all models here to ensure they're registered with SQLAlchemy metadata.

Trust & Reliability Focus:
- Author-centric models for reputation tracking
- Institution models for organizational trust
- Structured citation tracking for network analysis
"""

from app.models.base import DatabaseBase
from app.models.users import DBUser
from app.models.papers import DBPaper, DBPaperChunk, DBCitationMap, DBResearchQuery
from app.models.conversations import DBConversation
from app.models.messages import DBMessage
from app.models.refresh_tokens import DBRefreshToken
from app.models.message_papers import DBMessagePaper

# NEW: Trust-focused models
from app.models.authors import DBAuthor, DBAuthorPaper, DBAuthorInstitution
from app.models.institutions import DBInstitution
from app.models.citations import DBCitation

# DEPRECATED: Minimal usage, consider removing after migration
from app.models.queries import DBQuery
from app.models.answers import DBAnswer

__all__ = [
    "DatabaseBase",
    # User & Auth
    "DBUser",
    "DBRefreshToken",
    # Conversations
    "DBConversation",
    "DBMessage",
    # Papers (core)
    "DBPaper",
    "DBPaperChunk",
    "DBMessagePaper",
    # Trust & Reliability (NEW)
    "DBAuthor",
    "DBAuthorPaper",
    "DBAuthorInstitution",
    "DBInstitution",
    "DBCitation",
    # Legacy/Deprecated (keep for backward compatibility)
    "DBCitationMap",
    "DBResearchQuery",
    "DBQuery",
    "DBAnswer",
]
