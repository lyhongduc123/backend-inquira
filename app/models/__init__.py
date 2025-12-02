"""
Database models for the Exegent application.

All models inherit from DatabaseBase which itself inherits from SQLAlchemy's Base.
Import all models here to ensure they're registered with SQLAlchemy metadata.
"""

from app.models.base import DatabaseBase, Base
from app.models.users import DBUser
from app.models.papers import DBPaper, DBPaperChunk
from app.models.queries import DBQuery
from app.models.answers import DBAnswer
from app.models.messages import DBMessage
from app.models.conversations import DBConversation
from app.models.refresh_tokens import DBRefreshToken

__all__ = [
    "DatabaseBase",
    "Base",
    "DBUser",
    "DBPaper",
    "DBPaperChunk",
    "DBQuery",
    "DBAnswer",
    "DBConversation",
    "DBMessage",
    "DBRefreshToken"
]
