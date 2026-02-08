"""
Database models for the Exegent application.

All models inherit from DatabaseBase which itself inherits from SQLAlchemy's Base.
Import all models here to ensure they're registered with SQLAlchemy metadata.

"""

from app.models.base import DatabaseBase
from app.models.users import DBUser
from app.models.papers import DBPaper, DBPaperChunk
from app.models.conversations import DBConversation
from app.models.messages import DBMessage
from app.models.refresh_tokens import DBRefreshToken
from app.models.message_papers import DBMessagePaper

# NEW: Trust-focused models
from app.models.authors import DBAuthor, DBAuthorPaper, DBAuthorInstitution
from app.models.institutions import DBInstitution
from app.models.citations import DBCitation
from app.models.journals import DBJournal
from app.models.preprocessing_state import DBPreprocessingState
from app.models.answer_vaidations import DBAnswerValidation
from app.models.message_contexts import DBMessageContext

__all__ = [
    "DatabaseBase",
    # User & Auth
    "DBUser",
    "DBRefreshToken",
    
    # Conversations & Messages
    "DBConversation",
    "DBMessage",
    "DBMessageContext",
    
    # Core
    "DBPaper",
    "DBPaperChunk",
    "DBMessagePaper",
    "DBAuthor",
    "DBAuthorPaper",
    "DBAuthorInstitution",
    "DBInstitution",
    "DBAnswerValidation",
    "DBJournal",
    "DBPreprocessingState",
]
