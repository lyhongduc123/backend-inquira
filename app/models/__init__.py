"""
Database models for the application.

All models inherit from DatabaseBase which itself inherits from SQLAlchemy's Base.
Import all models here to ensure they're registered with SQLAlchemy metadata.

"""

from app.models.base import DatabaseBase
from app.models.users import DBUser
from app.models.papers import DBPaper, DBPaperChunk
from app.models.conversations import DBConversation
from app.models.messages import DBMessage
from app.models.refresh_tokens import DBRefreshToken
from app.models.email_otps import DBEmailOtp
from app.models.message_papers import DBMessagePaper
from app.models.authors import DBAuthor, DBAuthorPaper, DBAuthorInstitution
from app.models.institutions import DBInstitution
from app.models.citations import DBCitation
from app.models.journals import DBJournal
from app.models.preprocessing_state import DBPreprocessingState
from app.models.answer_vaidations import DBAnswerValidation
from app.models.message_contexts import DBMessageContext
from app.models.author_relationships import DBAuthorRelationship
from app.models.conferences import DBConference
from app.models.pipeline_tasks import DBPipelineTask, DBPipelineEvent
from app.models.bookmarks import DBBookmark
from app.models.user_settings import DBUserSettings
from app.models.benchmark_corpus import DBBenchmarkPaper
from app.models.benchmarks import (
    DBGroundTruthDataset,
    DBPaperRelevanceBenchmark,
    DBChunkQualityBenchmark,
    DBBreakdownQualityBenchmark,
    DBPipelineBenchmark,
)

__all__ = [
    "DatabaseBase",
    # User & Auth
    "DBUser",
    "DBRefreshToken",
    "DBEmailOtp",
    "DBBookmark",
    # Conversations & Messages
    "DBConversation",
    "DBMessage",
    "DBMessageContext",
    "DBUserSettings",
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
    
    # Benchmarking
    "DBGroundTruthDataset",
    "DBPaperRelevanceBenchmark",
    "DBChunkQualityBenchmark",
    "DBBreakdownQualityBenchmark",
    "DBPipelineBenchmark",
    "DBBenchmarkPaper",
]
