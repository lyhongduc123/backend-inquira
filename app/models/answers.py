from sqlalchemy import Boolean, Column, DateTime, Enum, Integer, String, func
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class DBAnswers(Base):
    __tablename__ = "answers"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, nullable=False, index=True)
    answer_text = Column(String, nullable=False)
    is_approved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())