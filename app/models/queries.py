from sqlalchemy import Boolean, Column, Integer, String, DateTime, Enum
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class DBQueries(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    query_text = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    status = Column(Enum, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())