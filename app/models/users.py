
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Boolean, String, Integer
from app.models.base import DatabaseBase as Base

class DBUser(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)