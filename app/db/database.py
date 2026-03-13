from contextlib import asynccontextmanager

from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text
from app.models.base import DatabaseBase

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_db_session():
    db = async_session()
    try:
        yield db
    finally:
        await db.close()

@asynccontextmanager
async def db_session_context():
    session = async_session()
    try:
        yield session
    finally:
        await session.close()

async def init_db():
    """Verify database connectivity and ensure required extensions exist"""
    async with engine.begin() as conn:
        # Required extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Health check
        await conn.execute(text("SELECT 1"))

        # Alembic check
        try:
            result = await conn.execute(
                text("SELECT version_num FROM alembic_version")
            )
            version = result.scalar_one()
        except Exception:
            raise RuntimeError(
                "Database schema not initialized. "
                "Run: alembic upgrade head"
            )