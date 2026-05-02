from contextlib import asynccontextmanager
from re import A
from typing import AsyncGenerator

from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text
from app.llm import provider
from app.models.base import DatabaseBase
from app.models.users import DBUser

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session():
    db = async_session()
    try:
        yield db
    finally:
        await db.close()


@asynccontextmanager
async def db_session_context() -> AsyncGenerator[AsyncSession, None]:
    session = async_session()
    try:
        yield session
    finally:
        await session.close()


async def init_db():
    """Verify database connectivity and ensure required extensions exist"""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))

        await conn.run_sync(DatabaseBase.metadata.create_all)

        await conn.execute(
            insert(DBUser)
            .values(
                id=1999,
                name="Anonymous User",
                email="anonymous@example.com",
                provider="email",
                provider_id="anonymous",
                is_active=True,
            )
            .on_conflict_do_nothing(
                index_elements=["id"]
            )
        )

        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_paper_chunks_bm25 on paper_chunks using bm25 (id, text, section_title) with (key_field='id');"
            )
        )

        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_papers_bm25 on papers using bm25 (id, title, abstract) with (key_field='id');"
            )
        )

        await conn.execute(text("SELECT 1"))

        # try:
        #     result = await conn.execute(
        #         text("SELECT version_num FROM alembic_version")
        #     )
        #     version = result.scalar_one()
        # except Exception:
        #     raise RuntimeError(
        #         "Database schema not initialized. "
        #         "Run: alembic upgrade head"
        #     )
