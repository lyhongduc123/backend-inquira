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

async def init_db():
    """Initialize database by creating pgvector extension and all tables"""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        DatabaseBase.metadata.bind = engine
        await conn.run_sync(DatabaseBase.metadata.create_all)