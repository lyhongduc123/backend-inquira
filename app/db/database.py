from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.models.users import DBUsers
from app.models.answers import DBAnswers
from app.models.queries import DBQueries

Base = declarative_base()

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
    async with engine.begin() as conn:
        Base.metadata.bind = engine
        await conn.run_sync(Base.metadata.create_all)