from sqlalchemy.orm import DeclarativeBase

class DatabaseBase(DeclarativeBase):
    __abstract__ = True
    