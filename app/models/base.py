from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class DatabaseBase(Base):
    __abstract__ = True
    