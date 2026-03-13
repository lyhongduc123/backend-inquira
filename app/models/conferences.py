from sqlalchemy import Column, Integer, String, Boolean, UniqueConstraint
from app.models.base import DatabaseBase as Base

class DBConference(Base):
    __tablename__ = "conferences"

    id = Column(Integer, primary_key=True, index=True)
    core_id = Column(Integer, index=True) # Cột đầu tiên trong CSV của bạn
    title = Column(String, index=True)
    acronym = Column(String, index=True)
    source = Column(String)  # Ví dụ: CORE2018, ICORE2026
    rank = Column(String)    # A*, A, B, C, Unranked
    is_primary = Column(Boolean, default=False)
    for_codes = Column(String) # Mã ngành (0806, 46, v.v.)

    __table_args__ = (
        UniqueConstraint('acronym', 'source', name='uq_conf_acronym_source'),
    )