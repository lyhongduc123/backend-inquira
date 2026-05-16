from typing import Optional
from app.core.model import CamelModel

class JournalMetadata(CamelModel):
    """Journal metadata"""
    title: Optional[str] = None
    sjr_score: Optional[float] = None
    sjr_best_quartile: Optional[str] = None
    is_open_access: Optional[bool] = None
    h_index: Optional[int] = None
    data_year: Optional[int] = None
    
    class Config:
        from_attributes = True
        ignore_extra = True


class ConferenceMetadata(CamelModel):
    """Conference metadata."""

    id: Optional[int] = None
    core_id: Optional[int] = None
    title: Optional[str] = None
    acronym: Optional[str] = None
    source: Optional[str] = None
    rank: Optional[str] = None
    is_primary: Optional[bool] = None
    for_codes: Optional[str] = None

    class Config:
        from_attributes = True
        ignore_extra = True
