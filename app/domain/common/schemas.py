from typing import Optional
from app.core.model import CamelModel

class SJRMetadata(CamelModel):
    """SJR journal metadata"""
    title: Optional[str] = None
    sjr_score: Optional[float] = None
    sjr_best_quartile: Optional[str] = None
    is_open_access: Optional[bool] = None
    h_index: Optional[int] = None
    data_year: Optional[int] = None
    
    class Config:
        from_attributes = True
        ignore_extra = True
