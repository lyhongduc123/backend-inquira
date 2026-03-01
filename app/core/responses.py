from typing import Generic, TypeVar, Optional, List
from pydantic import BaseModel, Field
from app.core.model import CamelModel
from datetime import datetime
from enum import Enum
from math import ceil

T = TypeVar('T')

class ErrorCode(str, Enum):
    """Standard error codes"""
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    BAD_REQUEST = "BAD_REQUEST"
    CONFLICT = "CONFLICT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

class ErrorDetail(BaseModel):
    """Error detail structure"""
    code: ErrorCode
    message: str
    details: Optional[dict] = None

class ApiResponse(CamelModel, Generic[T]):
    """Standard API response wrapper"""
    success: bool
    data: Optional[T] = None
    error: Optional[ErrorDetail] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"id": 1, "name": "Example"},
                "error": None,
                "timestamp": "2024-12-27T10:00:00Z"
            }
        }


class PaginatedData(CamelModel, Generic[T]):
    """Paginated response data"""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


# Helper functions
def success_response(data: T, request_id: Optional[str] = None) -> ApiResponse[T]:
    return ApiResponse(success=True, data=data, request_id=request_id)


def paginated_response(
    data: List[T],
    total: int,
    page: int,
    page_size: int,
    request_id: Optional[str] = None
) -> ApiResponse[PaginatedData[T]]:
    """Create a paginated response"""
    total_pages = ceil(total / page_size) if page_size > 0 else 0
    
    paginated_data = PaginatedData(
        items=data,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1
    )
    
    return ApiResponse(success=True, data=paginated_data, request_id=request_id)

def error_response(
    code: ErrorCode,
    message: str,
    details: Optional[dict] = None,
    request_id: Optional[str] = None
) -> ApiResponse[None]:
    return ApiResponse(
        success=False,
        error=ErrorDetail(code=code, message=message, details=details),
        request_id=request_id
    )