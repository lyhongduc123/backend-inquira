"""
Custom exception classes for API error handling
"""
from fastapi import HTTPException, status
from app.core.responses import ErrorCode


class BaseApiException(HTTPException):
    """
    Base exception for all API errors
    
    Provides structured error information including error codes
    """
    def __init__(self, message: str, code: ErrorCode, status_code: int, details: dict | None = None):
        self.code = code
        self.details = details
        super().__init__(status_code=status_code, detail=message)


class NotFoundException(BaseApiException):
    """Resource not found (404)"""
    def __init__(self, message: str = "Resource not found", details: dict | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class UnauthorizedException(BaseApiException):
    """Unauthorized access (401)"""
    def __init__(self, message: str = "Unauthorized", details: dict | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.UNAUTHORIZED,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class ForbiddenException(BaseApiException):
    """Forbidden access (403)"""
    def __init__(self, message: str = "Forbidden", details: dict | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.FORBIDDEN,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class ValidationException(BaseApiException):
    """Validation error (422)"""
    def __init__(self, message: str = "Validation error", details: dict | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class BadRequestException(BaseApiException):
    """Bad request (400)"""
    def __init__(self, message: str = "Bad request", details: dict | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.BAD_REQUEST,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class ConflictException(BaseApiException):
    """Conflict error (409)"""
    def __init__(self, message: str = "Resource conflict", details: dict | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.CONFLICT,
            status_code=status.HTTP_409_CONFLICT,
            details=details
        )


class InternalServerException(BaseApiException):
    """Internal server error (500)"""
    def __init__(self, message: str = "Internal server error", details: dict | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.INTERNAL_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class ServiceUnavailableException(BaseApiException):
    """Service unavailable (503)"""
    def __init__(self, message: str = "Service unavailable", details: dict | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.SERVICE_UNAVAILABLE,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )
