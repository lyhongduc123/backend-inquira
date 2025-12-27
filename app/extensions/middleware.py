"""
Middleware for request processing
"""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request ID tracking for debugging and logging
    
    Generates a unique request ID for each request or uses the one provided
    in the X-Request-ID header. Adds the request ID to both the request state
    and response headers.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        # Get request ID from header or generate new one
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        
        # Store in request state for access in handlers
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = request_id
        
        return response
