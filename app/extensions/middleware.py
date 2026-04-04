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
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers['X-Request-ID'] = request_id
        
        return response
