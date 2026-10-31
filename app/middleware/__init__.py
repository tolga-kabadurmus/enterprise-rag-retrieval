"""HTTP middleware package."""

from app.middleware.exceptions import UnhandledExceptionMiddleware

__all__ = ["UnhandledExceptionMiddleware"]
