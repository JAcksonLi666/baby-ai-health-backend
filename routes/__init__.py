from .health import router as health_router
from .records import router as records_router
from .daily import router as daily_router
from .ai import router as ai_router

__all__ = ["health_router", "records_router", "daily_router", "ai_router"]
