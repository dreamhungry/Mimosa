# -*- coding: utf-8 -*-
"""REST API route definitions for personality system."""

from fastapi import APIRouter

from .service_context import ServiceContext


def create_api_router(context: ServiceContext) -> APIRouter:
    """Create REST API router for personality endpoints.

    :param context: Service context.
    :returns: APIRouter with personality endpoints.
    """
    router = APIRouter(prefix="/api", tags=["personality"])

    @router.get("/personality")
    async def get_personality():
        """Get current personality state (Big Five + style)."""
        return {
            "status": "ok",
            "data": context.personality.get_state_dict(),
        }

    @router.post("/personality/reset")
    async def reset_personality():
        """Reset personality to baseline configuration."""
        context.personality.reset()
        return {
            "status": "ok",
            "message": "Personality reset to baseline",
            "data": context.personality.get_state_dict(),
        }

    return router
