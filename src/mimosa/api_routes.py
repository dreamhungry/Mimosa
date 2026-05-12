# -*- coding: utf-8 -*-
"""REST API route definitions for personality and configuration."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .service_context import ServiceContext


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PersonalityUpdateRequest(BaseModel):
    """Request body for updating personality state."""

    big_five: dict | None = None
    style: dict | None = None


class LLMConfigUpdateRequest(BaseModel):
    """Request body for updating LLM configuration."""

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_api_router(context: ServiceContext) -> APIRouter:
    """Create REST API router for personality and config endpoints.

    :param context: Service context.
    :returns: APIRouter with all API endpoints.
    """
    router = APIRouter(prefix="/api", tags=["api"])

    # -- Personality endpoints ------------------------------------------

    @router.get("/personality")
    async def get_personality():
        """Get current personality state (Big Five + style)."""
        return {
            "status": "ok",
            "data": context.personality.get_state_dict(),
        }

    @router.put("/personality")
    async def update_personality(body: PersonalityUpdateRequest):
        """Update personality state (Big Five traits and/or style)."""
        try:
            context.personality.update_state(
                big_five=body.big_five,
                style=body.style,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
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

    # -- LLM config endpoints ------------------------------------------

    @router.get("/config/llm")
    async def get_llm_config():
        """Get current LLM configuration (api_key masked)."""
        cfg = context.config.llm
        key = cfg.api_key
        masked_key = f"****{key[-4:]}" if len(key) > 4 else "****"
        return {
            "status": "ok",
            "data": {
                "provider": cfg.provider,
                "model": cfg.model,
                "base_url": cfg.base_url,
                "api_key": masked_key,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            },
        }

    @router.put("/config/llm")
    async def update_llm_config(body: LLMConfigUpdateRequest):
        """Update LLM configuration (hot-reload, no restart needed)."""
        try:
            context.update_llm_config(
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                api_key=body.api_key,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Return updated config (masked key)
        cfg = context.config.llm
        key = cfg.api_key
        masked_key = f"****{key[-4:]}" if len(key) > 4 else "****"
        return {
            "status": "ok",
            "data": {
                "provider": cfg.provider,
                "model": cfg.model,
                "base_url": cfg.base_url,
                "api_key": masked_key,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            },
        }

    return router
