from fastapi import APIRouter, Depends, HTTPException, status

from app.runtime_settings.models import RuntimeSettingsResponse, RuntimeSettingsUpdate
from app.runtime_settings.repository import SECRET_COLUMNS
from app.runtime_settings.service import (
    RuntimeSettingsConfigurationError,
    RuntimeSettingsService,
)


def create_settings_router(
    settings: RuntimeSettingsService,
    current_user_dependency,
) -> APIRouter:
    router = APIRouter(prefix="/api/settings", tags=["运行设置"])

    @router.get("/runtime", response_model=RuntimeSettingsResponse | None)
    async def get_runtime_settings(user=Depends(current_user_dependency)):
        return settings.get_public(user.id)

    @router.put("/runtime", response_model=RuntimeSettingsResponse)
    async def save_runtime_settings(
        payload: RuntimeSettingsUpdate,
        user=Depends(current_user_dependency),
    ):
        try:
            return settings.save(user.id, payload)
        except RuntimeSettingsConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    @router.delete("/runtime/secrets/{secret_name}", response_model=RuntimeSettingsResponse)
    async def clear_runtime_secret(secret_name: str, user=Depends(current_user_dependency)):
        if secret_name not in SECRET_COLUMNS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="密钥字段不存在",
            )
        try:
            return settings.clear_secret(user.id, secret_name)
        except RuntimeSettingsConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    return router
