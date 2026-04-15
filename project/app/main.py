from fastapi import Depends, FastAPI

from app.api.deps import get_app_settings
from app.core.config import Settings

app = FastAPI(title="Review Intelligence API", version="0.1.0")


@app.get("/health")
def health(settings: Settings = Depends(get_app_settings)) -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "device": settings.device,
    }
