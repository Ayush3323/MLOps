from fastapi import Depends, FastAPI

from app.api.deps import get_app_settings
from app.api.routes.analyze import router as analyze_router
from app.api.routes.index import router as index_router
from app.api.routes.query import router as query_router
from app.core.config import Settings

app = FastAPI(title="Review Intelligence API", version="0.1.0")
app.include_router(analyze_router)
app.include_router(index_router)
app.include_router(query_router)


@app.get("/health")
def health(settings: Settings = Depends(get_app_settings)) -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "device": settings.device,
    }
