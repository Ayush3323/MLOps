from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_app_settings
from app.api.routes.analyze import router as analyze_router
from app.api.routes.index import router as index_router
from app.api.routes.query import router as query_router
from app.core.config import Settings

app = FastAPI(title="Review Intelligence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)
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
