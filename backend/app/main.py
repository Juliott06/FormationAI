import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings


logger = logging.getLogger("uvicorn.error")


def create_app() -> FastAPI:
    settings = get_settings()
    logger.info(
        "FormationAI config: model=%s imgsz=%d confidence=%.2f iou=%.2f tracker=%s max_det=%d debug_video=%s",
        settings.yolo_model_name,
        settings.yolo_image_size,
        settings.yolo_confidence_threshold,
        settings.yolo_iou_threshold,
        settings.tracker_config,
        settings.yolo_max_detections,
        settings.generate_debug_video,
    )
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()
