from contextlib import asynccontextmanager
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReferenceStatsResponse,
)
from src.pipeline.inference_pipeline import (
    PlateRecognitionPipeline,
)
from src.reliability.reference_stats import (
    ReferenceStats,
)


MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pipeline = PlateRecognitionPipeline()
    app.state.reference_stats = ReferenceStats()
    yield


app = FastAPI(
    title="License Plate Recognition API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    information = app.state.pipeline.get_model_info()
    return ModelInfoResponse(**information)


@app.get(
    "/reference-stats",
    response_model=ReferenceStatsResponse,
)
def reference_stats() -> ReferenceStatsResponse:
    return ReferenceStatsResponse(
        **app.state.reference_stats.to_response()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    file: UploadFile = File(...),
) -> PredictionResponse:
    content_type = file.content_type or ""

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type.",
        )

    try:
        contents = file.file.read(MAX_FILE_SIZE + 1)
    finally:
        file.file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The uploaded file exceeds 10 MB.",
        )

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be decoded.",
        )

    try:
        result = app.state.pipeline.recognize(image)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    bbox = None

    if result.bbox is not None:
        bbox = {
            "xmin": result.bbox.xmin,
            "ymin": result.bbox.ymin,
            "xmax": result.bbox.xmax,
            "ymax": result.bbox.ymax,
        }

    confidence_percentile = (
        app.state.reference_stats.confidence_percentile(
            result.detector_confidence
        )
    )

    return PredictionResponse(
        plate_detected=result.plate_detected,
        plate_text=result.plate_text,
        bbox=bbox,
        detector_confidence=result.detector_confidence,
        confidence_percentile=confidence_percentile,
        ocr_confidence=result.ocr_confidence,
        number_of_detections=result.number_of_detections,
        image_width=result.image_width,
        image_height=result.image_height,
        latency={
            "detection_ms": result.detection_latency_ms,
            "ocr_ms": result.ocr_latency_ms,
            "total_ms": result.total_latency_ms,
        },
    )