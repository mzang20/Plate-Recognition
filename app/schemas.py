from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class BoundingBoxResponse(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class LatencyResponse(BaseModel):
    detection_ms: float
    ocr_ms: float
    total_ms: float


class PredictionResponse(BaseModel):
    plate_detected: bool
    plate_text: str
    bbox: Optional[BoundingBoxResponse] = None
    detector_confidence: Optional[float] = None
    confidence_percentile: Optional[float] = None
    ocr_confidence: Optional[float] = None
    number_of_detections: int
    image_width: int
    image_height: int
    latency: LatencyResponse


class ModelInfoResponse(BaseModel):
    detector: str
    image_size: int
    confidence_threshold: float
    device: str
    ocr_engine: str
    ocr_version: str
    ocr_config: str
    ocr_target_height: int
    crop_padding_ratio: float


class ConfidenceDistributionResponse(BaseModel):
    source: str
    sample_size: int
    mean: float
    median: float
    bin_edges: List[float]
    bin_counts: List[int]


class ReferenceStatsResponse(BaseModel):
    detector_confidence: ConfidenceDistributionResponse