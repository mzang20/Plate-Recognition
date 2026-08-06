from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Union

import cv2
import numpy as np

from src.detection.detector import BoundingBox, PlateDetector
from src.recognition.ocr import PlateOCR


@dataclass
class PlateRecognitionResult:
    plate_detected: bool
    plate_text: str
    bbox: Optional[BoundingBox]
    detector_confidence: Optional[float]
    ocr_confidence: Optional[float]
    number_of_detections: int
    image_width: int
    image_height: int
    detection_latency_ms: float
    ocr_latency_ms: float
    total_latency_ms: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class PlateRecognitionPipeline:
    def __init__(
        self,
        detector: Optional[PlateDetector] = None,
        ocr: Optional[PlateOCR] = None,
    ) -> None:
        self.detector = detector or PlateDetector()
        self.ocr = ocr or PlateOCR()

    def recognize(
        self,
        image: np.ndarray,
    ) -> PlateRecognitionResult:
        start_time = perf_counter()

        detection = self.detector.detect(image)

        if (
            not detection.detected
            or detection.bounding_box is None
        ):
            total_latency_ms = (
                perf_counter() - start_time
            ) * 1000.0

            return PlateRecognitionResult(
                plate_detected=False,
                plate_text="",
                bbox=None,
                detector_confidence=None,
                ocr_confidence=None,
                number_of_detections=0,
                image_width=detection.image_width,
                image_height=detection.image_height,
                detection_latency_ms=detection.latency_ms,
                ocr_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
            )

        ocr_result = self.ocr.recognize(
            image=image,
            box=detection.bounding_box,
        )

        total_latency_ms = (
            perf_counter() - start_time
        ) * 1000.0

        return PlateRecognitionResult(
            plate_detected=True,
            plate_text=ocr_result.text,
            bbox=detection.bounding_box,
            detector_confidence=detection.confidence,
            ocr_confidence=ocr_result.confidence,
            number_of_detections=(
                detection.number_of_detections
            ),
            image_width=detection.image_width,
            image_height=detection.image_height,
            detection_latency_ms=detection.latency_ms,
            ocr_latency_ms=ocr_result.latency_ms,
            total_latency_ms=total_latency_ms,
        )

    def recognize_from_path(
        self,
        image_path: Union[str, Path],
    ) -> PlateRecognitionResult:
        image_path = Path(image_path).resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image was not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not decode image: {image_path}"
            )

        return self.recognize(image)

    def get_model_info(self) -> Dict[str, object]:
        return {
            "detector": self.detector.model_path.name,
            "image_size": self.detector.image_size,
            "confidence_threshold": (
                self.detector.confidence_threshold
            ),
            "device": self.detector.device,
            "ocr_engine": "tesseract",
            "ocr_version": self.ocr.tesseract_version,
            "ocr_config": self.ocr.config_id,
            "ocr_target_height": self.ocr.target_height,
            "crop_padding_ratio": self.ocr.padding_ratio,
        }