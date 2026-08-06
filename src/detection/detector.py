from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_IMAGE_SIZE = 960
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_MAX_DETECTIONS = 5


@dataclass
class BoundingBox:

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class PlateDetection:

    detected: bool
    bounding_box: Optional[BoundingBox]
    confidence: Optional[float]
    number_of_detections: int
    latency_ms: float
    image_width: int
    image_height: int

    def to_dict(self) -> Dict[str, object]:

        result = asdict(self)

        # Keep the field name short for the later API response.
        result["bbox"] = result.pop("bounding_box")

        return result


class PlateDetector:

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        image_size: int = DEFAULT_IMAGE_SIZE,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.device = device or self._select_device()

        self._validate_configuration()

        print(f"Loading YOLO model from: {self.model_path}")
        print(f"Inference device: {self.device}")

        # The model is loaded once when PlateDetector is constructed.
        self.model = YOLO(str(self.model_path))

    @staticmethod
    def _select_device() -> str:

        if torch.cuda.is_available():
            return "0"

        return "cpu"

    def _validate_configuration(self) -> None:

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path}"
            )

        if self.model_path.suffix.lower() != ".pt":
            raise ValueError(
                "The YOLO model path must point to a .pt file."
            )

        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero.")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if self.max_detections <= 0:
            raise ValueError("max_detections must be greater than zero.")

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:

        if image is None:
            raise ValueError("The image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "The image must be provided as a NumPy array."
            )

        if image.size == 0:
            raise ValueError("The image is empty.")

        if image.ndim not in (2, 3):
            raise ValueError(
                "The image must be a grayscale or color image."
            )

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(
                "The image must have 1, 3, or 4 channels."
            )

    @staticmethod
    def _clamp_box(
        box: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> BoundingBox:

        xmin, ymin, xmax, ymax = box.tolist()

        xmin_int = max(0, min(int(round(xmin)), image_width - 1))
        ymin_int = max(0, min(int(round(ymin)), image_height - 1))
        xmax_int = max(0, min(int(round(xmax)), image_width))
        ymax_int = max(0, min(int(round(ymax)), image_height))

        if xmax_int <= xmin_int or ymax_int <= ymin_int:
            raise ValueError(
                "The detector returned an invalid bounding box."
            )

        return BoundingBox(
            xmin=xmin_int,
            ymin=ymin_int,
            xmax=xmax_int,
            ymax=ymax_int,
        )

    # Detect highest confidence license plate in an image
    def detect(self, image: np.ndarray) -> PlateDetection:

        self._validate_image(image)

        image_height, image_width = image.shape[:2]

        start_time = perf_counter()

        predictions = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            max_det=self.max_detections,
        )

        latency_ms = (perf_counter() - start_time) * 1000.0

        if not predictions:
            return PlateDetection(
                detected=False,
                bounding_box=None,
                confidence=None,
                number_of_detections=0,
                latency_ms=latency_ms,
                image_width=image_width,
                image_height=image_height,
            )

        result = predictions[0]
        number_of_detections = len(result.boxes)

        if number_of_detections == 0:
            return PlateDetection(
                detected=False,
                bounding_box=None,
                confidence=None,
                number_of_detections=0,
                latency_ms=latency_ms,
                image_width=image_width,
                image_height=image_height,
            )

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidence_scores = (
            result.boxes.conf.detach().cpu().numpy()
        )

        top_index = int(np.argmax(confidence_scores))
        top_box = boxes[top_index]
        top_confidence = float(confidence_scores[top_index])

        bounding_box = self._clamp_box(
            box=top_box,
            image_width=image_width,
            image_height=image_height,
        )

        return PlateDetection(
            detected=True,
            bounding_box=bounding_box,
            confidence=top_confidence,
            number_of_detections=number_of_detections,
            latency_ms=latency_ms,
            image_width=image_width,
            image_height=image_height,
        )

    def detect_from_path(
        self,
        image_path: Union[str, Path],
    ) -> PlateDetection:

        image_path = Path(image_path).resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image was not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"OpenCV could not decode the image: {image_path}"
            )

        return self.detect(image)

    def draw_detection(
        self,
        image: np.ndarray,
        detection: PlateDetection,
    ) -> np.ndarray:

        self._validate_image(image)

        annotated_image = image.copy()

        if not detection.detected or detection.bounding_box is None:
            return annotated_image

        box = detection.bounding_box

        cv2.rectangle(
            annotated_image,
            (box.xmin, box.ymin),
            (box.xmax, box.ymax),
            (0, 255, 0),
            2,
        )

        confidence_text = (
            f"License plate: {detection.confidence:.3f}"
        )

        text_y = max(25, box.ymin - 10)

        cv2.putText(
            annotated_image,
            confidence_text,
            (box.xmin, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return annotated_image