from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

import os
import re
import shutil

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from src.detection.detector import BoundingBox


DEFAULT_PADDING_RATIO = 0.05
DEFAULT_TARGET_HEIGHT = 96

TESSERACT_CONFIG = (
    "--oem 3 --psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

OCR_CONFIG_ID = "tesseract_psm7_clahe_h96_pad005_v1"


@dataclass
class OCRResult:
    text: str
    confidence: float
    crop_box: BoundingBox
    latency_ms: float
    config_id: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class PlateOCR:
    def __init__(
        self,
        padding_ratio: float = DEFAULT_PADDING_RATIO,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        tesseract_cmd: Optional[Union[str, Path]] = None,
    ) -> None:
        if padding_ratio < 0:
            raise ValueError("padding_ratio cannot be negative.")

        if target_height <= 0:
            raise ValueError("target_height must be greater than zero.")

        self.padding_ratio = padding_ratio
        self.target_height = target_height
        self.config = TESSERACT_CONFIG
        self.config_id = OCR_CONFIG_ID

        self.tesseract_path = self._configure_tesseract(
            tesseract_cmd
        )

        self.tesseract_version = str(
            pytesseract.get_tesseract_version()
        )

    @staticmethod
    def _configure_tesseract(
        tesseract_cmd: Optional[Union[str, Path]],
    ) -> Path:
        candidates: List[Path] = []

        if tesseract_cmd is not None:
            candidates.append(Path(tesseract_cmd))

        environment_path = os.environ.get("TESSERACT_CMD")

        if environment_path:
            candidates.append(Path(environment_path))

        discovered_path = shutil.which("tesseract")

        if discovered_path:
            candidates.append(Path(discovered_path))

        candidates.extend(
            [
                Path(
                    "C:/Program Files/"
                    "Tesseract-OCR/tesseract.exe"
                ),
                Path(
                    "C:/Program Files (x86)/"
                    "Tesseract-OCR/tesseract.exe"
                ),
                Path("/usr/bin/tesseract"),
                Path("/usr/local/bin/tesseract"),
                Path("/opt/homebrew/bin/tesseract"),
            ]
        )

        for candidate in candidates:
            candidate = candidate.expanduser().resolve()

            if candidate.exists():
                pytesseract.pytesseract.tesseract_cmd = str(
                    candidate
                )
                return candidate

        raise FileNotFoundError(
            "Tesseract was not found. Install it or set "
            "the TESSERACT_CMD environment variable."
        )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("The image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "The image must be a NumPy array."
            )

        if image.size == 0:
            raise ValueError("The image is empty.")

        if image.ndim not in (2, 3):
            raise ValueError(
                "The image must be grayscale or color."
            )

    @staticmethod
    def normalize_plate_text(value: object) -> str:
        return re.sub(
            r"[^A-Z0-9]",
            "",
            str(value).upper(),
        )

    @staticmethod
    def _get_box_values(
        box: Union[BoundingBox, Sequence[float]],
    ) -> Tuple[float, float, float, float]:
        if isinstance(box, BoundingBox):
            values = (
                float(box.xmin),
                float(box.ymin),
                float(box.xmax),
                float(box.ymax),
            )
        else:
            if len(box) != 4:
                raise ValueError(
                    "The bounding box must contain four values."
                )

            values = tuple(float(value) for value in box)

        if not all(isfinite(value) for value in values):
            raise ValueError(
                "The bounding box contains invalid values."
            )

        return values

    def clamp_and_pad_box(
        self,
        box: Union[BoundingBox, Sequence[float]],
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        xmin, ymin, xmax, ymax = self._get_box_values(box)

        if xmax <= xmin or ymax <= ymin:
            raise ValueError("The bounding box is invalid.")

        width = xmax - xmin
        height = ymax - ymin

        horizontal_padding = width * self.padding_ratio
        vertical_padding = height * self.padding_ratio

        padded_xmin = max(
            0,
            int(floor(xmin - horizontal_padding)),
        )
        padded_ymin = max(
            0,
            int(floor(ymin - vertical_padding)),
        )
        padded_xmax = min(
            image_width,
            int(ceil(xmax + horizontal_padding)),
        )
        padded_ymax = min(
            image_height,
            int(ceil(ymax + vertical_padding)),
        )

        if (
            padded_xmax <= padded_xmin
            or padded_ymax <= padded_ymin
        ):
            raise ValueError(
                "The padded bounding box is invalid."
            )

        return BoundingBox(
            xmin=padded_xmin,
            ymin=padded_ymin,
            xmax=padded_xmax,
            ymax=padded_ymax,
        )

    def extract_crop(
        self,
        image: np.ndarray,
        box: Union[BoundingBox, Sequence[float]],
    ) -> Tuple[np.ndarray, BoundingBox]:
        self._validate_image(image)

        image_height, image_width = image.shape[:2]

        crop_box = self.clamp_and_pad_box(
            box=box,
            image_width=image_width,
            image_height=image_height,
        )

        crop = image[
            crop_box.ymin:crop_box.ymax,
            crop_box.xmin:crop_box.xmax,
        ]

        if crop.size == 0:
            raise ValueError(
                "The extracted plate crop is empty."
            )

        return crop, crop_box

    @staticmethod
    def _to_grayscale(crop: np.ndarray) -> np.ndarray:
        if crop.ndim == 2:
            return crop

        channels = crop.shape[2]

        if channels == 1:
            return crop[:, :, 0]

        if channels == 3:
            return cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY,
            )

        if channels == 4:
            return cv2.cvtColor(
                crop,
                cv2.COLOR_BGRA2GRAY,
            )

        raise ValueError(
            "The crop has an unsupported channel count."
        )

    def preprocess_crop(
        self,
        crop: np.ndarray,
    ) -> np.ndarray:
        self._validate_image(crop)

        gray = self._to_grayscale(crop)

        scale = max(
            1.0,
            self.target_height / max(gray.shape[0], 1),
        )

        target_width = max(
            1,
            int(round(gray.shape[1] * scale)),
        )

        resized = cv2.resize(
            gray,
            (target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC,
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(resized)

        return cv2.copyMakeBorder(
            enhanced,
            12,
            12,
            12,
            12,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )

    def run_tesseract(
        self,
        processed_crop: np.ndarray,
    ) -> Tuple[str, float]:
        data = pytesseract.image_to_data(
            processed_crop,
            output_type=Output.DICT,
            config=self.config,
        )

        tokens: List[str] = []
        confidence_values: List[float] = []

        for token, confidence in zip(
            data["text"],
            data["conf"],
        ):
            normalized_token = self.normalize_plate_text(
                token
            )

            try:
                confidence_float = float(confidence)
            except (TypeError, ValueError):
                confidence_float = -1.0

            if normalized_token:
                tokens.append(normalized_token)

                if confidence_float >= 0:
                    confidence_values.append(
                        confidence_float
                    )

        text = self.normalize_plate_text(
            "".join(tokens)
        )

        if not text:
            text = self.normalize_plate_text(
                pytesseract.image_to_string(
                    processed_crop,
                    config=self.config,
                )
            )

        confidence = (
            float(np.mean(confidence_values)) / 100.0
            if confidence_values
            else 0.0
        )

        return text, confidence

    def recognize(
        self,
        image: np.ndarray,
        box: Union[BoundingBox, Sequence[float]],
    ) -> OCRResult:
        start_time = perf_counter()

        crop, crop_box = self.extract_crop(
            image=image,
            box=box,
        )

        processed_crop = self.preprocess_crop(crop)

        text, confidence = self.run_tesseract(
            processed_crop
        )

        latency_ms = (
            perf_counter() - start_time
        ) * 1000.0

        return OCRResult(
            text=text,
            confidence=confidence,
            crop_box=crop_box,
            latency_ms=latency_ms,
            config_id=self.config_id,
        )