from pathlib import Path

import cv2

from src.detection.detector import PlateDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ufpr_alpr"
    / "validation"
    / "track0061"
    / "track0061[01].png"
)

OUTPUT_PATH = PROJECT_ROOT / "tests" / "detector_test_output.jpg"


def main() -> None:
    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Current path: {TEST_IMAGE_PATH}"
        )

    detector = PlateDetector()

    image = cv2.imread(str(TEST_IMAGE_PATH))

    if image is None:
        raise ValueError(
            f"Could not read the test image: {TEST_IMAGE_PATH}"
        )

    detection = detector.detect(image)

    print("Detection result:")
    print(detection.to_dict())

    annotated_image = detector.draw_detection(
        image=image,
        detection=detection,
    )

    saved = cv2.imwrite(
        str(OUTPUT_PATH),
        annotated_image,
    )

    if not saved:
        raise RuntimeError(
            f"Could not save output image to: {OUTPUT_PATH}"
        )

    print(f"Annotated image saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()