from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


DATASET_DIR = Path("data/raw/ufpr_alpr")
OUTPUT_DIR = Path("data/processed")
SPLITS = ("training", "validation", "testing")


def calculate_image_features(image: np.ndarray) -> dict[str, float]:
    """Calculate simple image-quality features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur_score": float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
    }


def parse_integer_list(value: str, expected_length: int) -> list[int]:
    """Parse space-separated integers and validate their count."""
    values = [int(number) for number in value.split()]

    if len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} integers, found {len(values)}: {value}"
        )

    return values


def parse_corners(value: str) -> list[tuple[int, int]]:
    """Parse plate corner coordinates such as '912,526 986,531 ...'."""
    corners: list[tuple[int, int]] = []

    for coordinate in value.split():
        x_text, y_text = coordinate.split(",")
        corners.append((int(x_text), int(y_text)))

    if len(corners) != 4:
        raise ValueError(
            f"Expected four plate corners, found {len(corners)}."
        )

    return corners


def parse_annotation(
    annotation_path: Path,
) -> tuple[dict[str, object], list[dict[str, int]]]:
    """Parse one UFPR-ALPR annotation file."""
    annotation: dict[str, object] = {}
    characters: list[dict[str, int]] = []

    lines = annotation_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        character_match = re.fullmatch(
            r"char\s+(\d+):\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
            line,
        )

        if character_match:
            char_index, x, y, width, height = map(
                int,
                character_match.groups(),
            )

            characters.append(
                {
                    "char_index": char_index,
                    "char_x": x,
                    "char_y": y,
                    "char_width": width,
                    "char_height": height,
                }
            )
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", maxsplit=1)
        key = key.strip().lower()
        value = value.strip()

        # Typo in 5 files
        if key == "ccamera":
            key = "camera"

        if key == "camera":
            annotation["camera"] = value

        elif key == "position_vehicle":
            x, y, width, height = parse_integer_list(value, 4)

            annotation.update(
                {
                    "vehicle_x": x,
                    "vehicle_y": y,
                    "vehicle_width": width,
                    "vehicle_height": height,
                }
            )

        elif key == "type":
            annotation["vehicle_type"] = value

        elif key == "make":
            annotation["vehicle_make"] = value

        elif key == "model":
            annotation["vehicle_model"] = value

        elif key == "year":
            annotation["vehicle_year"] = int(value)

        elif key == "plate":
            annotation["plate_text"] = value

        elif key == "corners":
            annotation["plate_corners"] = parse_corners(value)

    required_fields = {
        "camera",
        "vehicle_x",
        "vehicle_y",
        "vehicle_width",
        "vehicle_height",
        "vehicle_type",
        "vehicle_make",
        "vehicle_model",
        "vehicle_year",
        "plate_text",
        "plate_corners",
    }

    missing_fields = required_fields - annotation.keys()

    if missing_fields:
        raise ValueError(
            f"Missing annotation fields: {sorted(missing_fields)}"
        )

    characters.sort(key=lambda row: row["char_index"])

    return annotation, characters


def get_frame_number(image_path: Path) -> int:
    """Extract 01 from a filename such as track0001[01].png."""
    match = re.search(r"\[(\d+)\]", image_path.stem)

    if match is None:
        raise ValueError(
            f"Could not extract frame number from {image_path.name}"
        )

    return int(match.group(1))


def process_image(
    image_path: Path,
    split: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build image-level and character-level records."""
    annotation_path = image_path.with_suffix(".txt")

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Missing annotation for {image_path}"
        )

    annotation, characters = parse_annotation(annotation_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"OpenCV could not read {image_path}")

    image_height, image_width = image.shape[:2]
    track_id = image_path.parent.name
    frame_number = get_frame_number(image_path)

    plate_text = str(annotation["plate_text"])
    corners = annotation["plate_corners"]

    if not isinstance(corners, list):
        raise TypeError("Plate corners were not parsed correctly.")

    corner_x_values = [point[0] for point in corners]
    corner_y_values = [point[1] for point in corners]

    plate_xmin = min(corner_x_values)
    plate_ymin = min(corner_y_values)
    plate_xmax = max(corner_x_values)
    plate_ymax = max(corner_y_values)

    plate_width = plate_xmax - plate_xmin
    plate_height = plate_ymax - plate_ymin

    corners_valid = all(
        0 <= x < image_width and 0 <= y < image_height
        for x, y in corners
    )

    plate_box_valid = (
        corners_valid
        and plate_width > 0
        and plate_height > 0
    )

    vehicle_x = int(annotation["vehicle_x"])
    vehicle_y = int(annotation["vehicle_y"])
    vehicle_width = int(annotation["vehicle_width"])
    vehicle_height = int(annotation["vehicle_height"])

    vehicle_box_valid = (
        vehicle_x >= 0
        and vehicle_y >= 0
        and vehicle_width > 0
        and vehicle_height > 0
        and vehicle_x + vehicle_width <= image_width
        and vehicle_y + vehicle_height <= image_height
    )

    plate_inside_vehicle = (
        plate_xmin >= vehicle_x
        and plate_ymin >= vehicle_y
        and plate_xmax <= vehicle_x + vehicle_width
        and plate_ymax <= vehicle_y + vehicle_height
    )

    full_image_features = calculate_image_features(image)

    plate_crop = image[
        plate_ymin : plate_ymax + 1,
        plate_xmin : plate_xmax + 1,
    ]

    if plate_crop.size == 0:
        raise ValueError(f"Empty plate crop for {image_path}")

    plate_features = calculate_image_features(plate_crop)

    character_count_matches = (
        len(characters) == len(plate_text)
    )

    all_character_boxes_valid = True
    character_rows: list[dict[str, object]] = []

    for character in characters:
        char_index = character["char_index"]
        char_x = character["char_x"]
        char_y = character["char_y"]
        char_width = character["char_width"]
        char_height = character["char_height"]

        char_xmax = char_x + char_width
        char_ymax = char_y + char_height

        box_valid = (
            char_x >= 0
            and char_y >= 0
            and char_width > 0
            and char_height > 0
            and char_xmax <= image_width
            and char_ymax <= image_height
        )

        all_character_boxes_valid &= box_valid

        char_label = (
            plate_text[char_index - 1]
            if 1 <= char_index <= len(plate_text)
            else None
        )

        character_rows.append(
            {
                "split": split,
                "track_id": track_id,
                "frame_number": frame_number,
                "image_name": image_path.name,
                "image_path": image_path.as_posix(),
                "plate_text": plate_text,
                "char_index": char_index,
                "char_label": char_label,
                "char_x": char_x,
                "char_y": char_y,
                "char_width": char_width,
                "char_height": char_height,
                "char_xmax": char_xmax,
                "char_ymax": char_ymax,
                "char_area": char_width * char_height,
                "char_relative_x": (
                    (char_x - plate_xmin) / plate_width
                    if plate_width > 0
                    else np.nan
                ),
                "char_relative_y": (
                    (char_y - plate_ymin) / plate_height
                    if plate_height > 0
                    else np.nan
                ),
                "char_box_valid": box_valid,
            }
        )

    image_area = image_width * image_height
    plate_area = plate_width * plate_height
    vehicle_area = vehicle_width * vehicle_height

    row: dict[str, object] = {
        "split": split,
        "track_id": track_id,
        "frame_number": frame_number,
        "image_name": image_path.name,
        "image_path": image_path.as_posix(),
        "annotation_path": annotation_path.as_posix(),
        "image_width": image_width,
        "image_height": image_height,
        "camera": annotation["camera"],
        "vehicle_type": annotation["vehicle_type"],
        "vehicle_make": annotation["vehicle_make"],
        "vehicle_model": annotation["vehicle_model"],
        "vehicle_year": annotation["vehicle_year"],
        "vehicle_x": vehicle_x,
        "vehicle_y": vehicle_y,
        "vehicle_width": vehicle_width,
        "vehicle_height": vehicle_height,
        "vehicle_area_ratio": vehicle_area / image_area,
        "plate_text": plate_text,
        "plate_character_count": len(plate_text),
        "annotated_character_count": len(characters),
        "plate_corners": json.dumps(corners),
        "corner_1_x": corners[0][0],
        "corner_1_y": corners[0][1],
        "corner_2_x": corners[1][0],
        "corner_2_y": corners[1][1],
        "corner_3_x": corners[2][0],
        "corner_3_y": corners[2][1],
        "corner_4_x": corners[3][0],
        "corner_4_y": corners[3][1],
        "plate_xmin": plate_xmin,
        "plate_ymin": plate_ymin,
        "plate_xmax": plate_xmax,
        "plate_ymax": plate_ymax,
        "plate_width": plate_width,
        "plate_height": plate_height,
        "plate_area_ratio": plate_area / image_area,
        "plate_aspect_ratio": (
            plate_width / plate_height
            if plate_height > 0
            else np.nan
        ),
        "relative_plate_width": plate_width / image_width,
        "relative_plate_height": plate_height / image_height,
        "image_brightness": full_image_features["brightness"],
        "image_contrast": full_image_features["contrast"],
        "image_blur_score": full_image_features["blur_score"],
        "plate_brightness": plate_features["brightness"],
        "plate_contrast": plate_features["contrast"],
        "plate_blur_score": plate_features["blur_score"],
        "corners_valid": corners_valid,
        "plate_box_valid": plate_box_valid,
        "vehicle_box_valid": vehicle_box_valid,
        "plate_inside_vehicle": plate_inside_vehicle,
        "character_count_matches": character_count_matches,
        "character_boxes_valid": all_character_boxes_valid,
    }

    row["annotation_valid"] = all(
        [
            plate_box_valid,
            vehicle_box_valid,
            character_count_matches,
            all_character_boxes_valid,
        ]
    )

    return row, character_rows


def build_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process all images in all official dataset splits."""
    image_rows: list[dict[str, object]] = []
    character_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, str]] = []

    for split in SPLITS:
        split_directory = DATASET_DIR / split
        image_paths = sorted(split_directory.rglob("*.png"))

        print(f"\nProcessing {split}: {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc=split):
            try:
                image_row, image_characters = process_image(
                    image_path=image_path,
                    split=split,
                )

                image_rows.append(image_row)
                character_rows.extend(image_characters)

            except (
                FileNotFoundError,
                ValueError,
                TypeError,
                UnicodeDecodeError,
            ) as error:
                error_rows.append(
                    {
                        "split": split,
                        "image_path": image_path.as_posix(),
                        "error": str(error),
                    }
                )

    return (
        pd.DataFrame(image_rows),
        pd.DataFrame(character_rows),
        pd.DataFrame(error_rows),
    )


def print_summary(
    image_metadata: pd.DataFrame,
    character_metadata: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    """Display a concise validation summary."""
    print("\nDataset summary")
    print("---------------")
    print(f"Images processed: {len(image_metadata):,}")
    print(f"Characters processed: {len(character_metadata):,}")
    print(f"Processing errors: {len(errors):,}")

    if not image_metadata.empty:
        print("\nImages by split:")
        print(image_metadata["split"].value_counts().to_string())

        print("\nTracks by split:")
        print(
            image_metadata.groupby("split")["track_id"]
            .nunique()
            .to_string()
        )

        print(
            "\nInvalid annotations: "
            f"{(~image_metadata['annotation_valid']).sum():,}"
        )

        print(
            "Character-count mismatches: "
            f"{(~image_metadata['character_count_matches']).sum():,}"
        )


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_metadata, character_metadata, errors = build_metadata()

    image_output = OUTPUT_DIR / "image_metadata.csv"
    character_output = OUTPUT_DIR / "character_annotations.csv"
    error_output = OUTPUT_DIR / "validation_errors.csv"

    image_metadata.to_csv(image_output, index=False)
    character_metadata.to_csv(character_output, index=False)
    errors.to_csv(error_output, index=False)

    print_summary(image_metadata, character_metadata, errors)

    print("\nSaved:")
    print(f"  {image_output.resolve()}")
    print(f"  {character_output.resolve()}")
    print(f"  {error_output.resolve()}")


if __name__ == "__main__":
    main()