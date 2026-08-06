import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_analysis.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "reference_stats.json"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction analysis not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "detected_960",
        "top_confidence_960",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if "split" in data.columns:
        data = data.loc[data["split"] == "validation"].copy()

    detected_column = data["detected_960"]

    if detected_column.dtype == bool:
        detected_mask = detected_column
    else:
        detected_mask = (
            detected_column.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    confidence_values = pd.to_numeric(
        data.loc[
            detected_mask,
            "top_confidence_960",
        ],
        errors="coerce",
    ).dropna()

    confidence_values = confidence_values.loc[
        confidence_values.between(0.0, 1.0)
    ]

    if confidence_values.empty:
        raise ValueError(
            "No valid confidence values were found."
        )

    confidence_values = (
        confidence_values
        .sort_values()
        .to_numpy(dtype=float)
    )

    bin_edges = np.linspace(0.0, 1.0, 21)

    bin_counts, bin_edges = np.histogram(
        confidence_values,
        bins=bin_edges,
    )

    output = {
        "detector_confidence": {
            "source": "validation_detected_960",
            "sample_size": int(len(confidence_values)),
            "mean": round(
                float(np.mean(confidence_values)),
                6,
            ),
            "median": round(
                float(np.median(confidence_values)),
                6,
            ),
            "bin_edges": [
                round(float(value), 4)
                for value in bin_edges
            ],
            "bin_counts": [
                int(value)
                for value in bin_counts
            ],
            "sorted_values": [
                round(float(value), 6)
                for value in confidence_values
            ],
        }
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            output,
            output_file,
            indent=2,
        )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Reference images: {len(confidence_values)}")


if __name__ == "__main__":
    main()