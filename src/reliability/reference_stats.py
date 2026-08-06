import json
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STATS_PATH = (
    PROJECT_ROOT
    / "models"
    / "reference_stats.json"
)


class ReferenceStats:
    def __init__(
        self,
        stats_path: Union[str, Path] = DEFAULT_STATS_PATH,
    ) -> None:
        self.stats_path = Path(stats_path).resolve()

        if not self.stats_path.exists():
            raise FileNotFoundError(
                f"Reference statistics not found: "
                f"{self.stats_path}"
            )

        with self.stats_path.open(
            "r",
            encoding="utf-8",
        ) as stats_file:
            data = json.load(stats_file)

        confidence_data = data["detector_confidence"]

        self.source = confidence_data["source"]
        self.sample_size = int(
            confidence_data["sample_size"]
        )
        self.mean = float(confidence_data["mean"])
        self.median = float(confidence_data["median"])

        self.bin_edges: List[float] = [
            float(value)
            for value in confidence_data["bin_edges"]
        ]

        self.bin_counts: List[int] = [
            int(value)
            for value in confidence_data["bin_counts"]
        ]

        self.sorted_values: List[float] = [
            float(value)
            for value in confidence_data["sorted_values"]
        ]

    def confidence_percentile(
        self,
        confidence: Optional[float],
    ) -> Optional[float]:
        if confidence is None or not self.sorted_values:
            return None

        rank = bisect_right(
            self.sorted_values,
            float(confidence),
        )

        return round(
            rank / len(self.sorted_values) * 100.0,
            1,
        )

    def to_response(self) -> Dict[str, object]:
        return {
            "detector_confidence": {
                "source": self.source,
                "sample_size": self.sample_size,
                "mean": self.mean,
                "median": self.median,
                "bin_edges": self.bin_edges,
                "bin_counts": self.bin_counts,
            }
        }