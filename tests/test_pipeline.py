from pathlib import Path

from src.pipeline.inference_pipeline import (
    PlateRecognitionPipeline,
)


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


def main() -> None:
    pipeline = PlateRecognitionPipeline()

    result = pipeline.recognize_from_path(
        TEST_IMAGE_PATH
    )

    print("Model information:")
    print(pipeline.get_model_info())

    print("\nPrediction:")
    print(result.to_dict())


if __name__ == "__main__":
    main()