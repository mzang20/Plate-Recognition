from pathlib import Path
from collections import Counter


DATASET_DIR = Path("data/raw/ufpr_alpr")
SPLITS = ["training", "validation", "testing"]


def inspect_split(split_name: str) -> None:
    split_dir = DATASET_DIR / split_name

    if not split_dir.exists():
        print(f"\n{split_name}: directory not found")
        return

    files = [path for path in split_dir.rglob("*") if path.is_file()]
    directories = [path for path in split_dir.rglob("*") if path.is_dir()]

    extension_counts = Counter(
        path.suffix.lower() if path.suffix else "<no extension>"
        for path in files
    )

    print(f"\n{split_name.upper()}")
    print("-" * len(split_name))
    print(f"Directories: {len(directories)}")
    print(f"Files: {len(files)}")
    print("Extensions:")

    for extension, count in extension_counts.most_common():
        print(f"  {extension}: {count}")

    print("\nExample files:")

    for path in files[:10]:
        print(f"  {path.relative_to(DATASET_DIR)}")


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR.resolve()}"
        )

    for split in SPLITS:
        inspect_split(split)


if __name__ == "__main__":
    main()