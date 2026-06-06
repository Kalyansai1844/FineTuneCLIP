import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

try:
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage, load_dataset
except ModuleNotFoundError:
    ClassLabel = Dataset = DatasetDict = Features = HFImage = load_dataset = None

DEFAULT_DATASET = "tanganke/stanford_cars"
SMOKE_DATASET_NAME = "smoke"
SMOKE_CLASSES = [
    ("red square", (220, 40, 40), "rectangle"),
    ("green circle", (40, 170, 80), "ellipse"),
    ("blue triangle", (50, 90, 220), "polygon"),
    ("yellow rectangle", (230, 210, 40), "rectangle"),
    ("purple circle", (140, 70, 180), "ellipse"),
    ("orange triangle", (230, 120, 40), "polygon"),
    ("cyan square", (40, 190, 200), "rectangle"),
    ("white circle", (235, 235, 235), "ellipse"),
    ("black rectangle", (35, 35, 35), "rectangle"),
    ("gray triangle", (130, 130, 130), "polygon"),
]


@dataclass
class FineGrainedData:
    train: Dataset
    test: Dataset
    class_names: List[str]
    label_column: str
    image_column: str


class MiniDataset:
    """List-backed dataset used only for dependency-free smoke tests."""

    def __init__(self, rows: List[dict]):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.rows[item]
        if isinstance(item, str):
            return [row[item] for row in self.rows]
        return self.rows[item]

    def select(self, indices) -> "MiniDataset":
        return MiniDataset([self.rows[i] for i in indices])


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_column(dataset: Dataset, preferred: List[str], fallback_kind: str) -> str:
    for name in preferred:
        if name in dataset.column_names:
            return name
    for name, feature in dataset.features.items():
        if fallback_kind in feature.__class__.__name__.lower():
            return name
    raise ValueError(f"Could not infer a {fallback_kind} column. Columns: {dataset.column_names}")


def _class_names(dataset: Dataset, label_column: str) -> List[str]:
    feature = dataset.features[label_column]
    if hasattr(feature, "names") and feature.names:
        return list(feature.names)
    unique = sorted(set(dataset[label_column]))
    return [str(x) for x in unique]


def _split_dataset(ds: DatasetDict) -> Tuple[Dataset, Dataset]:
    if "train" in ds and "test" in ds:
        return ds["train"], ds["test"]
    if "train" in ds and "validation" in ds:
        return ds["train"], ds["validation"]
    first = ds[list(ds.keys())[0]]
    split = first.train_test_split(test_size=0.25, seed=42, stratify_by_column=None)
    return split["train"], split["test"]


def _choose_classes(class_names: List[str], num_classes: int) -> List[int]:
    return list(range(min(num_classes, len(class_names))))


def _filter_and_remap(dataset: Dataset, label_column: str, keep_ids: List[int]) -> Dataset:
    id_map: Dict[int, int] = {old: new for new, old in enumerate(keep_ids)}
    filtered = dataset.filter(lambda row: int(row[label_column]) in id_map)
    return filtered.map(lambda row: {"label": id_map[int(row[label_column])]})


def _smoke_image(color: Tuple[int, int, int], shape: str, offset: int) -> Image.Image:
    image = Image.new("RGB", (64, 64), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    pad = 10 + (offset % 4)
    if shape == "ellipse":
        draw.ellipse((pad, pad, 64 - pad, 64 - pad), fill=color)
    elif shape == "polygon":
        draw.polygon(((32, pad), (64 - pad, 64 - pad), (pad, 64 - pad)), fill=color)
    else:
        draw.rectangle((pad, pad, 64 - pad, 64 - pad), fill=color)
    return image


def _make_smoke_split(class_defs, per_class: int) -> Dataset:
    rows = []
    for label, (_, color, shape) in enumerate(class_defs):
        for offset in range(per_class):
            rows.append({"image": _smoke_image(color, shape, offset), "label": label})
    return MiniDataset(rows)


def _load_smoke_dataset(num_classes: int, train_limit: int | None, test_limit: int | None) -> FineGrainedData:
    class_defs = SMOKE_CLASSES[: max(1, min(num_classes, len(SMOKE_CLASSES)))]
    train = _make_smoke_split(class_defs, per_class=3)
    test = _make_smoke_split(class_defs, per_class=2)
    if train_limit is not None:
        train = train.select(range(min(train_limit, len(train))))
    if test_limit is not None:
        test = test.select(range(min(test_limit, len(test))))
    return FineGrainedData(
        train=train,
        test=test,
        class_names=[name for name, _, _ in class_defs],
        label_column="label",
        image_column="image",
    )


def load_fine_grained_dataset(
    dataset_name: str = DEFAULT_DATASET,
    num_classes: int = 10,
    train_limit: int | None = 500,
    test_limit: int | None = 200,
) -> FineGrainedData:
    """Load a small, CPU-friendly fine-grained image classification subset."""
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")

    if dataset_name.lower() in {SMOKE_DATASET_NAME, "synthetic", "tiny"}:
        return _load_smoke_dataset(num_classes, train_limit, test_limit)

    if load_dataset is None:
        raise ImportError("Install the 'datasets' package to load HuggingFace datasets, or use --dataset smoke.")

    raw = load_dataset(dataset_name)
    train, test = _split_dataset(raw)
    label_column = _find_column(train, ["label", "labels", "class", "fine_label"], "classlabel")
    image_column = _find_column(train, ["image", "img"], "image")

    all_class_names = _class_names(train, label_column)
    keep_ids = _choose_classes(all_class_names, num_classes)
    class_names = [all_class_names[i].replace("_", " ") for i in keep_ids]

    train = _filter_and_remap(train, label_column, keep_ids)
    test = _filter_and_remap(test, label_column, keep_ids)

    if train_limit is not None:
        train = train.shuffle(seed=42).select(range(min(train_limit, len(train))))
    if test_limit is not None:
        test = test.shuffle(seed=123).select(range(min(test_limit, len(test))))

    return FineGrainedData(train=train, test=test, class_names=class_names, label_column="label", image_column=image_column)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare a 10-class fine-grained dataset subset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--metadata-out", default="artifacts/dataset_metadata.json")
    args = parser.parse_args()

    data = load_fine_grained_dataset(args.dataset, args.num_classes, args.train_limit, args.test_limit)
    payload = {
        "dataset": args.dataset,
        "num_train": len(data.train),
        "num_test": len(data.test),
        "class_names": data.class_names,
        "image_column": data.image_column,
        "label_column": data.label_column,
    }
    save_json(Path(args.metadata_out), payload)
    print(payload)


if __name__ == "__main__":
    main()
