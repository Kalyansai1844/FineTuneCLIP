import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(items, **kwargs):
        return items

from clip_utils import (
    argmax_indices,
    batch_iter,
    concat_logits,
    encode_images,
    encode_texts,
    load_clip,
    per_class_accuracy,
    save_json,
    simple_prompts,
    topk_accuracies,
)
from data_loader import load_fine_grained_dataset


def run_baseline(args) -> dict:
    data = load_fine_grained_dataset(args.dataset, args.num_classes, args.train_limit, args.test_limit)
    model, processor, device = load_clip(args.model)

    text_features = encode_texts(model, processor, simple_prompts(data.class_names), device)

    all_logits = []
    all_labels = []
    for batch in tqdm(batch_iter(data.test, args.batch_size), desc="Baseline CLIP"):
        images = [row[data.image_column] for row in batch]
        labels = [int(row[data.label_column]) for row in batch]
        image_features = encode_images(model, processor, images, device)
        all_logits.append(image_features @ text_features.T)
        all_labels.extend(labels)

    if not all_logits:
        raise ValueError("No test examples found. Increase --test-limit or choose a dataset with test rows.")
    logits = concat_logits(all_logits)
    preds = argmax_indices(logits)
    metrics = topk_accuracies(logits, all_labels)
    metrics["per_class_accuracy"] = per_class_accuracy(preds, all_labels, data.class_names)
    metrics["class_names"] = data.class_names
    metrics["method"] = "Baseline CLIP"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple-prompt zero-shot CLIP classification.")
    parser.add_argument("--dataset", default="tanganke/stanford_cars")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="artifacts/baseline_results.json")
    args = parser.parse_args()

    metrics = run_baseline(args)
    save_json(Path(args.out), metrics)
    print(metrics)


if __name__ == "__main__":
    main()
