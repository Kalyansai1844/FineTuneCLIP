import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(items, **kwargs):
        return items

from clip_utils import argmax_indices, batch_iter, concat_logits, encode_images, encode_texts, l2_normalize, load_clip, mean_embedding, per_class_accuracy, save_json, topk_accuracies
from data_loader import load_fine_grained_dataset
from prompt_generator import fallback_prompts, load_or_create_prompts


def build_enriched_prototypes(model, processor, class_names, prompts_by_class, device):
    prototypes = []
    for class_name in class_names:
        prompts = prompts_by_class.get(class_name) or fallback_prompts(class_name)
        text_features = encode_texts(model, processor, prompts, device)
        prototypes.append(l2_normalize(mean_embedding(text_features)))
    return concat_logits(prototypes)


def run_enriched(args) -> dict:
    data = load_fine_grained_dataset(args.dataset, args.num_classes, args.train_limit, args.test_limit)
    model, processor, device = load_clip(args.model)
    prompts_by_class = load_or_create_prompts(args.prompts, data.class_names)
    prototypes = build_enriched_prototypes(model, processor, data.class_names, prompts_by_class, device)

    all_logits = []
    all_labels = []
    for batch in tqdm(batch_iter(data.test, args.batch_size), desc="Enriched CLIP"):
        images = [row[data.image_column] for row in batch]
        labels = [int(row[data.label_column]) for row in batch]
        image_features = encode_images(model, processor, images, device)
        all_logits.append(image_features @ prototypes.T)
        all_labels.extend(labels)

    if not all_logits:
        raise ValueError("No test examples found. Increase --test-limit or choose a dataset with test rows.")
    logits = concat_logits(all_logits)
    preds = argmax_indices(logits)
    metrics = topk_accuracies(logits, all_labels)
    metrics["per_class_accuracy"] = per_class_accuracy(preds, all_labels, data.class_names)
    metrics["class_names"] = data.class_names
    metrics["method"] = "Enriched Prompts"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot CLIP with LLM-enriched class prototypes.")
    parser.add_argument("--dataset", default="tanganke/stanford_cars")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompts", default="artifacts/generated_prompts.json")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="artifacts/enriched_results.json")
    args = parser.parse_args()

    metrics = run_enriched(args)
    save_json(Path(args.out), metrics)
    print(metrics)


if __name__ == "__main__":
    main()
