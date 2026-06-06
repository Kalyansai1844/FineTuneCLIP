import argparse
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = nn = DataLoader = TensorDataset = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(items, **kwargs):
        return items

from clip_utils import argmax_indices, batch_iter, concat_logits, encode_images, encode_texts, l2_normalize, load_clip, per_class_accuracy, save_json, topk_accuracies
from data_loader import load_fine_grained_dataset
from prompt_generator import fallback_prompts, load_or_create_prompts


if nn is not None:
    class PromptAdapter(nn.Module):
        def __init__(self, dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return l2_normalize(self.net(x))
else:
    class PromptAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install 'torch' to train the prompt adapter with real CLIP models.")


def _prompt_training_tensors(model, processor, class_names, prompts_by_class, device):
    xs, ys = [], []
    for label, class_name in enumerate(class_names):
        prompts = prompts_by_class.get(class_name) or fallback_prompts(class_name)
        xs.append(encode_texts(model, processor, prompts, device))
        ys.extend([label] * len(prompts))
    return torch.cat(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def _run_smoke_adapter_without_torch(args) -> dict:
    from enriched_clip import build_enriched_prototypes

    data = load_fine_grained_dataset(args.dataset, args.num_classes, args.train_limit, args.test_limit)
    model, processor, device = load_clip(args.model)
    prompts_by_class = load_or_create_prompts(args.prompts, data.class_names)
    prototypes = build_enriched_prototypes(model, processor, data.class_names, prompts_by_class, device)

    all_logits, all_labels = [], []
    for batch in tqdm(batch_iter(data.test, args.batch_size), desc="Adapter smoke eval"):
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
    metrics["method"] = "Prompt Adapter"

    checkpoint = Path(args.checkpoint)
    if checkpoint.parent != Path("."):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("Smoke adapter fallback: no trainable torch checkpoint was created.\n", encoding="utf-8")
    return metrics


def run_adapter(args) -> dict:
    if args.prompt_batch_size < 1:
        raise ValueError("prompt_batch_size must be >= 1")
    if torch is None:
        if args.model == "smoke":
            return _run_smoke_adapter_without_torch(args)
        raise ImportError("Install 'torch' to train the prompt adapter with real CLIP models.")
    torch.manual_seed(0)

    data = load_fine_grained_dataset(args.dataset, args.num_classes, args.train_limit, args.test_limit)
    model, processor, device = load_clip(args.model)
    prompts_by_class = load_or_create_prompts(args.prompts, data.class_names)

    prompt_x, prompt_y = _prompt_training_tensors(model, processor, data.class_names, prompts_by_class, device)
    dim = prompt_x.shape[1]
    adapter = PromptAdapter(dim, args.hidden_dim).to(device)
    classifier = nn.Linear(dim, len(data.class_names), bias=False).to(device)
    optimizer = torch.optim.AdamW(list(adapter.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(prompt_x, prompt_y), batch_size=args.prompt_batch_size, shuffle=True, generator=generator)

    adapter.train()
    classifier.train()
    for _ in tqdm(range(args.epochs), desc="Training adapter"):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = classifier(adapter(xb))
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    adapter.eval()
    classifier.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(batch_iter(data.test, args.batch_size), desc="Adapter eval"):
            images = [row[data.image_column] for row in batch]
            labels = [int(row[data.label_column]) for row in batch]
            image_features = encode_images(model, processor, images, device)
            adapted = adapter(image_features)
            all_logits.append(classifier(adapted))
            all_labels.extend(labels)

    if not all_logits:
        raise ValueError("No test examples found. Increase --test-limit or choose a dataset with test rows.")
    logits = torch.cat(all_logits, dim=0)
    preds = logits.argmax(dim=1).tolist()
    metrics = topk_accuracies(logits, all_labels)
    metrics["per_class_accuracy"] = per_class_accuracy(preds, all_labels, data.class_names)
    metrics["class_names"] = data.class_names
    metrics["method"] = "Prompt Adapter"
    checkpoint = Path(args.checkpoint)
    if checkpoint.parent != Path("."):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "classifier": classifier.state_dict(), "class_names": data.class_names}, checkpoint)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny CPU-friendly prompt adapter.")
    parser.add_argument("--dataset", default="tanganke/stanford_cars")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompts", default="artifacts/generated_prompts.json")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="artifacts/prompt_adapter.pt")
    parser.add_argument("--out", default="artifacts/adapter_results.json")
    args = parser.parse_args()

    metrics = run_adapter(args)
    save_json(Path(args.out), metrics)
    print(metrics)


if __name__ == "__main__":
    main()
