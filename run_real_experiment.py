"""
End-to-end real CLIP experiment on CIFAR-10.

Runs baseline, enriched, and adapter pipelines with actual CLIP model
on real images, then generates comparison artifacts.
"""
import argparse
import json
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run full real CLIP experiment on CIFAR-10")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--dataset", default="uoft-cs/cifar10")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset
    model_name = "openai/clip-vit-base-patch32"
    prompts_file = str(out / "real_prompts.json")

    # ── Step 1: Load dataset ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Loading CIFAR-10 dataset")
    print("=" * 60)
    t0 = time.time()
    from data_loader import load_fine_grained_dataset, save_json
    data = load_fine_grained_dataset(
        dataset_name, args.num_classes, args.train_limit, args.test_limit
    )
    print(f"  Classes: {data.class_names}")
    print(f"  Train samples: {len(data.train)}")
    print(f"  Test samples:  {len(data.test)}")
    print(f"  Image column:  {data.image_column}")
    print(f"  Label column:  {data.label_column}")
    save_json(out / "real_dataset_metadata.json", {
        "dataset": dataset_name,
        "num_train": len(data.train),
        "num_test": len(data.test),
        "class_names": data.class_names,
    })
    print(f"  Dataset loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Load CLIP model ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading CLIP model")
    print("=" * 60)
    t0 = time.time()
    from clip_utils import (
        load_clip, encode_images, encode_texts, simple_prompts,
        l2_normalize, mean_embedding, concat_logits, argmax_indices,
        topk_accuracies, per_class_accuracy, batch_iter, save_json as save_json2
    )
    model, processor, device = load_clip(model_name)
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ── Step 3: Baseline CLIP ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Running Baseline CLIP (simple prompts)")
    print("=" * 60)
    t0 = time.time()
    baseline_prompts = simple_prompts(data.class_names)
    print(f"  Prompts: {baseline_prompts[:3]}...")
    text_features = encode_texts(model, processor, baseline_prompts, device)

    all_logits = []
    all_labels = []
    n_batches = 0
    for batch in batch_iter(data.test, args.batch_size):
        images = [row[data.image_column] for row in batch]
        labels = [int(row[data.label_column]) for row in batch]
        image_features = encode_images(model, processor, images, device)
        all_logits.append(image_features @ text_features.T)
        all_labels.extend(labels)
        n_batches += 1
        if n_batches % 5 == 0:
            print(f"    Processed {n_batches} batches ({len(all_labels)} images)...")

    logits = concat_logits(all_logits)
    preds = argmax_indices(logits)
    baseline_metrics = topk_accuracies(logits, all_labels)
    baseline_metrics["per_class_accuracy"] = per_class_accuracy(preds, all_labels, data.class_names)
    baseline_metrics["class_names"] = data.class_names
    baseline_metrics["method"] = "Baseline CLIP"
    save_json2(out / "real_baseline.json", baseline_metrics)
    print(f"  Baseline Top-1: {baseline_metrics['top_1'] * 100:.2f}%")
    print(f"  Baseline Top-5: {baseline_metrics['top_5'] * 100:.2f}%")
    print(f"  Baseline done in {time.time() - t0:.1f}s")

    # ── Step 4: Enriched CLIP ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Running Enriched CLIP (hand-crafted visual prompts)")
    print("=" * 60)
    t0 = time.time()

    # Load hand-crafted prompts
    prompts_path = Path(prompts_file)
    if prompts_path.exists():
        prompts_by_class = json.loads(prompts_path.read_text(encoding="utf-8"))
        print(f"  Loaded prompts from {prompts_file}")
    else:
        print(f"  WARNING: {prompts_file} not found, using fallback")
        from prompt_generator import fallback_prompts
        prompts_by_class = {name: fallback_prompts(name) for name in data.class_names}

    # Build enriched prototypes
    prototypes = []
    for class_name in data.class_names:
        class_prompts = prompts_by_class.get(class_name, [])
        if not class_prompts:
            from prompt_generator import fallback_prompts
            class_prompts = fallback_prompts(class_name)
        print(f"  {class_name}: {len(class_prompts)} prompts")
        text_feat = encode_texts(model, processor, class_prompts, device)
        prototypes.append(l2_normalize(mean_embedding(text_feat)))
    enriched_text_features = concat_logits(prototypes)

    all_logits = []
    all_labels = []
    n_batches = 0
    for batch in batch_iter(data.test, args.batch_size):
        images = [row[data.image_column] for row in batch]
        labels = [int(row[data.label_column]) for row in batch]
        image_features = encode_images(model, processor, images, device)
        all_logits.append(image_features @ enriched_text_features.T)
        all_labels.extend(labels)
        n_batches += 1
        if n_batches % 5 == 0:
            print(f"    Processed {n_batches} batches ({len(all_labels)} images)...")

    logits = concat_logits(all_logits)
    preds = argmax_indices(logits)
    enriched_metrics = topk_accuracies(logits, all_labels)
    enriched_metrics["per_class_accuracy"] = per_class_accuracy(preds, all_labels, data.class_names)
    enriched_metrics["class_names"] = data.class_names
    enriched_metrics["method"] = "Enriched Prompts"
    save_json2(out / "real_enriched.json", enriched_metrics)
    print(f"  Enriched Top-1: {enriched_metrics['top_1'] * 100:.2f}%")
    print(f"  Enriched Top-5: {enriched_metrics['top_5'] * 100:.2f}%")
    print(f"  Enriched done in {time.time() - t0:.1f}s")

    # ── Step 5: Prompt Adapter ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Training & evaluating Prompt Adapter")
    print("=" * 60)
    t0 = time.time()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Build training data from prompt text embeddings
    xs, ys = [], []
    for label, class_name in enumerate(data.class_names):
        class_prompts = prompts_by_class.get(class_name, [])
        if not class_prompts:
            from prompt_generator import fallback_prompts
            class_prompts = fallback_prompts(class_name)
        xs.append(encode_texts(model, processor, class_prompts, device))
        ys.extend([label] * len(class_prompts))
    prompt_x = torch.cat(xs, dim=0)
    prompt_y = torch.tensor(ys, dtype=torch.long)
    dim = prompt_x.shape[1]
    print(f"  Training data: {prompt_x.shape[0]} prompt embeddings, dim={dim}")

    # Build adapter
    torch.manual_seed(0)
    from prompt_adapter import PromptAdapter
    adapter = PromptAdapter(dim, 256).to(device)
    classifier = nn.Linear(dim, len(data.class_names), bias=False).to(device)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(classifier.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        TensorDataset(prompt_x, prompt_y),
        batch_size=16, shuffle=True, generator=generator
    )

    # Train
    adapter.train()
    classifier.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_batch = classifier(adapter(xb))
            loss = loss_fn(logits_batch, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{args.epochs}, loss={epoch_loss / n:.4f}")

    # Evaluate adapter
    adapter.eval()
    classifier.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in batch_iter(data.test, args.batch_size):
            images = [row[data.image_column] for row in batch]
            labels = [int(row[data.label_column]) for row in batch]
            image_features = encode_images(model, processor, images, device)
            adapted = adapter(image_features)
            all_logits.append(classifier(adapted))
            all_labels.extend(labels)

    logits = torch.cat(all_logits, dim=0)
    preds = logits.argmax(dim=1).tolist()
    adapter_metrics = topk_accuracies(logits, all_labels)
    adapter_metrics["per_class_accuracy"] = per_class_accuracy(preds, all_labels, data.class_names)
    adapter_metrics["class_names"] = data.class_names
    adapter_metrics["method"] = "Prompt Adapter"
    save_json2(out / "real_adapter.json", adapter_metrics)
    torch.save({
        "adapter": adapter.state_dict(),
        "classifier": classifier.state_dict(),
        "class_names": data.class_names,
    }, out / "real_prompt_adapter.pt")
    print(f"  Adapter Top-1: {adapter_metrics['top_1'] * 100:.2f}%")
    print(f"  Adapter Top-5: {adapter_metrics['top_5'] * 100:.2f}%")
    print(f"  Adapter done in {time.time() - t0:.1f}s")

    # ── Step 6: Evaluation & Comparison ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Generating comparison table, CSV, and plot")
    print("=" * 60)

    rows = [baseline_metrics, enriched_metrics, adapter_metrics]

    # Print table
    print("\n  Method              Top-1     Top-5")
    print("  -----------------------------------")
    for row in rows:
        method = row["method"]
        t1 = row["top_1"] * 100
        t5 = row["top_5"] * 100
        print(f"  {method:<18} {t1:6.2f}% {t5:7.2f}%")

    # Print per-class accuracy
    print("\n  Per-class Top-1 accuracy:")
    for cls in data.class_names:
        baseline_acc = baseline_metrics["per_class_accuracy"].get(cls, 0) * 100
        enriched_acc = enriched_metrics["per_class_accuracy"].get(cls, 0) * 100
        adapter_acc = adapter_metrics["per_class_accuracy"].get(cls, 0) * 100
        delta = enriched_acc - baseline_acc
        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"    {cls:<14} Baseline:{baseline_acc:5.1f}%  Enriched:{enriched_acc:5.1f}% ({marker}{abs(delta):+.1f}%)  Adapter:{adapter_acc:5.1f}%")

    # CSV
    import csv
    csv_path = out / "real_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "top_1", "top_5"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"method": row["method"], "top_1": row["top_1"], "top_5": row["top_5"]})
    print(f"\n  Saved CSV to {csv_path}")

    # Plot
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(Path("work/matplotlib").resolve()))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        class_names = data.class_names
        x = range(len(class_names))
        width = 0.8 / len(rows)
        fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 1.2), 6))
        colors_list = ["#4A90D9", "#50C878", "#E8833A"]
        for i, row in enumerate(rows):
            per_class = row.get("per_class_accuracy", {})
            y = [per_class.get(name, 0.0) * 100 for name in class_names]
            offsets = [v + i * width for v in x]
            ax.bar(offsets, y, width=width, label=row["method"], color=colors_list[i])

        center_offset = width * (len(rows) - 1) / 2
        ax.set_xticks([v + center_offset for v in x])
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("CIFAR-10: Per-Class Accuracy — Baseline vs Enriched vs Adapter")
        ax.legend()
        ax.set_ylim(0, 105)
        plt.tight_layout()
        plot_path = out / "real_per_class_accuracy.png"
        plt.savefig(str(plot_path), dpi=160)
        plt.close()
        print(f"  Saved plot to {plot_path}")
    except Exception as exc:
        print(f"  Plot failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    enriched_delta = (enriched_metrics["top_1"] - baseline_metrics["top_1"]) * 100
    adapter_delta = (adapter_metrics["top_1"] - baseline_metrics["top_1"]) * 100
    print(f"  Baseline  Top-1: {baseline_metrics['top_1'] * 100:.2f}%")
    print(f"  Enriched  Top-1: {enriched_metrics['top_1'] * 100:.2f}% (delta: {enriched_delta:+.2f}%)")
    print(f"  Adapter   Top-1: {adapter_metrics['top_1'] * 100:.2f}% (delta: {adapter_delta:+.2f}%)")
    if enriched_delta > 0:
        print("  [+] Enriched prompts IMPROVED over baseline")
    elif enriched_delta == 0:
        print("  [=] Enriched prompts matched baseline")
    else:
        print("  [-] Enriched prompts did NOT improve over baseline")
    if adapter_delta > 0:
        print("  [+] Prompt adapter IMPROVED over baseline")
    elif adapter_delta == 0:
        print("  [=] Prompt adapter matched baseline")
    else:
        print("  [-] Prompt adapter did NOT improve over baseline")


if __name__ == "__main__":
    main()
