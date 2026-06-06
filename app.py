import argparse
from types import SimpleNamespace

import numpy as np

try:
    import gradio as gr
except ModuleNotFoundError:
    gr = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

from clip_utils import encode_images, encode_texts, load_clip, simple_prompts
from data_loader import load_fine_grained_dataset
from enriched_clip import build_enriched_prototypes
from prompt_generator import fallback_prompts, load_or_create_prompts


def load_demo_state(args):
    data = load_fine_grained_dataset(args.dataset, args.num_classes, train_limit=1, test_limit=1)
    model, processor, device = load_clip(args.model)
    prompts = load_or_create_prompts(args.prompts, data.class_names)
    baseline_text = encode_texts(model, processor, simple_prompts(data.class_names), device)
    enriched_text = build_enriched_prototypes(model, processor, data.class_names, prompts, device)
    return SimpleNamespace(
        model=model,
        processor=processor,
        device=device,
        class_names=data.class_names,
        prompts=prompts,
        baseline_text=baseline_text,
        enriched_text=enriched_text,
    )


def _predict_with_prototypes(image_features, prototypes, class_names):
    logits = image_features @ prototypes.T
    if torch is not None and isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits * 100.0, dim=-1)[0]
        idx = int(probs.argmax().item())
        return class_names[idx], float(probs[idx].item()), idx

    logits = np.asarray(logits, dtype=np.float32) * 100.0
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    idx = int(probs[0].argmax().item())
    return class_names[idx], float(probs[0, idx].item()), idx


def build_predict_fn(state):
    def predict(image):
        if image is None:
            return "Upload an image.", "Upload an image.", ""
        try:
            image_features = encode_images(state.model, state.processor, [image], state.device)
            base_name, base_conf, _ = _predict_with_prototypes(image_features, state.baseline_text, state.class_names)
            rich_name, rich_conf, rich_idx = _predict_with_prototypes(image_features, state.enriched_text, state.class_names)
            winning_prompts = state.prompts.get(state.class_names[rich_idx]) or fallback_prompts(state.class_names[rich_idx])
            return (
                f"{base_name} ({base_conf:.1%})",
                f"{rich_name} ({rich_conf:.1%})",
                "\n".join(winning_prompts),
            )
        except Exception as exc:
            return f"Prediction failed: {exc}", f"Prediction failed: {exc}", ""

    return predict


def main() -> None:
    if gr is None:
        raise ImportError("Install 'gradio' to launch the demo app.")

    parser = argparse.ArgumentParser(description="Gradio demo for baseline vs enriched CLIP prompts.")
    parser.add_argument("--dataset", default="tanganke/stanford_cars")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompts", default="artifacts/generated_prompts.json")
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    state = load_demo_state(args)
    predict = build_predict_fn(state)
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Image"),
        outputs=[
            gr.Textbox(label="Baseline CLIP prediction"),
            gr.Textbox(label="Enriched CLIP prediction"),
            gr.Textbox(label="Prompts for enriched winning class", lines=6),
        ],
        title="FineTuneCLIP",
    )
    demo.launch()


if __name__ == "__main__":
    main()
