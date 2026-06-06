import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

from data_loader import load_fine_grained_dataset


PROMPT_TEMPLATE = """Generate 5 diverse, descriptive sentences about the visual appearance of a {class_name}.
Focus on color, shape, texture, distinguishing features.
Return only the sentences, one per line."""


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fallback_prompts(class_name: str) -> List[str]:
    return [
        f"A clear photo showing the distinctive shape and proportions of a {class_name}.",
        f"A close view of a {class_name} with visible color, texture, and surface details.",
        f"The {class_name} appears with its typical silhouette and identifying visual features.",
        f"A natural image of a {class_name} highlighting fine-grained patterns and markings.",
        f"A well-lit photograph of a {class_name} emphasizing features that separate it from similar classes.",
    ]


def _clean_lines(text: str, class_name: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        if line:
            lines.append(line)
    return (lines + fallback_prompts(class_name))[:5]


def normalize_prompts(prompts_by_class, class_names: Sequence[str]) -> dict:
    """Return exactly five non-empty prompts per requested class."""
    if not isinstance(prompts_by_class, dict):
        prompts_by_class = {}

    normalized = {}
    for class_name in class_names:
        raw = prompts_by_class.get(class_name)
        if isinstance(raw, str):
            prompts = _clean_lines(raw, class_name)
        elif isinstance(raw, list):
            prompts = [str(item).strip() for item in raw if str(item).strip()]
            prompts = (prompts + fallback_prompts(class_name))[:5]
        else:
            prompts = fallback_prompts(class_name)
        normalized[class_name] = prompts
    return normalized


def load_or_create_prompts(path: str | Path, class_names: Sequence[str]) -> dict:
    path = Path(path)
    try:
        prompts_by_class = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        prompts_by_class = {}

    prompts = normalize_prompts(prompts_by_class, class_names)
    if prompts != prompts_by_class:
        save_json(path, prompts)
    return prompts


def generate_with_openai(class_name: str, model: str) -> List[str]:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(class_name=class_name)}],
        temperature=0.8,
    )
    return _clean_lines(response.choices[0].message.content or "", class_name)


def generate_with_ollama(class_name: str, model: str) -> List[str]:
    import ollama

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(class_name=class_name)}],
    )
    return _clean_lines(response["message"]["content"], class_name)


def generate_prompts(class_names: List[str], provider: str, model: str) -> dict:
    prompts = {}
    for class_name in class_names:
        try:
            if provider == "openai":
                prompts[class_name] = generate_with_openai(class_name, model)
            elif provider == "ollama":
                prompts[class_name] = generate_with_ollama(class_name, model)
            else:
                prompts[class_name] = fallback_prompts(class_name)
        except Exception as exc:
            print(f"Falling back for {class_name}: {exc}")
            prompts[class_name] = fallback_prompts(class_name)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM-enriched visual prompts for each class.")
    parser.add_argument("--dataset", default="tanganke/stanford_cars")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--provider", choices=["openai", "ollama", "fallback"], default=os.getenv("PROMPT_PROVIDER", "fallback"))
    parser.add_argument("--model", default=os.getenv("PROMPT_MODEL", "gpt-3.5-turbo"))
    parser.add_argument("--out", default="artifacts/generated_prompts.json")
    args = parser.parse_args()

    data = load_fine_grained_dataset(args.dataset, args.num_classes, train_limit=1, test_limit=1)
    prompts = normalize_prompts(generate_prompts(data.class_names, args.provider, args.model), data.class_names)
    save_json(Path(args.out), prompts)
    print(f"Saved prompts for {len(prompts)} classes to {args.out}")


if __name__ == "__main__":
    main()
