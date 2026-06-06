from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None

try:
    from sklearn.metrics import accuracy_score
except ModuleNotFoundError:
    accuracy_score = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ModuleNotFoundError:
    CLIPModel = CLIPProcessor = None


MODEL_NAME = "openai/clip-vit-base-patch32"
SMOKE_MODEL_NAME = "smoke"
SMOKE_COLORS = ("red", "green", "blue", "yellow", "purple", "orange", "cyan", "white", "black", "gray")
SMOKE_RGB = np.array(
    [
        (220, 40, 40),
        (40, 170, 80),
        (50, 90, 220),
        (230, 210, 40),
        (140, 70, 180),
        (230, 120, 40),
        (40, 190, 200),
        (235, 235, 235),
        (35, 35, 35),
        (130, 130, 130),
    ],
    dtype=np.float32,
) / 255.0
SMOKE_DIM = 16


def get_device():
    """Keep the whole project CPU-first by default."""
    if torch is None:
        return "cpu"
    return torch.device("cpu")


def no_grad():
    if torch is not None:
        return torch.no_grad()

    def decorator(fn):
        return fn

    return decorator


class SmokeProcessor:
    """Tiny CLIPProcessor stand-in for offline smoke tests."""

    def __call__(self, images=None, text=None, **kwargs):
        if images is not None:
            values = []
            for image in images:
                arr = np.asarray(ensure_rgb(image)).astype(np.float32) / 255.0
                flat = arr.reshape(-1, 3)
                background = np.array([18, 18, 18], dtype=np.float32) / 255.0
                distances = np.linalg.norm(flat - background, axis=1)
                values.append(flat[int(distances.argmax())])
            array = np.asarray(values, dtype=np.float32)
            if torch is not None:
                return {"pixel_values": torch.tensor(array, dtype=torch.float32)}
            return {"pixel_values": array}
        if text is not None:
            array = np.asarray([_smoke_text_vector(t) for t in text], dtype=np.float32)
            if torch is not None:
                return {"text_features": torch.tensor(array, dtype=torch.float32)}
            return {"text_features": array}
        return {}


class SmokeCLIPModel:
    """Small deterministic model with the same methods this project uses."""

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_image_features(self, pixel_values):
        if torch is not None and isinstance(pixel_values, torch.Tensor):
            features = torch.zeros((pixel_values.shape[0], SMOKE_DIM), dtype=torch.float32)
            palette = torch.tensor(SMOKE_RGB, dtype=torch.float32, device=pixel_values.device)
            distances = torch.cdist(pixel_values[:, :3].float(), palette)
            color_idx = distances.argmin(dim=1)
            for row, idx in enumerate(color_idx.tolist()):
                features[row, idx] = 1.0
            return features

        pixel_values = np.asarray(pixel_values, dtype=np.float32)
        features = np.zeros((pixel_values.shape[0], SMOKE_DIM), dtype=np.float32)
        distances = np.linalg.norm(pixel_values[:, None, :3] - SMOKE_RGB[None, :, :], axis=2)
        color_idx = distances.argmin(axis=1)
        for row, idx in enumerate(color_idx.tolist()):
            features[row, idx] = 1.0
        return features

    def get_text_features(self, text_features):
        return text_features


def _smoke_text_vector(text: str) -> List[float]:
    lowered = text.lower()
    vector = [0.0] * SMOKE_DIM
    for idx, color in enumerate(SMOKE_COLORS):
        if color in lowered:
            vector[idx] = 1.0
            break
    else:
        digest = hashlib.md5(lowered.encode("utf-8")).digest()
        vector[digest[0] % min(len(SMOKE_COLORS), SMOKE_DIM)] = 1.0
    return vector


def load_clip(model_name: str = MODEL_NAME) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = get_device()
    if model_name == SMOKE_MODEL_NAME:
        return SmokeCLIPModel().to(device).eval(), SmokeProcessor(), device
    if torch is None or CLIPModel is None or CLIPProcessor is None:
        raise ImportError("Install 'torch' and 'transformers' to load real CLIP models, or use --model smoke.")
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def ensure_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    if torch is not None and isinstance(x, torch.Tensor):
        return F.normalize(x, p=2, dim=-1)
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def to_device(value, device):
    return value.to(device) if hasattr(value, "to") else value


def concat_logits(chunks: Sequence):
    if not chunks:
        raise ValueError("No logits to concatenate.")
    if torch is not None and isinstance(chunks[0], torch.Tensor):
        return torch.cat(chunks, dim=0)
    return np.concatenate(chunks, axis=0)


def argmax_indices(logits) -> List[int]:
    if torch is not None and isinstance(logits, torch.Tensor):
        return logits.argmax(dim=1).tolist()
    return np.asarray(logits).argmax(axis=1).tolist()


def mean_embedding(features):
    if torch is not None and isinstance(features, torch.Tensor):
        return features.mean(dim=0, keepdim=True)
    return np.asarray(features).mean(axis=0, keepdims=True)


def is_torch_available() -> bool:
    return torch is not None


def l2_normalize_torch_only(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _extract_features(output):
    """Extract tensor from model output, handling transformers 5.x BaseModelOutputWithPooling.

    In transformers 5.x, get_text_features/get_image_features apply the
    projection internally but wrap the result in BaseModelOutputWithPooling
    instead of returning a plain tensor.
    """
    if torch is not None and isinstance(output, torch.Tensor):
        return output
    # numpy array (smoke model path)
    if isinstance(output, np.ndarray):
        return output
    # transformers 5.x returns BaseModelOutputWithPooling
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    raise TypeError(f"Cannot extract features from {type(output)}")


@no_grad()
def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: Sequence[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(images=[ensure_rgb(img) for img in images], return_tensors="pt")
    inputs = {k: to_device(v, device) for k, v in inputs.items()}
    raw = model.get_image_features(**inputs)
    return l2_normalize(_extract_features(raw))


@no_grad()
def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(text=list(texts), padding=True, truncation=True, return_tensors="pt")
    inputs = {k: to_device(v, device) for k, v in inputs.items()}
    raw = model.get_text_features(**inputs)
    return l2_normalize(_extract_features(raw))


def batch_iter(items: Sequence, batch_size: int) -> Iterable[List[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        if isinstance(chunk, dict):
            keys = list(chunk.keys())
            size = len(chunk[keys[0]]) if keys else 0
            yield [{key: chunk[key][i] for key in keys} for i in range(size)]
        else:
            yield chunk


def topk_accuracies(logits: torch.Tensor, labels: Sequence[int], topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    if torch is not None and isinstance(logits, torch.Tensor):
        labels_t = torch.tensor(labels, dtype=torch.long)
        max_k = min(max(topk), logits.shape[1])
        pred = logits.topk(max_k, dim=1).indices
        results = {}
        for k in topk:
            kk = min(k, logits.shape[1])
            correct = pred[:, :kk].eq(labels_t.unsqueeze(1)).any(dim=1).float().mean().item()
            results[f"top_{k}"] = correct
        return results

    logits = np.asarray(logits)
    labels_np = np.asarray(labels)
    max_k = min(max(topk), logits.shape[1])
    pred = np.argsort(-logits, axis=1)[:, :max_k]
    results = {}
    for k in topk:
        kk = min(k, logits.shape[1])
        correct = (pred[:, :kk] == labels_np[:, None]).any(axis=1).mean().item()
        results[f"top_{k}"] = correct
    return results


def per_class_accuracy(preds: Sequence[int], labels: Sequence[int], class_names: Sequence[str]) -> Dict[str, float]:
    scores = {}
    for idx, name in enumerate(class_names):
        mask = [i for i, label in enumerate(labels) if label == idx]
        if not mask:
            scores[name] = 0.0
            continue
        y_true = [labels[i] for i in mask]
        y_pred = [preds[i] for i in mask]
        if accuracy_score is None:
            scores[name] = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
        else:
            scores[name] = accuracy_score(y_true, y_pred)
    return scores


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def simple_prompts(class_names: Sequence[str]) -> List[str]:
    return [f"a photo of a {name}" for name in class_names]
