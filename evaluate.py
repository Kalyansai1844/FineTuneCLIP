import argparse
import csv
import json
import os
from pathlib import Path


def _load_optional(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        row = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Skipping invalid JSON result file {p}: {exc}")
        return None
    if not isinstance(row, dict):
        print(f"Skipping result file {p}: expected a JSON object.")
        return None
    return row


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def print_table(rows):
    print("\nMethod              Top-1     Top-5")
    print("-----------------------------------")
    for row in rows:
        method = str(row.get("method", "Unknown"))
        top_1 = _as_float(row.get("top_1", 0.0))
        top_5 = _as_float(row.get("top_5", 0.0))
        print(f"{method:<18} {top_1 * 100:6.2f}% {top_5 * 100:7.2f}%")


def write_comparison_csv(rows, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "top_1", "top_5"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": str(row.get("method", "Unknown")),
                    "top_1": _as_float(row.get("top_1", 0.0)),
                    "top_5": _as_float(row.get("top_5", 0.0)),
                }
            )


def _class_names_from_rows(rows) -> list:
    class_names = []
    for row in rows:
        row_class_names = row.get("class_names", [])
        if not isinstance(row_class_names, list):
            row_class_names = []
        for name in row_class_names:
            if name not in class_names:
                class_names.append(name)
    return class_names


def _plot_per_class_with_pillow(rows, class_names, out_path: str) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        print("matplotlib and Pillow are not installed; skipping per-class plot.")
        return False

    width, height = 900, 520
    margin_left, margin_bottom, margin_top = 80, 120, 45
    plot_width = width - margin_left - 30
    plot_height = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = [(50, 100, 200), (30, 150, 95), (210, 120, 40)]

    draw.text((margin_left, 16), "Per-class accuracy", fill="black")
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_height), fill="black")
    draw.line((margin_left, margin_top + plot_height, width - 25, margin_top + plot_height), fill="black")

    group_width = plot_width / max(1, len(class_names))
    bar_width = max(8, int(group_width / max(1, len(rows) + 1)))
    for class_idx, class_name in enumerate(class_names):
        group_x = margin_left + class_idx * group_width
        for row_idx, row in enumerate(rows):
            per_class = row.get("per_class_accuracy") or {}
            if not isinstance(per_class, dict):
                per_class = {}
            score = max(0.0, min(1.0, _as_float(per_class.get(class_name, 0.0))))
            bar_height = int(score * plot_height)
            x0 = int(group_x + row_idx * bar_width + 6)
            y0 = margin_top + plot_height - bar_height
            x1 = x0 + bar_width - 2
            y1 = margin_top + plot_height
            draw.rectangle((x0, y0, x1, y1), fill=colors[row_idx % len(colors)])
        draw.text((int(group_x + 4), margin_top + plot_height + 8), str(class_name)[:18], fill="black")

    legend_x = margin_left
    legend_y = height - 48
    for idx, row in enumerate(rows):
        draw.rectangle((legend_x, legend_y, legend_x + 14, legend_y + 14), fill=colors[idx % len(colors)])
        draw.text((legend_x + 20, legend_y), str(row.get("method", "Unknown")), fill="black")
        legend_x += 190

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return True


def plot_per_class(rows, out_path: str) -> bool:
    class_names = _class_names_from_rows(rows)
    if not class_names:
        print("No per-class data found; skipping plot.")
        return False

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("work/matplotlib").resolve()))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _plot_per_class_with_pillow(rows, class_names, out_path)

    try:
        x = range(len(class_names))
        width = 0.8 / len(rows)
        plt.figure(figsize=(max(10, len(class_names) * 1.2), 5))
        for i, row in enumerate(rows):
            per_class = row.get("per_class_accuracy") or {}
            if not isinstance(per_class, dict):
                per_class = {}
            y = [_as_float(per_class.get(name, 0.0)) * 100 for name in class_names]
            offsets = [v + i * width for v in x]
            plt.bar(offsets, y, width=width, label=str(row.get("method", "Unknown")))
        center_offset = width * (len(rows) - 1) / 2
        plt.xticks([v + center_offset for v in x], class_names, rotation=35, ha="right")
        plt.ylabel("Top-1 accuracy (%)")
        plt.title("Per-class accuracy")
        plt.legend()
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160)
        return True
    except Exception as exc:
        print(f"Matplotlib plot failed ({exc}); using Pillow fallback.")
        return _plot_per_class_with_pillow(rows, class_names, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CLIP prompt tuning variants.")
    parser.add_argument("--baseline", default="artifacts/baseline_results.json")
    parser.add_argument("--enriched", default="artifacts/enriched_results.json")
    parser.add_argument("--adapter", default="artifacts/adapter_results.json")
    parser.add_argument("--plot", default="artifacts/per_class_accuracy.png")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    rows = [row for row in [_load_optional(args.baseline), _load_optional(args.enriched), _load_optional(args.adapter)] if row]
    if not rows:
        raise FileNotFoundError("No result JSON files found. Run baseline_clip.py and enriched_clip.py first.")
    print_table(rows)
    if args.csv:
        write_comparison_csv(rows, args.csv)
        print(f"\nSaved comparison CSV to {args.csv}")
    if plot_per_class(rows, args.plot):
        print(f"\nSaved per-class plot to {args.plot}")


if __name__ == "__main__":
    main()
