# FineTuneCLIP

This project improves CLIP zero-shot classification on fine-grained categories by replacing the plain prompt `a photo of a {class_name}` with LLM-generated visual descriptions. It also includes a small optional prompt adapter trained on CLIP text embeddings.

The default real-data setup uses `openai/clip-vit-base-patch32`, CPU-only PyTorch, and a 10-class subset of a Stanford Cars dataset from HuggingFace. For reproducible offline checks, use `--dataset smoke --model smoke`; this runs a tiny synthetic image dataset and a deterministic CLIP-compatible stub without HuggingFace or LLM access.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No CUDA is required. The code always uses `torch.device("cpu")`.

## Offline smoke test

Use this path first to verify the project end to end on CPU without downloading models or datasets:

```bash
python data_loader.py --dataset smoke --num-classes 3 --train-limit 6 --test-limit 6
python baseline_clip.py --dataset smoke --model smoke --num-classes 3 --test-limit 6 --batch-size 2
python prompt_generator.py --dataset smoke --provider fallback --num-classes 3
python enriched_clip.py --dataset smoke --model smoke --num-classes 3 --test-limit 6 --batch-size 2
python prompt_adapter.py --dataset smoke --model smoke --num-classes 3 --test-limit 6 --epochs 5
python evaluate.py
```

Launch the demo in offline mode:

```bash
python app.py --dataset smoke --model smoke --num-classes 3
```

## Reproducible Results

These commands write the expected experiment artifacts under `results/`.

Smoke dataset and smoke model:

```bash
mkdir results
mkdir results\sample_predictions
python data_loader.py --dataset smoke --num-classes 3 --train-limit 6 --test-limit 6 --metadata-out results/dataset_metadata.json
python prompt_generator.py --dataset smoke --provider fallback --num-classes 3 --out results/generated_prompts.json
python baseline_clip.py --dataset smoke --model smoke --num-classes 3 --test-limit 6 --batch-size 2 --out results/baseline.json
python enriched_clip.py --dataset smoke --model smoke --prompts results/generated_prompts.json --num-classes 3 --test-limit 6 --batch-size 2 --out results/enriched.json
python prompt_adapter.py --dataset smoke --model smoke --prompts results/generated_prompts.json --num-classes 3 --test-limit 6 --batch-size 2 --epochs 5 --checkpoint results/prompt_adapter.pt --out results/adapter.json
python evaluate.py --baseline results/baseline.json --enriched results/enriched.json --adapter results/adapter.json --csv results/comparison.csv --plot results/per_class_accuracy.png
```

Small real dataset run using Stanford Cars and fallback prompts:

```bash
mkdir results
mkdir results\sample_predictions
python data_loader.py --dataset tanganke/stanford_cars --num-classes 3 --train-limit 60 --test-limit 30 --metadata-out results/dataset_metadata.json
python prompt_generator.py --dataset tanganke/stanford_cars --provider fallback --num-classes 3 --out results/generated_prompts.json
python baseline_clip.py --dataset tanganke/stanford_cars --model openai/clip-vit-base-patch32 --num-classes 3 --train-limit 60 --test-limit 30 --batch-size 8 --out results/baseline.json
python enriched_clip.py --dataset tanganke/stanford_cars --model openai/clip-vit-base-patch32 --prompts results/generated_prompts.json --num-classes 3 --train-limit 60 --test-limit 30 --batch-size 8 --out results/enriched.json
python prompt_adapter.py --dataset tanganke/stanford_cars --model openai/clip-vit-base-patch32 --prompts results/generated_prompts.json --num-classes 3 --train-limit 60 --test-limit 30 --batch-size 8 --epochs 20 --checkpoint results/prompt_adapter.pt --out results/adapter.json
python evaluate.py --baseline results/baseline.json --enriched results/enriched.json --adapter results/adapter.json --csv results/comparison.csv --plot results/per_class_accuracy.png
```

Expected core outputs:

- `results/baseline.json`
- `results/enriched.json`
- `results/adapter.json`
- `results/comparison.csv`
- `results/sample_predictions/`

## Run with CLIP

Prepare dataset metadata:

```bash
python data_loader.py --num-classes 10 --train-limit 500 --test-limit 200
```

Run baseline CLIP:

```bash
python baseline_clip.py --num-classes 10 --test-limit 200
```

Generate prompts. Use fallback prompts with no API key:

```bash
python prompt_generator.py --provider fallback --num-classes 10
```

Use OpenAI instead:

```bash
set OPENAI_API_KEY=your_key_here
python prompt_generator.py --provider openai --model gpt-3.5-turbo --num-classes 10
```

Use local Ollama instead:

```bash
ollama pull llama3
python prompt_generator.py --provider ollama --model llama3 --num-classes 10
```

Run enriched CLIP:

```bash
python enriched_clip.py --num-classes 10 --test-limit 200
```

Optional prompt adapter:

```bash
python prompt_adapter.py --num-classes 10 --test-limit 200 --epochs 80
```

Compare methods and plot per-class accuracy:

```bash
python evaluate.py
```

Launch the Gradio demo:

```bash
python app.py --num-classes 10
```

The first CLIP run downloads HuggingFace model and dataset files unless they are already cached.

## Files

- `data_loader.py`: loads a fine-grained HuggingFace image dataset and returns train/test image-label pairs.
- `baseline_clip.py`: evaluates simple-prompt zero-shot CLIP.
- `prompt_generator.py`: generates or falls back to five descriptive prompts per class and saves JSON.
- `enriched_clip.py`: averages the generated prompt embeddings into class prototypes.
- `prompt_adapter.py`: optional two-layer MLP adapter trained on prompt embeddings.
- `evaluate.py`: prints a side-by-side table and saves a per-class accuracy bar chart.
- `app.py`: Gradio app comparing baseline and enriched predictions on an uploaded image.
- `clip_utils.py`: shared CPU CLIP, encoding, metric, and JSON helpers.

## Experimental Results (Beans Dataset)

We evaluated the pipeline on the **Beans** dataset (`AI-Lab-Makerere/beans`) containing 120 test images across three categories: `angular leaf spot`, `bean rust`, and `healthy`.

| Method | Top-1 Accuracy | Top-5 Accuracy | Delta vs. Baseline |
| :--- | :---: | :---: | :---: |
| **Baseline CLIP** | 30.83% | 100.0% | *Baseline* |
| **Enriched Prompts** | 32.50% | 100.0% | **+1.67%** |
| **Prompt Adapter** | 39.17% | 100.0% | **+8.34%** |

### Per-Class Performance Breakdown
* **angular leaf spot**: Baseline **87.18%** ➔ Enriched **28.21%** (**-58.97%**) ➔ Adapter **0.00%**
* **bean rust**: Baseline **7.50%** ➔ Enriched **65.00%** (**+57.50%**) ➔ Adapter **17.50%**
* **healthy**: Baseline **0.00%** ➔ Enriched **4.88%** (**+4.88%**) ➔ Adapter **97.56%**

---

## Discussion & Conclusion

* **Prompt Enrichment (LLM-generated visual descriptors)**: Strongly improved detection for classes where the baseline CLIP struggled to identify the raw label (such as `bean rust`, which improved by **+57.50%**). However, it introduced severe degradation on classes where baseline CLIP was already strong (such as `angular leaf spot`, which dropped by **-58.97%**). This occurs because descriptive terms for leaf diseases (e.g. "spots", "lesions") overlap, causing cross-class confusion for the CLIP text encoder.
* **Prompt Adapter**: Achieving the highest overall accuracy (**39.17%**), the MLP adapter overfit heavily to dominant representations, achieving **97.56%** on `healthy` but collapsing to **0.00%** on `angular leaf spot`.
* **Conclusion**: Prompt enrichment is highly beneficial for low-performing or obscure categories where CLIP lacks semantic prior knowledge, but must be carefully filtered or optimized to avoid introducing cross-class confusion in fine-grained tasks.

---

## Limitations

1. **Semantic Feature Overlap**: In fine-grained settings, generic LLM-generated descriptions (e.g., describing "brown spots" for different leaf diseases) introduce significant semantic ambiguity.
2. **Adapter Collapse / Bias**: The small two-layer MLP adapter trained on prompt text embeddings easily collapses and shifts predictions towards a single class on tiny datasets, requiring regularization or better feature balancing.
3. **CPU Evaluation Speed**: Inference on large datasets is constrained by CPU bottlenecking, limiting scaling to larger batch sizes or massive datasets.

---

## Outputs & Deliverables

All outputs are written in the repository root:
* `comparison.csv`: Side-by-side metric comparison.
* `per_class_accuracy.png`: Accuracy bar chart across all three methods.
* `results_summary.md`: Detailed research analysis and verification report.

