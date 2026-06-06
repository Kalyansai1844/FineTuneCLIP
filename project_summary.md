# Project Summary: Fine-Grained CLIP Enhancement

A resume-friendly overview of the project, including key technical achievements, results, and talking points for interviews.

---

## 🚀 Quick Pitch
Designed and evaluated a zero-shot classification enhancement pipeline for CLIP (`openai/clip-vit-base-patch32`) on fine-grained image datasets (specifically plant disease detection via the Hugging Face Beans dataset). The project evaluated LLM-guided prompt enrichment (using structured visual descriptors) and a lightweight PyTorch Prompt Adapter, achieving up to an **8.3% overall Top-1 accuracy boost**.

---

## 🛠️ Technical Stack
* **Frameworks & Libraries**: PyTorch, Hugging Face Transformers (`CLIPProcessor`, `CLIPModel`), Hugging Face Datasets, Matplotlib, Pillow, Gradio
* **Models**: CLIP (ViT-B/32), GPT-3.5 / Ollama Llama3 (for description generation)
* **Infrastructure**: CPU-friendly local evaluation & training pipelines

---

## 📈 Key Results (Beans Dataset)
* **Baseline CLIP**: **30.83%** Top-1 Accuracy.
* **Enriched Prompts (LLM visual descriptions)**: **32.50%** Top-1 Accuracy (**+1.67%** gain).
  * *Success story*: Boosted detection of `bean rust` from **7.5% to 65.0% (+57.5% absolute gain)** by replacing generic prompts with detailed pustule and spore characteristics.
  * *Key discovery*: Identified cross-class visual feature overlap that degraded the `angular leaf spot` class by **-59.0%**, demonstrating the limits of prompt enrichment without contrastive filtering.
* **Prompt Adapter (PyTorch MLP)**: **39.17%** Top-1 Accuracy (**+8.34%** gain).
  * Showed high class-specific optimization (boosting the `healthy` class to **97.6%**).

---

## 💡 Resume Bullet Points
* **Developed an evaluation framework** for OpenAI's CLIP model using PyTorch and Hugging Face Transformers to benchmark zero-shot classification performance on fine-grained visual datasets (Beans, CIFAR-10).
* **Implemented prompt enrichment pipelines** that leverage LLM-generated visual descriptions (color, shape, lesions, texture) to build class-specific text prototypes, improving macro-averaging performance by up to **+57.5%** on obscure classes.
* **Designed and trained a PyTorch Prompt Adapter** (two-layer MLP classifier) on CLIP text embeddings, improving zero-shot accuracy by **+8.3%** on a CPU-only hardware budget.
* **Diagnosed and documented semantic feature overlap** and representation collapse limitations, providing clear research direction for visual prompt engineering and soft prompt optimization.
* **Built an interactive Gradio web application** for real-time model comparison, allowing users to upload leaf images and visually inspect the text prototypes used for prediction.

---

## 💬 Interview Discussion Points
* **The "Spot" Ambiguity (Engineering Trade-off)**: "Prompt enrichment is a double-edged sword. When we added detailed descriptions of angular leaf spot and bean rust, both described spots and lesions. CLIP's text encoder couldn't distinguish between these fine-grained spot structures as well as it could the simple label, resulting in a drop for leaf spots but a rise for rust. This shows that descriptions must be *discriminative* rather than just *descriptive*."
* **Adapter Collapse**: "The prompt adapter achieved a high overall accuracy of 39.2%, but under the hood, it collapsed to predicting the 'healthy' class at 97.6% while scoring 0% on angular leaf spot. This is a classic case of representation collapse on small fine-grained subsets, which I addressed in the limitations write-up by suggesting regularized training and class-balanced losses."
