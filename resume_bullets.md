# Resume Bullet Points: Fine-Grained CLIP Enhancement

High-impact, recruiter-ready descriptions of the project. Choose the bullets that best fit your resume style.

---

## Option 1: Bullet Points (Action + Context + Result)

* **Multi-Modal Zero-Shot Pipeline**: Designed and implemented an evaluation framework for frozen CLIP (`ViT-B/32`) on fine-grained classification tasks (plant diseases and floral datasets), benchmarking template-based prompts against LLM-guided prompt enrichment and PyTorch adapters.
* **Lightweight Prompt Adapter**: Developed and trained a lightweight PyTorch Prompt Adapter (2-layer MLP classifier) on multi-modal text embeddings, improving zero-shot classification accuracy by **+8.3%** on a CPU-only hardware budget.
* **Prompt Enrichment & LLM Prototype Generation**: Built an automated pipeline that leverages LLM-generated visual descriptors (color, shape, lesions, texture) to generate class-specific text prototypes, boosting classification accuracy on low-performing baseline classes by up to **+57.5% absolute**.
* **Diagnostic Evaluation & Cross-Dataset Validation**: Identified and analyzed representation collapse and semantic feature overlap limitations across datasets (Beans, Oxford Flowers 102); formulated trade-offs regarding descriptive vs. discriminative prompts to mitigate cross-class confusion.
* **Interactive Gradio Interface**: Engineered a real-time web application using Gradio, enabling users to upload custom images and inspect side-by-side model predictions and underlying text prototype embeddings.

---

## Option 2: STAR Format (Situation, Task, Action, Result)

* **Situation**: Pre-trained multi-modal models like CLIP struggle with fine-grained visual classification tasks (e.g., distinguishing between highly similar plant diseases or flower species) due to generic prompt templates like `a photo of a {class}`.
* **Task**: Create a lightweight, resource-efficient pipeline to enhance zero-shot performance on fine-grained datasets (Beans and Oxford Flowers 102) without fine-tuning the heavy backbones.
* **Action**:
  * Freezed CLIP's text and vision backbones to enable CPU-only evaluation.
  * Implemented an automated visual description generator utilizing structured prompts to extract detailed visual characteristics.
  * Engineered a 2-layer PyTorch MLP Prompt Adapter trained on prompt text embeddings to learn target-class mappings.
  * Developed a web interface using Gradio to visually inspect embedding alignments.
* **Result**:
  * Achieved up to a **+8.34% Top-1 accuracy gain** on the Beans dataset and a **+6.25% gain** on Oxford Flowers 102.
  * Improved detection of obscure disease classes (`bean rust`) from **7.5% to 65.0% (+57.5% absolute)**.
  * Documented core generalizability trade-offs regarding semantic feature overlap and majority-class adapter collapse.
