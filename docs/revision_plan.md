# 📋 Comprehensive Revision Plan — Review 2 (6 Aug 2026)

> **Project Mancing-Tipu:** Deteksi AI-Generated Video dan Identifikasi Potensi Penipuan Digital Berbasis Multimodal Deep Learning  
> *Detailed action plan addressing every technical, analytical, and manuscript feedback item from Review 2.*

---

## 📁 Workspace Isolation Strategy: `notebook/revision1/`

To guarantee that all original submission notebooks under [`notebook/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook) remain **100% UNTOUCHED** and preserved as historical baselines, **all revised and new notebooks are organized inside a dedicated directory:** [`notebook/revision1/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/revision1).

```
notebook/                                      <-- Original Submission (UNTOUCHED)
├── gemastik_audio_ai_detection.ipynb
├── gemastik_video_ai_detection.ipynb
└── gemastik_multimodal_fusion_evaluation.ipynb

notebook/revision1/                            <-- All Revised & New Notebooks
├── gemastik_base_model_inference.ipynb        (New: Dedicated Base Model Zero-Shot Inference)
├── gemastik_video_ai_detection.ipynb          (Revised Video Detection)
├── gemastik_audio_ai_detection.ipynb          (Revised Audio Detection)
├── gemastik_multimodal_fusion_evaluation.ipynb (Revised Multimodal Fusion & Stage 2 AI-Only Scope)
└── kfold/                                     (New: 5-Fold Stratified GroupKFold Directory)
    ├── kfold_video_ai_detection.ipynb
    ├── kfold_audio_ai_detection.ipynb
    └── kfold_multimodal_evaluation.ipynb
```

---

## 🔍 Section 1: Pretrained Model Label Mapping & Baseline Corrections

### 1.1 Verified Hugging Face Config `id2label` Results
- **Video Model (`dima806/deepfake_vs_real_image_detection`):**
  - Config `id2label`: `{"0": "Real", "1": "Fake"}`
  - Dataset Convention: `0 = Real`, `1 = AI`
  - **Status: MATCHES PERFECTLY.** Index `0` is Real and Index `1` is Fake/AI.
- **Audio Model (`MelodyMachine/Deepfake-audio-detection-V2`):**
  - Config `id2label`: `{"0": "fake", "1": "real"}`
  - Dataset Convention: `0 = Real`, `1 = AI`
  - **Status: INVERTED / OPPOSITE.** Pretrained Audio model uses Index `0` for `"fake"` and Index `1` for `"real"`.

### 1.2 Resolution of Review 2 Finding ("Akurasi 33,65% diduga bug")
- When evaluating the raw pretrained audio model zero-shot before fine-tuning, interpreting output index `0` as `CLASSES[0] = "Real"` caused predictions to be **systematically inverted** ($100\% - 33.65\% = 66.35\%$).
- **Fine-Tuning Behavior:** PyTorch re-trains the head with `id2label={0: "Real", 1: "AI"}`, so fine-tuned checkpoints (`melody_audio_deepfake_aug`) output correct indices ($94.87\%$ accuracy).
- **Corrected Zero-Shot Baselines:**
  - Raw Video Baseline: **58.97%**
  - Raw Audio Baseline: **66.35%** (corrected from 33.65%)
  - Raw Ensemble Baseline: **63.78%** (corrected from 36.22%; lower than audio alone because raw video predictions dilute audio confidence).

---

## 🛠️ Section 2: Technical Code & Pipeline Tasks (`notebook/revision1/`)

### 2.1 New Dedicated Base Model Inference Notebook
- **Target File:** `notebook/revision1/gemastik_base_model_inference.ipynb`
- **Task:** Implement zero-shot base model evaluation notebook with explicit index remapping:
  - Video (`dima806`): `0 -> 0 (Real)`, `1 -> 1 (AI)`
  - Audio (`MelodyMachine`): `0 ("fake") -> 1 (AI)`, `1 ("real") -> 0 (Real)`
- Report true zero-shot baselines (Video: 58.97%, Audio: 66.35%, Ensemble: 63.78%).

### 2.2 Stage 2 (IndoBERT) Evaluation Scope Realignment
- **Target File:** `notebook/revision1/gemastik_multimodal_fusion_evaluation.ipynb`
- **Task:** Filter Stage 2 IndoBERT evaluation to run **strictly on AI-generated video transcripts** (122 test samples or Stage 1 AI outputs).
- Report **Macro F1-Score** alongside majority-class baseline comparisons ($225/312 = 72.1\%$ for overall data, or majority class ratio on AI subset).

### 2.3 Data Splitting, Augmentation, & Dual Cross-Validation Strategy

- **Verified Data Mechanics:**
  - **Data Splitting by Video ID:** Confirmed in `notebook/gemastik_video_ai_detection.ipynb` and `notebook/gemastik_audio_ai_detection.ipynb` via `val_split.json`. All frames and 5s audio chunks of the same video ID belong to the same split.
  - **Augmentation Isolation:** Confirmed applied on-the-fly only to `train_samples` (`augment=True`), while validation sets use `augment=False`.

- **Dual Cross-Validation Strategy:**
  1. **Option A (Dedicated K-Fold Experiments):**
     - Located in dedicated directory: `notebook/revision1/kfold/`
     - Implement `sklearn.model_selection.StratifiedGroupKFold(n_splits=5)` grouped strictly by `video_stem`.
     - Notebooks: `notebook/revision1/kfold/kfold_video_ai_detection.ipynb`, `notebook/revision1/kfold/kfold_audio_ai_detection.ipynb`, and `notebook/revision1/kfold/kfold_multimodal_evaluation.ipynb`.
     - Trains and evaluates 5 folds, reporting mean $\pm$ standard deviation.
  2. **Option B (Standard Deterministic Holdout Split + Manuscript Subsection):**
     - Located in `notebook/revision1/`
     - Retains deterministic 80-20 holdout split (`val_split.json`) for fixed multi-modal benchmark stability across visual, audio, and text modalities.
     - Documented in the manuscript's **Keterbatasan (Limitations)** subsection explaining training time constraints and referencing the K-Fold experiments.

### 2.4 Late Fusion Weight Ablation & End-to-End Pipeline Evaluation
- **Target File:** `notebook/revision1/gemastik_multimodal_fusion_evaluation.ipynb`
- **Late Fusion Mechanism:** Formally specify probability weighting formula:
  $$P_{\text{multimodal}} = w_{\text{video}} \cdot P_{\text{video}} + w_{\text{audio}} \cdot P_{\text{audio}}$$
- **Weight Ablation:** Run grid search on Validation set for $w_{\text{video}} \in [0.1, 0.9]$ and decision threshold $\tau \in [0.4, 0.6]$.
- **End-to-End Pipeline Evaluation:** Build joint execution pipeline (`Stage 1 Ensemble -> Stage 2 IndoBERT`) and compute missed scam rates.

---

## 📝 Section 3: Manuscript Revisions & Consistency Checklist

### 3.1 Data Tables & Metric Corrections
- **Table II (Dataset Composition):** Fix math total calculations so class and subset sums match.
- **Table IV & Table V:** Update with corrected zero-shot baselines, Macro F1 scores, and majority-class baselines.
- **Hyperparameter Table:** Add explicit table detailing epochs, learning rate ($1\times 10^{-4}$ / Cosine Annealing), batch size, optimizer (AdamW), weight decay, frame rate (1 FPS), audio chunking (5s), and early stopping patience.

### 3.2 Subhead Titles & Terminology Standardization
- Change subheader title: `"MATRIKS EVALUASI"` $\rightarrow$ `"METRIK EVALUASI"`.
- Standardize label terminology across text, tables, and figures to **`Fraud` / `Non-Fraud`** (or `Penipuan` / `Aman`).

### 3.3 Text, Equations, & Citation Fixes
- **Introduction:** Merge duplicated paragraphs prior to Objectives section.
- **Equations:** Fix typos in formula ("Precission" $\rightarrow$ "Precision", "Pr e cision" formatting). Number all equations (Equations 1–4) and reference them in text.
- **Affiliation:** Include full university name (*Universitas Pendidikan Indonesia*) and fix truncated email.
- **Section Numbering:** Fix Roman numeral sequence (IV, V, VI).
- **Figures:** Enlarge/re-render Figure 4 confusion matrices for print readability; add source citations for Figures 1 & 2; blur identifiable faces in dataset example figures for privacy compliance.
- **DOIs & URLs:** Fix missing hyphens in DOIs and broken URLs caused by PDF rendering.

### 3.4 Limitations Subsection (Subbab Keterbatasan)
Add a dedicated **Keterbatasan (Limitations)** subsection covering:
1. **Validation Scheme:** Explain rationale for primary holdout split (multimodal consistency) while referencing 5-fold cross-validation results from `notebook/revision1/kfold/`.
2. **Dataset Size:** Constraints of current Indonesian video dataset size.
3. **Shortcut Learning Risks:** Potential model reliance on platform compression artifacts or generator watermarks rather than deepfake features.
4. **Generative Model Generalization:** Evaluation limits on unseen/newer generative models.

---

## 🗂️ Workspace Target Files Summary

- 📂 **[`notebook/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook)**: Original submission notebooks (**100% UNTOUCHED**).
- 📂 **`notebook/revision1/`**: Directory containing all revised & new notebooks:
  - `gemastik_base_model_inference.ipynb`
  - `gemastik_video_ai_detection.ipynb`
  - `gemastik_audio_ai_detection.ipynb`
  - `gemastik_multimodal_fusion_evaluation.ipynb`
  - 📁 `kfold/`
    - `kfold_video_ai_detection.ipynb`
    - `kfold_audio_ai_detection.ipynb`
    - `kfold_multimodal_evaluation.ipynb`
- 📄 **[`docs/revision_plan.md`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/docs/revision_plan.md)**: Master revision plan document.
- 📄 **[`docs/agents.md`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/docs/agents.md)**: Multi-agent system architecture reference.
