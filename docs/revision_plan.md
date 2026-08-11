# 📋 Comprehensive Revision Plan — Review 2 (6 Aug 2026)

> **Project Mancing-Tipu:** Deteksi AI-Generated Video dan Identifikasi Potensi Penipuan Digital Berbasis Multimodal Deep Learning  
> *Detailed action plan addressing every technical, analytical, and manuscript feedback item from Review 2 using 5-Fold Stratified GroupKFold Cross-Validation.*

---

## 📁 Workspace & Checkpoint Isolation Strategy

To guarantee that all original submission notebooks under [`notebook/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook) and original trained model checkpoints under `/content/drive/MyDrive/Gemastik26/models/` remain **100% UNTOUCHED** and preserved as historical baselines:

1. **Notebooks Directory:** All revised notebooks are organized inside: [`notebook/revision1/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/revision1).
2. **Validation Framework:** **5-Fold Stratified GroupKFold Cross-Validation** is used exclusively across all pipeline stages.
3. **Model Checkpoint Directory:** All fine-tuned fold models for revision 1 will be saved inside a dedicated Google Drive subdirectory: **`MODEL_SAVE_DIR = "/content/drive/MyDrive/Gemastik26/models/revision1"`**.

```
notebook/                                      <-- Original Submission Notebooks (UNTOUCHED)
├── gemastik_audio_ai_detection.ipynb
├── gemastik_video_ai_detection.ipynb
└── gemastik_multimodal_fusion_evaluation.ipynb

notebook/revision1/                            <-- Revision 1 (5-Fold Stratified GroupKFold Pipeline)
├── gemastik_base_model_inference.ipynb        (Completed: Dedicated Base Model Zero-Shot Inference)
├── generate_kfold_split.ipynb                 (Standalone 5-Fold Stratified GroupKFold Split Generator)
├── kfold_video_ai_detection.ipynb             (5-Fold Video ViT Fine-Tuning Pipeline)
├── kfold_audio_ai_detection.ipynb             (5-Fold Audio Wav2Vec2 Fine-Tuning Pipeline)
└── kfold_multimodal_evaluation.ipynb          (5-Fold Multimodal Fusion & Stage 2 AI-Only Evaluation)

Drive Directory Structure:
/content/drive/MyDrive/Gemastik26/kfold_splits.json  <-- Exported 5-Fold Split Dictionary
/content/drive/MyDrive/Gemastik26/models/            <-- Original Checkpoints (UNTOUCHED)
/content/drive/MyDrive/Gemastik26/models/revision1/  <-- Revision 1 5-Fold Checkpoints
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
- **Fine-Tuning Behavior:** PyTorch re-trains the head with `id2label={0: "Real", 1: "AI"}`, so fine-tuned checkpoints output correct indices ($94.87\%$ accuracy).
- **Corrected Zero-Shot Baselines:**
  - Raw Video Baseline: **58.97%**
  - Raw Audio Baseline: **66.35%** (corrected from 33.65%)
  - Raw Ensemble Baseline: **63.78%** (corrected from 36.22%; lower than audio alone because raw video predictions dilute audio confidence).

---

## 🛠️ Section 2: Technical Code & Pipeline Tasks (`notebook/revision1/`)

### 2.1 Base Model Zero-Shot Inference Notebook [COMPLETED]
- **Target File:** `notebook/revision1/gemastik_base_model_inference.ipynb`
- **Status:** **IMPLEMENTED.**
- Evaluates raw `dima806` ViT and `MelodyMachine` Wav2Vec2 base models with explicit index remapping (`Audio: 0 -> 1, 1 -> 0`).

### 2.2 Standalone 5-Fold Stratified GroupKFold Split Generator & Distribution Audit
- **Target File:** `notebook/revision1/generate_kfold_split.ipynb`
- **Task & Distribution Mechanics:** 
  - Load all unique video stems from the dataset.
  - Run `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
  - **Group Constraint (Zero Leakage):** Grouped strictly by `video_stem` (`Path(v_folder).name`), guaranteeing that all frames and 5s audio chunks of the same video ID remain together in the exact same fold.
  - **Stratification Constraint (Preserving Main Distribution):** Optimizes video group assignments so that **every single validation fold (Folds 1–5) strictly mirrors the main overall dataset class ratio** (e.g. ~61% Real / ~39% AI).
  - **Review 2 Resolution:** Directly fixes the reviewer's criticism (*"Proporsi kelas train dan test sangat berbeda"*) by guaranteeing identical class proportions across all train and validation splits.
  - Run visual EDA benchmarking main dataset class percentages against Folds 1–5.
  - Verify 0 video overlap across folds.
  - Export `kfold_splits.json` to Google Drive (`/content/drive/MyDrive/Gemastik26/kfold_splits.json`).

### 2.3 5-Fold Video ViT Fine-Tuning Pipeline
- **Target File:** `notebook/revision1/kfold_video_ai_detection.ipynb`
- **Save Location:** `MODEL_SAVE_DIR = "/content/drive/MyDrive/Gemastik26/models/revision1"`
- **Task:** 
  - Load `kfold_splits.json`.
  - For each fold (1 to 5):
    - Train ViT (`dima806`) on fold training videos with train-set-only data augmentation (`aug_train_transform`: `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`).
    - Evaluate on fold validation videos without augmentation (`val_transform`).
    - Apply early stopping based on validation loss (patience=3).
    - Save fold models as `dima806_deepfake_raw_fold{K}` and `dima806_deepfake_aug_fold{K}`.
  - Report 5-fold mean $\pm$ standard deviation for Accuracy, Precision, Recall, and Macro F1-Score.

### 2.4 5-Fold Audio Wav2Vec2 Fine-Tuning Pipeline
- **Target File:** `notebook/revision1/kfold_audio_ai_detection.ipynb`
- **Save Location:** `MODEL_SAVE_DIR = "/content/drive/MyDrive/Gemastik26/models/revision1"`
- **Task:** 
  - Load `kfold_splits.json`.
  - For each fold (1 to 5):
    - Chunk audio into 5-second overlapping segments (16kHz mono).
    - Fine-tune Wav2Vec2 (`MelodyMachine`) with train-only Gaussian noise/gain augmentation (`augment=True`).
    - Evaluate fold validation audio without augmentation (`augment=False`).
    - Apply early stopping based on validation loss (patience=3).
    - Save fold models as `melody_audio_deepfake_raw_fold{K}` and `melody_audio_deepfake_aug_fold{K}`.
  - Report 5-fold mean $\pm$ standard deviation for Accuracy, Precision, Recall, and Macro F1-Score.

### 2.5 5-Fold Multimodal Fusion & Stage 2 (IndoBERT) Evaluation Pipeline
- **Target File:** `notebook/revision1/kfold_multimodal_evaluation.ipynb`
- **Task:**
  - Load fold models for both Video and Audio across all 5 folds.
  - Evaluate multimodal late fusion ensemble:
    $$P_{\text{multimodal}} = w_{\text{video}} \cdot P_{\text{video}} + w_{\text{audio}} \cdot P_{\text{audio}}$$
  - Run late fusion weight ablation ($w_{\text{video}} \in [0.1, 0.9]$) per fold.
  - Restrict Stage 2 IndoBERT evaluation strictly to **AI-generated video transcripts** (122 test samples or Stage 1 AI outputs).
  - Implement joint end-to-end pipeline evaluation (`Stage 1 Ensemble -> Stage 2 IndoBERT`) measuring missed scam rates per fold.
  - Report overall 5-fold mean $\pm$ standard deviation for Accuracy, Precision, Recall, and **Macro F1-Score**.

---

## 📝 Section 3: Manuscript Revisions & Consistency Checklist

### 3.1 Data Tables & Metric Corrections
- **Table II (Dataset Composition):** Update with exact 5-fold cross-validation split distributions (stems per fold, class balance).
- **Table IV & Table V:** Replace single holdout metrics with **5-Fold Cross-Validation Mean $\pm$ Standard Deviation** metrics (Accuracy, Precision, Recall, Macro F1-Score).
- **Hyperparameter Table:** Add explicit table detailing epochs, learning rate ($1\times 10^{-4}$ / Cosine Annealing), batch size, optimizer (AdamW), weight decay, frame rate (1 FPS), audio chunking (5s), early stopping patience, and 5-fold CV scheme.

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
1. **Dataset Size:** Constraints of current Indonesian video dataset size.
2. **Shortcut Learning Risks:** Potential model reliance on platform compression artifacts or generator watermarks rather than deepfake features.
3. **Generative Model Generalization:** Evaluation limits on unseen/newer generative models.

---

## 🗂️ Workspace Target Files Summary

- 📂 **[`notebook/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook)**: Original submission notebooks (**100% UNTOUCHED**).
- 📂 **`notebook/revision1/`**: 5-Fold Stratified GroupKFold Revision Pipeline:
  - `gemastik_base_model_inference.ipynb` [Done]
  - `generate_kfold_split.ipynb` (Standalone 5-Fold Split Generator)
  - `kfold_video_ai_detection.ipynb` (5-Fold Video ViT Training)
  - `kfold_audio_ai_detection.ipynb` (5-Fold Audio Wav2Vec2 Training)
  - `kfold_multimodal_evaluation.ipynb` (5-Fold Multimodal Fusion & Stage 2 Evaluation)
- 📁 **Google Drive Directory:**
  - Split File: `/content/drive/MyDrive/Gemastik26/kfold_splits.json`
  - Model Save Path: `/content/drive/MyDrive/Gemastik26/models/revision1/`
- 📄 **[`docs/revision_plan.md`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/docs/revision_plan.md)**: Master revision plan document.
- 📄 **[`docs/agents.md`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/docs/agents.md)**: Multi-agent system architecture reference.
