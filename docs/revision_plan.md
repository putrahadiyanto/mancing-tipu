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
├── generate_kfold_split.ipynb                 (Completed: 5-Fold Stratified GroupKFold Split Generator & EDA)
├── kfold_video_ai_detection.ipynb             (5-Fold Video ViT Fine-Tuning Pipeline — L4 Optimized)
├── kfold_audio_ai_detection.ipynb             (5-Fold Audio Wav2Vec2 Fine-Tuning Pipeline)
├── kfold_multimodal_evaluation.ipynb          (5-Fold Multimodal Fusion & Stage 2 AI-Only Evaluation)
└── hyperparameter/                            <-- Modality & GPU Hyperparameter Config Directory
    ├── video_colab_L4.json                    (Video ViT — NVIDIA L4 24GB VRAM Hyperparameter Config)
    ├── video_colab_T4.json                    (Video ViT — NVIDIA T4 16GB VRAM Hyperparameter Config)
    ├── audio_colab_L4.json                    (Audio Wav2Vec2 — NVIDIA L4 24GB VRAM Hyperparameter Config)
    └── audio_colab_T4.json                    (Audio Wav2Vec2 — NVIDIA T4 16GB VRAM Hyperparameter Config)

Drive Directory Structure:
/content/drive/MyDrive/Gemastik26/kfold_splits.json  <-- Exported 5-Fold Split Dictionary
/content/drive/MyDrive/Gemastik26/models/            <-- Original Checkpoints (UNTOUCHED)
/content/drive/MyDrive/Gemastik26/models/revision1/  <-- Revision 1 5-Fold Checkpoints
```

---

## 🎬 Internal Training Notebook Execution Flow (`kfold_video_ai_detection.ipynb`)

The training notebook follows a strict internal 8-stage execution flow:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Setup Environment & Hardware Check                                            │
│   • Mount Google Drive & detect GPU profile (NVIDIA L4 24GB VRAM)                       │
│   • Dynamically load hyperparameter config: hyperparameter/video_colab_L4.json         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Smart Unzip & 1 FPS Frame Extraction                                           │
│   • Unzip train.zip & trainAI.zip to Colab SSD (Smart Skip if target exists)           │
│   • Extract 1 FPS frames (max 30 frames/video) to dataset_frames/ (Smart Skip if extracted)│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Strict 5-Fold Split Loading (`kfold_splits.json`)                             │
│   • Strictly load kfold_splits.json from Drive                                         │
│   • Throws explicit FileNotFoundError if missing (No on-the-fly generation allowed)     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: PyTorch Dataset & Data Augmentations (Section 2B.2)                           │
│   • aug_train_transform: RandomResizedCrop + HorizontalFlip + Rotation + ColorJitter    │
│   • val_transform: Non-augmented Resize + ToTensor + Normalize                         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: 5-Fold ViT Fine-Tuning Engine (`train_vit_kfold_pipeline`)                     │
│   • Smart Skip: Check if fold checkpoint exists in models/revision1/ -> Skip training!  │
│   • Full Fine-Tuning: AdamW (all 86M ViT params) + AMP FP16                               │
│   • Dynamic Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=2)            │
│   • Early Stopping: Triggered if val loss fails to improve for 3 epochs (patience=3)  │
│   • Save Checkpoints: dima806_deepfake_aug_fold_1..5 to models/revision1/               │
│   • VRAM Cleanup: del model + torch.cuda.empty_cache() between folds                   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: 5-Fold Out-of-Fold (OOF) Metrics & Confusion Matrix Visualizations            │
│   • Subplot Heatmaps: Plot 5 individual fold confusion matrices                        │
│   • 🏆 Overall Combined Confusion Matrix: Plot master confusion matrix for 100% test set  │
│   • Paper Metrics: Compute 5-Fold Mean ± Std (Accuracy, Precision, Recall, Macro F1)   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: 5-Fold Soft Voting Ensemble Inference Helper                                  │
│   • Build predict_video_kfold_ensemble() helper                                        │
│   • Soft Probability Averaging across all 5 fold checkpoints for robust prediction      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 8: Independent Held-Out Test Set Evaluation (`testReal.zip` & `testAI.zip`)       │
│   • Unzip dedicated test archives from Drive (testReal.zip & testAI.zip)               │
│   • Run 5-Fold Soft Voting Ensemble inference on independent test videos               │
│   • Plot Dedicated Independent Test Set Confusion Matrix & Classification Report       │
└────────────────────────────────────────────────────────────────────────────────────────┘
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
  - **Strict Split Loading:** Strictly load `kfold_splits.json` from Google Drive (`/content/drive/MyDrive/Gemastik26/kfold_splits.json`). No on-the-fly split generation is allowed inside training notebooks; throws an explicit `FileNotFoundError` if missing.
  - For each fold (1 to 5):
    - Train ViT (`dima806`) on fold training videos with train-set-only data augmentation (`aug_train_transform`: `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`).
    - Evaluate on fold validation videos without augmentation (`val_transform`).
    - **Dynamic Learning Rate Scheduling:** Apply `ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-6)` on validation loss, dropping learning rate when loss plateaus for 2 epochs.
    - **Early Stopping:** Trigger early stopping if validation loss fails to improve for 3 consecutive epochs (`patience=3`).
    - Save fold models as `dima806_deepfake_aug_fold_{K}`.
  - Report 5-fold mean $\pm$ standard deviation for Accuracy, Precision, Recall, and Macro F1-Score.
  - **Independent Test Evaluation (Section 10):** Evaluates the 5-Fold Soft Voting Ensemble on the dedicated independent holdout test set (`testReal.zip` & `testAI.zip`), outputting an independent Test Set Confusion Matrix heatmap and classification report.

### 2.4 5-Fold Audio Wav2Vec2 Fine-Tuning Pipeline [COMPLETED]
- **Target File:** `notebook/revision1/kfold_audio_ai_detection.ipynb`
- **Status:** **IMPLEMENTED.**
- **Save Location:** `MODEL_SAVE_DIR = "/content/drive/MyDrive/Gemastik26/models/revision1"`
- **Task & Verification:** 
  - **Strict Split Loading:** Strictly load `kfold_splits.json` from Google Drive (`/content/drive/MyDrive/Gemastik26/kfold_splits.json`). No on-the-fly split generation fallback allowed; throws `FileNotFoundError` if missing.
  - **Pretrained Head Label Alignment:** Preserves `MelodyMachine`'s pretrained head alignment with `id2label = {0: "AI", 1: "Real"}` and `label2id = {"AI": 0, "Real": 1}`, setting sample target `0 = AI` and `1 = Real` to reuse pretrained classifier weights directly.
  - For each fold (1 to 5):
    - Chunk audio into non-overlapping 5-second segments (80,000 samples at 16kHz mono, `stride_samples = max_samples`).
    - Fine-tune Wav2Vec2 (`MelodyMachine`) with Section 2B.3 train-only waveform augmentations (`apply_audio_augmentations`: Additive Gaussian Noise, Gain, Time Shift).
    - Evaluate fold validation audio without augmentation (`augment=False`).
  - **Learning Rate & Optimizer:** AdamW optimizer with initial `learning_rate = 3e-5` (0.00003) matching the original baseline notebook, and `weight_decay = 0.01`.
    - **Learning Rate Scheduling:** Apply `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)` for smooth learning rate decay across all 10 epochs.
    - **Early Stopping:** Apply early stopping based on validation loss (`patience=3`).
    - Save fold models as `melody_audio_aug_fold_{K}` to `models/revision1/`.
  - Report 5-fold mean $\pm$ standard deviation for Accuracy, Precision, Recall, and Macro F1-Score with subplot heatmaps and master combined OOF confusion matrix.
  - **Independent Test Evaluation (Section 10):** Evaluates 5-Fold Soft Voting Audio Ensemble on dedicated test archives (`testReal.zip` & `testAI.zip`), plotting independent Test Set Confusion Matrix heatmap and classification report.

### 2.5 5-Fold Multimodal Video + Audio Late Fusion Evaluation Pipeline [COMPLETED]
- **Target File:** `notebook/revision1/kfold_multimodal_evaluation.ipynb`
- **Status:** **IMPLEMENTED.**
- **Task:**
  - Load fine-tuned fold checkpoints for both Video (`dima806_deepfake_aug_fold_1..5`) and Audio (`melody_audio_aug_fold_1..5`) across all 5 folds.
  - Evaluate 50/50 multimodal late fusion ensemble:
    $$P_{\text{multimodal}} = 0.5 \cdot P_{\text{video}} + 0.5 \cdot P_{\text{audio}}$$
  - Report overall 5-fold mean $\pm$ standard deviation for Accuracy, Precision, Recall, and **Macro F1-Score** with subplot heatmaps and master combined OOF confusion matrix.
  - **Independent Test Evaluation (Section 10):** Evaluates 5-Fold Multimodal Soft Voting Ensemble on dedicated test archives (`testReal.zip` & `testAI.zip`), plotting independent Test Set Confusion Matrix heatmap and classification report.

---

## 🎨 Section 2B: Multimodal Data Augmentation Strategy

To ensure model robustness against social media compression and real-world audio/video noise without causing artifact-erasing distortions:

### 2B.1 Core Augmentation Principles
1. **Strict Train-Set Scope:** Augmentations are applied strictly on-the-fly to the training DataLoader (`aug_train_transform` for video, `augment=True` for audio). Validation and test sets remain **100% un-augmented** to guarantee unbiased evaluation metrics.
2. **Artifact Preservation:** Augmentations are selected so they do not destroy subtle deepfake artifacts (e.g. boundary blending, lip-sync misalignment, facial flickering).

### 2B.2 Video Modality Augmentation Pipeline (`torchvision.transforms`)
- **Random Horizontal Flip:** `p=0.5` — Preserves facial deepfake geometry while doubling spatial orientation variability.
- **Random Rotation:** `degrees=15` (`p=0.5`) — Simulates slight handheld camera tilt without breaking face alignment.
- **Color Jitter:** `brightness=0.2, contrast=0.2, saturation=0.2` (`p=0.5`) — Simulates lighting and exposure variations.
- **Random Resized Crop:** Scale `[0.8, 1.0]` resized to `(224, 224)` — Simulates minor framing shifts and zoom levels.
- **Normalization:** ImageNet standard mean `[0.485, 0.456, 0.406]` & std `[0.229, 0.224, 0.225]`.

### 2B.3 Audio Modality Augmentation Pipeline (`torchaudio` / PyTorch)
- **Additive Gaussian Noise:** $\text{SNR} \in [15, 30]\text{ dB}$ (`p=0.5`) — Simulates background environmental noise (street noise, room reverberation).
- **Random Gain / Volume Adjustment:** Gain scale $\in [-6\text{ dB}, +6\text{ dB}]$ (`p=0.5`) — Simulates varying microphone sensitivities and distances.
- **Time Shift / Roll:** Shift up to $\pm 10\%$ in time (`p=0.5`) — Simulates random phrase alignment offsets.
- **5-Second Non-Overlapping Chunking:** Chunks 16kHz mono audio into non-overlapping 5s windows (80,000 samples) with 5s stride step size (`stride_samples = max_samples`), matching the original baseline notebook chunking structure.

---

## 📝 Section 3: Manuscript Revisions & Consistency Checklist

### 3.1 Data Tables & Metric Corrections
- **Table II (Dataset Composition):** Update with exact 5-fold cross-validation split distributions (stems per fold, class balance).
- **Table IV & Table V:** Replace single holdout metrics with **5-Fold Cross-Validation Mean $\pm$ Standard Deviation** metrics (Accuracy, Precision, Recall, Macro F1-Score).
- **Hyperparameter Table:** Add explicit table detailing epochs, learning rate ($1\times 10^{-4}$ for video, $5\times 10^{-5}$ for audio), LR scheduler (`ReduceLROnPlateau` with factor 0.5 and patience 2), batch size, optimizer (AdamW), weight decay, frame rate (1 FPS), audio chunking (5s), early stopping patience (3), and 5-fold CV scheme.

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
