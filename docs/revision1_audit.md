# 🔎 Revision 1 Notebook Audit — Findings & Planned Fixes

> **Date:** 11 August 2026
> **Scope:** All notebooks under `notebook/revision1/`, hyperparameter configs, and `docs/revision_plan.md`
> **Status:** Audit complete. Fixes not yet applied.

---

## 📋 Verified Correct (No Action Needed)

The following were checked and found consistent across all notebooks:

- **Audio label alignment:** MelodyMachine head preserved as `0 = AI, 1 = Real`; training labels mapped via `label_idx = 1 if cls == "Real" else 0`. All downstream inference (`probs[:, 0]`) uses the same convention.
- **Video label alignment:** `0 = Real, 1 = AI` matches `CLASSES = ["Real", "AI"]` and dima806 native mapping.
- **Split integrity:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `video_stem`, zero-leakage assertion, fold keys `fold_1..fold_5` consistent across split generator, both training notebooks, and the multimodal notebook.
- **Strict split loading:** All notebooks raise `FileNotFoundError` if `kfold_splits.json` is missing; no on-the-fly generation.
- **Fusion formula:** `P_multimodal = 0.5·P_video + 0.5·P_audio`, threshold 0.5, index-aligned between modalities.
- **Augmentation scope:** Train-only (video `aug_train_transform`, audio `augment=True`); validation/test unaugmented.
- **Hyperparameter configs:** `audio_colab_L4/T4.json` (LR 3e-5, CosineAnnealingLR), `video_colab_L4/T4.json` (LR 1e-4, ReduceLROnPlateau factor 0.5 patience 2) — match the revision plan.

---

## 🔴 CRITICAL FINDINGS

### F1. `kfold_video_ai_detection.ipynb` — Cell 4 NameError (`hp_cfg`)

**Problem:** The hyperparameter block loads config into `hp` (`hp = json.load(f)` / `hp = default_hp`) but all 18 reads use the undefined variable `hp_cfg` (e.g. `HF_MODEL_ID = hp_cfg.get("hf_model_id", ...)`). The notebook **fails at startup on any fresh run** with `NameError: name 'hp_cfg' is not defined`.

**Evidence:** All saved outputs (OOF 96.31% frame-level, test 97.65%) are stale from an older code version. Cell 4's stored output ("Config file ... not found. Using default Video L4 GPU profile.") is text that does not exist in the current source — the cell was edited after execution and never re-run.

**Planned Fix:** Rename `hp` → `hp_cfg` in the load block (or `hp_cfg` → `hp` in the 18 read sites) so the variable names are consistent. Re-execute the notebook to regenerate results with the current code.

---

### F2. `kfold_audio_ai_detection.ipynb` — Cell 20 kernel crash (independent test section)

**Problem:** The saved state of the independent test-evaluation cell contains:

```
The Kernel crashed while executing code in the current cell or a previous cell.
```

**Evidence:** The 312-sample ensemble loop and classification report completed, but the kernel died at the end of the cell (VS Code Jupyter, crash introduced in commit `0919d15`).

**Likely causes:**
1. All 5 Wav2Vec2 fold models (~6.3 GB fp32) loaded into VRAM simultaneously.
2. All audio chunks of each video stacked into a single batch (a long video = 40+ chunks in one forward pass) → CUDA OOM → kernel death.
3. See F3: the variable-length tail chunk can raise inside this same cell.

**Planned Fix:**
- Process chunks in sub-batches (e.g. `chunk_size <= 8`) instead of one giant `torch.stack` per video.
- Load/evaluate fold models one at a time (sequential) and `del` + `torch.cuda.empty_cache()` per fold, matching the multimodal notebook's strategy.
- Wrap per-video inference in `try/except` with a logged fallback instead of crashing the kernel.
- Re-run the cell and confirm the crash is gone before marking this section complete.

---

## 🟠 HIGH FINDINGS

### F3. Audio tail-chunk bug — variable-length `torch.stack` (all notebooks)

**Problem:** Chunking uses:

```python
for start in range(0, total_len - MAX_AUDIO_SAMPLES + 1, MAX_AUDIO_SAMPLES):
    chunks.append(waveform[:, start:start + MAX_AUDIO_SAMPLES])
```

Whenever the audio length is **not an exact multiple of 5 s**, the final chunk is shorter than 80,000 samples. `torch.stack(inputs_list)` on mixed lengths raises `RuntimeError`.

**Impact by location:**
| Location | Behavior |
|---|---|
| Audio notebook Cell 18 (`predict_audio_kfold_ensemble`) | Hard crash (no try/except) |
| Audio notebook Cell 20 (test eval) | Hard crash — likely contributed to F2 |
| Multimodal notebook Cell 7 (`predict_audio_ai_prob`) | Swallowed by `except: return 0.5` → **silent neutral predictions** → fusion silently degrades to video-only for affected videos |

**Why it matters:** The silent 0.5 fallback is the most probable reason the multimodal results (OOF 88.85%, test 91.03%) fall far below the unimodal video results (96.31% / 97.65%). The paper's "multimodal outperforms unimodal" claim is currently **unsupported by the data** until this is fixed and re-run.

**Planned Fix:** In all three locations, pad the tail chunk to `MAX_AUDIO_SAMPLES` before stacking:

```python
last = waveform[:, start:start + MAX_AUDIO_SAMPLES]
if last.shape[-1] < MAX_AUDIO_SAMPLES:
    last = F.pad(last, (0, MAX_AUDIO_SAMPLES - last.shape[-1]))
```

Then re-run both the audio test section and the full multimodal notebook, and compare the new fusion numbers against the unimodal baselines.

---

### F4. `kfold_multimodal_evaluation.ipynb` — Cell 14 NameError (helper never runs)

**Problem:** `predict_multimodal_single_video()` references `ensemble_video_models` and `ensemble_audio_models`, which are **never defined** in this notebook. The cell has no outputs (never executed) — any call raises `NameError`.

**Planned Fix:** Define the ensemble model lists inside the helper (loop over `kfold_splits.keys()`, load from `MODEL_SAVE_DIR`) or load them at the start of Cell 14 and pass them in.

---

## 🟡 MEDIUM FINDINGS

### F5. Audio OOF classification report & confusion matrices have swapped class labels

**Location:** Audio notebook Cell 16 (OOF section).

**Problem:** Audio model labels are `0 = AI, 1 = Real`, but the report uses `target_names=CLASSES=["Real", "AI"]` and heatmaps use `xticklabels/yticklabels=CLASSES`. sklearn assigns `target_names[i]` in **sorted label order**, so:

- The row labeled **"Real"** actually contains the **AI-class** statistics, and vice versa.
- Each fold CM + the master CM are transposed in display.

**Proof from saved output:** the report shows "Real" with support 1240 and "AI" with support 9726 — the AI class is the 9726 one, so the labels are swapped. Accuracy and macro-F1 values are unaffected, but **per-class numbers and Figure CMs in the paper would be mislabeled**.

**Note:** Cell 20 (test section) does this correctly with `["AI", "Real"]` — the two sections are inconsistent.

**Planned Fix (choose one):**
- Option A: plot/report with `labels=["AI", "Real"]` order matching the model head, or
- Option B (recommended): keep the dataset convention internally — remap targets/preds (`1 - label`) to `0 = Real, 1 = AI` before computing the report and CM, so `CLASSES = ["Real", "AI"]` is correct everywhere.

Apply the same fix to the fold-level and master heatmaps.

---

### F6. Metric granularity is inconsistent across sections

**Problem:** The reported metrics are computed at different units:

| Section | Unit | Saved result |
|---|---|---|
| Video OOF (Cell 16) | **Frame-level** | 21,624 frames, acc 96.31% |
| Video test (Cell 20) | Video-level | 255 videos, acc 97.65% |
| Audio OOF (Cell 16) | **Chunk-level** | 10,966 chunks, acc 96.84% |
| Audio test (Cell 20) | Video-level | 312 videos, acc 95.83% |
| Multimodal OOF / test | Video-level | 1067 / 312 videos, 88.85% / 91.03% |

**Consequences:**
- The 5-fold mean±std table and the independent test table mix units and are not directly comparable.
- Audio OOF is heavily AI-skewed at chunk level (9726 AI vs 1240 Real chunks), inflating the "96.84%" headline number.
- Video OOF is Real-skewed at frame level (18,201 Real vs 3,423 AI frames).

**Planned Fix:** Compute OOF metrics at **video level** consistently — aggregate frame/chunk softmax probabilities per video, then vote once per video — for all three modalities. Update the revision plan / paper tables to state the aggregation unit explicitly.

---

## 🟢 LOW FINDINGS

### F7. `gemastik_base_model_inference.ipynb` references non-existent `val_split.json`

**Problem:** The notebook loads `VAL_SPLIT_FILE = ".../Gemastik26/val_split.json"` (fallback `val_stems = None`), but the revision pipeline only ever creates `kfold_splits.json`. The `val_split.json` file is only used by the **original** submission notebooks. Result: the notebook silently evaluates the **entire training set** while labeling it "validation split". Baselines (video 58.97%, audio 66.35%, ensemble 63.78%) are full-dataset numbers.

**Planned Fix:** Either load fold-1 `val_stems` from `kfold_splits.json`, or rename the notebook's section/prints to "Full Dataset Zero-Shot Baseline" to reflect reality.

---

### F8. Docs mismatch — checkpoint naming

**Problem:** `docs/revision_plan.md` §2.5 (and §2.3/§2.4) writes `dima806_deepfake_aug_fold1..5` and `melody_audio_aug_fold1..5`, but the code saves `..._fold_1` (with underscore before the number).

**Planned Fix:** Update `docs/revision_plan.md` to use `fold_1..fold_5` naming. Also re-verify the §2.4/§2.5 `[COMPLETED]` / `[IMPLEMENTED]` status markers once F2/F4 are fixed.

---

### F9. Minor issues

- Audio notebook Cell 4 **rewrites the hyperparameter JSON at `CONFIG_PATH`** (`json.dump` side-effect) whenever the profile is found. If the config lives in the repo clone, running the notebook modifies the repo file. Prefer reading without writing back.
- All notebooks emit `FutureWarning: torch.cuda.amp.autocast(...) is deprecated` — use `torch.amp.autocast("cuda", ...)`.
- `confusion_matrix(y_true, y_pred)` without an explicit `labels=[0, 1]` in video/audio OOF sections — safe today (both classes present) but fragile if a fold ever contains a single class.

---

## 🗂️ Recommended Fix Order

1. **F1** — one-line variable rename in the video notebook (blocks all video re-runs).
2. **F4** — define ensemble models in multimodal Cell 14.
3. **F3** — tail-chunk padding in all three notebooks (changes results, so re-run affected sections).
4. **F2** — make audio test inference crash-proof (sub-batching + per-fold model loading) and re-run.
5. **F5** — fix audio OOF label ordering (display-only, numbers unchanged).
6. **F6** — video-level OOF aggregation for comparable paper metrics.
7. **F7, F8, F9** — docs and minor cleanups.

After the fixes, re-run: audio notebook Cell 20 → multimodal notebook (Cells 9–13) → compare new fusion vs unimodal numbers before finalizing the manuscript tables.
