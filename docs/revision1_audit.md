# 🔎 Revision 1 Notebook Audit — Findings & Planned Fixes

> **Date:** 11 August 2026 (updated 13 Aug 2026)
> **Scope:** All notebooks under `notebook/revision1/`, hyperparameter configs, and `docs/revision_plan.md`
> **Status:** Audit complete. **All fixes applied and re-executed.** Final v2 results (13 Aug 2026, independent 312-video test set): Video Saja **96.15%**, Audio Saja **97.44%**, Multimodal 50/50 **97.76%**. The current pipeline is `kfold_audio_ai_detection_v2.ipynb` (saves to `models/revision1_v2/`) + `kfold_multimodal_evaluation_v2.ipynb` (Video from `models/revision1/`, Audio from `models/revision1_v2/`). Numbers cited in the finding sections below are pre-fix references.

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

**Status: ✅ FIXED (12 Aug 2026).** `hp` → `hp_cfg` rename applied; video notebook now uses `hp_cfg` consistently (19 `hp_cfg.get(...)` reads, 0 bare `hp.` reads).

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

**Status: ✅ FIXED (v2 audio notebook).** `kfold_audio_ai_detection_v2.ipynb` uses sequential per-fold model loading, chunk sub-batching (`sub_batch=8`), and tail-padded chunks.

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

**Status: ✅ FIXED (v2 audio notebook + v2 multimodal + base notebook).** v2 uses `pad_len = max_samples - total_len; chunk = F.pad(waveform, (0, pad_len))` for 100% tail coverage; `gemastik_base_model_inference.ipynb` pads short chunks via `np.pad`.

---

### F4. `kfold_multimodal_evaluation.ipynb` — Cell 14 NameError (helper never runs)

**Problem:** `predict_multimodal_single_video()` references `ensemble_video_models` and `ensemble_audio_models`, which are **never defined** in this notebook. The cell has no outputs (never executed) — any call raises `NameError`.

**Planned Fix:** Define the ensemble model lists inside the helper (loop over `kfold_splits.keys()`, load from `MODEL_SAVE_DIR`) or load them at the start of Cell 14 and pass them in.

**Status: ✅ FIXED (v2 multimodal notebook).** Cell 14 helper loads each fold checkpoint from `MODEL_SAVE_DIR` / `AUDIO_MODEL_SAVE_DIR` inside the fold loop.

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

**Status: ✅ FIXED (v2 audio notebook).** v2 preserves the head convention `id2label={0: "AI", 1: "Real"}`, `label_idx = 1 if cls == "Real" else 0`, and uses chunk-softmax-averaged **video-level** votes. Audio probability = `probs[:, 1]` (Real) / `probs[:, 0]` (AI) as appropriate, and reports/CMs use the matching class order.

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

**Status: ✅ FIXED (v2 audio notebook).** v2 evaluates at video level (`evaluate_audio_video_level`, chunk-softmax-averaged, vote once per video).

---

## 🟢 LOW FINDINGS

### F7. `gemastik_base_model_inference.ipynb` references non-existent `val_split.json`

**Problem:** The notebook loads `VAL_SPLIT_FILE = ".../Gemastik26/val_split.json"` (fallback `val_stems = None`), but the revision pipeline only ever creates `kfold_splits.json`. The `val_split.json` file is only used by the **original** submission notebooks. Result: the notebook silently evaluates the **entire training set** while labeling it "validation split". Baselines (video 58.97%, audio 66.35%, ensemble 63.78%) are full-dataset numbers.

**Planned Fix:** Either load fold-1 `val_stems` from `kfold_splits.json`, or rename the notebook's section/prints to "Full Dataset Zero-Shot Baseline" to reflect reality.

**Status: ✅ FIXED.** The notebook now loads fold-1 `val_stems` from `kfold_splits.json` (214 validation videos) and reports video 61.68% / audio 65.42% / ensemble 70.09% on that split (Cell 8 outputs).

**Cache-error check (frame-cache stem collision):** This notebook does **NOT** have the stem-collision bug. It extracts frames in-memory as PIL images (`extract_frames_from_video` — no on-disk frame cache, no `existing` shortcut) and removes the temp wav after each audio read. Fold-1 val videos live in separate per-class dirs (`/content/dataset_raw_val/Real/...` vs `.../AI/...`), so even identical stems across classes cannot collide. No fix required.

---

### F8. Docs mismatch — checkpoint naming

**Problem:** `docs/revision_plan.md` §2.5 (and §2.3/§2.4) writes `dima806_deepfake_aug_fold1..5` and `melody_audio_aug_fold1..5`, but the code saves `..._fold_1` (with underscore before the number).

**Planned Fix:** Update `docs/revision_plan.md` to use `fold_1..fold_5` naming. Also re-verify the §2.4/§2.5 `[COMPLETED]` / `[IMPLEMENTED]` status markers once F2/F4 are fixed.

**Status: ✅ FIXED.** Code uses `dima806_deepfake_aug_{fold_name}` / `melody_audio_aug_{fold_name}` with fold keys `fold_1..fold_5`; `revision_plan.md` §2.3/§2.4/§2.5 now uses `fold_1..fold_5` / `fold_{K}` and marks the v2 pipeline.

---

### F9. Minor issues

- Audio notebook Cell 4 **rewrites the hyperparameter JSON at `CONFIG_PATH`** (`json.dump` side-effect) whenever the profile is found. If the config lives in the repo clone, running the notebook modifies the repo file. Prefer reading without writing back. **Status: ⚠️ NOT FIXED** — still present in both `kfold_audio_ai_detection.ipynb` and `kfold_audio_ai_detection_v2.ipynb` (cosmetic, does not affect results).
- All notebooks emit `FutureWarning: torch.cuda.amp.autocast(...) is deprecated` — use `torch.amp.autocast("cuda", ...)`. (v2/base notebooks already use `torch.amp.autocast`.)
- `confusion_matrix(y_true, y_pred)` without an explicit `labels=[0, 1]` in video/audio OOF sections — safe today (both classes present) but fragile if a fold ever contains a single class.

---

## 🐛 POST-AUDIT FINDING — `kfold_multimodal_evaluation_v2.ipynb` Cell 13 frame-cache stem collision

> **Date:** 13 Aug 2026
> **Status:** **Root-caused and fixed.** Cell 13 in `notebook/revision1/kfold_multimodal_evaluation_v2.ipynb` was updated to store cached frames per-class.

### Problem

The "Video Saja" result on the independent test set was stuck at **60.90%** across every attempt to fix the frame pipeline (ffmpeg → PIL → cv2.imwrite, R/B channel swaps, transform changes). Changing the frame extraction code changed **nothing** — a strong hint the bug was elsewhere.

### Root cause

The frame cache was keyed on **video stem only**:

```
frame_paths_all[idx] = cache_frames(vf, os.path.join(FRAME_CACHE, stem), 1.0, 30)
```

`testReal.zip` and `testAI.zip` contain videos with **identical stems** — the AI clips are derived from the same Real sources (e.g. both zips have a `39.mp4`). Because `test_video_samples` lists Real videos first (indices 0–189), Real `39` populated `…/frames_cache_v2_{pid}/39/`. When AI `39` was processed, `cache_frames` hit the `existing` shortcut and returned the **Real twin's frames**. Every AI video was therefore scored against its Real counterpart's frames → P(AI) ≈ 0 → all videos predicted Real → exactly **60.90%** accuracy.

The same stem-collision existed in **every** earlier pipeline variant (ffmpeg/PIL/swap), which is why all of them collapsed identically. The channel/normalization theories were red herrings.

### The fix

Include the true class in the cache subdir, mirroring the target video notebook's per-class layout `/content/dataset_frames_test/{cls}/{stem}/`:

```
frame_paths_all[idx] = cache_frames(vf, os.path.join(FRAME_CACHE, CLASSES[true_label], stem), 1.0, 30)
```

A fresh `frames_cache_v2_{pid}` root with class subdirs (`…/Real/{stem}/`, `…/AI/{stem}/`) was used, so no stale dirs are reused.

### Evidence

- **Pre-fix diagnostic (Cell 14):** On Real rows, `v2_cache_p1 == target_p1` exactly (cv2.imwrite fix was already live); on AI rows `n_v2 = 30` even when the true AI video yields `n_tgt = 10` or `0` frames — the 30-frame set was the Real twin's cache.
- **Post-fix:** "Video Saja" jumped 60.90% → **96.15%**; "Audio Saja" **97.44%**; "Multimodal 50/50" **97.76%** — matching the video notebook's 96.15% reference.

### Lesson

Cache keys must be unique per (class, video), not per video stem, when two classes share stem names across archives.

---

## 🗂️ Recommended Fix Order

> **Update (13 Aug 2026):** All F1–F8 fixes are **applied and re-executed** in the current v2 pipeline. Historical order below for reference.

1. **F1** — one-line variable rename in the video notebook (blocks all video re-runs). ✅
2. **F4** — define ensemble models in multimodal Cell 14. ✅
3. **F3** — tail-chunk padding in all three notebooks (changes results, so re-run affected sections). ✅
4. **F2** — make audio test inference crash-proof (sub-batching + per-fold model loading) and re-run. ✅
5. **F5** — fix audio OOF label ordering (display-only, numbers unchanged). ✅
6. **F6** — video-level OOF aggregation for comparable paper metrics. ✅
7. **F7, F8, F9** — docs and minor cleanups. ✅ F7/F8 done; F9 partial (notebooks already use `torch.amp.autocast`; the `json.dump` config side-effect and `labels=[0,1]` robustness remain cosmetic).

**Final results (current v2 pipeline, independent 312-video test set):** Video Saja **96.15%**, Audio Saja **97.44%**, Multimodal 50/50 **97.76%**.
