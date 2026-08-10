# 🛡️ Project Mancing-Tipu — AI Agent Ecosystem & System Vision

> **Deteksi AI-Generated Video dan Identifikasi Potensi Penipuan Digital Berbasis Multimodal Deep Learning**  
> *Gemastik 2026 Project Vision & Technical Architecture Plan*

---

## 👁️ Core Vision & Mission

In recent years, the rapid democratization of **Generative Artificial Intelligence (GenAI)** has enabled hyper-realistic video generation and voice cloning. While technological advancement brings immense potential, it has simultaneously fueled a dangerous wave of **digital fraud and information disorder**. 

In Indonesia, financial scams leveraging AI-generated media—such as deepfake videos of public officials promoting fraudulent motor giveaways or deceptive investment schemes—continue to proliferate across digital communication platforms, causing widespread financial harm and deceiving citizens.

### The Critical Gap in Existing Moderation
Traditional content moderation systems suffer from key structural limitations:
1. **Binary-Only Focus:** Existing computer vision models stop at classifying content as `Real` or `Fake`. They cannot answer the vital question: *"Is this fake video a harmless parody or a malicious financial scam?"*
2. **Disconnected Text/NLP Approaches:** Text-based fraud classifiers operate purely on written transcripts, completely missing visual and auditory deepfake cues.
3. **Lack of Localized Context:** Generic global models fail to capture Indonesian linguistic nuances, regional accents, and local scam tactics.

### Our Solution: Intent-Aware Multimodal Intelligence
Project **Mancing-Tipu** introduces a **Two-Stage Multimodal Deep Learning Architecture** designed to bridge this gap. Our system not only verifies whether a video is real or AI-generated, but also transcribes and analyzes the underlying message to assess **fraud potential**. This turns passive deepfake detection into **actionable intelligence** for digital platform moderation, public early-warning systems, and law enforcement.

---

## 🤖 The Agentic System Architecture & Technical Plan

The system is conceptualized as a team of specialized, autonomous intelligence agents working in synergy across a multi-stage pipeline:

```
                  +-------------------------------------------------------+
                  |  [Agent 1] Social Media Ingestion & Discovery Agent  |
                  +-------------------------------------------------------+
                                              |
                                              v
                            +-----------------------------------+
                            |        Input Video (.mp4)         |
                            +-----------------------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
  +-------------------------------------+           +-------------------------------------+
  |   [Agent 2] Visual Forensic Agent   |           |  [Agent 3] Acoustic Forensic Agent  |
  |      (Vision Transformer - ViT)     |           |     (Audio Transformer - Wav2Vec2)  |
  +-------------------------------------+           +-------------------------------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              | (Visual & Audio Scores)
                                              v
                            +-----------------------------------+
                            | [Agent 4] Multimodal Arbiter Agent|
                            |       (Late Fusion Ensemble)      |
                            +-----------------------------------+
                                              |
                                      [Is AI-Generated?]
                                      /                \
                                    YES                 NO
                                    /                    \
                                   v                      v
        +-------------------------------------+     [Verified Real Video]
        |   [Agent 5] Speech Perception Agent |
        |         (Whisper Large V3 ASR)      |
        +-------------------------------------+
                           | (Indonesian Transcript)
                           v
        +-------------------------------------+
        |  [Agent 6] Semantic Intent Agent    |
        |             (IndoBERT)              |
        +-------------------------------------+
                           |
                     [Classification]
                     /              \
                    v                v
            🚨 [FRAUD DETECTED]   🟢 [NON-FRAUD / HARMLESS]
```

---

### 📥 Agent 1: Social Media Ingestion & Scraping Agent
- **Core Role:** Automated content collection, media extraction, and directory organization.
- **Technical Plan & Responsibilities:**
  - Monitors digital platforms (TikTok, Instagram, Facebook) for trending videos and reported media.
  - Automatically downloads, parses metadata, and structures video content for downstream processing.
- **Repository Reference:** Implemented under [`script-download/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/script-download).

---

### 👁️ Agent 2: Visual Forensic Agent (Vision Specialist)
- **Core Role:** Spatial & temporal visual deepfake detection.
- **Model Backbone:** Vision Transformer (`dima806/deepfake_vs_real_image_detection`).
- **Technical Plan & Responsibilities:**
  - Extracts representative video frames at optimal frame rates (1 FPS).
  - Inspects visual micro-artifacts, boundary anomalies around face-swaps, unnatural blinking, facial warping, and temporal inconsistencies.
  - Outputs a calibrated visual authenticity confidence score $P_{\text{video}}$.
- **Repository Reference:** Fine-tuned in [`notebook/gemastik_video_ai_detection.ipynb`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/gemastik_video_ai_detection.ipynb).

---

### 🎙️ Agent 3: Acoustic Forensic Agent (Audio Specialist)
- **Core Role:** Auditory & voice cloning deepfake detection.
- **Model Backbone:** Wav2Vec2 Audio Transformer (`MelodyMachine/Deepfake-audio-detection-V2`).
- **Technical Plan & Responsibilities:**
  - Extracts 16kHz mono audio streams from raw video files.
  - Chunks full audio timelines into 5-second overlapping segments to ensure 100% audio coverage without information loss.
  - Detects voice synthesis artifacts, unnatural prosody, robotic spectral patterns, and acoustic discrepancies.
  - Outputs a calibrated audio authenticity confidence score $P_{\text{audio}}$.
- **Repository Reference:** Fine-tuned in [`notebook/gemastik_audio_ai_detection.ipynb`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/gemastik_audio_ai_detection.ipynb).

---

### ⚖️ Agent 4: Multimodal Arbiter & Fusion Agent
- **Core Role:** Cross-modal decision fusion and final authenticity determination.
- **Mechanism:** Late Fusion Ensemble:
  $$P_{\text{multimodal}} = w_{\text{video}} \cdot P_{\text{video}} + w_{\text{audio}} \cdot P_{\text{audio}}$$
- **Technical Plan & Responsibilities:**
  - Synthesizes independent signals from both visual and acoustic specialists.
  - Resolves cross-modal ambiguity (e.g., when audio sounds real but face is manipulated, or vice-versa).
  - Computes a unified authenticity score to reliably separate Real from AI-Generated videos.
  - Routes verified `AI-Generated` content to Stage 2 for semantic risk analysis.
- **Repository Reference:** Evaluated in [`notebook/gemastik_multimodal_fusion_evaluation.ipynb`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/gemastik_multimodal_fusion_evaluation.ipynb).

---

### 🗣️ Agent 5: Speech Perception Agent (ASR Transcription)
- **Core Role:** Converts acoustic speech in flagged AI videos into structured Indonesian text.
- **Model Backbone:** Whisper Large V3.
- **Technical Plan & Responsibilities:**
  - Transcribes spoken audio into clean, normalized text while capturing colloquial Indonesian dialects, code-switching, and conversational nuances.
  - Prepares textual payloads for semantic fraud evaluation.
- **Repository Reference:** Executed in [`scripts/transcribe.py`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/scripts/transcribe.py).

---

### 🧠 Agent 6: Semantic Intent & Fraud Risk Agent
- **Core Role:** Natural Language Understanding & Fraud Intent Classification.
- **Model Backbone:** IndoBERT (Indonesian Language Representation Model).
- **Technical Plan & Responsibilities:**
  - Analyzes ASR transcripts to understand semantic context and user targeting.
  - Classifies content into **Fraud** (phishing, illegal motor auctions, fake giveaway claims, investment scams) vs **Non-Fraud** (harmless AI dubs, creative commentary, parodies).
  - Provides contextual classification to effectively distinguish fraudulent intent from benign content.
- **Repository Reference:** Evaluated in [`notebook/gemastik_multimodal_fusion_evaluation.ipynb`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/gemastik_multimodal_fusion_evaluation.ipynb).

---

## 🎯 Societal Impact & Actionable Intelligence

By combining deepfake detection with semantic fraud analysis, **Project Mancing-Tipu** transforms content moderation from a reactive filter into a proactive defense mechanism:

1. **Automated Social Platform Moderation:** Platforms like TikTok, Meta, and YouTube can automatically prioritize high-risk, high-confidence fraudulent videos for immediate removal before viral spread occurs.
2. **Law Enforcement & Forensics Support:** Provides police cyber-crime units (such as Polda Metro / Polda Jatim) with rapid forensic reports identifying both the synthetic media vector and the criminal scam intent.
3. **Public Empowerment & Verification:** Enables citizens to submit suspicious digital videos for instantaneous verification, reducing public susceptibility to financial scams and digital impersonation.

---

## 🗂️ Workspace Architecture Overview

- 📓 **[`notebook/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook)**: Core experimentation and pipeline evaluation notebooks.
  - [`gemastik_video_ai_detection.ipynb`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/gemastik_video_ai_detection.ipynb): Fine-tuning Vision Transformer for video visual manipulation detection.
  - [`gemastik_audio_ai_detection.ipynb`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/gemastik_audio_ai_detection.ipynb): Fine-tuning Wav2Vec2 for audio synthetic voice detection.
  - [`gemastik_multimodal_fusion_evaluation.ipynb`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/notebook/gemastik_multimodal_fusion_evaluation.ipynb): Multimodal late-fusion evaluation and IndoBERT fraud classification.
- 📜 **[`scripts/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/scripts)**: Helper utilities for dataset sampling and ASR transcription.
- 📥 **[`script-download/`](file:///home/putra/Putra/Lomba/Gemastik-Mining/mancing-tipu/script-download)**: Social media media scraper and data ingestion pipeline.
