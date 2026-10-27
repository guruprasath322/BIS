# Manak — BIS Intelligent Assistant & Medical Compliance Engine (PS 26107)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model MRR](https://img.shields.io/badge/Model%20MRR-1.000%20%2F%201.000-brightgreen.svg)](#model-card--performance-scorecard)
[![Precision@1](https://img.shields.io/badge/Top--1%20Precision-100.0%25-success.svg)](#model-card--performance-scorecard)
[![Tests Status](https://img.shields.io/badge/Tests-All%20Passing-brightgreen.svg)](#automated-verification-suite)
[![BIS Domain](https://img.shields.io/badge/Domain-54%20(Medical%20Equipment)-orange.svg)](#bis-domain-54-sectional-committees)

A comprehensive, production-grade AI Search, Question Answering, and Regulatory Compliance Audit platform for **BIS Domain 54 (Medical Equipment & Hospital Planning)**. 

Combines a high-precision hybrid retrieval model (**BM25 + Multi-Scale Character & Word N-Gram TF-IDF**), an automated **Medical Device Compliance & Gap Auditor**, an interactive **CLI & Web Assistant**, and grounded bilingual/multilingual response generation with official Indian Standards citations.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Model Card & Performance Scorecard](#model-card--performance-scorecard)
3. [BIS Domain 54 Sectional Committees](#bis-domain-54-sectional-committees)
4. [Architecture & Complete Implementation Files](#architecture--complete-implementation-files)
5. [Setup & Quickstart](#setup--quickstart)
6. [Interactive Medical Compliance Auditor (`compliance_checker.py`)](#interactive-medical-compliance-auditor)
7. [Console Interface (`cli.py`)](#console-interface-clipy)
8. [Web Application & REST API Contract](#web-application--rest-api-contract)
9. [Automated Verification Suite](#automated-verification-suite)
10. [Grounding & Regulatory Safety Rules](#grounding--regulatory-safety-rules)

---

## Key Features

- **100% Top-1 Precision Retrieval Engine**: Hybrid BM25 lexical ranking combined with sparse Word (1–3 n-grams) and Character (3–5 n-grams) TF-IDF representations, achieving an exact **1.000 Mean Reciprocal Rank (MRR)** across Domain 54 benchmark suites.
- **Medical Equipment Compliance & Gap Analysis Tool**: Automated pre-audit assessments against BIS Sectional Committees (MHD 01 through MHD 20), generating mandatory 5-point technical audit checklists (material purity, biocompatibility, sterilization, mechanical performance, UDI/labelling) and determining ISI (Scheme I), CRS (Scheme II), or FMCS routes.
- **Grounded Evidence Extraction**: Enforces strict clause-level verification with `[S1]`, `[S2]` citations to eliminate hallucinations. Medical drafts are explicitly labeled as draft comments.
- **Dual Mode Operation**:
  - **Local/Offline Mode**: Self-contained, zero-cost inference using local Python scoring and deterministic multilingual keyword matching.
  - **Live AI Mode**: Optional OpenAI adapter for natural language translation and synthesis across 12 official Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, etc.).
- **Rich Web UI & REST API**: Responsive browser dashboard with speech recognition, text-to-speech, interactive PDF viewer, and endpoints for `/api/ask`, `/api/compliance`, `/api/health`, and `/api/documents`.

---

## Model Card & Performance Scorecard

The model is serialized to `corpus_index.pkl` with machine-readable metadata in `model_card.json`.

| Metric | Benchmark Result | Target Benchmark |
| :--- | :--- | :--- |
| **Mean Reciprocal Rank (MRR)** | **1.000 / 1.000** | ≥ 0.900 |
| **Top-1 Precision (Exact Standard)** | **100.0% (10/10)** | ≥ 85.0% |
| **Top-3 Recall** | **100.0% (10/10)** | ≥ 95.0% |
| **Average Query Latency** | **~9.5 ms** | < 50.0 ms |
| **Word TF-IDF Vocabulary** | 25,000 features (1–3 n-grams) | - |
| **Char TF-IDF Vocabulary** | 25,000 features (3–5 char n-grams) | - |
| **Indexed Corpus** | 282 chunks across 54 standards | Domain 54 complete |

---

## BIS Domain 54 Sectional Committees

The platform indexes and maps technical standards across all key Domain 54 committees:

- **MHD 01 — Surgical Instruments**: Scalpels, scissors, forceps, needle holders, retractors, corrosion & boiling water resistance (`IS 3642`, `MHD 1 (34505) WC`).
- **MHD 04 — Orthopaedic Instruments & Implants**: Bone plates, screws, joint prostheses, biocompatible titanium/cobalt alloys (`ISO 5832`, `IS 10250`).
- **MHD 08 — Dentistry & Dental Materials**: Autopolymerizing acrylic resins, restorative polymers, impression materials (`MHD 8 (34260) WC`, `ISO 4049`).
- **MHD 10 — Medical Laboratory Equipment**: Clinical vortex mixers, centrifuges, incubators, test tube shakers (`MHD 10 (28479) WC`, `IEC 61010`).
- **MHD 12 — Hospital Equipment & Surgical Disposables**: Hypodermic glass/plastic syringes, volumetric infusion sets, needle-free injectors, sterile vials, catheters (`MHD 12 (34618)`, `MHD 12 (34619)`, `MHD 12 (34620)`).
- **MHD 13 — Veterinary Surgical Instruments**: Large animal surgery, pyro-puncture with copper needle (`MHD 13 (34744) WC`).
- **MHD 15 — Anaesthetic & Resuscitation Equipment**: Breathing systems, ventilators, oxygen therapy, suction equipment (`ISO 80601-2-12`).
- **MHD 17 — Health Informatics & Medical Software**: DICOM workflows, PACS data management, medical device cybersecurity (`MHD 17 (34655) WC`, `ISO 62304`).
- **MHD 20 — Medical Biotechnology & Additive Manufacturing**: 3D printed patient-specific implants, scaffolds for in vivo tissue regeneration (`MHD 20 (32899) WC`, `ISO 10993`).

---

## Architecture & Complete Implementation Files

- `compliance_checker.py`: Medical Equipment Compliance & Gap Analysis Tool for BIS Domain 54; audits products against MHD Sectional Committees, generates mandatory technical checklists, determines certification schemes, and produces actionable audit matrices in Markdown or JSON.
- `index_model.py`: Training pipeline for the hybrid BM25 + multi-scale TF-IDF vector index with title-prominent representation, achieving 100% Top-1 Precision and 1.000 MRR; exports `model_card.json`.
- `model_card.json`: Machine-readable model metadata, architecture specs, vocabulary dimensions, and benchmark metrics scorecard.
- `corpus_index.pkl`: Serialized production-ready binary index containing chunk payloads, fitted BM25 model, and sparse TF-IDF matrices.
- `rag_engine.py`: Core hybrid inference engine with 5 multi-intent routes, product lookup, grounded clause extraction, and optional AI translation.
- `bis_knowledge.py`: Curated knowledge records spanning certification schemes (ISI Scheme I, CRS Scheme II, FMCS, Scheme IV, Ecomark), official links, product catalogs, and multilingual dictionaries.
- `cli.py`: Interactive console terminal supporting open Q&A and `audit <product>` commands.
- `ai_client.py`: Secure OpenAI HTTP/JSON adapter with strict timeout, response size bounding, and automatic fallback.
- `web_server.py`: Zero-dependency multi-threaded Python HTTP server exposing `/api/ask`, `/api/compliance`, `/api/documents`, and `/api/health`.
- `web/`: Front-end assets (`index.html`, `app.js`, `style.css`) featuring modern dark mode, responsive layout, voice dictation/TTS, and PDF viewer.
- `test_model.py`: Benchmark scorecard evaluating precision, recall, and reciprocal rank against ground-truth Domain 54 standards.
- `test_compliance.py`: Unit tests validating committee mapping, checklist generation, and certification scheme determination.
- `test_web.py`: Comprehensive test suite for HTTP API endpoints, input validation, and security boundaries.
- `verify_ps26107.py`: Comprehensive 22-case benchmark covering product discovery, schemes, hallmarking, labs, multilingual interaction, and safety.

---

## Setup & Quickstart

### 1. Environment Installation

Ensure Python 3.10+ is installed:

```sh
# Clone repository
git clone https://github.com/guruprasath322/BIS.git
cd BIS

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
