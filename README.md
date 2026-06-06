# SecurTalent Framework

**A Deterministic Defense Framework Against Sub-Visual Whitelisting and Semantic Injection Attacks in Automated Document Parsing Pipelines via Low-Level Object Metadata Auditing and Vector Space Similarity Mapping**

---

## Overview

SecurTalent is an academic research prototype that detects and neutralizes adversarial content hidden within PDF documents before they enter automated parsing pipelines (e.g., resume screeners, document intake systems, RAG ingestion flows).

## Threat Model

This framework addresses three primary attack surfaces:

### 1. Sub-Visual Whitelisting Attacks
Adversaries embed text rendered at near-zero contrast (e.g., white-on-white, `#fefefe` on `#ffffff`) to inject keywords, skills, or credentials that are invisible to human reviewers but parsed and indexed by automated systems. The framework detects these by computing **W3C relative luminance** and **WCAG 2.1 contrast ratios** at the individual glyph level, flagging any character whose contrast ratio against the page background falls below a configurable threshold.

### 2. Semantic Injection Attacks
Attackers craft prompt-like or instruction-like payloads within documents designed to manipulate downstream NLP models, LLM-based extractors, or classification pipelines. SecurTalent employs **vector space similarity mapping** using sentence-level embeddings to measure the semantic distance between document segments and known adversarial prompt templates, flagging segments that fall within a suspicious similarity radius.

### 3. Provenance Spoofing
Malicious actors tamper with PDF metadata fields (author, creator, producer, modification dates) to fabricate document provenance, bypass origin-based trust checks, or evade forensic attribution. The framework audits raw PDF metadata objects and cross-references structural markers to detect inconsistencies indicative of forgery.

## Architecture

```
securtalent-framework/
├── core/                    # Detection engines
│   ├── integrity_guard.py   # Sub-visual attack detection (contrast auditing)
│   ├── nlp_engine.py        # Semantic injection detection (embedding similarity)
│   └── provenance_audit.py  # Metadata provenance verification
├── pipeline/
│   └── fusion_engine.py     # Multi-signal fusion and scoring
├── interface/
│   └── dashboard.py         # Streamlit-based analysis dashboard
├── research/
│   ├── datasets/            # Evaluation corpora (adversarial + benign)
│   └── evaluation_bench.py  # Benchmarking and metric computation
├── requirements.txt
└── README.md
```

## Status

🔬 **Academic prototype** — not intended for production deployment. This software is provided for research and educational purposes under the terms of the accompanying license.

## Getting Started

```bash
pip install -r requirements.txt
python -m core.integrity_guard   # Run Module 1 self-test
```

## Citation

If you use this framework in academic work, please cite the accompanying paper (forthcoming).
