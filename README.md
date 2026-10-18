# 🎭 Sarcasm Sentiment Engine (SSE)

> A rule-based NLP pipeline for entity-focused sentiment analysis, using heuristic reference tracking, sarcasm detection, and configurable domain lexicons.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=flat-square)](https://spacy.io/)
[![NLTK VADER](https://img.shields.io/badge/Sentiment-NLTK_VADER-orange?style=flat-square)](https://www.nltk.org/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Overview

SSE is a lightweight NLP project that analyzes sentiment towards a specified entity in a piece of text. It identifies relevant sentences, tracks simple references to the entity, detects predefined sarcasm patterns, and combines the available evidence into an entity-focused sentiment result.

The implementation uses spaCy, NLTK VADER, handcrafted rules, and configurable domain lexicons rather than a trained end-to-end sentiment or sarcasm model. It is intended as an exploration of rule-based entity-aware sentiment analysis, including the challenges of reference tracking and interpreting sentiment in context.

---

## Features

- Entity-focused sentiment analysis
- Heuristic pronoun and institutional reference tracking
- Rule-based sarcasm detection
- Sentiment scoring using VADER and domain-specific lexicons
- Configuration-driven processing

---

## Pipeline

```text
Input Text
    │
    ▼
Evidence Extraction
    │
    ▼
Entity Tracking
    │
    ▼
Sarcasm Detection
    │
    ▼
Sentiment Scoring
    │
    ▼
Entity Sentiment Report
```

---

## Tech Stack

- Python
- spaCy
- NLTK VADER
- JSON Configuration

---

## Project Structure

```text
sarcasm_sentiment_engine/
├── sarcasm_sentiment_engine.py
├── domain_lexicons.json
├── engine_settings.json
└── requirements.txt
```
---

## Design

SSE separates evidence extraction, entity reference tracking, sarcasm detection, and sentiment scoring into distinct stages. Configuration files define domain-specific vocabulary and processing settings, allowing the behaviour of the pipeline to be adjusted without changing its core implementation.

The current approach relies on handcrafted rules and heuristic reference tracking. Its results therefore depend on the text, the patterns covered by the rules, and the quality of the configured lexicons.

---
## Limitations

- Entity reference tracking is heuristic and may misidentify pronouns or indirect references.
- Sarcasm detection relies on predefined linguistic patterns and may miss context-dependent or subtle sarcasm.
- Sentiment scores may not capture the intended meaning of complex or ambiguous statements.
- Domain coverage depends on the available lexicons and configuration.
- Performance has not been established across all domains and text types.
  
---

## Future Improvements

- Transformer-based sarcasm detection
- Support for analyzing sentiment towards multiple entities
- Additional domain lexicons
- Web interface
---

*An exploration of rule-based NLP for entity-aware sentiment analysis.*
