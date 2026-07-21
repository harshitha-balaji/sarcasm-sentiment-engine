# 🎭 Sarcasm Sentiment Engine (SSE)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=flat-square)](https://spacy.io/)
[![NLTK VADER](https://img.shields.io/badge/Sentiment-NLTK_VADER-orange?style=flat-square)](https://www.nltk.org/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

> A rule-based NLP engine for entity-focused sentiment analysis using heuristic coreference, sarcasm detection, and configurable domain lexicons.

---

## Overview

Most sentiment analyzers score an entire article. SSE instead evaluates sentiment **towards a specific entity** by identifying relevant sentences, tracking simple references, detecting common sarcasm patterns, and aggregating evidence into a final sentiment score.

The project was built to explore how far a rule-based NLP pipeline can go before requiring transformer-based models.

---

## Features

- Entity-focused sentiment analysis
- Heuristic pronoun and institutional reference tracking
- Rule-based sarcasm detection
- Configurable domain-specific lexicons
- Fully configuration-driven architecture

---

## Pipeline

```
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
Sentiment Scoring (VADER + Domain Lexicons)
    │
    ▼
Entity Sentiment Report
```

---

## Tech Stack

- Python
- spaCy
- NLTK VADER
- Regular Expressions
- JSON Configuration

---

## Project Structure

```
sarcasm_sentiment_engine/
├── sarcasm_sentiment_engine.py
├── domain_lexicons.json
├── engine_settings.json
└── requirements.txt
```

---

## Design

SSE intentionally uses a rule-based architecture to better understand the challenges of entity-aware sentiment analysis before introducing machine learning models. The engine separates configuration from implementation, allowing behaviour to be tuned without modifying the core logic.

---

## Limitations

- Coreference resolution is heuristic rather than model-based.
- Sarcasm detection relies on predefined linguistic patterns.
- Domain support depends on manually defined lexicons.

---

## Future Improvements

- Transformer-based sarcasm detection
- Multi-entity sentiment tracking
- Additional domain lexicons
- Web interface

---

*Built as an exploration of rule-based NLP for entity-aware sentiment analysis.*
