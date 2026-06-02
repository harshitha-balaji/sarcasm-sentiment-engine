# 🎭 Sarcasm Sentiment Engine (SSE)

> **Entity-aware political sentiment analysis pipeline** — tracks named entities through news text, resolves pronoun coreference, detects sarcasm, and generates a structured sentiment report. No ML models, pure linguistic architecture.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=flat-square)](https://spacy.io/)
[![NLTK VADER](https://img.shields.io/badge/Sentiment-NLTK_VADER-orange?style=flat-square)](https://www.nltk.org/)
[![HuggingFace](https://img.shields.io/badge/Lexicons-Config_Driven-yellow?style=flat-square)](https://github.com/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## What is this?

Most sentiment tools score an entire article and stop there. SSE tracks **a specific named entity** through a news paragraph — resolving pronouns, following institutional references, and detecting sarcasm — then scores only the sentences actually *about* that entity.

```
Enter News Paragraph:
> The administration promised reform. They delivered a flawless masterclass in gridlock.

Enter Political Entity:
> administration

═══════════════════════════════════════════════════════════════
POLITICAL SENTIMENT REPORT: ADMINISTRATION
OVERALL STANCE: NEGATIVE (-0.6231)
═══════════════════════════════════════════════════════════════
Sentence Evidence                                  | Sarcasm? | Sentiment  | Score
-----------------------------------------------------------
The administration promised reform.                | No       | POSITIVE   | 0.2960
They delivered a flawless masterclass in gridlock. | Yes      | NEGATIVE   | -0.7100
═══════════════════════════════════════════════════════════════
```

---

## Features

- **Entity-aware evidence extraction** — isolates only sentences relevant to the target entity, not the full article
- **Three-stage coreference window** — tracks pronouns and institutional synonyms that refer back to the target across sentence boundaries
- **Sarcasm detection** — surface cue matching, structural marker detection (ALL-CAPS + `!!`), and conditional regex patterns (`if X... disaster`)
- **Domain lexicons** — custom VADER score overrides for political and financial vocabulary via `domain_lexicons.json`
- **Fully config-driven** — all thresholds, penalties, cues, and triggers live in `engine_settings.json`; tune behavior without touching code
- **Multi-domain support** — switch between political and finance lexicon profiles

---

## How It Works

The engine runs three core linguistic stages:

### 1. Evidence Extraction (The "Lexicon" Layer)
Three-condition sentence collector:
- **Condition A** — sentence directly names the target entity → include
- **Condition B** — sentence contains an institutional synonym before the entity appears → include as context setup
- **Condition C** — sentence follows a matched sentence and starts with a tracked pronoun → include via coreference window

### 2. Sarcasm Detection (The "Heuristic" Layer)
Three independent sarcasm signals, checked per sentence:
- **Cue matching** — surface phrase list from `engine_settings.json` (e.g. "flawless masterclass", "oh great")
- **Structural markers** — `!!` combined with ALL-CAPS words signals exaggerated irony
- **Conditional regex** — pattern `if [X]... [disaster-word]` with a positive base score triggers a sentiment flip

All signals are gated by a **negative backdrop check** — sarcasm only fires if the article contains at least one global crisis word, preventing false flips in genuinely positive articles.

### 3. Scoring (The "Engine" Layer)
VADER compound score per sentence, adjusted by:
- Domain lexicon overrides (political/financial vocabulary)
- Sarcasm flip: positive score → `−|score|`
- Irony floor penalty if flipped score is too mild
- Conditional sarcasm multiplier for regex-triggered flips

---

## Project Structure

```
sarcasm_sentiment_engine/
├── sarcasm_sentiment_engine.py   # Core engine + CLI entry point
├── domain_lexicons.json          # Political & finance VADER overrides + institutional synonyms
├── engine_settings.json          # All thresholds, cues, penalties, and triggers
└── requirements.txt
```

The engine is deliberately config-driven — `sarcasm_sentiment_engine.py` contains only logic, never hardcoded values.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Named Entity Recognition & tokenization | `spaCy` (en_core_web_sm) |
| Base sentiment scoring | `NLTK VADER` |
| Regex pattern matching | `re` (stdlib) |
| Config loading | `json` (stdlib) |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon
```

### 2. Run the engine

```bash
python sarcasm_sentiment_engine.py
```

### 3. Follow the prompts

```
Enter News Paragraph/Article Snippet:
> [paste your text here]

Enter Political Entity (Person/Party/Country):
> [e.g. Biden, Labour Party, Treasury]
```

Press `0` to exit.

---

## Configuration

### `engine_settings.json`

| Parameter | What it controls |
|-----------|-----------------|
| `memory_limit` | How many sentences the pronoun tracking window stays open after a match |
| `leading_chunk_size` | Character window scanned for pronoun subjects at sentence start |
| `irony_floor_penalty` | Minimum negative score applied when sarcasm flips a weak positive |
| `sentiment_threshold` | Boundary between POSITIVE / NEUTRAL / NEGATIVE labels |
| `conditional_multiplier` | Score amplifier for regex-triggered conditional sarcasm |
| `negative_backdrop_triggers` | Words that activate global sarcasm gating for the article |
| `sarcasm_cues` | Surface phrase list for direct sarcasm detection |

### `domain_lexicons.json`

Add or edit domain profiles under `political` or `finance`. Each domain has:
- `lexicon` — word → VADER score override pairs
- `institutional_synonyms` — terms treated as stand-ins for the target entity

To add a new domain (e.g. sports), add a new top-level key with the same structure and load it in the engine constructor.

---

## Design Decisions

**Why no ML model?**
This project was built deliberately rule-based to understand the linguistic problem from first principles. The ceiling of this approach — sarcasm that doesn't match surface cues, context-dependent irony — directly motivates why transformer models exist. Consider this the architectural foundation for a future hybrid system.

**Why config-driven?**
Hardcoded thresholds and lexicons make tuning impossible without reading code. Externalizing them lets you adjust behavior for a new domain, test edge cases, or hand the project to someone else without touching the engine logic.

---

## Roadmap

- [ ] Transformer sarcasm layer — replace cue matching with `cardiffnlp/twitter-roberta-base-irony`
- [ ] Multi-entity tracking in one pass
- [ ] Web UI / Streamlit interface
- [ ] Support for longer articles with paragraph-level chunking
- [ ] Sports and legal domain lexicons

---

## Limitations

- Sarcasm detection relies on surface cues — novel irony patterns not in the cue list will be missed
- Negative backdrop check is article-wide, not sentence-local — may cause false flips in mixed-tone articles
- Coreference resolution is heuristic (pronoun + position), not structural — complex multi-paragraph chains may break
- Works best on political and financial news; other domains need a new lexicon profile

---

*Built as a rule-based NLP project to explore entity-aware sentiment analysis and the linguistic boundaries of lexicon-driven sarcasm detection.*
