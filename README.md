# 🎭 Sarcasm Sentiment Engine (SSE)

> **Domain-aware, entity-tracking NLP engine** — follows a named entity through news text, resolves pronoun coreference, detects sarcasm, and scores only the sentences that are actually *about* that entity. No ML models. Pure linguistic architecture.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=flat-square)](https://spacy.io/)
[![NLTK VADER](https://img.shields.io/badge/Sentiment-NLTK_VADER-orange?style=flat-square)](https://www.nltk.org/)
![Config](https://img.shields.io/badge/architecture-config--driven-yellow?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## What is this?

Standard sentiment tools score an entire article. SSE scores a specific *entity* inside it.

Give it a paragraph and a target — a politician, institution, or organisation — and it tracks that entity sentence by sentence: resolving pronouns, catching institutional synonyms, detecting sarcasm, and producing a per-sentence evidence table with a final stance verdict.

```
Enter News Paragraph:
> The administration promised reform. They delivered a flawless masterclass in gridlock.

Enter Political Entity:
> administration

════════════════════════════════════════════════════════════════════════════════════
POLITICAL SENTIMENT REPORT: ADMINISTRATION
OVERALL STANCE: NEGATIVE (-0.6231)
════════════════════════════════════════════════════════════════════════════════════
Sentence Evidence                                  | Sarcasm? | Sentiment  | Score
------------------------------------------------------------------------------------
The administration promised reform.                | No       | POSITIVE   | 0.2960
They delivered a flawless masterclass in gridlock. | Yes      | NEGATIVE   | -0.7100
════════════════════════════════════════════════════════════════════════════════════
```

---

## Features

- **Entity-aware evidence extraction** — collects only sentences relevant to the target entity, not the full article
- **Three-stage coreference window** — tracks pronouns and institutional synonyms that refer back to the target across sentence boundaries
- **Sarcasm detection** — surface cue matching, structural marker detection (ALL-CAPS + `!!`), and conditional regex patterns (`if X... disaster`)
- **Domain lexicons** — custom VADER score overrides for political and financial vocabulary via `domain_lexicons.json`
- **Fully config-driven** — all thresholds, penalties, cues, and triggers live in `engine_settings.json`; tune behavior without touching code
- **Multi-domain support** — switch between political and finance lexicon profiles; extend to any domain by adding a new profile

---

## How It Works

Three sequential stages process each input:

### 1. Evidence Extraction
Three-condition sentence collector that builds the entity's evidence set:
- **Condition A** — sentence directly names the target entity → include
- **Condition B** — sentence contains an institutional synonym before the entity appears → include as context setup
- **Condition C** — sentence follows a matched sentence and opens with a tracked pronoun → include via coreference window

### 2. Sarcasm Detection
Three independent signals checked per sentence, each gated by a **negative backdrop check** — sarcasm only fires if the article contains at least one global crisis word, preventing false flips in genuinely positive text:
- **Cue matching** — surface phrase list from `engine_settings.json` (e.g. "flawless masterclass", "oh great")
- **Structural markers** — `!!` combined with ALL-CAPS words signals exaggerated irony
- **Conditional regex** — pattern `if [X]... [disaster-word]` with a positive base score triggers a sentiment flip

### 3. Scoring
VADER compound score per sentence, adjusted by:
- Domain lexicon overrides (political / financial vocabulary)
- Sarcasm flip: positive score → `−|score|`
- Irony floor penalty if flipped score is too mild
- Conditional sarcasm multiplier for regex-triggered flips

---

## Project Structure

```
sarcasm_sentiment_engine/
├── sarcasm_sentiment_engine.py   # Core engine + CLI entry point
├── domain_lexicons.json          # Domain VADER overrides + institutional synonyms
├── engine_settings.json          # All thresholds, cues, penalties, and triggers
└── requirements.txt
```

`sarcasm_sentiment_engine.py` contains only logic — never hardcoded values. All operational parameters live in config.

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

Enter Political Entity (Person/Party/Institution):
> [e.g. Treasury, Labour Party, Fed]
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

Domain profiles live under `political` or `finance`. Each profile contains:
- `lexicon` — word → VADER score override pairs
- `institutional_synonyms` — terms treated as stand-ins for the target entity

To add a new domain (e.g. `sports`, `legal`), add a top-level key with the same structure and load it in the engine constructor.

---

## Design Decisions

**Why no ML model?**
SSE was built rule-based deliberately — to understand the linguistic problem from first principles before reaching for a transformer. The ceiling of this approach (novel irony, context-dependent sarcasm) directly motivates why models like `cardiffnlp/twitter-roberta-base-irony` exist. The architecture is designed so that stage 2 can be swapped for a model-based classifier without touching stages 1 or 3.

**Why config-driven?**
Hardcoded thresholds make tuning impossible without reading engine logic. Externalising them means you can adjust behavior for a new domain, stress-test edge cases, or hand the project to someone else — without touching the core pipeline.

---

## Roadmap

- [ ] Swap stage 2 for `cardiffnlp/twitter-roberta-base-irony` — model-based sarcasm classifier
- [ ] Multi-entity tracking in a single pass
- [ ] Paragraph-level chunking for full articles
- [ ] Sports and legal domain lexicon profiles
- [ ] Streamlit interface for non-CLI use

---

## Limitations

- Sarcasm detection relies on surface cues — novel irony not in the cue list will be missed
- Negative backdrop check is article-wide, not sentence-local — may cause false flips in mixed-tone articles
- Coreference resolution is heuristic (pronoun + position), not structural — complex multi-paragraph chains may break
- Works best on political and financial news; other domains need a new lexicon profile
---
*Built rule-first. The ceiling of this approach is the reason transformers exist.*
