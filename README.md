# Intell Audio Inference & Retrieval System

A local-first, intent-aware temporal audio intelligence and grounded RAG platform.

## What Is Intell Audio?

Originally created in 2022 as a basic PocketSphinx + Gentle audio search tool, Intell Audio has evolved in 2026 into an enterprise-grade audio intelligence platform combining temporal forced alignment, semantic chunking, speaker-turn segmentation, automatic chapters, content understanding, and grounded RAG.

## Why This Project Was Rebuilt

**2022:**
- PocketSphinx
- Basic transcript generation
- Gentle alignment
- CSV-based search
- Streamlit monolith

**2026:**
- Modular multi-tier Python architecture
- Pluggable ASR / alignment engine abstractions
- Word-boundary preserving temporal chunking
- Heuristic speaker-turn segmentation & speaker assignment
- Semantic content analysis (topics, intents, actions, targets, tools, procedure steps)
- Automatic chapter boundary detection & title synthesis
- BM25 Okapi lexical indexing
- Local Qwen3 vector embeddings + Qdrant vector storage
- Hybrid Reciprocal Rank Fusion (RRF) retrieval
- Deterministic application-side timestamp citations
- FastAPI backend + Streamlit frontend

## Core Pipeline

Audio
→ ASR (PocketSphinx / Whisper-ready)
→ Gentle Forced Alignment
→ Temporal Transcript Generation
→ Word-Boundary Chunking
→ Speaker-Turn Segmentation
→ Speaker Assignment
→ Content Semantic Analysis (optional)
→ BM25 Lexical Indexing
→ Qwen3 Embedding Generation
→ Qdrant Vector Upsert (enriched payloads)
→ Chapter Boundary & Title Generation
→ Hybrid Retrieval (BM25 + Qdrant)
→ Reranking
→ Qwen3 Reasoning
→ Grounded Answer with Resolved Timestamps


## Features

### Audio ingestion
### Transcription
### Forced alignment
### Temporal indexing
### Hybrid retrieval
### Ask the Audio
### Timestamp citations

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| API | FastAPI |
| Storage | SQLite |
| Lexical retrieval | BM25 |
| Vector DB | Qdrant |
| Embeddings | Qwen3-Embedding-0.6B |
| LLM | Qwen3-8B |
| Local inference | LM Studio |
| Alignment | Gentle |
| Audio | MoviePy / Pydub |
| Testing | Pytest |

## Project Structure

[tree]

## Requirements

Hardware:
- NVIDIA RTX 4060 8GB recommended
- 16GB+ system RAM recommended
- Docker Desktop
- Python 3.11+

Software:
- LM Studio
- Qdrant
- Gentle
- Python

## Installation

git clone ...
cd ...
python -m venv .venv
...
pip install -r requirements.txt

## Local Services

### 1. Gentle

docker ...

### 2. Qdrant

docker ...

### 3. LM Studio

Explain model configuration.

## Configuration

.env example

## Running

Terminal 1:
Gentle

Terminal 2:
Qdrant

Terminal 3:
LM Studio

Terminal 4:
FastAPI

Terminal 5:
Streamlit

## Usage

1. Upload audio
2. Process
3. Index
4. Ask questions
5. Open timestamp citation

## API

Health
Ingest
Index
Search
Ask

## Testing

pytest

## Example Questions

"What topics were discussed?"
"When was X mentioned?"
"What did the speaker say about Y?"
"Summarize the discussion about Z."

## Design Principles

### Local-first
### Evidence-grounded
### Temporal retrieval
### Deterministic citations
### Modular providers

## Current Limitations

Be honest about:
- ASR engine
- speaker diarization
- model context limits
- local inference performance
- YouTube dependency
- Gentle limitations

## Roadmap

Phase 1 — Architecture ✓
Phase 2 — Modern ASR
Phase 3 — Temporal Transcript ✓
Phase 4 — Intelligence
Phase 5 — Intelligent Retrieval ✓
Phase 6 — Ask the Audio ✓
Phase 7 — Speaker Intelligence
Phase 8 — Productionization

## Project Evolution

2022 → 2026