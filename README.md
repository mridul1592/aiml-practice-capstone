# Agricultural Knowledge RAG System — Multilingual Edition

A production-grade Retrieval-Augmented Generation (RAG) system that delivers reliable agricultural advisory to farmers in India in their preferred language (English, Hindi, or Punjabi). Supports **text and audio** queries, hybrid retrieval, cross-encoder reranking, and source attribution.

---

## 🎯 Overview

The system addresses agricultural knowledge accessibility by:
- Aggregating curated agricultural literature (PoP manuals, advisory bulletins, IPM guides)
- Parsing, cleaning, and chunking documents with metadata extraction
- Retrieving the most relevant passages via **hybrid search** (BM25 + dense semantic) and **cross-encoder reranking**
- Generating grounded, source-attributed responses in the user's language
- Suppressing hallucinations by requiring every claim to map back to retrieved context

### Target Users
Farmers in Punjab and Haryana

### Supported Crops
- Wheat
- Paddy (Rice)

### Use Cases
- Pest / disease diagnosis & control advisory
- Fertilizer & nutrient guidance
- Irrigation scheduling
- Seasonal best practices
- Variety selection

### Supported Languages
Primary: **English, Hindi, Punjabi**
Detected: English, Hindi, Punjabi, Marathi, Bengali, Odia, Tamil, Telugu, Kannada, Malayalam

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
   PDFs / Audio ──▶ │  Ingestion Layer    │
                    │  • PDF Parser       │
                    │  • Chunking         │
                    │  • Metadata Extract │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Embedding Layer    │
                    │  BAAI/bge-large-    │
                    │  en-v1.5 (1024-d)   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  FAISS Vector Store │ ◀── metadata.json
                    └──────────┬──────────┘
                               │
   Query ──▶ Lang Detect ──▶  ▼
                    ┌─────────────────────┐
                    │  Hybrid Retriever   │
                    │  BM25 + Dense → RRF │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  BGE Cross-Encoder  │
                    │  Reranker (top-k)   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  LLM Generator      │
                    │  Ollama / OpenAI    │
                    └──────────┬──────────┘
                               ▼
                  ┌────────────────────────┐
                  │ Grounded Response with │
                  │  Source Attribution    │
                  └────────────────────────┘
```

---

## 📋 Project Structure

```
aiml-practice-capstone/
├── data/
│   ├── raw/                       # Source PDFs
│   ├── processed/                 # Cleaned chunks (JSON)
│   └── embeddings/                # FAISS index + metadata.json
│
├── ingestion/                     # Document ingestion pipeline
│   ├── pdf_parser.py              # PyMuPDF-based extraction
│   ├── metadata_extractor.py      # Crop / region / season tagging
│   ├── chunking.py                # Token-aware chunking
│   └── orchestrator.py            # Batch ingestion driver
│
├── rag/                           # Retrieval-Augmented Generation
│   ├── embedder.py                # BAAI/BGE + SentenceTransformer routing
│   ├── vector_store.py            # FAISS index management
│   ├── retriever.py               # Hybrid BM25 + Dense + RRF + lang detect
│   ├── reranker.py                # BAAI/bge-reranker-base cross-encoder
│   ├── generator.py               # LLM client (Ollama / OpenAI)
│   ├── prompt_templates.py        # Multilingual prompt scaffolding
│   ├── embeddings_orchestrator.py # End-to-end ingest + embed + index
│   └── rag_orchestrator.py        # Query-time pipeline coordinator
│
├── audio/                         # Voice query support
│   ├── audio_processor.py         # Whisper transcription
│   └── audio_rag_handler.py       # Audio → RAG bridge
│
├── api/                           # FastAPI backend
│   └── fastapi_app.py
│
├── ui/                            # Streamlit frontend
│   └── streamlit_app.py           # Chat interface with reranker toggle
│
├── tests/                         # Unit / integration tests
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   └── test_rag.py
│
├── utils/
│   ├── config.py                  # Pydantic settings
│   └── logger.py
│
├── requirements.txt
├── .env                           # Environment configuration
└── main.py                        # CLI entry point
```

---

## 🚀 Quick Start

### 1. Setup Python Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Setup LLM (Ollama recommended)

```bash
# Install from https://ollama.ai
ollama pull mistral        # default model
ollama serve               # start the daemon
```

Or use OpenAI by setting `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`.

### 3. Configure `.env`

Defaults work out of the box. Adjust as needed:

```bash
# LLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Embedding (1024-dim, English-optimized)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024

# Reranker
USE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-base

# Retrieval
CHUNK_SIZE=200
CHUNK_OVERLAP=40
SIMILARITY_THRESHOLD=0.5
TOP_K_RETRIEVAL=5
LANGUAGE_DETECT_THRESHOLD=0.5
```

### 4. Build the Index

Place PDFs in `data/raw/` (or any directory you point `--pdf-dir` at), then:

```bash
# One-shot: ingest + embed + build FAISS index
python main.py embed --pdf-dir Agri_docs
```

### 5. Query

```bash
# Text query
python main.py query --query "How to control wheat pests?"

# Multilingual
python main.py query --query "गेहूँ में कीटों का नियंत्रण कैसे करें?"

# With filters
python main.py query --query "Paddy irrigation" --crop paddy --region punjab --k 10

# Audio query (Whisper transcribes, then RAG answers)
python main.py audio-query --audio-file query.mp3 --model-size base
```

### 6. Launch UI / API

```bash
# Streamlit chat UI (http://localhost:8501)
python main.py ui

# FastAPI backend (http://localhost:8000/docs)
python main.py api
```

---

## 🧩 CLI Reference

The `main.py` entry point exposes seven subcommands:

| Command | Description |
|---------|-------------|
| `ingest`      | Parse PDFs into cleaned JSON chunks (no embeddings) |
| `embed`       | Full pipeline: parse → chunk → embed → build FAISS index |
| `query`       | Run a text RAG query against the existing index |
| `audio-query` | Run a RAG query from an audio file (mp3/wav/m4a/flac/ogg) |
| `api`         | Start the FastAPI backend server |
| `ui`          | Start the Streamlit chat interface |
| `test`        | Run a specific test module |

Run `python main.py <command> --help` for per-command flags.

---

## 🔑 Key Features

### Hybrid Retrieval (BM25 + Dense + RRF)
First-stage retrieval combines lexical (BM25) and semantic (FAISS dense) search, merged via **Reciprocal Rank Fusion (RRF)**. This captures both exact-keyword matches (crop varieties, chemical names, dosages) and paraphrased semantic matches.

### Cross-Encoder Reranking
Top-`k` candidates from hybrid retrieval are rescored by **BAAI/bge-reranker-base**, a cross-encoder that jointly attends over `(query, passage)` pairs — significantly more accurate than bi-encoder similarity. Toggleable via `USE_RERANKER` in `.env` or per-request from the UI.

### BGE Embedding Pipeline
- **Model:** `BAAI/bge-large-en-v1.5` (1024-dim)
- Routed through native HuggingFace `transformers` (not SentenceTransformers) for correct CLS-token pooling
- Query-side prefixing: `"Represent this sentence for searching relevant passages: "`
- GPU-accelerated with FP16 when CUDA is available

### Multilingual Language Detection
Script-based detector for 10 Indian languages (Devanagari, Gurmukhi, Bengali, Odia, Tamil, Telugu, Kannada, Malayalam) plus English. The response language defaults to the detected query language unless overridden.

### Audio Queries (Whisper)
Voice queries are transcribed via OpenAI Whisper (configurable model size: `tiny` / `base` / `small` / `medium` / `large`), with language preservation in the generated response.

### Source Attribution
Every response cites the source PDFs it drew from. A runtime range-map (`vector_id → filename`) is built from the processed-JSON directory so attribution survives index changes without re-embedding.

### Streamlit Chat UI
- Sidebar controls for filters (crop, region, season, disease), `k`, threshold, temperature, and reranker toggle
- Expandable per-chunk panel showing both semantic and rerank scores
- Per-message metadata badge (`🎯 Reranked` / `⚡ No Rerank`)
- Auto-scroll to user's latest question after generation

---

## 📊 Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon URL |
| `OLLAMA_MODEL` | `mistral` | Generation model |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | HuggingFace model id |
| `EMBEDDING_DIMENSION` | `1024` | Must match the model's output dim |
| `USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder checkpoint |
| `CHUNK_SIZE` | `200` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `40` | Overlap between consecutive chunks |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum semantic similarity to retain |
| `TOP_K_RETRIEVAL` | `5` | Chunks returned to the LLM |
| `LANGUAGE_DETECT_THRESHOLD` | `0.5` | Script-detection confidence floor |
| `FAISS_INDEX_PATH` | `./data/embeddings/faiss_index.bin` | FAISS index location |
| `METADATA_INDEX_PATH` | `./data/embeddings/metadata.json` | Chunk metadata location |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | FastAPI binding |
| `STREAMLIT_PORT` | `8501` | Streamlit port |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

> ⚠️ **`EMBEDDING_DIMENSION` must match the chosen `EMBEDDING_MODEL`.** Common values: `BAAI/bge-large-en-v1.5` → **1024**, `BAAI/bge-base-en-v1.5` → **768**, `paraphrase-multilingual-MiniLM-L12-v2` → **384**. Mismatched values will fail at FAISS index load with a dimension-mismatch error.

---

## 📦 Dependencies

| Component | Library |
|-----------|---------|
| Embedding & reranking | `transformers`, `sentence-transformers`, `torch` |
| Vector search | `faiss-cpu` (or `faiss-gpu`) |
| Lexical search | `rank-bm25` |
| PDF parsing | `pymupdf` |
| Audio transcription | `openai-whisper`, `ffmpeg` |
| LLM clients | `ollama`, `openai` |
| API | `fastapi`, `uvicorn` |
| UI | `streamlit` |
| Config | `pydantic`, `pydantic-settings`, `python-dotenv` |

Full pinned list lives in `requirements.txt`.

---

## 🧪 Testing

```bash
# Module-level test suites
python main.py test --test-module tests.test_ingestion
python main.py test --test-module tests.test_embeddings
python main.py test --test-module tests.test_rag

# Or directly with pytest
pytest tests/ -v
pytest tests/ --cov=.
```

---

## 🛠️ Troubleshooting

**FAISS index dimension mismatch on load**
Index was built with a different `EMBEDDING_DIMENSION` than the current setting. Rebuild:
```bash
python main.py embed --pdf-dir Agri_docs
```

**Sources show as "Unknown"**
The runtime source-map needs `data/processed/*_processed.json` to exist. Re-run `ingest` or `embed` to regenerate the processed JSONs.

**`tensor a (1024) must match tensor b (512)`**
A previous build hardcoded `max_length=1024`. The current `embedder.py` respects `tokenizer.model_max_length` (capped at 512 for BERT-based models). Pull the latest code.

**Ollama connection refused**
Start the daemon: `ollama serve`. Verify with `curl http://localhost:11434/api/tags`.

**CUDA not available / slow embeddings**
- Verify `torch.cuda.is_available()` returns `True`
- Set `DEVICE=cpu` in `.env` for explicit CPU mode
- Use `faiss-cpu` if `faiss-gpu` is unavailable for your CUDA version

**Whisper missing ffmpeg**
Install FFmpeg system-wide:
- Windows: `choco install ffmpeg` or download from ffmpeg.org
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg`

---

## 🎓 Example Queries

```text
EN  →  "What is the optimal irrigation schedule for wheat during winter?"
EN  →  "Recommended fungicides for paddy blast in Punjab"
HI  →  "गेहूँ में शूट फ्लाई का नियंत्रण कैसे करें?"
HI  →  "धान में नाइट्रोजन की मात्रा कितनी होनी चाहिए?"
PA  →  "ਕਣਕ ਦੀ ਫ਼ਸਲ ਲਈ ਬੀਜ ਦੀ ਮਾਤਰਾ ਕਿੰਨੀ ਚਾਹੀਦੀ ਹੈ?"
```

---

## 📝 Recent Changes

- **Switched embedding model** to `BAAI/bge-large-en-v1.5` (1024-dim) — superior retrieval quality over the previous multilingual MiniLM (384-dim) and instruct-E5 models.
- **Added BGE cross-encoder reranker** (`BAAI/bge-reranker-base`) as a second-stage scorer. Retrieves exactly `k` chunks instead of `k×3` and reorders — 3× fewer FAISS/BM25/cross-encoder operations.
- **Fixed source attribution** — chunks no longer display "Unknown" filenames. Three-layer fix: ingestion injects `source_file`, embedder propagates it, runtime range-map enriches missing entries from `data/processed/`.
- **Fixed Streamlit UI issues** — header overlap on first chat message, auto-scroll now lands on the user's question (not the assistant reply), and `height=1` iframe workaround for Chromium's zero-height suppression.
- **Hardened tokenization** — `max_length` now respects `tokenizer.model_max_length` (capped at 512) instead of a hardcoded value, eliminating position-embedding shape mismatches on BERT-based models.
- **Corrected BGE pooling** — uses `last_hidden_state[:, 0, :]` (CLS) instead of `pooler_output` for BAAI models, restoring retrieval quality.
- **Added audio query support** via Whisper with language preservation in responses.
- **Vector store dimension guard** — `vector_store.py` now raises a clear error if the loaded index dimension differs from the current embedding model.

---

## 📄 License

MIT License — see `LICENSE` for details.

## 👥 Contributors

AI ML FAS Team
