# Architecture

This document describes the system architecture of the Agricultural RAG System using Mermaid diagrams (renders inline on GitHub).

---

## 1. System Overview

End-to-end view of the offline indexing pipeline and the online query pipeline. The two pipelines meet at the **FAISS vector store + metadata index**.

```mermaid
flowchart TB
    %% ─────────────────────── INPUTS ───────────────────────
    subgraph INPUTS[" "]
        direction LR
        PDFS["📄 Agricultural PDFs<br/>data/raw/"]
        TEXT["⌨️ Text Query"]
        AUDIO["🎙️ Audio Query<br/>mp3 / wav / m4a"]
    end

    %% ─────────────────────── OFFLINE INDEXING ───────────────────────
    subgraph OFFLINE["🛠️ Offline Indexing Pipeline (python main.py embed)"]
        direction TB
        PARSE["PDF Parser<br/>(PyMuPDF)"]
        CHUNK["Token-Aware Chunker<br/>chunk_size=200, overlap=40"]
        META["Metadata Extractor<br/>crop · region · season · disease"]
        PROC[("📁 data/processed/<br/>*_processed.json")]
        EMBED["Embedder<br/>BAAI/bge-large-en-v1.5<br/>1024-d · GPU FP16"]
    end

    %% ─────────────────────── STORAGE ───────────────────────
    subgraph STORAGE["💾 Persistent Storage"]
        direction LR
        FAISS[("FAISS Index<br/>IndexFlatL2<br/>12k+ vectors")]
        METAJSON[("metadata.json<br/>chunk → filename map")]
        SRCMAP["Runtime Source Map<br/>vector_id → PDF filename"]
    end

    %% ─────────────────────── ONLINE QUERY ───────────────────────
    subgraph ONLINE["🔍 Online Query Pipeline"]
        direction TB
        WHISPER["Whisper Transcriber<br/>(audio only)"]
        LANGDET["Language Detector<br/>script-based, 10 languages"]
        RAGORCH["RAG Orchestrator<br/>rag_orchestrator.py"]

        subgraph RETRIEVAL["Hybrid Retrieval"]
            direction LR
            BM25["BM25<br/>(lexical)"]
            DENSE["FAISS Dense<br/>(semantic)"]
            RRF["Reciprocal Rank<br/>Fusion (RRF)"]
            BM25 --> RRF
            DENSE --> RRF
        end

        RERANK["BGE Cross-Encoder Reranker<br/>BAAI/bge-reranker-base"]
        GEN["Response Generator<br/>Ollama / OpenAI"]
    end

    %% ─────────────────────── INTERFACES ───────────────────────
    subgraph INTERFACES["🖥️ User Interfaces"]
        direction LR
        UI["Streamlit Chat UI<br/>:8501"]
        API["FastAPI REST API<br/>:8000"]
        CLI["CLI<br/>main.py"]
    end

    LLM[("☁️ LLM Backend<br/>Ollama (local)<br/>or OpenAI")]
    RESPONSE["📤 Grounded Response<br/>+ Source Attribution"]

    %% ─────────────────────── EDGES ───────────────────────
    PDFS --> PARSE --> CHUNK --> META --> PROC
    PROC --> EMBED --> FAISS
    PROC --> METAJSON
    PROC -.boot.-> SRCMAP

    TEXT --> UI
    TEXT --> API
    TEXT --> CLI
    AUDIO --> WHISPER --> CLI
    AUDIO --> WHISPER --> UI

    UI --> RAGORCH
    API --> RAGORCH
    CLI --> RAGORCH

    RAGORCH --> LANGDET
    LANGDET --> RETRIEVAL
    FAISS --> DENSE
    METAJSON --> DENSE
    RETRIEVAL --> RERANK
    SRCMAP -.enrich.-> RERANK
    RERANK --> GEN
    GEN <--> LLM
    GEN --> RESPONSE

    %% ─────────────────────── STYLING ───────────────────────
    classDef storage fill:#fff4d6,stroke:#b8860b,stroke-width:2px,color:#000
    classDef llm fill:#e6e6fa,stroke:#6a5acd,stroke-width:2px,color:#000
    classDef ui fill:#d4f4dd,stroke:#2e8b57,stroke-width:2px,color:#000
    classDef offline fill:#ffe4e1,stroke:#cd5c5c,stroke-width:1px,color:#000
    classDef online fill:#e0f0ff,stroke:#1e90ff,stroke-width:1px,color:#000
    classDef output fill:#fffacd,stroke:#daa520,stroke-width:2px,color:#000

    class FAISS,METAJSON,SRCMAP,PROC storage
    class LLM llm
    class UI,API,CLI ui
    class PARSE,CHUNK,META,EMBED offline
    class WHISPER,LANGDET,RAGORCH,BM25,DENSE,RRF,RERANK,GEN online
    class RESPONSE output
```

---

## 2. Query Lifecycle (Sequence Diagram)

Runtime flow when a user submits a query. Shows how the RAG orchestrator coordinates the retriever, reranker, and generator.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit UI / API
    participant Orch as RAG Orchestrator
    participant Lang as Language Detector
    participant Emb as Embedder<br/>(BGE-large)
    participant BM25 as BM25 Index
    participant FAISS as FAISS Store
    participant RR as BGE Reranker
    participant Gen as Generator
    participant LLM as Ollama / OpenAI

    User->>UI: Submit query<br/>(text or audio)
    Note over UI: If audio:<br/>Whisper transcribes first
    UI->>Orch: query(text, filters, k=10)

    Orch->>Lang: detect_language(text)
    Lang-->>Orch: "hi" / "en" / "pa"

    Orch->>Emb: embed_query(text)
    Note over Emb: Prepends BGE prefix:<br/>"Represent this sentence..."
    Emb-->>Orch: query vector (1024-d)

    par Hybrid retrieval
        Orch->>BM25: keyword search
        BM25-->>Orch: ranked candidates
    and
        Orch->>FAISS: vector.search(k)
        FAISS-->>Orch: ranked candidates
    end

    Note over Orch: Reciprocal Rank Fusion<br/>merges both lists
    Note over Orch: Enrich chunks with<br/>filename from source map

    alt USE_RERANKER=true
        Orch->>RR: rerank(query, chunks)
        Note over RR: Cross-encoder scores<br/>(query, passage) pairs
        RR-->>Orch: reordered chunks<br/>+ rerank_score
    end

    Orch->>Gen: generate(query, chunks, language)
    Gen->>LLM: prompt with context
    LLM-->>Gen: response text
    Gen-->>Orch: response + sources + confidence

    Orch-->>UI: {response, sources, chunks, scores}
    UI-->>User: Render answer<br/>+ source attribution
```

---

## 3. Module Layout

Internal module dependencies — which Python packages call which.

```mermaid
flowchart LR
    subgraph entry["Entry Points"]
        MAIN["main.py<br/>(CLI dispatcher)"]
        APP["api/fastapi_app.py"]
        ST["ui/streamlit_app.py"]
    end

    subgraph audio["audio/"]
        AP["audio_processor.py<br/>Whisper wrapper"]
        ARH["audio_rag_handler.py"]
    end

    subgraph ingest["ingestion/"]
        PP["pdf_parser.py"]
        CK["chunking.py"]
        ME["metadata_extractor.py"]
        IO["orchestrator.py"]
    end

    subgraph rag["rag/"]
        EM["embedder.py"]
        VS["vector_store.py"]
        RT["retriever.py<br/>+ LanguageDetector"]
        RR["reranker.py"]
        GN["generator.py"]
        PT["prompt_templates.py"]
        EO["embeddings_orchestrator.py"]
        RO["rag_orchestrator.py"]
    end

    subgraph utils["utils/"]
        CF["config.py<br/>(Pydantic Settings)"]
        LG["logger.py"]
    end

    subgraph external["External"]
        HF["🤗 HuggingFace<br/>BGE models"]
        OL["Ollama daemon"]
        OAI["OpenAI API"]
    end

    %% Entry → orchestrators
    MAIN --> IO
    MAIN --> EO
    MAIN --> RO
    MAIN --> ARH
    APP --> RO
    ST --> RO
    ARH --> AP
    ARH --> RO

    %% Ingestion graph
    IO --> PP
    IO --> CK
    IO --> ME

    %% Embedding graph
    EO --> IO
    EO --> EM
    EO --> VS

    %% RAG runtime graph
    RO --> EM
    RO --> VS
    RO --> RT
    RO --> RR
    RO --> GN
    RT --> EM
    RT --> VS
    GN --> PT

    %% Settings used everywhere
    EM -.-> CF
    VS -.-> CF
    RT -.-> CF
    RR -.-> CF
    GN -.-> CF
    RO -.-> CF
    AP -.-> CF
    EO -.-> CF
    IO -.-> CF

    %% External calls
    EM --> HF
    RR --> HF
    AP --> HF
    GN --> OL
    GN --> OAI

    classDef cfg fill:#fff4d6,stroke:#b8860b,color:#000
    classDef ext fill:#e6e6fa,stroke:#6a5acd,color:#000
    classDef ep fill:#d4f4dd,stroke:#2e8b57,color:#000
    class CF,LG cfg
    class HF,OL,OAI ext
    class MAIN,APP,ST ep
```

---

## 4. Data Storage Layout

What lives on disk and how it relates.

```mermaid
flowchart TB
    subgraph raw["data/raw/"]
        PDF1["wheat_pop.pdf"]
        PDF2["paddy_advisory.pdf"]
        PDFN["...N PDFs"]
    end

    subgraph processed["data/processed/"]
        JSON1["wheat_pop_processed.json<br/>{source_file, chunks[]}"]
        JSON2["paddy_advisory_processed.json"]
        JSONN["..."]
    end

    subgraph embeddings["data/embeddings/"]
        FBIN["faiss_index.bin<br/>(IndexFlatL2, 1024-d)"]
        MJSON["metadata.json<br/>[{vector_id, content,<br/>filename, crop, ...}]"]
    end

    subgraph runtime["Runtime (in-memory)"]
        SMAP["Source Map<br/>List[(start_id, end_id, filename)]"]
    end

    PDF1 -.parsed.-> JSON1
    PDF2 -.parsed.-> JSON2
    PDFN -.parsed.-> JSONN

    JSON1 -.embedded.-> FBIN
    JSON1 -.embedded.-> MJSON
    JSON2 -.embedded.-> FBIN
    JSON2 -.embedded.-> MJSON

    JSON1 -.boot.-> SMAP
    JSON2 -.boot.-> SMAP
    JSONN -.boot.-> SMAP

    classDef diskfile fill:#fff4d6,stroke:#b8860b,color:#000
    classDef memory fill:#e0f0ff,stroke:#1e90ff,color:#000
    class FBIN,MJSON,JSON1,JSON2,JSONN,PDF1,PDF2,PDFN diskfile
    class SMAP memory
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Two-stage retrieval** (hybrid then rerank) | Bi-encoder is fast over large corpora; cross-encoder is accurate but slow — restrict it to the top-`k` candidates. |
| **BM25 + Dense + RRF** | Lexical search catches exact terminology (chemical names, dosages, variety codes); semantic search handles paraphrases. RRF merges without tuning weights. |
| **BGE over multilingual MiniLM** | bge-large-en-v1.5 is ~3× the dimensions (1024 vs 384) and trained on a much larger corpus. For agricultural English text it consistently retrieves better; Hindi/Punjabi queries still work through transliterated/translated terms. |
| **Native HF Transformers for BGE** | SentenceTransformers wraps BGE incorrectly (uses `pooler_output`); we need the CLS token from `last_hidden_state[:, 0, :]` for retrieval-tuned BGE checkpoints. |
| **Runtime source map** | Avoids re-embedding when filename data was missing from older builds. The map is rebuilt from `data/processed/*.json` on startup. |
| **FAISS IndexFlatL2 (not IVF/HNSW)** | At ~12k vectors the corpus is small enough that exact search is sub-millisecond. No recall trade-off, no index-build complexity. |
| **Pydantic Settings** | Single source of truth for config across modules; auto-loads from `.env`; type-validated at startup so misconfiguration fails fast. |
