# Agricultural Knowledge RAG System - Multilingual Edition

A production-grade Retrieval-Augmented Generation (RAG) system designed to provide reliable agricultural knowledge to farmers in India in their preferred language (Hindi, Punjabi, or English).

## 🎯 Overview

This system addresses the challenge of agricultural knowledge accessibility by:
- Aggregating agricultural information from multiple sources
- Processing and chunking documents intelligently
- Retrieving relevant information using semantic search
- Generating context-aware responses in the farmer's language
- Preventing hallucinations through grounded responses

### Target Users
Farmers in Punjab and Haryana

### Supported Crops
- Wheat
- Paddy (Rice)

### Features
- Pest/disease advisory
- Fertilizer guidance
- Irrigation recommendations
- Seasonal best practices

### Supported Languages
- English
- Hindi
- Punjabi

## 📋 Project Structure

```
project/
├── data/
│   ├── raw/              # Raw PDFs and documents
│   ├── processed/        # Processed and cleaned JSON documents
│   └── embeddings/       # FAISS indices and metadata
│
├── ingestion/            # Document ingestion pipeline
│   ├── pdf_parser.py
│   ├── metadata_extractor.py
│   └── chunking.py
│
├── rag/                  # RAG pipeline components
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompt_templates.py
│   └── generator.py
│
├── api/                  # FastAPI backend
│   └── fastapi_app.py
│
├── ui/                   # Streamlit frontend
│   └── streamlit_app.py
│
├── evaluation/           # Evaluation and testing
│
├── utils/                # Utility modules
│   ├── config.py
│   ├── logger.py
│   └── __init__.py
│
├── tests/                # Unit and integration tests
│
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
└── main.py              # Entry point
```

## 🚀 Quick Start

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Edit `.env` file with your settings:

```bash
# LLM Configuration
LLM_PROVIDER=ollama              # or 'openai'
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral             # or 'llama2', 'neural-chat'

# If using OpenAI:
# OPENAI_API_KEY=your_key_here

# Embedding Model (multilingual)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# RAG Settings
CHUNK_SIZE=400
CHUNK_OVERLAP=100
TOP_K_RETRIEVAL=5
SUPPORTED_LANGUAGES=en,hi,pa
```

### 4. Setup LLM Provider

#### Option A: Using Ollama (Recommended for Local Development)

```bash
# Install Ollama from https://ollama.ai
# Download a model
ollama pull mistral
# or
ollama pull llama2

# Start Ollama service
ollama serve
```

#### Option B: Using OpenAI

Set your OpenAI API key in `.env`:
```
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
```

## 📦 Dependencies Overview

### Core Components
- **FastAPI**: REST API framework
- **Streamlit**: Frontend framework
- **LangChain**: LLM orchestration
- **FAISS**: Vector similarity search
- **SentenceTransformers**: Multilingual embeddings
- **PyMuPDF**: PDF processing

### Supporting Libraries
- **Pydantic**: Data validation
- **python-dotenv**: Environment management
- **pandas/numpy**: Data processing
- **pytest**: Testing framework

## 🔄 Development Workflow

This project is built incrementally. Current status:

### ✅ Step 1: Project Structure & Setup (COMPLETED)
- Created folder structure
- Generated requirements.txt
- Setup configuration management
- Setup logging infrastructure

### 📌 Step 2: PDF Ingestion Module (NEXT)
- PDF text extraction
- Header/footer removal
- OCR artifact cleaning
- JSON output generation

### Coming Steps
3. Chunking & metadata extraction
4. Embedding generation & FAISS storage
5. Retrieval pipeline
6. RAG response generation
7. FastAPI backend
8. Streamlit UI
9. Evaluation module

## 📝 Configuration Details

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | ollama | LLM provider (ollama/openai) |
| `OLLAMA_MODEL` | mistral | Ollama model name |
| `EMBEDDING_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | SentenceTransformer model |
| `CHUNK_SIZE` | 400 | Document chunk size in tokens |
| `TOP_K_RETRIEVAL` | 5 | Number of chunks to retrieve |
| `SUPPORTED_LANGUAGES` | en,hi,pa | Supported language codes |

### Directory Structure

- **data/raw/**: Place your PDF files here
- **data/processed/**: Cleaned JSON documents (auto-generated)
- **data/embeddings/**: FAISS indices (auto-generated)
- **logs/**: Application logs

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=.

# Run specific test
pytest tests/test_module.py -v
```

## 📚 Documentation

Each module includes:
- Type hints for all functions
- Comprehensive docstrings
- Inline comments for complex logic
- Exception handling with logging

## 🛠️ Development Notes

### Code Quality Standards
- PEP 8 compliant code
- Black formatting for consistency
- Type hints throughout
- Comprehensive logging
- Error handling at all layers

### Dependencies
- Python 3.10+
- CUDA support optional (for GPU embeddings)

## 📞 Support & Troubleshooting

### Common Issues

**Issue: CUDA not available**
- Solution: Install `faiss-cpu` or `faiss-gpu` as needed

**Issue: Ollama connection failed**
- Solution: Ensure Ollama is running: `ollama serve`

**Issue: Missing embeddings**
- Solution: Run embedding generation step first

## 🎓 Example Agriculture Queries

The system is designed to handle queries like:

- "गेहूँ में शूट फ्लाई का नियंत्रण कैसे करें?" (Hindi)
- "ਪੱਧਾ ਵਿੱਚ ਜੀਵਾਣੂ ਭਿੱਜ ਕਰਨ ਦਾ ਸਮਾ ਕਿਹੜਾ ਹੈ?" (Punjabi)
- "What is the optimal irrigation schedule for wheat during winter?" (English)

## 📄 License

MIT License - See LICENSE file for details

## 👥 Contributors

AI ML FAS Team

---

## Next Steps

Proceed to **Step 2: PDF Ingestion Module** when ready.

To start:
1. Place sample agricultural PDFs in `data/raw/`
2. Run the PDF ingestion pipeline
3. Verify processed JSON outputs
