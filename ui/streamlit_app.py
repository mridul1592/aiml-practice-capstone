"""
Beautiful Streamlit UI for Agricultural RAG System.

Features:
- 10-language support (English, Hindi, Tamil, Telugu, Odia, Kannada, Marathi, Malayalam, Bengali, Punjabi)
- Text and audio input
- Automatic language detection
- Beautiful response formatting
- Confidence indicators
- Source attribution
- Responsive design
"""

import sys
from pathlib import Path
from typing import Optional
import os
import shutil
import subprocess
import tempfile
import json

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import logger FIRST before using it
from utils.logger import setup_logger
logger = setup_logger(__name__)

# Import other modules
from audio.audio_rag_handler import AudioRAGHandler
from rag.rag_orchestrator import RAGPipeline
from rag.retriever import LanguageDetector
from utils.config import settings

# Find FFmpeg executable
def get_ffmpeg_path():
    """Get the full path to FFmpeg executable."""
    # Common installation locations for Windows (check most likely first)
    common_paths = [
        r"C:\Users\mridu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "FFmpeg" / "bin" / "ffmpeg.exe"),
    ]

    # Check hardcoded paths first (most reliable for installed packages)
    for path in common_paths:
        if os.path.exists(path):
            logger.info(f"Found FFmpeg at: {path}")
            return path

    # Try Python's shutil.which() as fallback
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        logger.info(f"Found FFmpeg in PATH: {ffmpeg}")
        return ffmpeg

    logger.warning("FFmpeg not found - audio recording will not work")
    return None

FFMPEG_PATH = get_ffmpeg_path()

# Display FFmpeg status in sidebar for debugging
with st.sidebar:
    if FFMPEG_PATH:
        st.success(f"✅ FFmpeg found: {FFMPEG_PATH.split(os.sep)[-3:]}")
    else:
        st.error("❌ FFmpeg not found - audio recording unavailable")

# Streamlit page configuration
st.set_page_config(
    page_title="🌾 Agricultural AI Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Agricultural RAG System - 10 Languages, Smart Farming Advice"
    }
)

# Language Configuration
LANGUAGES = {
    "en": "🇬🇧 English",
    "hi": "🇮🇳 Hindi (हिंदी)",
    "pa": "🇮🇳 Punjabi (ਪੰਜਾਬੀ)",
    "ta": "🇮🇳 Tamil (தமிழ்)",
    "te": "🇮🇳 Telugu (తెలుగు)",
    "or": "🇮🇳 Odia (ଓଡ଼ିଆ)",
    "kn": "🇮🇳 Kannada (ಕನ್ನಡ)",
    "mr": "🇮🇳 Marathi (मराठी)",
    "ml": "🇮🇳 Malayalam (മലയാളം)",
    "bn": "🇮🇳 Bengali (বাংলা)",
}

# Beautiful Custom CSS
st.markdown(
    """
    <style>
    /* Main styling */
    :root {
        --primary-color: #10b981;
        --secondary-color: #059669;
        --accent-color: #f59e0b;
        --text-dark: #1f2937;
        --text-light: #6b7280;
        --bg-light: #f9fafb;
        --border-color: #e5e7eb;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 40px 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }

    .header-title {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }

    .header-subtitle {
        font-size: 1.1em;
        margin-top: 10px;
        text-align: center;
        opacity: 0.95;
    }

    /* Card styling */
    .card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
    }

    .card:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }

    .response-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #f0fdf4 100%);
        border-left: 4px solid #10b981;
    }

    /* Language badge */
    .language-badge {
        display: inline-block;
        background: #dbeafe;
        color: #0c4a6e;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
        margin: 5px 5px 5px 0;
    }

    /* Confidence indicator */
    .confidence-high {
        color: #10b981;
        font-weight: 700;
    }

    .confidence-medium {
        color: #f59e0b;
        font-weight: 700;
    }

    .confidence-low {
        color: #ef4444;
        font-weight: 700;
    }

    /* Source box */
    .source-item {
        background: #f3f4f6;
        border-left: 4px solid #3b82f6;
        padding: 12px 15px;
        margin: 10px 0;
        border-radius: 6px;
        font-size: 0.95em;
    }

    /* Chunk box */
    .chunk-item {
        background: #f3f4f6;
        color: #1f2937;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 15px;
        margin: 12px 0;
        line-height: 1.6;
    }

    .chunk-header {
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .similarity-score {
        background: #dbeafe;
        color: #0c4a6e;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }

    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #bbf7d0;
    }

    .metric-label {
        font-size: 0.9em;
        color: var(--text-light);
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.8em;
        color: var(--primary-color);
        font-weight: 700;
        margin-top: 5px;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.3);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 8px 8px 0 0;
    }

    /* Alert styling */
    .stSuccess {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
    }

    .stError {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
    }

    .stWarning {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-light);
        font-size: 0.9em;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid var(--border-color);
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 2px solid var(--border-color);
        margin: 20px 0;
    }

    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #000000 0%, #f3f4f6 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_pipeline(provider: str, model_name: str) -> RAGPipeline:
    """Initialize RAG pipeline with caching."""
    return RAGPipeline(llm_provider=provider, llm_model_name=model_name)


@st.cache_resource
def initialize_audio_handler() -> AudioRAGHandler:
    """Initialize audio handler with caching."""
    return AudioRAGHandler(audio_model_size="base")


def get_confidence_color(confidence: str) -> str:
    """Get color class for confidence level."""
    if confidence.lower() == "high":
        return "confidence-high"
    elif confidence.lower() == "medium":
        return "confidence-medium"
    else:
        return "confidence-low"


def display_header():
    """Display beautiful header."""
    st.markdown(
        """
        <div class='header-container'>
            <h1 class='header-title'>🌾 Agricultural AI Assistant</h1>
            <p class='header-subtitle'>Get expert farming advice in 10 languages with AI-powered recommendations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_response(result: dict):
    """Display query response with beautiful formatting."""
    # Response
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 💬 Response")
    with col2:
        lang_code = result.get("language", "en")
        lang_name = LANGUAGES.get(lang_code, "Unknown")
        st.markdown(f"<span class='language-badge'>{lang_name}</span>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='card response-card'>
        {result['response']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class='metric-container'>
                <div class='metric-label'>📊 Confidence</div>
                <div class='metric-value {get_confidence_color(result.get("confidence", "low"))}'>
                    {result.get("confidence", "N/A").upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-container'>
                <div class='metric-label'>📖 Context Chunks</div>
                <div class='metric-value'>{result.get("num_context_chunks", 0)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-container'>
                <div class='metric-label'>📚 Sources</div>
                <div class='metric-value'>{len(result.get("sources", []))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Sources
    if result.get("sources"):
        st.markdown("### 📚 Sources")
        for source in result["sources"]:
            st.markdown(
                f"<div class='source-item'>📄 {source}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Retrieved Chunks
    if result.get("retrieved_chunks"):
        with st.expander(f"📖 Retrieved Chunks ({len(result['retrieved_chunks'])})"):
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                similarity = chunk.get("similarity_score", 0)
                content = chunk.get("content", "")
                filename = chunk.get("filename", "Unknown")

                st.markdown(
                    f"""
                    <div class='chunk-item'>
                        <div class='chunk-header'>
                            <span><b>Chunk {i}</b> | 📄 {filename}</span>
                            <span class='similarity-score'>Similarity: {similarity:.1%}</span>
                        </div>
                        <p>{content}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _transcribe_audio_to_query(audio_bytes, suffix: str, language: str) -> None:
    """
    Transcribe audio bytes using Whisper and write result to the query text area.
    This is the ONLY job of this function — no RAG, no answering, just text.
    """
    import time as _time

    env = os.environ.copy()
    if FFMPEG_PATH:
        env['PATH'] = os.path.dirname(FFMPEG_PATH) + os.pathsep + env.get('PATH', '')

    raw_path = ""
    wav_path = ""
    transcribed = ""
    error_msg = ""

    # ── All heavy work happens inside the spinner ────────────────────────────
    with st.spinner("🎙️ Transcribing audio…"):
        try:
            # 1. Save raw audio to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw_f:
                raw_f.write(bytes(audio_bytes))
                raw_f.flush()
                raw_path = raw_f.name
            _time.sleep(0.05)
            logger.info(f"Audio saved: {raw_path} ({os.path.getsize(raw_path)} bytes)")

            # 2. Convert to WAV via FFmpeg (Whisper needs it internally)
            wav_path = raw_path.rsplit(".", 1)[0] + "_in.wav"
            if FFMPEG_PATH:
                conv = subprocess.run(
                    [FFMPEG_PATH, "-y", "-i", raw_path,
                     "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
                    capture_output=True, text=True, timeout=30
                )
                if conv.returncode != 0:
                    error_msg = f"Audio conversion failed: {conv.stderr[:200]}"
                    logger.error(f"FFmpeg error: {conv.stderr}")
                else:
                    audio_input = wav_path
                    logger.info(f"Converted to WAV: {wav_path}")
            else:
                audio_input = raw_path
            _time.sleep(0.05)

            if not error_msg:
                # 3. Run Whisper
                output_dir = os.path.abspath(".")
                cmd = [sys.executable, "-m", "whisper", audio_input,
                       "--model", "base",
                       "--output_format", "json",
                       "--output_dir", output_dir,
                       "--verbose", "False"]
                if language and language != "Auto-detect":
                    cmd += ["--language", language]

                logger.info(f"Whisper cmd: {' '.join(cmd)}")
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
                logger.info(f"Whisper exit={proc.returncode} | stdout={proc.stdout} | stderr={proc.stderr[:200]}")

                if proc.returncode != 0:
                    error_msg = f"Whisper failed: {proc.stderr[:300]}"
                else:
                    # 4. Parse JSON
                    stem = Path(audio_input).stem
                    json_path = os.path.join(output_dir, stem + ".json")
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as jf:
                            transcribed = json.load(jf).get("text", "").strip()
                        os.remove(json_path)
                        logger.info(f"Transcription ({len(transcribed)} chars): {transcribed[:100]}")
                    else:
                        error_msg = "Whisper ran but produced no output file."
                        logger.error(f"Expected JSON not found: {json_path}")

        except subprocess.TimeoutExpired:
            error_msg = "Transcription timed out (120 s). Try a shorter clip."
        except Exception as exc:
            import traceback
            error_msg = str(exc)
            logger.error(traceback.format_exc())
        finally:
            for p in (raw_path, wav_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    # ── Spinner has closed by here ───────────────────────────────────────────

    if error_msg:
        st.error(error_msg)
    elif transcribed:
        st.session_state.query_text_value = transcribed
        st.rerun()   # spinner is gone; rerun refreshes text area cleanly
    else:
        st.warning("⚠️ No speech detected — try speaking more clearly.")


def main():
    """Main Streamlit application."""
    display_header()

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # LLM Settings
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["ollama", "openai"],
            help="Select the LLM provider for response generation",
            key="llm_provider_sidebar"
        )

        if llm_provider == "ollama":
            llm_model = st.selectbox(
                "Ollama Model",
                options=["neural-chat", "mistral", "llama2", "orca2"],
                help="Select model available in Ollama",
                key="ollama_model_sidebar"
            )
        else:
            llm_model = st.text_input(
                "OpenAI Model",
                value="gpt-3.5-turbo",
                help="Requires OPENAI_API_KEY in .env",
                key="openai_model_sidebar"
            )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = deterministic, Higher = creative",
            key="temperature_sidebar"
        )

        st.markdown("---")
        st.markdown("### 📊 System Information")

        # System Stats
        with st.expander("View System Settings", expanded=False):
            st.json({
                "Supported Languages": len(settings.supported_languages),
                "Embedding Model": settings.embedding_model,
                "Chunk Size": settings.chunk_size,
                "Top K": settings.top_k_retrieval,
            })

        # Vector Store Stats
        try:
            with st.expander("Vector Store Statistics", expanded=False):
                pipeline = initialize_pipeline(llm_provider, llm_model)
                stats = pipeline.get_stats()

                if "vector_store" in stats:
                    st.metric(
                        "Total Vectors",
                        stats['vector_store'].get('total_vectors', 0)
                    )
                    st.metric(
                        "Embedding Dimension",
                        stats['vector_store'].get('embedding_dimension', 0)
                    )
        except Exception as e:
            st.warning(f"Could not load stats: {str(e)}")

    # ==================== GOOGLE-STYLE SEARCH INTERFACE ====================
    st.markdown("<br>", unsafe_allow_html=True)

    # Centered search box area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 20px;'>
                <h2 style='font-size: 2.5em; margin: 0; color: #10b981;'>🌾 Ask Anything</h2>
                <p style='color: #6b7280; font-size: 1.1em; margin-top: 5px;'>Text or Audio - 10 Languages Supported</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Search input area (Google-style)
    # Initialize query text in session state
    if "query_text_value" not in st.session_state:
        st.session_state.query_text_value = ""

    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        # NO key= here — value= controls content; audio transcription writes to query_text_value
        query_text = st.text_area(
            "Enter your question",
            value=st.session_state.query_text_value,
            placeholder="e.g., How to control wheat pests? / गेहूँ में कीटों का नियंत्रण कैसे करें?",
            height=80,
            label_visibility="collapsed",
        )
        # Keep session state in sync so typing also updates it
        st.session_state.query_text_value = query_text

    st.markdown("<br>", unsafe_allow_html=True)

    # Language selector for audio (helps Whisper detect correctly)
    audio_lang_col1, audio_lang_col2, audio_lang_col3 = st.columns([1, 2, 1])
    with audio_lang_col2:
        st.write("**Select language for audio input:**")
        audio_language = st.selectbox(
            "Audio language",
            options=["Auto-detect", "en", "hi", "pa"],
            format_func=lambda x: "🔍 Auto-detect" if x == "Auto-detect" else LANGUAGES.get(x, x),
            key="audio_language"
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # Audio input section - simple transcription to text
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🎙️ Record Audio")
        try:
            from streamlit_mic_recorder import mic_recorder
            audio_data = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                just_once=True,          # only fires once per recording
                use_container_width=True,
                format="webm",
                key="mic_recorder_widget"
            )
            if audio_data:
                # Use audio id to avoid re-transcribing the same clip on every rerun
                audio_id = audio_data.get("id", id(audio_data['bytes']))
                if st.session_state.get("last_audio_id") != audio_id:
                    st.session_state.last_audio_id = audio_id
                    _transcribe_audio_to_query(audio_data['bytes'], ".webm", audio_language)
        except ImportError:
            st.info("📍 Audio recording requires `streamlit-mic-recorder`")

    with col2:
        st.markdown("#### 📁 Upload Audio File")
        uploaded_audio = st.file_uploader(
            "Upload audio (MP3, WAV, M4A, FLAC, OGG)",
            type=["mp3", "wav", "m4a", "flac", "ogg", "opus", "aac"],
            key="audio_upload"
        )
        if uploaded_audio:
            upload_id = f"{uploaded_audio.name}_{uploaded_audio.size}"
            if st.session_state.get("last_audio_id") != upload_id:
                st.session_state.last_audio_id = upload_id
                ext = Path(uploaded_audio.name).suffix.lower()
                _transcribe_audio_to_query(uploaded_audio.getbuffer(), ext, audio_language)

    st.markdown("---")

    # Optional Filters (in expandable section)
    with st.expander("🎯 Filters & Settings (Optional)"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            crop = st.selectbox(
                "Crop",
                options=["", "wheat", "paddy", "rice", "maize", "cotton", "sugarcane"],
                help="Filter by crop type",
                key="crop_filter"
            )

        with col2:
            region = st.text_input(
                "Region",
                placeholder="e.g., Punjab",
                help="Filter by region",
                key="region_filter"
            )

        with col3:
            season = st.selectbox(
                "Season",
                options=["", "kharif", "rabi", "summer"],
                help="Filter by season",
                key="season_filter"
            )

        with col4:
            disease = st.text_input(
                "Disease/Pest",
                placeholder="e.g., rust",
                help="Filter by disease/pest",
                key="disease_filter"
            )

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            k = st.slider(
                "Context Chunks",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="k_text_query"
            )

        with col2:
            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="threshold_text_query"
            )

        with col3:
            st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Get Answer Button ────────────────────────────────────────────────────
    # Completely independent of audio. Reads only what is in the text area.
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.2, 1.6, 1.2])
    with col2:
        submit_button = st.button("🚀 Get Answer", use_container_width=True, type="primary", key="main_submit")

    if submit_button:
        current_query = query_text.strip()
        if not current_query:
            st.error("❌ Please type a question or use audio input to populate the text area first.")
        else:
            logger.info(f"Get Answer clicked | query='{current_query[:80]}'")
            try:
                with st.spinner("🔄 Finding the best answer for you..."):
                    # Detect language from the text
                    lang_detector = LanguageDetector()
                    detected_lang = lang_detector.detect_language(current_query)
                    logger.info(f"Detected language: {detected_lang}")

                    # Read filter values
                    crop     = st.session_state.get("crop_filter", "")
                    region   = st.session_state.get("region_filter", "")
                    season   = st.session_state.get("season_filter", "")
                    disease  = st.session_state.get("disease_filter", "")
                    k        = st.session_state.get("k_text_query", 5)
                    threshold = st.session_state.get("threshold_text_query", 0.2)

                    logger.info(f"Filters: crop={crop} region={region} season={season} disease={disease} k={k} threshold={threshold}")

                    # Initialize pipeline
                    pipeline = initialize_pipeline(llm_provider, llm_model)
                    logger.info("Pipeline initialized")

                    # Execute RAG query
                    result = pipeline.query(
                        current_query,
                        k=k,
                        similarity_threshold=threshold,
                        language=detected_lang,
                        crop=crop if crop else None,
                        region=region if region else None,
                        season=season if season else None,
                        disease=disease if disease else None,
                        temperature=temperature,
                    )
                    logger.info(f"Query complete | response length={len(result.get('response',''))}")

                st.success("✅ Got your answer!")
                st.markdown("---")
                display_response(result)

            except Exception as e:
                import traceback
                logger.error(f"Query error: {traceback.format_exc()}")
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Error details"):
                    st.code(traceback.format_exc())
                st.info("💡 Make sure you have:")
                st.write("1. Built embeddings: `python main.py embed --pdf-dir Agri_docs`")
                st.write(f"2. Started {llm_provider} server (`ollama serve`)")

    # Footer
    st.markdown(
        """
        <div class='footer'>
            <p>🌾 Agricultural AI Assistant | Supporting 10 Languages | Text & Audio Input</p>
            <p style='font-size: 0.85em;'>Powered by SentenceTransformers + FAISS + LLM + Whisper</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
