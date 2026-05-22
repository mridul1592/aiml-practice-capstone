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
        background: #000000;
        border: 1px solid #e5e7eb;
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
        background: #000000;
        color: #166534;
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
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        query_text = st.text_area(
            "Enter your question",
            placeholder="e.g., How to control wheat pests? / गेहूँ में कीटों का नियंत्रण कैसे करें?",
            height=80,
            label_visibility="collapsed",
            key="main_query"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Audio input section - simple transcription to text
    col1, col2 = st.columns([1, 1])

    transcribed_text = ""

    with col1:
        st.markdown("#### 🎙️ Record Audio")
        try:
            from streamlit_mic_recorder import mic_recorder

            audio_data = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                just_once=False,
                use_container_width=False,
                format="webm"
            )

            if audio_data:
                try:
                    with st.spinner("🔄 Transcribing audio..."):
                        # CHECKPOINT 1: Audio data received
                        st.write("**[DEBUG] Checkpoint 1:** Audio data received")
                        logger.info("CHECKPOINT 1: Audio data received from mic_recorder")
                        logger.info(f"  - Audio data type: {type(audio_data)}")
                        logger.info(f"  - Audio data keys: {audio_data.keys() if isinstance(audio_data, dict) else 'N/A'}")
                        logger.info(f"  - Bytes length: {len(audio_data.get('bytes', b'')) if isinstance(audio_data, dict) else 'N/A'}")

                        # Save webm temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
                            f.write(audio_data['bytes'])
                            f.flush()  # Flush to disk
                            webm_path = f.name

                        # CRITICAL: Ensure file is fully released before using it
                        import time
                        time.sleep(0.1)  # Small delay to ensure file handle release

                        # CHECKPOINT 2: File written to disk
                        st.write("**[DEBUG] Checkpoint 2:** File written to temp location")
                        logger.info("CHECKPOINT 2: Audio file written to temp location")
                        logger.info(f"  - Temp file path: {webm_path}")
                        logger.info(f"  - File exists: {os.path.exists(webm_path)}")
                        logger.info(f"  - File size: {os.path.getsize(webm_path) if os.path.exists(webm_path) else 'N/A'}")
                        logger.info(f"  - Absolute path: {os.path.abspath(webm_path)}")

                        try:
                            # Convert webm to wav using FFmpeg (Whisper handles WAV better)
                            try:
                                st.write("**[DEBUG] Checkpoint 2.5:** Converting WebM to WAV")
                                logger.info("CHECKPOINT 2.5: Converting WebM to WAV using FFmpeg")

                                wav_path = webm_path.replace(".webm", ".wav")
                                if FFMPEG_PATH:
                                    convert_cmd = [
                                        FFMPEG_PATH,
                                        "-i", webm_path,
                                        "-acodec", "pcm_s16le",
                                        "-ar", "16000",
                                        "-ac", "1",
                                        "-y",  # Overwrite output
                                        wav_path
                                    ]
                                    logger.info(f"  - FFmpeg command: {' '.join(convert_cmd)}")

                                    convert_result = subprocess.run(
                                        convert_cmd,
                                        capture_output=True,
                                        text=True,
                                        timeout=30
                                    )

                                    if convert_result.returncode == 0:
                                        logger.info(f"  - Conversion successful")
                                        logger.info(f"  - WAV path: {wav_path}")
                                        logger.info(f"  - WAV exists: {os.path.exists(wav_path)}")
                                        time.sleep(0.1)  # Small delay after conversion
                                        audio_input_path = wav_path  # Use WAV for Whisper
                                    else:
                                        logger.error(f"  - Conversion failed: {convert_result.stderr}")
                                        st.warning("FFmpeg conversion failed, trying with WebM")
                                        audio_input_path = webm_path  # Fallback to WebM
                                else:
                                    logger.warning("  - FFmpeg not available, using WebM directly")
                                    audio_input_path = webm_path
                            except Exception as e:
                                logger.error(f"EXCEPTION at CP2.5: {str(e)}", exc_info=True)
                                st.error(f"Conversion error: {str(e)}")
                                audio_input_path = webm_path  # Fallback

                            # CHECKPOINT 3: About to run Whisper
                            try:
                                st.write("**[DEBUG] Checkpoint 3:** Running Whisper CLI")
                                logger.info("CHECKPOINT 3: About to run Whisper CLI command")

                                # Use absolute path for output directory to avoid path issues
                                output_dir = os.path.abspath(".")
                                whisper_cmd = [sys.executable, "-m", "whisper", audio_input_path, "--model", "base", "--output_format", "json", "--output_dir", output_dir, "--verbose", "False"]
                                logger.info(f"  - Command: {' '.join(whisper_cmd)}")
                                logger.info(f"  - Current working directory: {os.getcwd()}")
                                logger.info(f"  - Output directory: {output_dir}")
                                logger.info(f"  - Python executable: {sys.executable}")

                                # CRITICAL: Add FFmpeg directory to PATH so Whisper can find it
                                env = os.environ.copy()
                                if FFMPEG_PATH:
                                    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
                                    env['PATH'] = ffmpeg_dir + os.pathsep + env.get('PATH', '')
                                    logger.info(f"  - Added FFmpeg dir to PATH: {ffmpeg_dir}")

                                result = subprocess.run(
                                    whisper_cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=60,
                                    env=env  # Pass environment with FFmpeg in PATH
                                )

                                # CHECKPOINT 4: Whisper execution completed
                                st.write("**[DEBUG] Checkpoint 4:** Whisper execution completed")
                                logger.info("CHECKPOINT 4: Whisper execution completed")
                                logger.info(f"  - Return code: {result.returncode}")
                                logger.info(f"  - Stdout length: {len(result.stdout)}")
                                logger.info(f"  - Stderr length: {len(result.stderr)}")
                                logger.info(f"  - Full Stdout:\n{result.stdout}")
                                logger.info(f"  - Full Stderr:\n{result.stderr}")

                                if result.returncode != 0:
                                    st.write(f"**[DEBUG] Whisper Error (code {result.returncode}):**")
                                    st.code(result.stderr, language="text")
                            except Exception as e:
                                logger.error(f"EXCEPTION at CP3-4: {str(e)}", exc_info=True)
                                st.error(f"Whisper execution error: {str(e)}")
                                result = None

                            if result and result.returncode == 0:
                                # CHECKPOINT 5: Looking for JSON output
                                # JSON is created in output_dir with the basename of the input file
                                webm_basename = os.path.basename(webm_path)
                                json_filename = webm_basename.replace(".webm", ".json")
                                json_file = os.path.join(output_dir, json_filename)

                                st.write("**[DEBUG] Checkpoint 5:** Looking for JSON output")
                                logger.info("CHECKPOINT 5: Looking for JSON output file")
                                logger.info(f"  - Input file basename: {webm_basename}")
                                logger.info(f"  - JSON filename: {json_filename}")
                                logger.info(f"  - Output directory: {output_dir}")
                                logger.info(f"  - Expected JSON path: {json_file}")
                                logger.info(f"  - JSON file exists: {os.path.exists(json_file)}")
                                logger.info(f"  - Files in output directory: {os.listdir(output_dir)[:15]}")  # List first 15 files

                                if os.path.exists(json_file):
                                    # CHECKPOINT 6: JSON file found, reading contents
                                    st.write("**[DEBUG] Checkpoint 6:** JSON file found, reading")
                                    logger.info("CHECKPOINT 6: JSON file found and readable")
                                    logger.info(f"  - File size: {os.path.getsize(json_file)}")

                                    with open(json_file, 'r', encoding='utf-8') as f:
                                        whisper_output = json.load(f)
                                        transcribed_text = whisper_output.get("text", "").strip()

                                        # CHECKPOINT 7: Text extracted
                                        st.write("**[DEBUG] Checkpoint 7:** Text extracted from JSON")
                                        logger.info("CHECKPOINT 7: Text extracted from JSON")
                                        logger.info(f"  - Transcribed text length: {len(transcribed_text)}")
                                        logger.info(f"  - Transcribed text: {transcribed_text[:200]}")

                                        if transcribed_text:
                                            st.success("✅ Audio transcribed!")
                                        else:
                                            st.warning("⚠️ No speech detected in audio")

                                    # Clean up JSON file
                                    os.remove(json_file)
                                    logger.info("CHECKPOINT 8: JSON file cleaned up")
                                else:
                                    st.error(f"JSON file not created. Looking in: {os.getcwd()}")
                                    logger.error("JSON file not found after Whisper execution")
                                    logger.error(f"  - Expected: {json_file}")
                                    logger.error(f"  - Directory contents: {os.listdir('.')}")
                            else:
                                st.error(f"Whisper error (code {result.returncode}): {result.stderr[:200]}")
                                logger.error(f"Whisper failed with return code {result.returncode}")
                                logger.error(f"  - Error: {result.stderr}")

                        except subprocess.TimeoutExpired:
                            st.error("Whisper transcription timed out (60 seconds)")
                            logger.error("Whisper transcription timed out")
                        finally:
                            # Clean up temp files
                            if os.path.exists(webm_path):
                                os.remove(webm_path)
                                logger.info("CHECKPOINT 9: Temp webm file cleaned up")
                            wav_path = webm_path.replace(".webm", ".wav")
                            if os.path.exists(wav_path):
                                os.remove(wav_path)
                                logger.info("CHECKPOINT 9.5: Temp wav file cleaned up")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Audio transcription error: {e}", exc_info=True)

        except ImportError:
            st.info("📍 Audio recording not available")

    with col2:
        st.markdown("#### 📁 Upload Audio File")
        uploaded_audio = st.file_uploader(
            "Upload audio (MP3, WAV, M4A, FLAC, OGG)",
            type=["mp3", "wav", "m4a", "flac", "ogg", "opus", "aac"],
            key="audio_upload"
        )

        if uploaded_audio:
            try:
                with st.spinner("🔄 Transcribing audio..."):
                    # CHECKPOINT 1: File upload received
                    st.write("**[DEBUG] Checkpoint 1:** File uploaded")
                    logger.info("CHECKPOINT 1: Audio file uploaded")
                    logger.info(f"  - File name: {uploaded_audio.name}")
                    logger.info(f"  - File size: {uploaded_audio.size}")

                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(uploaded_audio.getbuffer())
                        f.flush()  # Flush to disk
                        audio_path = f.name

                    # CRITICAL: Ensure file is fully released before using it
                    import time
                    time.sleep(0.1)  # Small delay to ensure file handle release

                    # CHECKPOINT 2: File saved
                    st.write("**[DEBUG] Checkpoint 2:** File saved to temp location")
                    logger.info("CHECKPOINT 2: Uploaded file saved to temp location")
                    logger.info(f"  - Temp file path: {audio_path}")
                    logger.info(f"  - File exists: {os.path.exists(audio_path)}")
                    logger.info(f"  - File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'}")

                    try:
                        # CHECKPOINT 3: Running Whisper
                        st.write("**[DEBUG] Checkpoint 3:** Running Whisper CLI")
                        logger.info("CHECKPOINT 3: About to run Whisper on uploaded file")

                        # Use absolute path for output directory to avoid path issues
                        output_dir = os.path.abspath(".")
                        whisper_cmd = [sys.executable, "-m", "whisper", audio_path, "--model", "base", "--output_format", "json", "--output_dir", output_dir, "--verbose", "False"]
                        logger.info(f"  - Command: {' '.join(whisper_cmd)}")
                        logger.info(f"  - Working dir: {os.getcwd()}")
                        logger.info(f"  - Output directory: {output_dir}")

                        # CRITICAL: Add FFmpeg directory to PATH so Whisper can find it
                        env = os.environ.copy()
                        if FFMPEG_PATH:
                            ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
                            env['PATH'] = ffmpeg_dir + os.pathsep + env.get('PATH', '')
                            logger.info(f"  - Added FFmpeg dir to PATH: {ffmpeg_dir}")

                        result = subprocess.run(
                            whisper_cmd,
                            capture_output=True,
                            text=True,
                            timeout=60,
                            env=env  # Pass environment with FFmpeg in PATH
                        )

                        # CHECKPOINT 4: Whisper completed
                        st.write("**[DEBUG] Checkpoint 4:** Whisper completed")
                        logger.info("CHECKPOINT 4: Whisper execution completed")
                        logger.info(f"  - Return code: {result.returncode}")
                        logger.info(f"  - Full Stdout:\n{result.stdout}")
                        logger.info(f"  - Full Stderr:\n{result.stderr}")

                        if result.returncode != 0:
                            st.write(f"**[DEBUG] Whisper Error (code {result.returncode}):**")
                            st.code(result.stderr, language="text")

                        if result.returncode == 0:
                            # CHECKPOINT 5: Looking for JSON
                            # JSON is created in output_dir with the basename of the input file
                            audio_basename = os.path.basename(audio_path)
                            json_filename = audio_basename.replace(".wav", ".json")
                            json_file = os.path.join(output_dir, json_filename)

                            st.write("**[DEBUG] Checkpoint 5:** Looking for JSON output")
                            logger.info("CHECKPOINT 5: Searching for JSON output")
                            logger.info(f"  - Input file basename: {audio_basename}")
                            logger.info(f"  - JSON filename: {json_filename}")
                            logger.info(f"  - Output directory: {output_dir}")
                            logger.info(f"  - Expected JSON: {json_file}")
                            logger.info(f"  - Exists: {os.path.exists(json_file)}")

                            if os.path.exists(json_file):
                                # CHECKPOINT 6: Reading JSON
                                st.write("**[DEBUG] Checkpoint 6:** Reading JSON")
                                logger.info("CHECKPOINT 6: JSON file found, reading")

                                with open(json_file, 'r', encoding='utf-8') as f:
                                    whisper_output = json.load(f)
                                    transcribed_text = whisper_output.get("text", "").strip()

                                    st.write("**[DEBUG] Checkpoint 7:** Text extracted")
                                    logger.info(f"CHECKPOINT 7: Text extracted: {len(transcribed_text)} chars")

                                    if transcribed_text:
                                        st.success("✅ Audio transcribed!")
                                    else:
                                        st.warning("⚠️ No speech detected")
                                os.remove(json_file)
                                logger.info("CHECKPOINT 8: JSON cleaned up")
                        else:
                            st.error(f"Whisper error (code {result.returncode}): {result.stderr[:200]}")
                            logger.error(f"Whisper failed: {result.stderr}")

                    except subprocess.TimeoutExpired:
                        st.error("Whisper transcription timed out (60 seconds)")
                        logger.error("Whisper timed out")
                    finally:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                            logger.info("CHECKPOINT 9: Temp file cleaned up")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Audio upload error: {e}", exc_info=True)

    # If audio was transcribed, show it and add to query
    if transcribed_text:
        st.markdown("---")
        st.markdown("### 📝 What you said:")
        st.markdown(
            f"""
            <div class='card'>
                <p style='margin: 0; font-size: 1.1em; line-height: 1.6;'><strong>{transcribed_text}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Auto-populate query with transcribed text
        if not query_text.strip():
            query_text = transcribed_text

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

    # Submit Button (Google-style)
    col1, col2, col3 = st.columns([1.2, 1.6, 1.2])
    with col2:
        submit_button = st.button("🚀 Get Answer", use_container_width=True, type="primary", key="main_submit")

    # Process query when button is clicked
    if submit_button:
        if not query_text.strip():
            st.error("❌ Please enter a question or record/upload audio")
        else:
            try:
                with st.spinner("🔄 Finding the best answer for you..."):
                    # Detect language
                    lang_detector = LanguageDetector()
                    detected_lang = lang_detector.detect_language(query_text)

                    # Initialize pipeline
                    pipeline = initialize_pipeline(llm_provider, llm_model)

                    # Get filter values from expander (set defaults if not in expander)
                    crop = st.session_state.get("crop_filter", "")
                    region = st.session_state.get("region_filter", "")
                    season = st.session_state.get("season_filter", "")
                    disease = st.session_state.get("disease_filter", "")
                    k = st.session_state.get("k_text_query", 5)
                    threshold = st.session_state.get("threshold_text_query", 0.2)

                    # Execute query
                    result = pipeline.query(
                        query_text,
                        k=k,
                        similarity_threshold=threshold,
                        language=detected_lang,
                        crop=crop if crop else None,
                        region=region if region else None,
                        season=season if season else None,
                        disease=disease if disease else None,
                        temperature=temperature,
                    )

                st.success("✅ Got your answer!")
                st.markdown("---")
                display_response(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure you have:")
                st.write("1. Built embeddings: `python main.py embed --pdf-dir Agri_docs`")
                st.write(f"2. Started {llm_provider} server")
                logger.error(f"Query error: {str(e)}")

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
