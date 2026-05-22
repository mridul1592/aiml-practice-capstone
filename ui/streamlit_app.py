"""
Agricultural AI Assistant — ChatGPT-style Streamlit UI.

Features:
- Persistent chat history (like ChatGPT)
- st.chat_message / st.chat_input for native chat layout
- Audio input (mic + upload) in sidebar → auto-submits as chat message
- 10-language support with automatic detection
- Hybrid BM25 + semantic retrieval
- Filters, k-chunks, similarity threshold in sidebar
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Path setup (must happen before project imports) ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config MUST be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="AgriBot 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Agricultural RAG System — 10 Languages, Hybrid Search"},
)

# ── Project imports (after sys.path) ────────────────────────────────────────
from utils.logger import setup_logger          # noqa: E402
logger = setup_logger(__name__)

from rag.rag_orchestrator import RAGPipeline   # noqa: E402
from rag.retriever import LanguageDetector     # noqa: E402
from utils.config import settings              # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
LANGUAGES = {
    "en": "🇬🇧 English",
    "hi": "🇮🇳 Hindi",
    "pa": "🇮🇳 Punjabi",
    "ta": "🇮🇳 Tamil",
    "te": "🇮🇳 Telugu",
    "or": "🇮🇳 Odia",
    "kn": "🇮🇳 Kannada",
    "mr": "🇮🇳 Marathi",
    "ml": "🇮🇳 Malayalam",
    "bn": "🇮🇳 Bengali",
}

EXAMPLE_QUESTIONS = [
    "How to control wheat pests?",
    "गेहूँ में कीटों का नियंत्रण कैसे करें?",
    "Best practices for paddy irrigation?",
    "Organic pest management for cotton?",
    "ਕਣਕ ਦੀ ਫਸਲ ਵਿੱਚ ਕੀੜਿਆਂ ਦੀ ਰੋਕਥਾਮ?",
]

# ── FFmpeg detection ─────────────────────────────────────────────────────────
def _get_ffmpeg_path() -> Optional[str]:
    candidates = [
        r"C:\Users\mridu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "FFmpeg" / "bin" / "ffmpeg.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return shutil.which("ffmpeg")

FFMPEG_PATH = _get_ffmpeg_path()

# ── CSS — ChatGPT-inspired layout ────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────── */
    .block-container { padding-top: 1rem; padding-bottom: 0; }

    /* ── Sidebar ────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #171717;
    }
    [data-testid="stSidebar"] * {
        color: #ececec !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stToggle label {
        color: #ababab !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #3a3a3a !important;
        margin: 0.6rem 0 !important;
    }
    [data-testid="stSidebar"] .stExpander {
        background: #202020 !important;
        border: 1px solid #3a3a3a !important;
        border-radius: 8px !important;
    }

    /* New-Chat button */
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        background: #2a2a2a !important;
        border: 1px solid #3a3a3a !important;
        color: #ececec !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
        background: #3a3a3a !important;
    }

    /* ── Welcome screen ─────────────────────── */
    .welcome-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 1rem 2rem;
        gap: 0.5rem;
    }
    .welcome-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .welcome-sub {
        color: #6b7280;
        font-size: 1rem;
        margin: 0 0 1.5rem;
    }
    .example-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        justify-content: center;
        max-width: 680px;
    }
    .example-btn {
        background: transparent;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 0.55rem 1rem;
        font-size: 0.88rem;
        cursor: pointer;
        color: inherit;
        transition: background 0.2s;
    }
    .example-btn:hover { background: #f3f4f6; }

    /* ── Chat message tweaks ────────────────── */
    [data-testid="stChatMessage"] {
        padding: 0.6rem 0.2rem;
    }

    /* ── Chat input bar ─────────────────────── */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        font-size: 1rem !important;
    }

    /* ── Response detail blocks ─────────────── */
    .resp-meta {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 0.5rem 0;
        font-size: 0.82rem;
        color: #6b7280;
    }
    .resp-tag {
        background: #f3f4f6;
        border-radius: 6px;
        padding: 2px 8px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached pipeline ───────────────────────────────────────────────────────────
@st.cache_resource
def initialize_pipeline(provider: str, model_name: str) -> RAGPipeline:
    return RAGPipeline(llm_provider=provider, llm_model_name=model_name)


# ── Audio transcription ───────────────────────────────────────────────────────
def _transcribe_audio(audio_bytes, suffix: str, language: str) -> str:
    """
    Transcribe audio bytes via Whisper CLI.
    Returns the transcribed string (may be empty) or raises on error.
    """
    import time as _time

    env = os.environ.copy()
    if FFMPEG_PATH:
        env["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + env.get("PATH", "")

    raw_path = wav_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(bytes(audio_bytes))
            f.flush()
            raw_path = f.name
        _time.sleep(0.05)

        wav_path = raw_path.rsplit(".", 1)[0] + "_in.wav"
        audio_input = raw_path
        if FFMPEG_PATH:
            conv = subprocess.run(
                [FFMPEG_PATH, "-y", "-i", raw_path,
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, text=True, timeout=30,
            )
            if conv.returncode == 0:
                audio_input = wav_path
        _time.sleep(0.05)

        output_dir = os.path.abspath(".")
        cmd = [sys.executable, "-m", "whisper", audio_input,
               "--model", "base", "--output_format", "json",
               "--output_dir", output_dir, "--verbose", "False"]
        if language and language != "Auto-detect":
            cmd += ["--language", language]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Whisper failed: {proc.stderr[:300]}")

        stem = Path(audio_input).stem
        json_path = os.path.join(output_dir, stem + ".json")
        if os.path.exists(json_path):
            text = json.load(open(json_path, encoding="utf-8")).get("text", "").strip()
            os.remove(json_path)
            return text
        raise RuntimeError("Whisper ran but produced no output file.")

    finally:
        for p in (raw_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# ── Response renderer (inside st.chat_message context) ───────────────────────
def _render_result(result: dict) -> None:
    """Render a RAG result dict inside an assistant chat bubble."""
    # Main answer text
    st.markdown(result.get("response", "No response returned."))

    # Compact metadata row
    lang_code = result.get("language", "en")
    lang_name = LANGUAGES.get(lang_code, lang_code)
    confidence = result.get("confidence", "low").upper()
    num_chunks = result.get("num_context_chunks", 0)
    mode = "Hybrid" if result.get("retrieval_mode") == "hybrid" else "Semantic"

    st.markdown(
        f'<div class="resp-meta">'
        f'<span class="resp-tag">🌐 {lang_name}</span>'
        f'<span class="resp-tag">📊 {confidence}</span>'
        f'<span class="resp-tag">📖 {num_chunks} chunks</span>'
        f'<span class="resp-tag">🔍 {mode}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Sources (top 3 unique, with doc name + section) ─────────────────────
    chunks = result.get("retrieved_chunks", [])

    if chunks:
        # Build deduplicated source list: (filename, section, chunk_index, score)
        seen_files: set = set()
        top_sources = []
        for i, chunk in enumerate(chunks):
            fname = chunk.get("filename", "Unknown")
            if fname not in seen_files:
                seen_files.add(fname)
                top_sources.append({
                    "filename": fname,
                    "section":  chunk.get("section", "").strip()[:60] or "—",
                    "chunk_no": i + 1,
                    "score":    chunk.get("similarity_score", 0),
                })
            if len(top_sources) == 3:
                break

        # Short display name: strip path, drop ".pdf"
        def _short(name: str) -> str:
            return Path(name).stem.replace("_", " ").replace("-", " ")

        with st.expander(f"📚 Sources — top {len(top_sources)} of {len(chunks)} chunks"):
            for s in top_sources:
                short = _short(s["filename"])
                st.markdown(
                    f"**📄 {short}**  \n"
                    f"Section: *{s['section']}* · "
                    f"Chunk #{s['chunk_no']} · "
                    f"Score: `{s['score']:.4f}`"
                )
                st.divider() if s != top_sources[-1] else None

    # ── Retrieved Chunks (all, collapsible) ──────────────────────────────────
    if chunks:
        with st.expander(f"📖 Retrieved Chunks ({len(chunks)})"):
            for i, chunk in enumerate(chunks, 1):
                score   = chunk.get("similarity_score", 0)
                content = chunk.get("content", "")
                fname   = _short(chunk.get("filename", "Unknown"))
                section = chunk.get("section", "").strip()[:50]
                st.markdown(
                    f"**Chunk {i}** · 📄 `{fname}` · *{section}* · Score: `{score:.4f}`"
                )
                st.markdown(content)
                if i < len(chunks):
                    st.divider()


# ── Core query handler ────────────────────────────────────────────────────────
def _process_query(query: str, pipeline: RAGPipeline, cfg: dict) -> None:
    """
    Append the user message, call the pipeline, append the assistant message,
    all with proper chat bubble rendering.
    """
    # Append + render user bubble immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run pipeline and render assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                lang = LanguageDetector().detect_language(query)
                result = pipeline.query(
                    query,
                    k=cfg["k"],
                    similarity_threshold=cfg["threshold"],
                    language=lang,
                    crop=cfg["crop"] or None,
                    region=cfg["region"] or None,
                    season=cfg["season"] or None,
                    disease=cfg["disease"] or None,
                    temperature=cfg["temperature"],
                    use_hybrid=cfg["use_hybrid"],
                )
                _render_result(result)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.get("response", ""), "result": result}
                )
            except Exception as exc:
                import traceback
                err = traceback.format_exc()
                logger.error(f"Query error:\n{err}")

                err_str = str(exc)
                # Friendly guidance for common Ollama errors
                if "500" in err_str or "Internal Server Error" in err_str:
                    st.error(
                        "❌ Ollama returned a 500 error — the model context overflowed or "
                        "is still loading.\n\n"
                        "**Quick fixes:**\n"
                        "- Reduce **Context Chunks (k)** in *Advanced Settings* (try k=5)\n"
                        "- Wait a few seconds and try again (model may still be loading)\n"
                        "- Restart Ollama: `ollama serve`"
                    )
                elif "Connection" in err_str or "refused" in err_str.lower():
                    st.error(
                        "❌ Cannot reach Ollama. Make sure it is running: `ollama serve`"
                    )
                else:
                    st.error(f"❌ {exc}")

                with st.expander("🔍 Error details"):
                    st.code(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {exc}", "result": None}
                )


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:

    # ── Session state init ───────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None
    if "pending_audio_query" not in st.session_state:
        st.session_state.pending_audio_query = None

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # Logo + new chat
        st.markdown("## 🌾 AgriBot")
        st.caption("Agricultural AI · 10 languages")
        if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
            st.session_state.messages = []
            st.session_state.last_audio_id = None
            st.session_state.pending_audio_query = None
            st.rerun()

        st.divider()

        # ── Model ────────────────────────────────────────────────────────────
        st.markdown("**Model**")
        llm_provider = st.selectbox(
            "Provider", ["ollama", "openai"],
            key="sb_provider", label_visibility="collapsed",
        )
        if llm_provider == "ollama":
            llm_model = st.selectbox(
                "Model", ["neural-chat", "mistral", "llama2", "orca2"],
                key="sb_model", label_visibility="collapsed",
            )
        else:
            llm_model = st.text_input(
                "Model name", value="gpt-3.5-turbo",
                key="sb_model_oai", label_visibility="collapsed",
            )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.7, 0.1, key="sb_temp",
        )

        st.divider()

        # ── Search mode ──────────────────────────────────────────────────────
        st.markdown("**Search Mode**")
        use_hybrid = st.toggle(
            "Hybrid (BM25 + Semantic)",
            value=True, key="sb_hybrid",
            help="Combines keyword and embedding search via Reciprocal Rank Fusion.",
        )
        st.caption("🔀 BM25 + FAISS → RRF" if use_hybrid else "🧠 FAISS embeddings only")

        st.divider()

        # ── Advanced ─────────────────────────────────────────────────────────
        with st.expander("⚙️ Advanced Settings"):
            k = st.slider("Context chunks (k)", 1, 20, 10, key="sb_k")
            threshold = st.slider(
                "Similarity threshold", 0.0, 1.0, 0.2, 0.05, key="sb_thresh",
            )
            st.markdown("**Filters**")
            crop = st.selectbox(
                "Crop", ["", "wheat", "paddy", "rice", "maize", "cotton", "sugarcane"],
                key="sb_crop",
            )
            region  = st.text_input("Region", placeholder="e.g. Punjab", key="sb_region")
            season  = st.selectbox("Season", ["", "kharif", "rabi", "summer"], key="sb_season")
            disease = st.text_input("Disease / Pest", placeholder="e.g. rust", key="sb_disease")

        st.divider()

        # ── Audio ────────────────────────────────────────────────────────────
        with st.expander("🎙️ Audio Input"):
            if FFMPEG_PATH:
                st.caption(f"✅ FFmpeg ready")
            else:
                st.caption("❌ FFmpeg not found — conversion may fail")

            audio_lang = st.selectbox(
                "Audio language",
                ["Auto-detect", "en", "hi", "pa"],
                format_func=lambda x: "🔍 Auto-detect" if x == "Auto-detect"
                            else LANGUAGES.get(x, x),
                key="sb_audio_lang",
            )

            st.markdown("**Record**")
            try:
                from streamlit_mic_recorder import mic_recorder
                audio_data = mic_recorder(
                    start_prompt="🎤 Start", stop_prompt="⏹ Stop",
                    just_once=True, use_container_width=True,
                    format="webm", key="sb_mic",
                )
                if audio_data:
                    aid = audio_data.get("id", id(audio_data["bytes"]))
                    if st.session_state.last_audio_id != aid:
                        st.session_state.last_audio_id = aid
                        with st.spinner("🎙️ Transcribing…"):
                            try:
                                text = _transcribe_audio(
                                    audio_data["bytes"], ".webm", audio_lang
                                )
                                if text:
                                    st.session_state.pending_audio_query = text
                                    st.rerun()
                                else:
                                    st.warning("No speech detected.")
                            except Exception as e:
                                st.error(str(e))
            except ImportError:
                st.info("Install `streamlit-mic-recorder` for recording.")

            st.markdown("**Upload**")
            uploaded = st.file_uploader(
                "Audio file", type=["mp3", "wav", "m4a", "flac", "ogg", "opus", "aac"],
                key="sb_upload", label_visibility="collapsed",
            )
            if uploaded:
                uid = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.last_audio_id != uid:
                    st.session_state.last_audio_id = uid
                    with st.spinner("🎙️ Transcribing…"):
                        try:
                            text = _transcribe_audio(
                                uploaded.getbuffer(),
                                Path(uploaded.name).suffix.lower(),
                                audio_lang,
                            )
                            if text:
                                st.session_state.pending_audio_query = text
                                st.rerun()
                            else:
                                st.warning("No speech detected.")
                        except Exception as e:
                            st.error(str(e))

        st.divider()

        # ── System info ──────────────────────────────────────────────────────
        with st.expander("📊 System Info"):
            try:
                pipeline = initialize_pipeline(llm_provider, llm_model)
                stats = pipeline.get_stats()
                vs = stats.get("vector_store", {})
                st.metric("Vectors", vs.get("total_vectors", "—"))
                st.metric("Embedding dim", vs.get("embedding_dimension", "—"))
                st.caption(f"Model: {settings.embedding_model}")
                st.caption(f"Languages: {len(settings.supported_languages)}")
            except Exception as e:
                st.warning(str(e))

    # ── MAIN CHAT AREA ────────────────────────────────────────────────────────

    # Build pipeline config dict (read once, passed everywhere)
    cfg = {
        "k":           st.session_state.get("sb_k", 10),
        "threshold":   st.session_state.get("sb_thresh", 0.2),
        "use_hybrid":  st.session_state.get("sb_hybrid", True),
        "temperature": st.session_state.get("sb_temp", 0.7),
        "crop":        st.session_state.get("sb_crop", ""),
        "region":      st.session_state.get("sb_region", ""),
        "season":      st.session_state.get("sb_season", ""),
        "disease":     st.session_state.get("sb_disease", ""),
    }

    try:
        pipeline = initialize_pipeline(llm_provider, llm_model)
    except Exception as e:
        st.error(f"Could not initialise pipeline: {e}")
        st.stop()

    # Welcome screen (only when no messages yet)
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-wrap">
                <p style="font-size:3rem;margin:0">🌾</p>
                <p class="welcome-title">How can I help you today?</p>
                <p class="welcome-sub">Ask farming questions in any of 10 Indian languages</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Clickable example questions
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, q in zip(cols, EXAMPLE_QUESTIONS):
            with col:
                if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
                    _process_query(q, pipeline, cfg)
                    st.rerun()

    # ── Replay chat history ───────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                result = msg.get("result")
                if result:
                    _render_result(result)
                else:
                    st.markdown(msg.get("content", ""))

    # ── Auto-submit pending audio transcription ───────────────────────────────
    if st.session_state.pending_audio_query:
        query = st.session_state.pending_audio_query
        st.session_state.pending_audio_query = None
        _process_query(query, pipeline, cfg)
        st.rerun()

    # ── Chat input (sticky at bottom) ─────────────────────────────────────────
    if prompt := st.chat_input(
        "Ask about farming, crops, pests, irrigation… (any language)"
    ):
        _process_query(prompt, pipeline, cfg)
        st.rerun()


if __name__ == "__main__":
    main()
