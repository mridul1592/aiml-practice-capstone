"""
Streamlit UI for Agricultural RAG System.

Provides web interface for:
- Query submission with metadata filters
- Result display with retrieved chunks
- Configuration management
- System statistics
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_orchestrator import RAGPipeline
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Agricultural RAG System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #2ecc71;
        margin-bottom: 30px;
    }
    .result-box {
        background-color: #000000;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 10px;
        border-left: 4px solid #3498db;
        margin: 5px 0;
    }
    .chunk-box {
        background-color: #f9f9f9;
        padding: 12px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_pipeline(provider: str, model_name: str) -> RAGPipeline:
    """Initialize RAG pipeline with caching."""
    return RAGPipeline(llm_provider=provider, llm_model_name=model_name)


def main():
    """Main Streamlit application."""
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h1 style='text-align: center; color: #2ecc71;'>🌾 Agricultural RAG System</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; color: #666;'>Smart farming advice powered by AI</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Provider
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["ollama", "openai"],
            help="Select the LLM provider for response generation",
        )
        
        # LLM Model (conditional)
        if llm_provider == "ollama":
            llm_model = st.text_input(
                "Ollama Model",
                value="mistral",
                help="Model name available in Ollama (e.g., mistral, llama2, neural-chat)",
            )
        else:
            llm_model = st.text_input(
                "OpenAI Model",
                value="gpt-3.5-turbo",
                help="OpenAI model (requires OPENAI_API_KEY in .env)",
            )
        
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative",
        )
        
        st.markdown("---")
        st.header("📊 System Info")
        
        # Display settings
        with st.expander("View Settings"):
            st.json({
                "Embedding Model": settings.embedding_model,
                "Embedding Dimension": settings.embedding_dimension,
                "Chunk Size": settings.chunk_size,
                "Top K Retrieval": settings.top_k_retrieval,
                "Vector Store Path": settings.faiss_index_path,
            })
        
        # Vector store stats
        try:
            pipeline = initialize_pipeline(llm_provider, llm_model)
            stats = pipeline.get_stats()
            
            with st.expander("Vector Store Stats", expanded=False):
                if "vector_store" in stats:
                    st.write(f"**Total Vectors:** {stats['vector_store'].get('total_vectors', 0)}")
                    st.write(f"**Total Metadata:** {stats['vector_store'].get('total_metadata', 0)}")
                    st.write(f"**Embedding Dimension:** {stats['vector_store'].get('embedding_dimension', 0)}")
                else:
                    st.warning("Vector store not initialized. Run embedding pipeline first.")
        except Exception as e:
            st.warning(f"Could not load stats: {str(e)}")

    # Main Content Area
    st.header("🔍 Query Agricultural Database")
    
    # Query Input
    query_text = st.text_area(
        "What would you like to know?",
        placeholder="e.g., How to control wheat pests? What is the best time to plant paddy?",
        height=100,
        label_visibility="collapsed",
    )

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        crop = st.selectbox(
            "Crop",
            options=["", "wheat", "paddy", "rice", "maize", "cotton"],
            help="Filter by crop type",
        )
    
    with col2:
        region = st.text_input(
            "Region",
            placeholder="e.g., Punjab, Gujarat",
            help="Filter by region (optional)",
        )
    
    with col3:
        season = st.selectbox(
            "Season",
            options=["", "kharif", "rabi", "summer"],
            help="Filter by season (optional)",
        )
    
    with col4:
        disease = st.text_input(
            "Disease/Pest",
            placeholder="e.g., rust, blight",
            help="Filter by disease/pest (optional)",
        )

    # Retrieval Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        k = st.slider(
            "Number of Context Chunks",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="More chunks = more context but longer response time",
        )
    
    with col2:
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Lower = broader results, Higher = stricter matching",
        )

    # Submit Button
    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if not query_text.strip():
            st.error("❌ Please enter a query")
        else:
            try:
                # Initialize pipeline
                with st.spinner("🔄 Processing query..."):
                    pipeline = initialize_pipeline(llm_provider, llm_model)
                    
                    # Build filters
                    filters_dict = {}
                    if crop:
                        filters_dict["crop"] = crop
                    if region:
                        filters_dict["region"] = region
                    if season:
                        filters_dict["season"] = season
                    if disease:
                        filters_dict["disease"] = disease
                    
                    # Execute query
                    result = pipeline.query(
                        query_text,
                        k=k,
                        similarity_threshold=threshold,
                        crop=crop if crop else None,
                        region=region if region else None,
                        season=season if season else None,
                        disease=disease if disease else None,
                        temperature=temperature,
                    )
                
                # Display Results
                st.success("✅ Query processed successfully!")
                
                st.markdown("---")
                
                # Response Section
                st.subheader("💬 Response")
                st.markdown(
                    f"<div class='result-box'>{result['response']}</div>",
                    unsafe_allow_html=True,
                )
                
                # Confidence and Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", result.get("confidence", "N/A"))
                with col2:
                    st.metric("Retrieved Chunks", result["num_context_chunks"])
                with col3:
                    st.metric("Sources", len(result.get("sources", [])))
                
                st.markdown("---")
                
                # Retrieved Sources
                if result.get("sources"):
                    st.subheader("📚 Sources")
                    for source in result["sources"]:
                        st.markdown(
                            f"<div class='source-box'>📄 {source}</div>",
                            unsafe_allow_html=True,
                        )
                
                st.markdown("---")
                
                # Retrieved Chunks (expandable)
                if result.get("retrieved_chunks"):
                    with st.expander(f"📖 Retrieved Chunks ({len(result['retrieved_chunks'])})"):
                        for i, chunk in enumerate(result["retrieved_chunks"], 1):
                            similarity = chunk.get("similarity_score", 0)
                            content = chunk.get("content", "")[:500]  # First 500 chars
                            filename = chunk.get("filename", "Unknown")
                            
                            st.markdown(
                                f"""
                                <div class='chunk-box'>
                                <b>Chunk {i}</b> | 📄 {filename} | 🎯 Similarity: {similarity:.2f}
                                <br/><br/>
                                {content}...
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.info("💡 Tip: Make sure you have:")
                st.write("1. Built embeddings: `python main.py embed --pdf-dir Agri_docs`")
                st.write(f"2. Started {llm_provider} server")
                if llm_provider == "openai":
                    st.write("3. Set OPENAI_API_KEY in .env file")
                logger.error(f"Query error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #999; font-size: 12px;'>
        Agricultural RAG System | Powered by SentenceTransformers + FAISS + LLM
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
