"""
FastAPI application for Agricultural RAG System.

Provides REST endpoints for:
- Query execution
- Health checks
- System statistics
- Embeddings management
"""

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.rag_orchestrator import RAGPipeline
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agricultural RAG System API",
    description="Multilingual RAG system for agricultural advisory (wheat, paddy)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or initialize RAG pipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        logger.info("Initializing RAG pipeline...")
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(..., description="The query text", min_length=1)
    language: Optional[str] = Field(
        None, description="Response language (en, hi, pa)"
    )
    crop: Optional[str] = Field(None, description="Filter by crop")
    region: Optional[str] = Field(None, description="Filter by region")
    season: Optional[str] = Field(None, description="Filter by season")
    disease: Optional[str] = Field(None, description="Filter by disease")
    k: int = Field(5, description="Number of retrieved chunks", ge=1, le=20)
    threshold: float = Field(
        0.2, description="Similarity threshold", ge=0.0, le=1.0
    )
    temperature: float = Field(0.7, description="LLM temperature", ge=0.0, le=1.0)

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "query": "How to control wheat pests?",
                "language": "en",
                "crop": "wheat",
                "k": 5,
                "temperature": 0.7,
            }
        }


class RetrievedChunk(BaseModel):
    """Retrieved chunk model."""

    content: str
    filename: str
    similarity_score: float
    crop: Optional[str] = None
    language: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response model."""

    query: str
    response: str
    language: str
    confidence: str
    sources: List[str]
    num_context_chunks: int
    retrieved_chunks: Optional[List[RetrievedChunk]] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str
    components: Dict[str, str]


class StatsResponse(BaseModel):
    """System statistics response model."""

    embedder: Dict
    vector_store: Dict
    retriever_support: Dict


# Routes


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status and component information
    """
    try:
        pipeline = get_rag_pipeline()
        return HealthResponse(
            status="healthy",
            message="Agricultural RAG System is operational",
            components={
                "embedder": "ready",
                "vector_store": "ready",
                "retriever": "ready",
                "generator": "ready",
            },
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get system statistics.

    Returns:
        System statistics and component information
    """
    try:
        pipeline = get_rag_pipeline()
        stats = pipeline.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest) -> QueryResponse:
    """
    Execute a RAG query.

    Args:
        request: Query request with filters and parameters

    Returns:
        Query response with answer, sources, and confidence

    Raises:
        HTTPException: If query execution fails
    """
    try:
        logger.info(f"Processing query: {request.query}")

        pipeline = get_rag_pipeline()
        result = pipeline.query(
            query_text=request.query,
            k=request.k,
            similarity_threshold=request.threshold,
            language=request.language,
            crop=request.crop,
            region=request.region,
            season=request.season,
            disease=request.disease,
            temperature=request.temperature,
        )

        # Format retrieved chunks
        retrieved_chunks = None
        if result.get("retrieved_chunks"):
            retrieved_chunks = [
                RetrievedChunk(
                    content=chunk.get("content", ""),
                    filename=chunk.get("filename", "Unknown"),
                    similarity_score=chunk.get("similarity_score", 0),
                    crop=chunk.get("crop"),
                    language=chunk.get("language"),
                )
                for chunk in result["retrieved_chunks"]
            ]

        response = QueryResponse(
            query=result["query"],
            response=result["response"],
            language=result["language"],
            confidence=result["confidence"],
            sources=result["sources"],
            num_context_chunks=result["num_context_chunks"],
            retrieved_chunks=retrieved_chunks,
        )

        logger.info(f"Query processed successfully")
        return response

    except ValueError as e:
        logger.error(f"Invalid query: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Query execution failed")


@app.get("/supported-languages")
async def get_supported_languages() -> Dict:
    """
    Get supported languages.

    Returns:
        Dictionary with supported language codes
    """
    return {
        "supported_languages": settings.supported_languages,
        "descriptions": {
            "en": "English",
            "hi": "Hindi (हिंदी)",
            "pa": "Punjabi (ਪੰਜਾਬੀ)",
        },
    }


@app.get("/supported-crops")
async def get_supported_crops() -> Dict:
    """
    Get supported crops.

    Returns:
        Dictionary with supported crops (based on ingested documents)
    """
    try:
        pipeline = get_rag_pipeline()
        stats = pipeline.get_stats()
        return {
            "message": "Supported crops depend on ingested documents",
            "common_crops": ["wheat", "paddy", "rice"],
            "vector_store_stats": stats["vector_store"],
        }
    except Exception as e:
        logger.error(f"Failed to retrieve crops: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve crops")


@app.get("/")
async def root() -> Dict:
    """Root endpoint with API information."""
    return {
        "name": "Agricultural RAG System API",
        "version": "1.0.0",
        "description": "Multilingual RAG for agricultural advisory",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "stats": "/stats",
            "query": "/query",
            "languages": "/supported-languages",
            "crops": "/supported-crops",
        },
    }


# Error handlers


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors."""
    logger.error(f"Value error: {str(exc)}")
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle generic exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level="info",
    )
