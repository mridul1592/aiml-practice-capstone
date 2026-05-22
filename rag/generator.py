"""
Response Generator for RAG system.

Handles:
- LLM integration (OpenAI, Ollama)
- Context-aware response generation
- Hallucination prevention
- Multilingual response generation
"""

from typing import Dict, List, Optional

from rag.prompt_templates import PromptTemplates
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMClient:
    """Abstract LLM client interface."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize LLM client.

        Args:
            model_name: Model name/identifier
            **kwargs: Additional configuration
        """
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Generation parameters

        Returns:
            Generated response
        """
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Ollama LLM client."""

    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.

        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
        """
        super().__init__(model_name)
        self.base_url = base_url

        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests library required for Ollama client")

        logger.info(f"Initialized Ollama client: {model_name}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate response using Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters

        Returns:
            Generated response

        Raises:
            Exception: If Ollama connection fails
        """
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                },
                timeout=120,
            )

            response.raise_for_status()
            result = response.json()

            return result.get("response", "").strip()

        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self.requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


class OpenAIClient(LLMClient):
    """OpenAI LLM client."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI client.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (default: from settings)

        Raises:
            ImportError: If openai not installed
            ValueError: If API key not provided
        """
        super().__init__(model_name)

        try:
            import openai
        except ImportError:
            raise ImportError("openai library required for OpenAI client")

        api_key = api_key or settings.openai_api_key

        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client: {model_name}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        Generate response using OpenAI.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum response length
            **kwargs: Additional parameters

        Returns:
            Generated response

        Raises:
            Exception: If API call fails
        """
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise


class ResponseGenerator:
    """Generate RAG responses with context grounding."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize response generator.

        Args:
            llm_client: LLM client instance

        Raises:
            ValueError: If LLM client is None
        """
        if llm_client is None:
            raise ValueError("LLM client cannot be None")

        self.llm_client = llm_client
        logger.info("Response generator initialized")

    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate RAG response.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            temperature: LLM temperature

        Returns:
            Dictionary with response and metadata

        Raises:
            ValueError: If query or chunks empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info("Generating response for query")

        # Check if we have sufficient context
        if not context_chunks:
            logger.warning("No context chunks provided")
            response_text = (
                "I do not have enough information to answer this question. "
                "Please consult a local agricultural expert or contact your nearest "
                "agricultural extension center."
            )
            return {
                "query": query,
                "response": response_text,
                "num_context_chunks": 0,
                "sources": [],
                "confidence": "low",
            }

        # Assemble context
        context_text = self._assemble_context(context_chunks)
        sources = [c.get("filename", "Unknown") for c in context_chunks]

        # Create system and retrieval prompts
        system_prompt = (
            "You are an agricultural expert providing farming advice based on provided context."
        )
        retrieval_template = (
            "Based on the following context:\n\n{context}\n\nAnswer this question: {query}"
        )

        # Format prompt
        formatted_prompt = retrieval_template.format(context=context_text, query=query)

        # Generate response
        try:
            response_text = self.llm_client.generate(
                formatted_prompt, system_prompt=system_prompt, temperature=temperature
            )

            # Check for hallucination indicators
            confidence = self._assess_confidence(response_text, context_chunks)

            return {
                "query": query,
                "response": response_text,
                "num_context_chunks": len(context_chunks),
                "sources": sources,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise

    def _assemble_context(self, context_chunks: List[Dict]) -> str:
        """
        Assemble context text from retrieved chunks.

        Args:
            context_chunks: List of retrieved chunk dicts

        Returns:
            Assembled context string
        """
        parts = []
        for i, chunk in enumerate(context_chunks, 1):
            content = chunk.get("content", "").strip()
            filename = chunk.get("filename", "Unknown")
            parts.append(f"[Source {i}: {filename}]\n{content}")
        return "\n\n".join(parts)

    def _assess_confidence(
        self, response: str, context_chunks: List[Dict], language: str = "en"
    ) -> str:
        """
        Assess confidence in generated response.

        Args:
            response: Generated response
            context_chunks: Context used for generation
            language: Response language code (en, hi, pa)

        Returns:
            Confidence level: high, medium, low
        """
        uncertainty_phrases = {
            "en": ["i don't know", "no information", "not available", "insufficient"],
            "hi": ["नहीं पता", "कोई जानकारी नहीं", "उपलब्ध नहीं", "अपर्याप्त"],
            "pa": ["ਪਤਾ ਨਹੀਂ", "ਕੋਈ ਜਾਣਕਾਰੀ ਨਹੀਂ", "ਉਪਲਬਧ ਨਹੀਂ"],
        }

        phrases = uncertainty_phrases.get(language, uncertainty_phrases["en"])
        response_lower = response.lower()

        for phrase in phrases:
            if phrase.lower() in response_lower:
                return "low"

        if len(context_chunks) >= 5:
            return "high"
        elif len(context_chunks) >= 3:
            return "medium"
        else:
            return "low"


def create_llm_client(provider: Optional[str] = None, **kwargs) -> LLMClient:
    """
    Create an LLM client based on provider.

    Args:
        provider: LLM provider (ollama, openai) (default: from settings)
        **kwargs: Additional configuration

    Returns:
        LLM client instance

    Raises:
        ValueError: If provider not supported
    """
    provider = provider or settings.llm_provider

    if provider == "ollama":
        return OllamaClient(
            model_name=kwargs.get("model_name", settings.ollama_model),
            base_url=kwargs.get("base_url", settings.ollama_base_url),
        )
    elif provider == "openai":
        return OpenAIClient(
            model_name=kwargs.get("model_name", "gpt-3.5-turbo"),
            api_key=kwargs.get("api_key", settings.openai_api_key),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def create_generator(provider: Optional[str] = None, **kwargs) -> ResponseGenerator:
    """
    Create a response generator.

    Args:
        provider: LLM provider
        **kwargs: Additional configuration

    Returns:
        ResponseGenerator instance
    """
    llm_client = create_llm_client(provider, **kwargs)
    return ResponseGenerator(llm_client)


if __name__ == "__main__":
    try:
        print("Testing Ollama client...")
        ollama_client = OllamaClient(model_name="mistral")

        if ollama_client.is_available():
            generator = ResponseGenerator(ollama_client)

            context_chunks = [
                {
                    "content": "Wheat pests can be controlled using integrated pest management.",
                    "filename": "wheat_guide.pdf",
                    "similarity_score": 0.85,
                }
            ]

            result = generator.generate("How to control wheat pests?", context_chunks)

            print("Query:", result["query"])
            print("Response:", result["response"])
            print("Confidence:", result["confidence"])
        else:
            print("Ollama not available. Please install and run Ollama.")

    except Exception as e:
        print(f"Error: {str(e)}")
