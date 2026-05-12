"""
Prompt templates for RAG system.

Provides templates for:
- Query understanding
- Context grounding
- Multilingual response generation
- Hallucination prevention
"""

from typing import Dict, List, Optional


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(self, template: str, variables: List[str]):
        """
        Initialize prompt template.

        Args:
            template: Template string with {variable} placeholders
            variables: List of required variables
        """
        self.template = template
        self.variables = variables

    def format(self, **kwargs) -> str:
        """
        Format template with provided variables.

        Args:
            **kwargs: Variable values

        Returns:
            Formatted prompt

        Raises:
            ValueError: If required variables missing
        """
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        return self.template.format(**kwargs)


class PromptTemplates:
    """Collection of prompt templates for RAG system."""

    # English templates
    ENGLISH_SYSTEM_PROMPT = """You are an expert agricultural advisor for Indian farmers.
Your role is to provide accurate, practical, and science-based agricultural guidance.

IMPORTANT RULES:
1. Only use information from the provided context. Do not make up information.
2. If the answer is not in the context, clearly state: "I do not have enough reliable information to answer this question."
3. Always cite the source document when providing information.
4. Provide practical, actionable advice suitable for small and marginal farmers.
5. Consider regional and seasonal variations when relevant.
6. If multiple solutions exist, explain the pros and cons of each.

You are speaking to farmers in India, so:
- Use simple, clear language
- Avoid jargon or explain technical terms
- Provide specific measurements and timings
- Include cost considerations when relevant
"""

    ENGLISH_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""Based on the following agricultural context, answer the question. 
Use ONLY the information provided in the context. If the answer is not in the context, say "I do not have enough reliable information."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:""",
        variables=["context", "query"],
    )

    # Hindi templates
    HINDI_SYSTEM_PROMPT = """आप भारतीय किसानों के लिए एक कृषि विशेषज्ञ सलाहकार हैं।
आपकी भूमिका सटीक, व्यावहारिक और विज्ञान-आधारित कृषि मार्गदर्शन प्रदान करना है।

महत्वपूर्ण नियम:
1. केवल प्रदान किए गए संदर्भ से जानकारी का उपयोग करें। बनाई हुई जानकारी न दें।
2. यदि उत्तर संदर्भ में नहीं है, तो स्पष्ट रूप से कहें: "मेरे पास इस सवाल का जवाब देने के लिए पर्याप्त विश्वसनीय जानकारी नहीं है।"
3. जानकारी देते समय हमेशा स्रोत दस्तावेज़ का उल्लेख करें।
4. छोटे और सीमांत किसानों के लिए उपयुक्त व्यावहारिक, कार्यान्वयन योग्य सलाह प्रदान करें।
5. प्रासंगिक होने पर क्षेत्रीय और मौसमी भिन्नताओं पर विचार करें।
6. यदि कई समाधान मौजूद हैं, तो प्रत्येक के फायदे और नुकसान समझाएं।
"""

    HINDI_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""निम्नलिखित कृषि संदर्भ के आधार पर सवाल का जवाब दें।
केवल संदर्भ में प्रदान की गई जानकारी का उपयोग करें। यदि उत्तर संदर्भ में नहीं है, तो कहें "मेरे पास पर्याप्त जानकारी नहीं है।"

संदर्भ:
{context}

सवाल: {query}

जवाब:""",
        variables=["context", "query"],
    )

    # Punjabi templates
    PUNJABI_SYSTEM_PROMPT = """ਤੁਸੀਂ ਭਾਰਤੀ ਕਿਸਾਨਾਂ ਲਈ ਇੱਕ ਖੇਤੀਬਾੜੀ ਸਲਾਹਕਾਰ ਹੋ।
ਤੁਹਾਡੀ ਭੂਮਿਕਾ ਸਟੀਕ, ਵਿਹਾਰਕ ਅਤੇ ਵਿਗਿਆਨ-ਆਧਾਰਿਤ ਖੇਤੀ ਮਾਗਦਰਸ਼ਨ ਪ್ਰਦਾਨ ਕਰਨੀ ਹੈ।

ਮਹੱਤਵਪੂਰਨ ਨਿਯਮ:
1. ਸਿਰਫ਼ ਪ੍ਰਦਾਨ ਕੀਤੇ ਸੰਦਰਭ ਤੋਂ ਜਾਣਕਾਰੀ ਦੀ ਵਰਤੋਂ ਕਰੋ। ਬਣਾਈ ਗਈ ਜਾਣਕਾਰੀ ਨਾ ਦਿਓ।
2. ਜੇ ਜਵਾਬ ਸੰਦਰਭ ਵਿੱਚ ਨਹੀਂ ਹੈ, ਤਾਂ ਸਪੱਸ਼ਟ ਰੂਪ ਨਾਲ ਕਹੋ: "ਮੇਰ ਕੋਲ ਇਸ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦੇਣ ਲਈ ਕਾਫ਼ੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।"
3. ਜਾਣਕਾਰੀ ਦਿੰਦੇ ਸਮੇਂ ਹਮੇਸ਼ਾ ਸਰੋਤ ਦਸਤਾਵੇਜ਼ ਦਾ ਹਵਾਲਾ ਦਿਓ।
4. ਛੋਟੇ ਅਤੇ ਹਾਸ਼ੀਏ ਕਿਸਾਨਾਂ ਲਈ ਉਪਯੁਕਤ ਵਿਹਾਰਕ, ਕਾਰਜਸ਼ੀਲ ਸਲਾਹ ਪ੍ਰਦਾਨ ਕਰੋ।
5. ਪ್ਰਾਸੰਗਿਕ ਹੋਣ ਸਮੇਂ ਖੇਤਰੀ ਅਤੇ ਮੌਸਮੀ ਭਿੰਨਤਾਵਾਂ ਨੂੰ ਧਿਆਨ ਵਿੱਚ ਰੱਖੋ।
6. ਜੇ ਕਈ ਹੱਲ ਮੌਜੂਦ ਹਨ, ਤਾਂ ਹਰੇਕ ਦੇ ਪੱਖ ਅਤੇ ਵਿਪੱਖ ਸਮਝਾਓ।
"""

    PUNJABI_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""ਹੇਠ ਲਿਖੇ ਖੇਤੀ ਸੰਦਰਭ ਦੇ ਆਧਾਰ ਤੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦਿਓ।
ਸਿਰਫ਼ ਸੰਦਰਭ ਵਿੱਚ ਪ੍ਰਦਾਨ ਕੀਤੀ ਜਾਣਕਾਰੀ ਦੀ ਵਰਤੋਂ ਕਰੋ। ਜੇ ਜਵਾਬ ਸੰਦਰਭ ਵਿੱਚ ਨਹੀਂ ਹੈ, ਤਾਂ ਕਹੋ "ਮੇਰ ਕੋਲ ਕਾਫ਼ੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।"

ਸੰਦਰਭ:
{context}

ਸਵਾਲ: {query}

ਜਵਾਬ:""",
        variables=["context", "query"],
    )

    # Context assembly templates
    CONTEXT_ASSEMBLY_TEMPLATE = PromptTemplate(
        template="""RETRIEVED DOCUMENTS:

{context}

---
These documents are retrieved based on semantic similarity to the query.
Use this information to answer the user's question accurately.""",
        variables=["context"],
    )

    CONTEXT_WITH_SOURCES_TEMPLATE = PromptTemplate(
        template="""RETRIEVED DOCUMENTS:

{context}

---
DOCUMENT SOURCES:
{sources}

These documents are retrieved based on semantic similarity to the query.""",
        variables=["context", "sources"],
    )

    # Guardrail templates
    INSUFFICIENT_CONTEXT_RESPONSE = {
        "en": "I do not have enough reliable information to answer this question. Please consult a local agricultural expert or contact your nearest agricultural extension center.",
        "hi": "मेरे पास इस सवाल का जवाब देने के लिए पर्याप्त विश्वसनीय जानकारी नहीं है। कृपया किसी स्थानीय कृषि विशेषज्ञ से संपर्क करें या अपने निकटतम कृषि विस्तार केंद्र से संपर्क करें।",
        "pa": "ਮੇਰ ਕੋਲ ਇਸ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦੇਣ ਲਈ ਕਾਫ਼ੀ ਭਰੋਸੇਮੰਦ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕਿਸੇ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਮਾਹਰ ਨੂੰ ਸੱਪਰਸ਼ ਕਰੋ ਜਾਂ ਆਪਣੇ ਨਜ਼ਦੀਕੀ ਖੇਤੀ ਵਿਸਤਾਰ ਕੇਂਦਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
    }

    QUERY_UNCLEAR_RESPONSE = {
        "en": "I didn't fully understand your question. Could you please clarify? For example:\n- Which crop are you asking about? (wheat, paddy, etc.)\n- What is the specific problem? (pests, diseases, irrigation, etc.)\n- Which region are you in?",
        "hi": "मुझे आपके सवाल को पूरी तरह समझ नहीं आया। क्या आप स्पष्ट कर सकते हैं? उदाहरण के लिए:\n- आप किस फसल के बारे में पूछ रहे हैं? (गेहूँ, धान, आदि)\n- विशिष्ट समस्या क्या है? (कीटों, बीमारियों, सिंचाई, आदि)\n- आप किस क्षेत्र में हैं?",
        "pa": "ਮੈਨੂੰ ਤੁਹਾਡੇ ਸਵਾਲ ਨੂੰ ਪੂਰੀ ਤਰ੍ਹਾਂ ਸਮਝ ਨਹੀਂ ਆਇਆ। ਕੀ ਤੁਸੀਂ ਸਪੱਸ਼ਟ ਕਰ ਸਕਦੇ ਹੋ? ਉਦਾਹਰਨ ਲਈ:\n- ਤੁਸੀਂ ਕਿਸ ਫਸਲ ਬਾਰੇ ਪੁੱਛ ਰਹੇ ਹੋ? (ਗੇਹੂੰ, ਚਾਵਲ, ਆਦਿ)\n- ਕੀ ਮਖਸੂਸ ਮਸਲਾ ਹੈ? (ਕੀਟ, ਬਿਮਾਰੀ, ਸਿੰਚਾਈ, ਆਦਿ)\n- ਤੁਸੀਂ ਕਿਸ ਖੇਤਰ ਵਿੱਚ ਹੋ?",
    }

    @classmethod
    def get_system_prompt(cls, language: str = "en") -> str:
        """
        Get system prompt for a language.

        Args:
            language: Language code (en, hi, pa)

        Returns:
            System prompt
        """
        prompts = {
            "en": cls.ENGLISH_SYSTEM_PROMPT,
            "hi": cls.HINDI_SYSTEM_PROMPT,
            "pa": cls.PUNJABI_SYSTEM_PROMPT,
        }

        return prompts.get(language, cls.ENGLISH_SYSTEM_PROMPT)

    @classmethod
    def get_retrieval_template(cls, language: str = "en") -> PromptTemplate:
        """
        Get retrieval prompt template for a language.

        Args:
            language: Language code

        Returns:
            PromptTemplate instance
        """
        templates = {
            "en": cls.ENGLISH_RETRIEVAL_TEMPLATE,
            "hi": cls.HINDI_RETRIEVAL_TEMPLATE,
            "pa": cls.PUNJABI_RETRIEVAL_TEMPLATE,
        }

        return templates.get(language, cls.ENGLISH_RETRIEVAL_TEMPLATE)

    @classmethod
    def get_insufficient_context_response(cls, language: str = "en") -> str:
        """
        Get insufficient context response.

        Args:
            language: Language code

        Returns:
            Response text
        """
        return cls.INSUFFICIENT_CONTEXT_RESPONSE.get(language, cls.INSUFFICIENT_CONTEXT_RESPONSE["en"])

    @classmethod
    def get_unclear_query_response(cls, language: str = "en") -> str:
        """
        Get unclear query response.

        Args:
            language: Language code

        Returns:
            Response text
        """
        return cls.QUERY_UNCLEAR_RESPONSE.get(language, cls.QUERY_UNCLEAR_RESPONSE["en"])

    @classmethod
    def assemble_context(cls, chunks: List[Dict], include_sources: bool = True) -> str:
        """
        Assemble retrieved chunks into context string.

        Args:
            chunks: List of retrieved chunks
            include_sources: Whether to include source information

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            source = chunk.get("filename", "Unknown")
            similarity = chunk.get("similarity_score", 0)

            part = f"[Document {i}] (Source: {source}, Relevance: {similarity:.2%})\n{content}"
            context_parts.append(part)

        return "\n\n".join(context_parts)

    @classmethod
    def assemble_sources(cls, chunks: List[Dict]) -> str:
        """
        Assemble source information from chunks.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted sources string
        """
        sources = set()

        for chunk in chunks:
            source = chunk.get("filename", "Unknown")
            sources.add(source)

        if not sources:
            return "No sources available"

        return "\n".join(f"- {source}" for source in sorted(sources))


if __name__ == "__main__":
    # Example usage
    print("=== English Template ===")
    template = PromptTemplates.get_retrieval_template("en")
    prompt = template.format(
        context="Wheat pests can be controlled using pesticides.",
        query="How to control wheat pests?"
    )
    print(prompt)

    print("\n=== Hindi Template ===")
    template = PromptTemplates.get_retrieval_template("hi")
    prompt = template.format(
        context="गेहूँ के कीटों को कीटनाशकों का उपयोग करके नियंत्रित किया जा सकता है।",
        query="गेहूँ में कीटों का नियंत्रण कैसे करें?"
    )
    print(prompt)

    print("\n=== Punjabi Template ===")
    template = PromptTemplates.get_retrieval_template("pa")
    print(template.format(
        context="ਗੇਹੂੰ ਦੇ ਕੀਟਾਂ ਨੂੰ ਕੀਟਨਾਸ਼ਕਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਕੰਟਰੋਲ ਕੀਤਾ ਜਾ ਸਕਦਾ ਹੈ।",
        query="ਗੇਹੂੰ ਵਿੱਚ ਕੀਟਾਂ ਦਾ ਨਿਯੰਤਰਣ ਕਿਵੇਂ ਕਰਨਾ ਹੈ?"
    ))
