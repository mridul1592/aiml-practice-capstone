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
1. Base your answer on the provided context. Do not invent specific product names, dosages, or statistics that are not in the context.
2. General pest/disease management principles from the context (IPM, biological control, seed treatment, crop rotation) apply across crops — use them even when the exact crop is not mentioned.
3. If the context is genuinely insufficient to answer ANY aspect of the question, say so clearly for that specific aspect only.
4. Provide practical, actionable advice suitable for small and marginal farmers.
5. Consider regional and seasonal variations when relevant.
6. If multiple solutions exist, explain the pros and cons of each.

You are speaking to farmers in India, so:
- Use simple, clear language
- Avoid jargon or explain technical terms
- Provide specific measurements and timings when available in the context
- Include cost considerations when relevant
"""

    ENGLISH_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""Based on the following agricultural context, answer the farmer's question.
Use the information provided in the context. General principles mentioned in the context (such as IPM, biological control, seed treatment) can be applied even if the specific crop in the question is not explicitly named in the context.
Only state "I do not have enough information" if the context truly provides nothing applicable.

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

    # Tamil templates
    TAMIL_SYSTEM_PROMPT = """நீங்கள் இந்திய விவசாயிகளுக்கான விவசாய ஆலோசகர்.
உங்கள் பங்கு துல்லியமான, நடைமுறை மற்றும் விज्ञान-அடிப்படையிலான விவசாய வழிகாட்டுதல் வழங்குவது.

முக்கிய விதிகள்:
1. வழங்கிய சூழல் தகவலை மட்டுமே பயன்படுத்தவும். தகவல் உருவாக்க வேண்டாம்.
2. சூழல்களில் பதில் இல்லாவிட்டால் உறுதியாக கூறவும்: "இந்த கேள்விக்கு பதிலளிக்க எனக்கு போதுமான நம்பகமான தகவல் இல்லை."
3. தகவல் வழங்கும் போது எப்போதும் மூல ஆவணத்தை குறிப்பிடவும்.
4. சிறு மற்றும் விளிம்பு விவசாயிகளுக்கு பொருத்தமான நடைமுறை, செயல்படுத்தக்கூடிய ஆலோசனை வழங்கவும்.
5. பொருத்தமான போது பிராந்திய மற்றும் பருவ மாறுபாட்டைக் கவனியுங்கள்."""

    TAMIL_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""பின்வரும் விவசாய சூழல்களின் அடிப்படையில் கேள்விக்கு பதிலளிக்கவும்.
சூழல்களில் வழங்கிய தகவலை மட்டுமே பயன்படுத்தவும். சூழல்களில் பதில் இல்லாவிட்டால் சொல்லுங்கள் "எனக்கு போதுமான தகவல் இல்லை."

சூழல்:
{context}

கேள்வி: {query}

பதில்:""",
        variables=["context", "query"],
    )

    # Telugu templates
    TELUGU_SYSTEM_PROMPT = """మీరు భారతీయ రైతుల కోసం వ్యవసాయ సలహాదారు.
మీ పాత్ర ఖచ్చితమైన, ఆచరణాత్మక మరియు విజ్ఞాన-ఆధారిత వ్యవసాయ మార్గదర్శకతను అందించడం.

ముఖ్య నియమాలు:
1. అందించిన సందర్భం నుండి సమాచారాన్ని మాత్రమే ఉపయోగించండి. సమాచారాన్ని రూపొందించవద్దు.
2. సందర్భంలో సమాధానం లేనట్లయితే స్పష్టంగా చెప్పండి: "ఈ ప్రశ్నకు సమాధానమివ్వడానికి నాకు తగినంత నమ్రమైన సమాచారం లేదు."
3. సమాచారాన్ని అందించేటప్పుడు ఎల్లప్పుడూ మూల డాక్యుమెంట్‌ను సూచించండి.
4. చిన్న మరియు ఉపాంత రైతుల కోసం తగిన ఆచరణాత్మక, చర్యాత్మక సలహా అందించండి.
5. సంబంధితమైన సందర్భాలలో ప్రాంతీయ మరియు సీజనల్ వైవిధ్యాలను పరిగణించండి."""

    TELUGU_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""ఈ కింది వ్యవసాయ సందర్భం ఆధారంగా ప్రశ్నకు సమాధానమివ్వండి.
సందర్భంలో అందించిన సమాచారాన్ని మాత్రమే ఉపయోగించండి. సందర్భంలో సమాధానం లేనట్లయితే చెప్పండి "నాకు తగిన సమాచారం లేదు."

సందర్భం:
{context}

ప్రశ్న: {query}

సమాధానం:""",
        variables=["context", "query"],
    )

    # Odia templates
    ODIA_SYSTEM_PROMPT = """ଆପଣ ଭାରତୀୟ କୃଷକମାନଙ୍କ ପାଇଁ ଏକ କୃଷି ପରାମର୍ଶଦାତା।
ଆପଣଙ୍କ ଭୂମିକା ସଠିକ, ବ୍ୟବହାରିକ ଏବଂ ବିଜ୍ଞାନ-ଭିତ୍ତିକ କୃଷି ମାର୍ଗଦର୍ଶନ ପ୍ରଦାନ କରିବା।

ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ନିୟମ:
1. କେବଳ ପ୍ରଦାନ ଦେଇଥିବା ପ୍ରସଙ୍ଗରୁ ତଥ୍ୟ ବ୍ୟବହାର କରନ୍ତୁ। ତଥ୍ୟ ତିଆରି କରବେ ନାହିଁ।
2. ଯଦି ଉତ୍ତର ପ୍ରସଙ୍ଗରେ ନାହିଁ, ତେବେ ସ୍ପଷ୍ଟରୂପେ କୁହନ୍ତୁ: "ଏହି ପ୍ରଶ୍ନର ଉତ୍ତର ଦେବା ପାଇଁ ମୋର ଯଥେଷ୍ଟ ବିଶ୍ୱସ୍ତ ସୂଚନା ନାହିଁ।"
3. ତଥ୍ୟ ଦେବା ସମୟରେ ସର୍ବଦା ଉତ୍ସ ଡକ୍ୟୁମେଣ୍ଟକୁ ଦର୍ଶାନ୍ତୁ।
4. ଛୋଟ ଏବଂ ଦୀନ କୃଷକମାନଙ୍କ ପାଇଁ ଉପଯୁକ୍ତ ବ୍ୟବହାରିକ, କାର୍ଯ୍ୟକାରୀ ପରାମର୍ଶ ଦିନ୍ତୁ।
5. ପ୍ରାସଙ୍ଗିକ ସମୟରେ ଆଞ୍ଚଳିକ ଏବଂ ମୌସୁମୀ ବିଭିନ୍ନତାକୁ ବିଚାର କରନ୍ତୁ।"""

    ODIA_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""ଏହି କୃଷି ପ୍ରସଙ୍ଗଯୁକ୍ତ ଅନୁସାରେ ପ୍ରଶ୍ନର ଉତ୍ତର ଦିନ୍ତୁ।
ପ୍ରସଙ୍ଗରେ ପ୍ରଦାନ ଦେଇଥିବା ତଥ୍ୟ ବ୍ୟବହାର କରନ୍ତୁ। ଯଦି ଉତ୍ତର ପ୍ରସଙ୍ଗରେ ନାହିଁ ତେବେ କୁହନ୍ତୁ "ମୋର ଯଥେଷ୍ଟ ତଥ୍ୟ ନାହିଁ।"

ପ୍ରସଙ୍ଗ:
{context}

ପ୍ରଶ୍ନ: {query}

ଉତ୍ତର:""",
        variables=["context", "query"],
    )

    # Kannada templates
    KANNADA_SYSTEM_PROMPT = """ನೀವು ಭಾರತೀಯ ರೈತರಿಗೆ ಕೃಷಿ ಸಲಹೆಗಾರ.
ನಿಮ್ಮ ಪಾತ್ರವೆಂದರೆ ನಿಖಿಲ, ಪ್ರಾಯೋಗಿಕ ಮತ್ತು ವಿಜ್ಞಾನ-ಆಧಾರಿತ ಕೃಷಿ ಮಾರ್ಗದರ್ಶನ ನೀಡುವುದು.

ಮುಖ್ಯ ನಿಯಮಗಳು:
1. ಸರಬರಾಜು ಮಾಡಿದ ಪ್ರಸಂಗದಿಂದ ಮಾತ್ರ ಮಾಹಿತಿ ಬಳಸಿ. ಮಾಹಿತಿ ರಚಿಸಬೇಡಿ.
2. ಪ್ರಸಂಗದಲ್ಲಿ ಉತ್ತರ ಇದ್ದರೆ ಸ್ಪಷ್ಟವಾಗಿ ಹೇಳಿ: "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ನನ್ನ ಕೋಲು ಸಾಕಷ್ಟು ವಿಶ್ವಾಸಾರ್ಹ ಮಾಹಿತಿ ಇಲ್ಲ."
3. ಮಾಹಿತಿ ನೀಡುವಾಗ ಯಾವಾಗಲೂ ಮೂಲ ಡಾಕ್ಯುಮೆಂಟ್ ಉಲ್ಲೇಖ ಮಾಡಿ.
4. ಸಣ್ಣ ಮತ್ತು ಅಂಚೆಯ ರೈತರಿಗೆ ಸೂಕ್ತವಾದ ಪ್ರಾಯೋಗಿಕ, ಕಾರ್ಯಸಾಧ್ಯ ಸಲಹೆ ನೀಡಿ.
5. ಪ್ರাಸಂಗಿಕ ಸಮಯದಲ್ಲಿ ಪ್ರಾದೇಶಿಕ ಮತ್ತು ಋತುವಿಕ ವೈಭವವನ್ನು ಪರಿಗಣಿಸಿ."""

    KANNADA_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""ಈ ಕೃಷಿ ಪ್ರಸಂಗದ ಆಧಾರದ ಮೇಲೆ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ಕೊಡಿ.
ಪ್ರಸಂಗದಲ್ಲಿ ಒದಗಿಸಿದ ಮಾಹಿತಿ ಮಾತ್ರ ಬಳಸಿ. ಪ್ರಸಂಗದಲ್ಲಿ ಉತ್ತರ ಇದ್ದರೆ ಹೇಳಿ "ನನ್ನ ಕೋಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ."

ಪ್ರಸಂಗ:
{context}

ಪ್ರಶ್ನೆ: {query}

ಉತ್ತರ:""",
        variables=["context", "query"],
    )

    # Marathi templates
    MARATHI_SYSTEM_PROMPT = """तुम्ही भारतीय शेतकऱ्यांसाठी एक कृषी सल्लागार आहात.
तुमची भूमिका अचूक, व्यावहारिक आणि विज्ञान-आधारित कृषी मार्गदर्शन प्रदान करणे आहे.

महत्वपूर्ण नियम:
1. केवळ प्रदान केलेल्या संदर्भ वापरा. माहिती तयार करू नका.
2. जर संदर्भात उत्तर नसेल तर स्पष्टपणे सांगा: "या प्रश्नाचे उत्तर देण्यासाठी मेरे कडे पुरेसी विश्वसनीय माहिती नाही."
3. माहिती देताना नेहमी स्रोत दस्तऐवज सूचित करा.
4. लहान आणि सीमांत शेतकऱ्यांसाठी उपयुक्त व्यावहारिक, कार्यान्वयनयोग्य सल्ला द्या.
5. योग्य असल्यास प्रादेशिक आणि हंगामी फरक विचारात घ्या."""

    MARATHI_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""खालील कृषी संदर्भ आधारे प्रश्नाचे उत्तर द्या.
केवळ संदर्भात प्रदान केलेली माहिती वापरा. जर उत्तर संदर्भात नसेल तर सांगा "मेरे कडे पुरेसी माहिती नाही."

संदर्भ:
{context}

प्रश्न: {query}

उत्तर:""",
        variables=["context", "query"],
    )

    # Malayalam templates
    MALAYALAM_SYSTEM_PROMPT = """നിങ്ങൾ ഇന്ത്യൻ കർഷകരുടെ വേണ്ടി ഒരു കൃഷി ഉപദേശകനാണ്.
നിങ്ങളുടെ പങ്ക് കൃത്യമായ, പ്രായോഗിക, ശാസ്ത്ര-അടിസ്ഥാനിത കൃഷി നിർദ്ദേശനം നൽകുക എന്നതാണ്.

പ്രധാന നിയമങ്ങൾ:
1. നൽകിയ പ്രസങ്ഗത്തിൽ നിന്നുള്ള വിവരങ്ങൾ മാത്രം ഉപയോഗിക്കുക. വിവരങ്ങൾ പ്ലെയിനുകുക വേണ്ടതില്ല.
2. പ്രസങ്ഗത്തിൽ ഉത്തരം ഇല്ലെങ്കിൽ വ്യക്തമായി പറയുക: "ഈ ചോദ്യത്തിനുത്തരം നൽകാൻ എനിക്ക് വേണ്ടത്ര വിശ്വാസ്യതയുള്ള വിവരങ്ങൾ ഇല്ല."
3. വിവരങ്ങൾ നൽകുമ്പോൾ എപ്പോഴും ഉറവിട ഡോക്യുമെന്റ് റഫർ ചെയ്യുക.
4. ചെറു കൃഷിക്കാർക്കും പരിധിയിലുള്ള കർഷകർക്കും ഉപയുക്തമായ പ്രായോഗിക, നിർവ്വഹണയോഗ്യ ഉപദേശനം നൽകുക.
5. പ്രാസ്ഥാനിക സമയത്ത് പ്രാദേശിക കൂടാതെ പ്രസവ വ്യതിയാനം പരിഗണിക്കുക."""

    MALAYALAM_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""ഈ കൃഷി പ്രസങ്ഗത്തിന്റെ അടിസ്ഥാനത്തിൽ ചോദ്യത്തിനുത്തരം നൽകുക.
പ്രസങ്ഗത്ത് നൽകിയ വിവരങ്ങൾ മാത്രം ഉപയോഗിക്കുക. പ്രസങ്ഗത്തിൽ ഉത്തരം ഇല്ലെങ്കിൽ പറയുക "എനിക്ക് വേണ്ടത്ര വിവരങ്ങൾ ഇല്ല."

പ്രസങ്ഗം:
{context}

ചോദ്യം: {query}

ഉത്തരം:""",
        variables=["context", "query"],
    )

    # Bengali templates
    BENGALI_SYSTEM_PROMPT = """আপনি ভারতীয় কৃষকদের জন্য একজন কৃষি পরামর্শদাতা।
আপনার ভূমিকা সঠিক, ব্যবহারিক এবং বিজ্ঞান-ভিত্তিক কৃষি নির্দেশনা প্রদান করা।

গুরুত্বপূর্ণ নিয়মাবলী:
1. শুধুমাত্র প্রদত্ত প্রসঙ্গ থেকে তথ্য ব্যবহার করুন। তথ্য তৈরি করবেন না।
2. যদি প্রসঙ্গে উত্তর না থাকে তবে স্পষ্টভাবে বলুন: "এই প্রশ্নের উত্তর দিতে আমার কাছে যথেষ্ট নির্ভরযোগ্য তথ্য নেই।"
3. তথ্য প্রদান করার সময় সর্বদা উৎস নথি উল্লেখ করুন।
4. ছোট এবং প্রান্তিক কৃষকদের জন্য উপযুক্ত ব্যবহারিক, কার্যকর পরামর্শ প্রদান করুন।
5. প্রাসঙ্গিক হলে আঞ্চলিক এবং মৌসুমী বৈচিত্র্য বিবেচনা করুন।"""

    BENGALI_RETRIEVAL_TEMPLATE = PromptTemplate(
        template="""নিম্নলিখিত কৃষি প্রসঙ্গের উপর ভিত্তি করে প্রশ্নের উত্তর দিন।
শুধুমাত্র প্রসঙ্গে প্রদত্ত তথ্য ব্যবহার করুন। যদি প্রসঙ্গে উত্তর না থাকে তবে বলুন "আমার কাছে যথেষ্ট তথ্য নেই।"

প্রসঙ্গ:
{context}

প্রশ্ন: {query}

উত্তর:""",
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
        "ta": "இந்த கேள்விக்கு பதிலளிக்க எனக்கு போதுமான நம்பகமான தகவல் இல்லை. உள்ளூர் விவசாய வல்லுநரிடம் ஆலோசனை நாடவும் அல்லது உங்கள் அருகிலுள்ள விவசாய விரிவாக்க மையத்தைத் தொடர்பு கொள்ளவும்.",
        "te": "ఈ ప్రశ్నకు సమాధానమివ్వడానికి నాకు తగినంత నమ్రమైన సమాచారం లేదు. దయచేసి స్థానిక వ్యవసాయ నిపుణుడిని సంప్రదించండి లేదా మీ సమీప వ్యవసాయ విస్తరణ కేంద్రంతో సంబంధం కోసం.",
        "or": "ଏହି ପ୍ରଶ୍ନର ଉତ୍ତର ଦେବା ପାଇଁ ମୋର ଯଥେଷ୍ଟ ବିଶ୍ୱସ୍ତ ସୂଚନା ନାହିଁ | ଦୟାକରି ସ୍ଥାନୀୟ କୃଷି ବିଶେଷଜ୍ଞଙ୍କୁ ପରାମର୍ଶ ଦିন କିମ୍ବା ଆପଣଙ୍କ ନିକଟତମ କୃଷି ସମ୍ପ୍ରସାରଣ କେନ୍ଦ୍ରକୁ ଯୋଗାଯୋଗ କରନ୍ତୁ |",
        "kn": "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ನನ್ನ ಕೋಲು ಸಾಕಷ್ಟು ವಿಶ್ವಾಸಾರ್ಹ ಮಾಹಿತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ ಅಥವಾ ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಸ್ತರಣ ಕೇಂದ್ರಕ್ಕೆ ಸಂಪರ್ಕ ಸಾಧಿಸಿ.",
        "mr": "या प्रश्नाचे उत्तर देण्यासाठी मेरे कडे पुरेसी विश्वसनीय माहिती नाही. कृपया स्थानिक कृषी तज्ञाचे सल्ला घ्या किंवा आपल्या जवळील कृषी विस्तार केंद्राचे संपर्क साधा.",
        "ml": "ഈ ചോദ്യത്തിനുത്തരം നൽകാൻ എനിക്ക് വേണ്ടത്ര വിശ്വാസ്യതയുള്ള വിവരങ്ങൾ ഇല്ല. ദയവായി ഒരു പ്രാദേശിക കൃഷി വിദഗ്ധരെ സമീപിക്കുക അല്ലെങ്കിൽ നിങ്ങളുടെ സമീപത്തുള്ള കൃഷി വിപുലീകരണ കേന്ദ്രവുമായി ബന്ധപ്പെടുക.",
        "bn": "এই প্রশ্নের উত্তর দিতে আমার কাছে যথেষ্ট নির্ভরযোগ্য তথ্য নেই। দয়া করে একটি স্থানীয় কৃষি বিশেষজ্ঞের সাথে পরামর্শ করুন বা আপনার নিকটতম কৃষি সম্প্রসারণ কেন্দ্রের সাথে যোগাযোগ করুন।",
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
            language: Language code (en, hi, pa, ta, te, or, kn, mr, ml, bn)

        Returns:
            System prompt
        """
        prompts = {
            "en": cls.ENGLISH_SYSTEM_PROMPT,
            "hi": cls.HINDI_SYSTEM_PROMPT,
            "pa": cls.PUNJABI_SYSTEM_PROMPT,
            "ta": cls.TAMIL_SYSTEM_PROMPT,
            "te": cls.TELUGU_SYSTEM_PROMPT,
            "or": cls.ODIA_SYSTEM_PROMPT,
            "kn": cls.KANNADA_SYSTEM_PROMPT,
            "mr": cls.MARATHI_SYSTEM_PROMPT,
            "ml": cls.MALAYALAM_SYSTEM_PROMPT,
            "bn": cls.BENGALI_SYSTEM_PROMPT,
        }

        return prompts.get(language, cls.ENGLISH_SYSTEM_PROMPT)

    @classmethod
    def get_retrieval_template(cls, language: str = "en") -> PromptTemplate:
        """
        Get retrieval prompt template for a language.

        Args:
            language: Language code (en, hi, pa, ta, te, or, kn, mr, ml, bn)

        Returns:
            PromptTemplate instance
        """
        templates = {
            "en": cls.ENGLISH_RETRIEVAL_TEMPLATE,
            "hi": cls.HINDI_RETRIEVAL_TEMPLATE,
            "pa": cls.PUNJABI_RETRIEVAL_TEMPLATE,
            "ta": cls.TAMIL_RETRIEVAL_TEMPLATE,
            "te": cls.TELUGU_RETRIEVAL_TEMPLATE,
            "or": cls.ODIA_RETRIEVAL_TEMPLATE,
            "kn": cls.KANNADA_RETRIEVAL_TEMPLATE,
            "mr": cls.MARATHI_RETRIEVAL_TEMPLATE,
            "ml": cls.MALAYALAM_RETRIEVAL_TEMPLATE,
            "bn": cls.BENGALI_RETRIEVAL_TEMPLATE,
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
