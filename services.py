import os
from google import genai
from google.genai import types
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import re

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1beta")
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def process_file_to_chunks(file_content: bytes):
    try:
        text = file_content.decode("utf-8")
        import re

        # ✅ Split by Q&A, Scenario, and Step patterns
        chunks = re.split(
            r'(?=Q\d+[\:\.])|(?=Scenario\s+\d+[\:\.])|(?=Step\s+\d+[\:\.])',
            text
        )

        # Clean and filter
        chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

        return chunks
    except Exception as e:
        print(f"Chunking Error: {e}")
        raise e
#def process_file_to_chunks(file_content: bytes):
#    try:
#        text = file_content.decode("utf-8")
#        splitter = RecursiveCharacterTextSplitter(
#            chunk_size=1000,
#            chunk_overlap=250,
#        )
#        return splitter.split_text(text)
#    except Exception as e:
#        print(f"Chunking Error: {e}")
#        raise e

async def get_embeddings(text: str, task_type: str = "RETRIEVAL_DOCUMENT"):
    """Embed text using Gemini gemini-embedding-001 at 768 dims"""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        raise e
# ========================
# GREETING DETECTION
# ========================
#I am using this previoulsy i was using api for handling that kind of greeting which was just increasing the latency.
def is_greeting(query: str) -> bool:
    query_clean = query.lower().strip()
    query_clean = re.sub(r'[^\w\s]', '', query_clean)
    query_words = query_clean.split()

    # Single word greetings — exact match only
    single_word_greetings = [
        "hi", "hello", "hey", "helo", "hii", "hiii",
        "salam", "sup", "greetings", "aoa","howdy"
    ]

    # Multi word greetings — exact phrase match
    multi_word_greetings = [
        "good morning", "good evening", "good afternoon", "good night",
        "how are you", "how r you", "how are u",
        "whats up", "what's up",
        "who are you", "what are you", "introduce yourself",
        "assalam", "assalamualaikum", "assalam o alaikum",
        "assalam-o-alaikum", "assalamoalaikum", 
        "walaikum", "walaikumsalam", "walaikum salam",
        "kya haal hai", "kaise ho", "aoa"
    ]

    if len(query_words) <= 4:
        if query_words[0] in single_word_greetings:
            return True
    for phrase in multi_word_greetings:
        if query_clean == phrase:
            return True

    return False
def get_greeting_response() -> str:
    return """Hello! 👋 Welcome to CopayHealth Payment Assistant!

I'm here to help you with:
- 💳 Payment plans and installments
- 🤝 Family contributions
- 📅 Payment scheduling
- 🧾 Receipts and documentation
- ⚠️ Payment troubleshooting
- 🔐 Account security

Feel free to ask me anything — I'm happy to help! 😊"""



def is_unclear(query: str) -> bool:
    query_clean = query.lower().strip()
    # If query is less than 3 characters
    if len(query_clean) < 3:
        return True
    # If query is just random letters
    import re
    if not re.search(r'[aeiou]', query_clean):
        return True
    return False

async def rerank_chunks(query: str, chunks: list):
    """
    Reranking logic to improve precision. 
    Standard vector search (Recall) finds candidates; 
    LLM-based Reranking (Precision) selects the absolute best match.
    """
    if not chunks or len(chunks) <= 1:
        return chunks

    # Prepare candidates for the reranker model
    choices = "\n".join([f"[{i}] {c.payload['text']}" for i, c in enumerate(chunks)])
    
    prompt = f"""
    You are an expert Re-ranker. Rank these context chunks based on their relevance to the user query.
    User Query: {query}
    
    Chunks:
    {choices}
    
    Return ONLY the index number (e.g., '0') of the most relevant chunk. 
    If none are relevant, return '0'.
    """
    
    try:
        # Using a faster model for reranking to minimize latency
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        best_index = int(re.search(r'\d+', response.choices[0].message.content).group())
        return [chunks[best_index]]
    except Exception as e:
        # Fallback to the top result from vector search if reranking fails
        return [chunks[0]]