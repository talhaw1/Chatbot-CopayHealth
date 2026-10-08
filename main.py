import uuid
import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from services import get_embeddings, process_file_to_chunks,groq_client,is_greeting,get_greeting_response,is_unclear,rerank_chunks
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Gemini client
client_gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1beta")
)

# Qdrant client
client_qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
COLLECTION_NAME = "file"

# ========================
# LIFESPAN
# ========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    app.state.q_client = AsyncQdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    try:
        cols = await app.state.q_client.get_collections()
        if not any(c.name == COLLECTION_NAME for c in cols.collections):
            await app.state.q_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        logger.info("Connected to Qdrant Cloud")
    except Exception as e:
        logger.error(f"Startup Error: {e}")

    yield

    # SHUTDOWN
    await app.state.q_client.close()


app = FastAPI(title="RAG Bot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bot.farmacyapp.com",   
        "http://localhost:3000",         
        "http://localhost:5403",        
        "http://127.0.0.1:5403",        
        "http://127.0.0.1:8000",        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ========================
# REQUEST MODELS
# ========================
class ChunksRequest(BaseModel):
    chunks: list[str]
    filename: str


# ========================
# ENDPOINT 1: Load txt + Chunk
# ========================
@app.post("/load-and-chunk")
async def load_and_chunk(file: UploadFile = File(...)):
    """
    Upload a PDF → extract text → return chunks
    Send these chunks to /embed-and-store
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only txt files are supported.")

    try:
        content = await file.read()
        chunks = process_file_to_chunks(content)

        logger.info(f"TXT loaded and chunked: {file.filename} → {len(chunks)} chunks")

        return {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "chunks": chunks          # ← send these to /embed-and-store
        }

    except Exception as e:
        logger.error(f"Load and Chunk Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")


# ========================
# ENDPOINT 2: Embed + Store in Qdrant
# ========================
@app.post("/embed-and-store")
async def embed_and_store(request: ChunksRequest):
    """
    Receive chunks from /load-and-chunk
    → embed each chunk
    → store in Qdrant
    """
    if not request.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided.")

    try:
        
        points = []
        for i,chunk in enumerate(request.chunks):
            vector = await get_embeddings(chunk, task_type="RETRIEVAL_DOCUMENT")
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "filename": request.filename
                }
            ))
            if (i+1)%90==0:
                logger.info(f"Rate limit pause at chunk {i+1}...Waiting 65 seconds ")
                await asyncio.sleep(65)
            else:
                await asyncio.sleep(0.7)    

        await app.state.q_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        logger.info(f" Stored {len(points)} chunks from {request.filename}")

        return {
            "message": f" Successfully embedded and stored {len(points)} chunks",
            "filename": request.filename,
            "total_stored": len(points)
        }

    except Exception as e:
        logger.error(f"Embed and Store Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to embed and store: {e}")

# ========================
# ENDPOINT 3: Ask a Question
# ========================
@app.get("/ask")
async def ask(query: str):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if is_greeting(query):
        return {
            "answer": get_greeting_response(),
            "sources": []
        }
    if is_unclear(query):
        return {
        "answer": "I'm sorry, I didn't quite understand that. Could you please rephrase your question? 😊",
        "sources": []
        }    
    try:
        # 1. Embed query
        response = client_gemini.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            ),
        )
        query_vector = response.embeddings[0].values
        # 2. Search Qdrant
        results = client_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5
        )
        candidate_points = [r for r in results.points if r.score > 0.45]

        if not candidate_points:
            return {
                "answer": "I'm sorry, I don't have enough information right now. Please contact our support team — we're always happy to help! 😊",
                "sources": []
            }
        
        #points = rerank_chunks(query,points)    
        final_points = await rerank_chunks(query, candidate_points)

        # 3. Build context
        context = final_points[0].payload["text"] 
        sources = [final_points[0].payload.get("filename", "unknown")]
        # 4. Generate answer
        chat_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": """You are a friendly and professional customer support assistant for CopayHealth Payment System. Your job is to help patients understand their payment options with warmth and care.
TONE RULES:
- Always be warm, polite and empathetic
- ALWAYS start every response with a warm phrase — 
  never start directly with the answer
- Rotate between these warm openers naturally:
  * "Great question!"
  * "I'd be happy to help!"
  * "Of course!"
  * "Absolutely!"
  * "Sure thing!"
  * "Happy to assist!"
  * "I can help with that!"
  * "Certainly!"
  * "No problem at all!"
- For troubleshooting and 'What if' questions start with:
  * "No worries!"
  * "Don't worry!"
  * "I've got you covered!"
  * "Let me help you with that!"
- Understand that customers may be stressed about payments — be reassuring
- Always end every response with: 'Is there anything else I can help you with? 😊'

STRICT RULES:
1. Answer ONLY using the information from the context provided
2. NEVER add information not present in the context
3. Only start with 'Yes,' or 'No,' if the question is a genuine yes/no question
4. For 'What if' and 'What happens' questions - start with a direct explanation
5. Never repeat the question in your answer - start directly with the answer
6. If answer is not in context, say: 'I'm sorry, I don't have enough information right now. Please contact our support team - we're always happy to help! 😊'

ANSWER FORMAT:
- Yes/No questions: Start with 'Yes,' or 'No,' followed by full details on same line
- What if/What happens questions: Start with direct explanation then bullet points
- Step by step questions: Use numbered steps
- Feature/option questions: Use bullet points
- Troubleshooting questions: Start with action statement then bullet points
- Simple fact questions: One clear sentence with all details

QUALITY RULES:
- Return COMPLETE answer with ALL bullet points - never cut short
- ALWAYS use bullet points when context has bullet points - never convert to sentences
- Keep answer CONCISE - do not over explain
- Do not add extra commentary after bullet points"""
                    #"content": "You are a helpful payment assistant. Answer the question using ONLY the information from the context provided. Always return the complete answer with all bullet points- never cut it short. If the context has a list, return the FULL list. Do not add any information that is not in the context. If the answer is not in the context, say 'I don't have enough information.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )

        answer = chat_response.choices[0].message.content  

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Ask Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")