CopayHealth AI: Advanced RAG Payment Assistant
An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to assist patients with medical billing, payment plans, and financial troubleshooting. This project implements Advanced RAG patterns to ensure high precision and reliability in a healthcare-finance context.

🚀 Key Technical Features
1. Two-Stage Retrieval (Recall & Precision)
Unlike basic RAG systems that rely solely on vector similarity, this architecture uses a two-stage process:

Stage 1 (Recall): Semantic search using Gemini-1.5-Embeddings and Qdrant Vector DB to find the top 5 relevant candidates.

Stage 2 (Precision): A Llama-3.1-8B Reranker cross-references candidates against the query to select the absolute best context, drastically reducing hallucinations.

2. Pattern-Based Semantic Chunking
Recognizing that medical billing docs are often structured as FAQs and Scenarios, I implemented a custom RegEx-driven Chunking Engine:

Preserves the integrity of Q&A pairs.

Identifies and isolates Step-by-Step procedures and Scenario-based troubleshooting.

Prevents context fragmentation that occurs with standard recursive character splitting.

3. Production-Ready Backend
Asynchronous Orchestration: Built with FastAPI and asyncio to handle concurrent embedding and retrieval tasks.

Resilient API Logic: Built-in rate-limit management and intelligent pauses for high-volume data ingestion.

Vector Cloud Integration: Uses Qdrant Cloud with a Cosine Distance metric for high-speed similarity matching.

🛠 Tech Stack
LLM: Llama-3.3-70B (via Groq)

Embeddings: Gemini-Embedding-001

Orchestration: Python, FastAPI

Vector Database: Qdrant

Reranker: Llama-3.1-8B

📂 Project Structure
Plaintext
├── main.py              # FastAPI endpoints & RAG Orchestration
├── services.py          # Embedding, Chunking, and Reranking logic
├── requirements.txt     # Dependency management
├── .env                 # Environment variables (Protected)
└── .gitignore           # Security rules for sensitive data
⚙️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/talhaw1/Chatbot-CopayHealth
Install dependencies:

Bash
pip install -r requirements.txt
Configure Environment Variables:
Create a .env file and add your keys:

Code snippet
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
QDRANT_URL=your_url
QDRANT_API_KEY=your_key
Run the API:

Bash
uvicorn main:app --reload
