# Multi-Agent RAG Chat Engine

Production-oriented prototype for the Allianz intern chat-engine assignment. The app lets a user upload a small set of PDFs/images, indexes extracted text with RAG, and answers questions through Groq while showing retrieval sources.

## Tech Stack

| Layer | Choice | Why |
| --- | --- | --- |
| UI | Streamlit | Fast upload/chat interface with minimal frontend code. |
| PDF parsing | PyMuPDF | Efficient page-count validation and page-wise text extraction. |
| OCR | EasyOCR | Handles image files and scanned PDF pages without requiring a separate Tesseract binary. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight semantic embedding model suitable for local MVP RAG. |
| Vector DB | ChromaDB | Local persistent collections with chunk metadata and similarity search. |
| LLM | Groq Chat Completions API | Fast hosted LLM inference using `GROQ_API_KEY`. |
| Orchestration | Plain Python OOP | Keeps agent responsibilities explicit, typed, and testable. |

## Project Structure

```text
.
|-- app.py
|-- agents/
|   |-- ingestion_agent.py
|   |-- extraction_agent.py
|   |-- chunking_embedding_agent.py
|   |-- retrieval_agent.py
|   `-- conversation_agent.py
|-- core/
|   |-- config.py
|   |-- schemas.py
|   |-- ports.py
|   |-- vector_store.py
|   |-- llm_client.py
|   `-- prompts.py
|-- utils/
|   |-- file_utils.py
|   |-- ocr_utils.py
|   |-- text_utils.py
|   `-- transcript_export.py
|-- tests/
|-- requirements.txt
`-- pytest.ini
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set your Groq API key in the same terminal:

```powershell
$env:GROQ_API_KEY="your_groq_key_here"
```

Optional Groq overrides:

```powershell
$env:GROQ_MODEL="llama-3.3-70b-versatile"
$env:GROQ_API_BASE_URL="https://api.groq.com/openai/v1"
$env:GROQ_TIMEOUT_SECONDS="120"
```

If you manually downloaded the embedding model, place it at:

```text
models/all-MiniLM-L6-v2
```

The current config points to that local path. To let Hugging Face download it automatically instead, change `EMBEDDING_MODEL` in `core/config.py` to:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

## Run

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use fake embeddings and a fake LLM client, so they do not require Groq or model downloads.

## Design Decisions

- Retrieval and response generation are split into `RetrievalAgent` and `ConversationAgent`. Each class has one reason to change, which keeps the chat pipeline closer to SOLID principles.
- `ConversationAgent` depends on the `LanguageModelClient` protocol and `RetrievalAgent` depends on the `VectorSearchStore` protocol, not concrete Groq or Chroma classes. This follows dependency inversion and keeps provider coupling low.
- `RetrievalAgent` chooses a retrieval mode before searching:
  - `focused` for direct factual questions.
  - `broad` for summaries, explanations, comparisons, and many-point questions.
  - `exhaustive` for counts, totals, complete lists, and questions containing "all" or similar completeness language.
- `GroqLLMClient` is an adapter around Groq's OpenAI-compatible Chat Completions endpoint. It uses the standard library HTTP client to avoid another runtime dependency.
- Agent handoffs use typed dataclasses in `core/schemas.py`, giving each stage a clear input/output contract.
- Retrieval is grounded by design: if no chunk passes the similarity threshold, the app returns a clear not-found response.
- Groq receives only retrieved context, retrieval mode, recent chat history, and the user query, reducing token use and limiting hallucination risk.
- Conversation export supports TXT and PDF using standard-library code, avoiding a new dependency for this feature.
- Runtime data, uploaded files, local models, `.env`, and `changes.md` are ignored by git.

## Required Environment

- Python 3.11 recommended.
- A valid Groq API key with access to the selected chat model.
- Internet access for Groq API calls.
- Local model files under `models/all-MiniLM-L6-v2`, or Hugging Face access for first-run embedding download.

## Troubleshooting

- `HTTP 403: error code 1010` means the request reached Groq but was blocked before normal API handling. Check VPN/proxy/network restrictions, confirm the key is active, and retry from a clean terminal with `GROQ_API_KEY` set.
- If your account does not have access to the configured model, set `GROQ_MODEL` to a model enabled for your Groq account.
