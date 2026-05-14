# Multi-Agent RAG Chat Engine

A lightweight prototype for the intern assignment in `Chat_Engine_Intern_Usecase.docx`.

The app accepts up to 3 PDF/image documents, validates page and format limits, extracts text, chunks and indexes the content, then answers questions with source-grounded retrieval.

## Install

Create and activate a virtual environment, then install the project packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You also need the Ollama desktop/server installed separately, plus a local model:

```powershell
ollama pull mistral
```

## Run

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## What It Implements

- **Agent 1: Ingestion & Validation** validates file type, file size, PDF page count, and image readability.
- **Agent 2: Text Extraction** extracts digital PDF text with PyMuPDF and uses EasyOCR for image/scanned content.
- **Agent 3: Chunking & Embedding** chunks text by paragraph/sentence boundaries, embeds with `sentence-transformers/all-MiniLM-L6-v2`, and stores vectors in ChromaDB.
- **Agent 4: Retrieval & Conversation** retrieves top matching chunks, keeps recent chat history, builds a transparent prompt, and asks Ollama/Mistral for a grounded answer with document/page citations.

## Notes

- The prototype intentionally avoids inventing facts. If retrieved context is weak, it returns a "not found in documents" response.
- EasyOCR and sentence-transformers may download model weights the first time they run.
- Ollama must be running locally for generated answers. Retrieval and prompt construction are still shown in the UI if Ollama is unavailable.
