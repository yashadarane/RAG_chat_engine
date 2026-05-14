# Multi-Agent RAG Chat Engine

A lightweight prototype for the intern assignment in `Chat_Engine_Intern_Usecase.docx`.

The app accepts up to 3 PDF/image documents, validates page and format limits, extracts text, chunks and indexes the content, then answers questions with source-grounded retrieval. It is designed to run locally with the packages already present in this workspace.

## Run

```powershell
streamlit run app.py
```

## What It Implements

- **Agent 1: Ingestion & Validation** validates file type, file size, PDF page count, and image readability.
- **Agent 2: Text Extraction** extracts digital PDF text with `pdfplumber`; image OCR is supported when the Tesseract binary is installed.
- **Agent 3: Chunking & Embedding** chunks text by paragraph/sentence boundaries and indexes chunks with a local TF-IDF vector store.
- **Agent 4: Retrieval & Conversation** retrieves top matching chunks, keeps recent chat history, builds a transparent prompt, and returns an extractive grounded answer with document/page citations.

## Notes

- The prototype intentionally avoids inventing facts. If retrieved context is weak, it returns a "not found in documents" response.
- OCR gracefully degrades when Tesseract is unavailable. Digitally encoded PDFs work without OCR.
- TF-IDF is used as a local embedding fallback so the assignment can be run without downloading external models.
