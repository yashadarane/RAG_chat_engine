# Autonomous Multi-Agent Document-Grounded Chat Engine using RAG and Agentic AI

**Project Title:** Autonomous Multi-Agent Document-Grounded Chat Engine using RAG and Agentic AI  
**Internship Assignment Name:** Chat Engine Assignment - RAG + Agentic AI  

---

## 2. Abstract

This project implements a document-grounded conversational chat engine based on Retrieval-Augmented Generation (RAG) and a multi-agent architecture. The system allows users to upload a limited set of PDF or image documents, validates the uploaded files, extracts page-level text using PDF parsing and OCR, chunks the extracted text, indexes the chunks using semantic embeddings and keyword search, and answers user questions using only retrieved document evidence.

The main goal of the project is to avoid hallucinated or unsupported answers by grounding responses in uploaded source documents. Instead of sending entire documents directly to a large language model (LLM), the system retrieves only relevant document chunks and constructs a controlled prompt with strict grounding, citation, and anti-hallucination rules. The implementation also includes retrieval explainability, source citations, transcript export, LangGraph-based orchestration, and a RAGAS evaluation workflow.

---

## 3. Introduction

Document-grounded chat systems are useful when users need to ask natural-language questions about policies, reports, manuals, technical documentation, scanned documents, or other organization-specific files. A general-purpose LLM may answer from prior knowledge or make unsupported assumptions, which is unsuitable when answers must be faithful to uploaded documents.

Retrieval-Augmented Generation solves this problem by separating document retrieval from answer generation. The system first identifies the relevant pieces of source text, then passes only that context to the LLM. This improves factual grounding, reduces prompt size, and makes the answer traceable to document/page sources.

This project uses a multi-agent design to keep responsibilities separated. Each agent performs a specific role: ingestion, extraction, indexing, retrieval, and conversation. This separation improves testability, maintainability, and production readiness because each stage has clear input/output contracts.

---

## 4. Problem Statement

The problem addressed by this project is the need for a lightweight, intelligent chat engine that can ingest user-provided documents and answer questions strictly from those documents. The system must handle PDFs and image files, extract text reliably, index the content for retrieval, retrieve relevant context for each user query, and generate answers with citations while avoiding hallucination.

The challenge is not only to generate answers, but to ensure that answers are:

- grounded in uploaded documents,
- traceable to document/page sources,
- robust across different query types,
- efficient within token limits,
- explainable through retrieval transparency,
- organized as independently testable agents.

---

## 5. Objectives

The main objectives of the project are:

- To provide a simple Streamlit interface for document upload and chat interaction.
- To validate uploaded documents for supported format, maximum document count, and maximum page count.
- To extract text from digital PDFs using PyMuPDF.
- To apply OCR using EasyOCR for image files and scanned PDF pages.
- To clean, chunk, embed, and index extracted text.
- To store semantic embeddings in ChromaDB.
- To support keyword retrieval using a local BM25-style keyword index.
- To implement hybrid retrieval using semantic search and keyword search.
- To select retrieval strategy based on query type: focused, broad, or exhaustive.
- To select relevant document scope before broad or exhaustive retrieval.
- To use query rewriting for improved retrieval recall.
- To rerank retrieved candidates before passing them to the answer-generation stage.
- To generate grounded answers using Groq Chat Completions.
- To cite document name and page number in answers.
- To orchestrate workflows using LangGraph.
- To evaluate the system using RAGAS and custom retrieval metrics.

---

## 6. Scope of the Project

### Supported Inputs

The system supports the following document types:

| Type | Support |
| --- | --- |
| PDF | Supported |
| PNG | Supported |
| JPG / JPEG | Supported |
| TIFF / TIF | Supported |
| BMP | Supported |

### Constraints Implemented

The implemented constraints are defined in `core/config.py`:

| Constraint | Value |
| --- | --- |
| Maximum documents per session | 3 |
| Maximum pages per PDF | 5 |
| File size threshold | 20 MB |
| Chunk size | 420 words |
| Chunk overlap | 60 words |
| Default top-k | 5 |
| Default minimum similarity | 0.25 |

### Query Scope

The system supports:

- direct factual questions,
- summaries,
- comparisons,
- complete lists,
- counts and totals,
- document-identification questions,
- multi-turn follow-up questions with limited recent history.

### Grounded Response Scope

The conversation agent is instructed to answer only from retrieved context. If sufficient evidence is not present, it returns a clear not-found response instead of inventing an answer.

---

## 7. System Architecture

### High-Level Architecture

```text
User
 |
 | Upload PDFs/images
 v
Streamlit UI
 |
 v
LangGraph Upload/Indexing Graph
 |
 +--> Agent 1: Ingestion and Validation
 |
 +--> Agent 2: Text Extraction / OCR
 |
 +--> Agent 3: Chunking + Embedding + Indexing
          |
          +--> ChromaDB semantic vector store
          |
          +--> Local BM25-style keyword index

User query
 |
 v
LangGraph Chat Graph
 |
 +--> Agent 4A: Retrieval Agent
 |       - query mode selection
 |       - document scope selection
 |       - query rewriting
 |       - semantic retrieval
 |       - keyword retrieval
 |       - deduplication
 |       - reranking
 |
 +--> Agent 4B: Conversation Agent
         - grounded prompt construction
         - Groq LLM call
         - source-cited answer
```

### Data Flow

1. The user uploads documents through Streamlit.
2. Agent 1 validates file type, document count, page count, and readability.
3. Agent 2 extracts page-level text using PyMuPDF or EasyOCR.
4. Agent 3 cleans and chunks text, generates embeddings, and indexes chunks.
5. The user submits a question.
6. Agent 4A determines retrieval mode and document scope, retrieves and reranks evidence.
7. Agent 4B builds a grounded prompt and generates an answer from the retrieved context.
8. The UI displays the answer, citations, retrieval transparency, and export options.
9. The graph records completed steps, agent handoffs, and errors for transparency and debugging.

---

## 8. Agent-Based Design

### Agent 1: Document Ingestion and Validation Agent

**File:** `agents/ingestion_agent.py`

| Aspect | Description |
| --- | --- |
| Responsibility | Validate uploaded documents and produce a structured session-level contract. |
| Input | Uploaded file objects or file paths. |
| Processing | Saves uploads, checks max document count, validates file extension, checks PDF page count, checks image readability. |
| Output | `IngestionOutput` containing `ValidatedDocument` objects. |
| Tools/Libraries | `pathlib`, Pillow, PyMuPDF through utility validation functions. |

The agent rejects unsupported formats and documents exceeding constraints, while preserving issue messages for user-facing reporting.

### Agent 2: Text Extraction Agent

**File:** `agents/extraction_agent.py`

| Aspect | Description |
| --- | --- |
| Responsibility | Extract page-level text from validated PDFs and images. |
| Input | Validated documents from Agent 1. |
| Processing | Uses PyMuPDF for digital PDF extraction; falls back to OCR for scanned PDF pages; uses EasyOCR for image files. |
| Output | `ExtractionOutput` containing extracted page text and warnings. |
| Tools/Libraries | PyMuPDF (`fitz`), EasyOCR, Pillow. |

This agent also reports warnings when OCR fails or a page produces no extractable text.

### Agent 3: Chunking and Indexing Agent

**File:** `agents/chunking_embedding_agent.py`

| Aspect | Description |
| --- | --- |
| Responsibility | Convert extracted text into chunks and index those chunks. |
| Input | Extracted documents with page-level text. |
| Processing | Cleans text, chunks by word count with overlap, assigns chunk ids, creates embeddings, stores chunks in ChromaDB and builds keyword index. |
| Output | `ChromaVectorStore` and `IndexingOutput`. |
| Tools/Libraries | Sentence Transformers, ChromaDB, local BM25-style keyword index. |

Each chunk stores:

- `chunk_id`,
- `doc_id`,
- `filename`,
- `page_number`,
- `text`.

### Agent 4A: Retrieval Agent

**File:** `agents/retrieval_agent.py`

| Aspect | Description |
| --- | --- |
| Responsibility | Retrieve the best document evidence for a query. |
| Input | User query and indexed vector store. |
| Processing | Selects retrieval mode, selects document scope, rewrites query, performs dense and keyword search, deduplicates candidates, reranks results. |
| Output | `RetrievalOutput` containing mode, scope, selected documents, matches, and sources. |
| Tools/Libraries | ChromaDB, local BM25-style search, Groq-backed query rewriting, optional Sentence Transformers cross-encoder reranker. |

Agent 4A separates two important decisions:

```text
Retrieval mode = how much evidence is needed
Document scope = which documents are eligible
```

This prevents exhaustive queries from blindly retrieving every uploaded document.

### Agent 4B: Conversation Agent

**File:** `agents/conversation_agent.py`

| Aspect | Description |
| --- | --- |
| Responsibility | Generate the final grounded response. |
| Input | User query, recent history, and `RetrievalOutput`. |
| Processing | Builds a grounded prompt with retrieved context, calls Groq, handles missing context and LLM errors. |
| Output | `RagOutput` containing answer, sources, prompt, matches, retrieval mode, scope, and selected documents. |
| Tools/Libraries | Groq Chat Completions API via custom HTTP client. |

If no context is retrieved, the agent returns:

```text
I could not find that in the uploaded documents.
```

---

## 9. Retrieval Strategy

### Semantic Retrieval

Semantic retrieval is performed using Sentence Transformers embeddings and ChromaDB. Each text chunk is embedded and stored in a Chroma collection. At query time, the query is embedded using the same embedding model and compared against stored chunk embeddings.

### Keyword / BM25 Retrieval

The current implementation uses a local BM25-style keyword search implemented in `core/vector_store.py`. It uses tokenization, document frequency, term frequency, and BM25 scoring. This is not Elasticsearch. Elasticsearch was considered but intentionally not implemented to keep the prototype lightweight and avoid requiring a separate server.

### Elasticsearch Status

Elasticsearch keyword search is **not implemented** in the current codebase. It is a suitable future enhancement if the system is deployed with an external search service.

### Hybrid Search

Focused retrieval uses both:

- dense semantic search through ChromaDB,
- keyword search through the local BM25-style index.

The results are merged, deduplicated, and reranked. Hybrid retrieval improves recall because semantic search can capture meaning while keyword search can capture exact terms such as API endpoint names, policy names, error codes, or document-specific terminology.

### Retrieval Modes

| Mode | Used For | Behavior |
| --- | --- | --- |
| Focused | Specific factual questions | Uses query rewriting, dense search, keyword search, deduplication, reranking, and returns top-k. |
| Broad | Summaries, explanations, comparisons | Retrieves broader context from selected document scope. |
| Exhaustive | Counts, totals, complete lists, “all” questions | Fetches complete context from selected relevant documents. |

### Document Scope Selection

Document scope is separate from retrieval mode.

| Scope | Meaning |
| --- | --- |
| `selected_documents` | Only documents relevant to the query are eligible. |
| `all_documents` | All uploaded documents are eligible, used only when the query explicitly asks for all documents. |

For example, a question such as “How many types of leave are mentioned?” is exhaustive, but the scope should be the employee handbook only. This avoids retrieving unrelated API or financial report chunks.

### Query Rewriting

`core/query_rewriter.py` implements an LLM-backed query rewriter. It asks the LLM to produce 3 to 5 search query variants while preserving user intent. This helps retrieve relevant context when the document uses different wording than the user query.

### Deduplication

Retrieved candidates are deduplicated by:

```text
(filename, page_number, chunk_id)
```

If duplicate chunks are found, the highest-scoring version is retained.

### Reranking

The default reranker is `ScoreReranker`, which sorts candidates by attached score. An optional cross-encoder reranker is available through `CrossEncoderReranker` and can be enabled using:

```powershell
$env:ENABLE_CROSS_ENCODER_RERANKER="true"
```

This optional feature depends on downloading or having access to the cross-encoder model.

---

## 10. Prompt Engineering

The grounding prompt is defined in `core/prompts.py`. It contains detailed instructions for:

- using retrieved context as the only source of truth,
- avoiding outside knowledge,
- not inventing facts,
- handling incomplete evidence,
- supporting semantic matching,
- respecting retrieval mode,
- citing document name and page number.

### Anti-Hallucination Strategy

The prompt explicitly instructs the model:

- not to invent numbers, dates, policies, requirements, or conclusions,
- not to use unstated assumptions,
- to say the answer is not found when evidence is missing,
- to avoid completeness claims unless retrieved context supports completeness.

### Semantic Matching

The prompt allows semantic matching between user wording and document wording when supported by context. For example, “holiday” may be answered using “leave” only if the retrieved evidence clearly describes time off.

### Citation Rules

The model is instructed to cite sources in the format:

```text
Source: <document_name>, page <page_number>
```

The system also stores source metadata separately in `Source` objects.

### Conversation History Handling

The conversation agent receives recent history, limited to the last few turns. The prompt instructs the LLM to use history only to resolve follow-up references, not as independent factual evidence.

---

## 11. LangGraph Orchestration

LangGraph is used in `core/graph_orchestration.py` to coordinate the agent workflows as explicit state graphs.

### Why LangGraph Was Used

The system is naturally agentic: each stage reads and writes structured state. LangGraph provides a clear way to model these stages as nodes in a workflow.

### State Management Design

The implementation uses two separate state schemas because document processing and question answering happen at different times.

The upload/indexing graph uses `DocumentPipelineState`:

| State Field | Purpose |
| --- | --- |
| `doc_paths` | File paths saved from the current upload. |
| `ingestion` | Agent 1 validation output. |
| `extraction` | Agent 2 page-level text extraction output. |
| `indexing` | Agent 3 indexing summary. |
| `vector_store` | Session vector store produced by indexing. |
| `current_step` | Most recent graph step. |
| `completed_steps` | Ordered list of completed graph nodes. |
| `agent_steps` | Human-readable handoff records for each agent. |
| `errors` | Runtime errors captured during graph execution. |

The query graph uses `QueryPipelineState`:

| State Field | Purpose |
| --- | --- |
| `query` | Current user question. |
| `history` | Recent chat turns used for follow-up resolution. |
| `vector_store` | Indexed store created by the upload graph. |
| `retrieval` | Agent 4A retrieval output. |
| `response` | Agent 4B grounded answer output. |
| `current_step` | Most recent graph step. |
| `completed_steps` | Ordered list of completed graph nodes. |
| `agent_steps` | Handoff records with agent, status, and summary message. |
| `errors` | Runtime errors captured during graph execution. |

Each node updates only its own part of the state. For example, the retrieval node writes `retrieval`, while the conversation node writes `response`. The orchestration layer appends a step record after each node:

```text
step
agent
status
message
```

This makes the runtime flow inspectable without making the agents depend on Streamlit.

### Upload / Indexing Graph

```text
upload_received
  |
  v
ingest
  |
  v
extract
  |
  v
index
  |
  v
END
```

This graph validates documents, extracts text, and indexes chunks.

### Chat Graph

```text
question_received
  |
  v
retrieve
  |
  v
converse
  |
  v
END
```

This graph retrieves evidence and generates the final answer.

### Error Handling

The graph uses conditional edges after each upstream node. If a node records an error, the graph stops before running downstream nodes on missing or incomplete state.

For example:

```text
retrieve fails
  |
  v
errors updated
  |
  v
END
```

In Streamlit, this graph state is stored separately from the chat transcript. The UI keeps:

- `vector_store`,
- `pipeline_report`,
- `chat_history`,
- `latest_question`,
- `latest_response`,
- `latest_query_state`,
- `live_ragas_result`.

This separates UI session state from internal agent workflow state.

The individual agents remain independently testable, while LangGraph coordinates the end-to-end workflow.

---

## 12. Database and Storage Design

### ChromaDB Vector Store

The vector store is implemented in `core/vector_store.py`. It creates a Chroma collection per session:

```text
session_<session_id>
```

Each chunk is stored with:

| Field | Purpose |
| --- | --- |
| `chunk_id` | Unique chunk identifier |
| `doc_id` | Document identifier |
| `filename` | Original document name |
| `page_number` | Source page |
| `text` | Chunk content |
| embedding | Semantic vector representation |

### Keyword Index

The keyword index is a local BM25-style index built over the same chunks. It stores tokenized chunks, document lengths, average document length, and document frequency statistics.

### Elasticsearch Index

Elasticsearch is not part of the current implementation. It is listed as a future enhancement for scalable keyword search.

---

## 13. Evaluation

The evaluation workflow is implemented in:

- `evals/manual_eval_set.json`
- `evals/run_eval.py`
- `evals/README.md`

### Evaluation Dataset

The manual evaluation set contains 15 questions with:

- question id,
- user question,
- reference answer guidance,
- expected documents,
- expected answer keywords.

### RAGAS Metrics

| Metric | What It Evaluates | Target |
| --- | --- | --- |
| Faithfulness | Whether the answer is supported by retrieved context | > 0.85 |
| Response Relevancy | Whether the answer addresses the question | > 0.80 |
| Context Precision | Whether retrieved chunks are relevant | > 0.75 |
| Context Recall | Whether required evidence was retrieved | > 0.75 |
| Retrieval Hit Rate | Whether expected document was retrieved | > 0.90 |
| Avg Retrieved Docs | Whether retrieval is noisy | Lower is better |

### RAGAS Implementation

The evaluation runner:

- runs the actual RAG pipeline,
- logs question, answer, retrieved contexts, sources, expected documents, and expected keywords,
- runs RAGAS metrics,
- computes custom retrieval hit rate and average retrieved documents,
- supports rate-limit controls,
- skips failed Groq generations from RAGAS scoring in the updated runner,
- serializes invalid numeric values as JSON `null`.

### Available Evaluation Results

The file `evals/outputs/eval_report_20260521_164224.json` contains a previous evaluation run. The custom retrieval metrics were:

| Metric | Value |
| --- | --- |
| Retrieval Hit Rate | 0.9333 |
| Average Retrieved Documents | 2.2667 |

This indicates that the expected document was retrieved in most cases, but retrieval was still somewhat noisy on average.

The same report includes several incomplete RAGAS metric values (`NaN`) and several failed generations due to Groq rate limits (`HTTP 429`). Therefore, those RAGAS values should be interpreted cautiously. The evaluation runner was later improved with batching, delays, retry/backoff, memory reset per question, and failed-generation filtering.

---

## 14. Testing

Automated tests are implemented in `tests/test_agents.py`.

Covered test areas include:

| Test Area | Status |
| --- | --- |
| Invalid format rejection | Implemented |
| Grounded retrieval with source citation | Implemented |
| No-match hallucination avoidance | Implemented |
| Retrieval mode selection | Implemented |
| Query rewrite deduplication | Implemented |
| Transcript export to TXT/PDF | Implemented |
| Exhaustive retrieval with selected document scope | Implemented |
| Explicit all-document scope | Implemented |
| LangGraph query orchestration | Implemented |
| LangGraph state tracking and error recording | Implemented |

Recommended additional tests:

- valid upload through Streamlit,
- explicit page-limit rejection,
- scanned PDF OCR quality,
- synonym query evaluation,
- multi-document comparison,
- rate-limit handling during evaluation,
- RAGAS runner with a small sample set.

The latest recorded verification in the conversation showed:

```text
10 passed
```

---

## 15. Results and Discussion

The project successfully implements a working multi-agent RAG chat engine with document upload, text extraction, indexing, hybrid retrieval, grounded answer generation, source citation, transcript export, LangGraph orchestration, and RAGAS evaluation support.

### Strengths

- Clear agent separation improves maintainability.
- Structured dataclasses make handoffs explicit.
- ChromaDB supports semantic retrieval.
- Local BM25-style keyword search improves retrieval recall without external infrastructure.
- Retrieval mode and document scope are separated, reducing irrelevant context.
- Prompt engineering strongly enforces document grounding.
- LangGraph models the pipeline as explicit workflows.
- Graph state records current step, completed steps, agent handoffs, and errors.
- Evaluation infrastructure is present and extensible.

### Observed Challenges

- RAGAS evaluation can be token-intensive when using a hosted LLM judge.
- Groq API rate limits can interrupt evaluation runs.
- Retrieval can still become noisy when document topics overlap.
- OCR quality depends on input image clarity.
- Some RAGAS scores in earlier reports were incomplete due to failed LLM generations.

---

## 16. Limitations

The current system has the following limitations:

- OCR accuracy depends on document scan quality and layout clarity.
- The system assumes small document sessions: maximum 3 documents and 5 pages per PDF.
- The keyword index is local and in-memory rather than Elasticsearch-backed.
- Elasticsearch was not implemented.
- Cross-encoder reranking is optional and disabled by default.
- Groq rate limits may affect long evaluations or repeated usage.
- ChromaDB collections are local and session-based.
- There is no user authentication or multi-user session isolation beyond local session identifiers.
- Layout-aware extraction for complex tables, forms, or multi-column documents is limited.
- RAGAS results may require careful interpretation when API failures occur.

---

## 17. Future Enhancements

Potential future improvements include:

- Add Elasticsearch or OpenSearch for scalable keyword search.
- Add a production-grade cross-encoder reranker by default.
- Improve layout-aware parsing for tables and forms.
- Add multilingual OCR support.
- Add persistent session management and user authentication.
- Add Docker-based deployment.
- Add cloud deployment with managed vector and keyword stores.
- Add evaluation dashboards for RAGAS results.
- Expand the manual evaluation dataset beyond 15 questions.
- Add regression tests for RAGAS metrics over time.
- Add document-level summarization and metadata extraction.
- Improve UI with document previews and source highlighting.

---

## 18. Conclusion

This internship project demonstrates a practical implementation of an autonomous multi-agent document-grounded chat engine using RAG and agentic AI principles. The system processes uploaded documents through validation, extraction, chunking, embedding, retrieval, and grounded response generation. It uses semantic and keyword retrieval, document scope selection, query rewriting, reranking, anti-hallucination prompting, and source citations to improve answer reliability.

The project also demonstrates modern AI engineering practices such as modular agent design, protocol-based abstraction, LangGraph orchestration, local evaluation workflows, and RAGAS-based quality assessment. While the system is a prototype, it provides a strong foundation for a production-ready document-grounded assistant.

---

## 19. References

- Streamlit - user interface and chat components.
- PyMuPDF - PDF parsing and page-level text extraction.
- EasyOCR - OCR for images and scanned pages.
- Sentence Transformers - embedding model and optional cross-encoder reranking.
- ChromaDB - local vector database for semantic retrieval.
- BM25 - keyword retrieval scoring approach, implemented locally.
- Groq Chat Completions API - LLM-based answer generation and query rewriting.
- LangGraph - graph-based orchestration of agent workflows.
- RAGAS - RAG evaluation metrics.
- HuggingFace / local sentence-transformer embeddings - evaluator embeddings for RAGAS.
- Python dataclasses and protocols - structured contracts and low-coupling design.
