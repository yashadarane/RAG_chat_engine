# RAGAS Evaluation

This folder contains a manual RAG evaluation workflow:

1. `manual_eval_set.json` defines 15 manual questions, reference answers, expected documents, and keyword hints.
2. `run_eval.py` runs the real RAG pipeline against uploaded documents.
3. Raw RAG logs are saved to `evals/outputs/`.
4. RAGAS scores are saved to `evals/outputs/` when evaluator credentials are available.
5. Custom retrieval metrics are saved with every run.

## Run RAG Logs Only

```powershell
.\.venv\Scripts\python.exe evals\run_eval.py `
  --docs data\uploads\your_session\01_quarterly_report.pdf `
         data\uploads\your_session\02_employee_handbook.pdf `
         data\uploads\your_session\03_api_documentation.pdf `
  --skip-ragas
```

## Run Full RAGAS

RAGAS uses Groq as the judge LLM and local sentence-transformer embeddings by default.

```powershell
$env:GROQ_API_KEY="your_groq_key_here"

.\.venv\Scripts\python.exe evals\run_eval.py `
  --docs data\uploads\your_session\01_quarterly_report.pdf `
         data\uploads\your_session\02_employee_handbook.pdf `
         data\uploads\your_session\03_api_documentation.pdf
```

Optional overrides:

```powershell
$env:RAGAS_LLM_MODEL="llama-3.3-70b-versatile"
$env:RAGAS_LOCAL_EMBEDDING_MODEL="models/all-MiniLM-L6-v2"
```

If you later want OpenAI embeddings as the RAGAS evaluator embedding model, set:

```powershell
$env:OPENAI_API_KEY="your_openai_key_here"
$env:RAGAS_USE_OPENAI_EMBEDDINGS="true"
$env:RAGAS_EMBEDDING_MODEL="text-embedding-3-small"
```

## Report Targets

| Metric | What It Evaluates | Target |
| --- | --- | --- |
| Faithfulness | Whether Agent 4B hallucinated | > 0.85 |
| Response Relevancy | Whether answer addresses question | > 0.80 |
| Context Precision | Whether retrieved chunks were relevant | > 0.75 |
| Context Recall | Whether required evidence was retrieved | > 0.75 |
| Retrieval Hit Rate | Whether expected document was retrieved | > 0.90 |
| Avg Retrieved Docs | Whether retrieval is noisy | Lower is better |
