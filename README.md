# EduRAG

EduRAG is an educational Retrieval-Augmented Generation system for Thai high-school science and mathematics. It combines scraped/normalized curriculum content, BGE-M3 embeddings, Chroma retrieval, OpenRouter-compatible LLM generation, and evaluator workflows for both dataset-based and live query testing.

## What This Project Does

- Scrapes and normalizes OpenStax and SciMath documents into one canonical document format.
- Chunks normalized lessons and stores them in a Chroma vector database using `BAAI/bge-m3`.
- Routes English queries toward OpenStax and Thai queries toward SciMath, with fallback retrieval.
- Generates RAG answers using retrieved curriculum context and answer preferences.
- Compares RAG answers against baseline LLM answers through live query evaluation.
- Runs offline evaluation datasets such as `physics_bee_20.csv`.
- Serves a simple demo website at `/demo`.

## Project Map

```text
ragaib/
  chat/              FastAPI app, chat endpoints, live eval endpoint, demo mount
  config/            Settings, curriculum maps, environment loading
  data/              Raw, normalized, processed, and evaluation artifacts
  demo/              Static demo website for live RAG vs baseline evaluation
  docs/              Architecture, workflows, and presentation notes
  embeddings/        Document normalization and chunk/document processing
  evaluation/        Dataset loaders, judge logic, retrieval eval, live query eval
  retrieval/         RAG chain, prompts, language/source routing
  scraper/           OpenStax and SciMath scrapers
  scripts/           Human-friendly workflow command wrappers
  tests/             Unit and smoke tests
  vector_store/      Chroma vector DB manager and indexing CLI
```

## Quick Start

```powershell
cd C:\Users\BKAWIN\Desktop\RAG\ragaib
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m uvicorn chat.api:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/demo/
```

## Main Workflows

- Build or refresh the vector DB:
  ```powershell
  .\.venv\Scripts\python.exe -m vector_store.indexer
  ```

- Run retrieval evaluation:
  ```powershell
  .\.venv\Scripts\python.exe -m evaluation.retrieval_eval --dataset .\physics_bee_20.csv --top-k 6
  ```

- Run full answer evaluation:
  ```powershell
  .\.venv\Scripts\python.exe -m evaluation.run_eval --dataset .\physics_bee_20.csv
  ```

- Resume interrupted answer evaluation:
  ```powershell
  .\.venv\Scripts\python.exe -m evaluation.run_eval --dataset .\physics_bee_20.csv --resume-from .\data\eval\results\YOUR_FILE.jsonl
  ```
