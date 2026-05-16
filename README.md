# Inquira Backend

Backend service for Inquira, an AI research assistant for scholarly search, paper ingestion, citation-grounded RAG answers, and evaluation. For frontend, you can find it [here](https://github.com/lyhongduc123/backend-inquira)

This README documents files that are tracked in this backend repository. It intentionally avoids local-only logs, generated outputs, and helper files that are not present in git.

## Table of Contents

- [Core Intent](#core-intent)
- [Tracked Project Structure](#tracked-project-structure)
- [Main API](#main-api)
- [Install](#install)
- [Run Backend](#run-backend)
- [Data Setup](#data-setup)
  - [Import CORE conference data](#1-import-core-conference-data)
  - [Download SJR data](#2-download-sjr-data)
  - [Import SJR journal data](#3-import-sjr-journal-data)
- [LitQA Workflow](#litqa-workflow)
  - [Preprocess LitQA DOIs](#1-preprocess-litqa-dois)
  - [Start the backend](#2-start-the-backend)
  - [Generate answer batches](#3-generate-answer-batches)
- [Answer Validation Export and Scoring](#answer-validation-export-and-scoring)
  - [Export validation rows](#1-export-validation-rows)
  - [Run RAGAS/project evaluation](#2-run-ragasproject-evaluation)
- [Retrieval Benchmark Workflow](#retrieval-benchmark-workflow)
- [Optional SciFact Query Breakdown](#optional-scifact-query-breakdown)
- [Notes](#notes)
<!-- - [License](#license) -->

## Core Intent

The backend has four main responsibilities:

1. **Retrieve research papers**
   - Query local papers and external scholarly sources.
   - Normalize paper metadata, authors, institutions, journals, conferences, citations, and identifiers.
   - Rank results with hybrid search, vector search, RRF fusion, and reranking.

2. **Process papers**
   - Preprocess selected papers.
   - Extract text and metadata.
   - Chunk documents and create embeddings for retrieval.

3. **Answer research questions**
   - Serve chat and RAG APIs.
   - Retrieve paper/chunk context.
   - Generate evidence-based answers with citations.
   - Store conversations, messages, context, and answer validation records.

4. **Evaluate retrieval and answers**
   - Run BEIR-style retrieval benchmarks.
   - Run LitQA/SciFact batch answer generation.
   - Export answer-validation datasets.
   - Score answer quality with RAGAS and project heuristics.

## Tracked Project Structure

```text
backend-exegent/
+-- app/
|   +-- main.py                  FastAPI application entrypoint
|   +-- core/                    config, database, exceptions, responses
|   +-- auth/                    auth, OAuth, OTP, refresh tokens
|   +-- domain/                  API domains: chat, papers, messages, authors, etc.
|   +-- rag_pipeline/            RAG pipeline implementations
|   +-- retriever/               paper retrieval providers and services
|   +-- search/                  local search, filters, fusion, query building
|   +-- processor/               preprocessing, extraction, chunking, embeddings
|   +-- llm/                     LLM clients, prompts, decomposition, answer generation
|   +-- models/                  SQLAlchemy models
|   +-- workers/                 task queue and worker helpers
|   +-- utils/                   shared utilities
+-- scripts/
|   +-- import_core_data.py
|   +-- download_sjr_data.py
|   +-- import_sjr_data.py
|   +-- preprocess_litqa_dois.py
|   +-- run_database_pipeline_stream_batch.py
|   +-- cf_breakdown.py
+-- rag_eval/
|   +-- ir_benchmark.py
|   +-- export_answer_validations.py
|   +-- evals.py
|   +-- evals/datasets/
|   +-- evals/experiments/
+-- data/
|   +-- CORE.csv
|   +-- beir_datasets/scifact/
|   +-- litqa/
|   +-- scimagojr/scimagojr_1999.csv
+-- run_e2e_benchmark.py
+-- pyproject.toml
+-- requirements.txt
+-- .env.example
```

## Main API

Entry point:

```text
app.main:app
```

The installed console script is:

```powershell
run
```

Important routes registered in `app/main.py`:

- `/health`
- `/docs`
- `/api/v1/auth`
- `/api/v1/chat`
- `/api/v1/conversations`
- `/api/v1/messages`
- `/api/v1/papers`
- `/api/v1/authors`
- `/api/v1/bookmarks`
- `/api/v1/user/settings`
- `/api/v1/admin/preprocessing`
- `/api/v1/admin/validation`

## Install

From `backend-exegent/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Alternative:

```powershell
pip install -r requirements.txt
pip install -e .
```

Create `.env` from the tracked template:

```powershell
Copy-Item .env.example .env
```

Minimum values to configure:

```env
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5432/exegent
DATABASE_SYNC_URL=postgresql://USER:PASSWORD@localhost:5432/exegent
FRONTEND_URL=http://localhost:3000
JWT_SECRET_KEY=change-me
```

Provider keys are defined in `.env.example` and `app/core/config.py`. Fill only the providers needed for the workflow you run.

## Run Backend

Start PostgreSQL then run:

```powershell
run (development)
```

Equivalent:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Check the server:

```powershell
curl http://localhost:8000/
```

Open docs:

```text
http://localhost:8000/docs
```

## Data Setup

Only tracked data/setup files are listed here.

### 1. Import CORE conference data

Tracked input:

```text
data/CORE.csv
```

Run:

```powershell
python scripts/import_core_data.py --file data/CORE.csv
```

### 2. Download SJR data

Tracked script:

```text
scripts/download_sjr_data.py
```

Run one of:

```powershell
python scripts/download_sjr_data.py --latest
python scripts/download_sjr_data.py --years 2024
python scripts/download_sjr_data.py --range 2020 2024
```

### 3. Import SJR journal data

Tracked sample data:

```text
data/scimagojr/scimagojr_1999.csv
```

Import one year:

```powershell
python scripts/import_sjr_data.py --year 1999 --update
```

Import all local SJR files:

```powershell
python scripts/import_sjr_data.py --all --update
```

Dry run:

```powershell
python scripts/import_sjr_data.py --year 1999 --dry-run
```

## LitQA Workflow

Tracked LitQA inputs:

```text
data/litqa/litqa-v2-public.jsonl
data/litqa/info.json
data/litqa/exception.json
```

### 1. Preprocess LitQA DOIs

Smoke test:

```powershell
python scripts/preprocess_litqa_dois.py --limit 10
```

Full run:

```powershell
python scripts/preprocess_litqa_dois.py
```

Check existing synced/processed DOIs:

```powershell
python scripts/preprocess_litqa_dois.py --check
```

### 2. Start the backend

The batch script calls the API, so the backend must be running:

```powershell
run
```

### 3. Generate answer batches

Tracked script:

```text
scripts/run_database_pipeline_stream_batch.py
```

Small LitQA run:

```powershell
python scripts/run_database_pipeline_stream_batch.py --dataset litqa --max-questions 10 --continue-on-error
```

Small SciFact run:

```powershell
python scripts/run_database_pipeline_stream_batch.py --dataset scifact --max-questions 10 --continue-on-error
```

Resume:

```powershell
python scripts/run_database_pipeline_stream_batch.py --resume --continue-on-error
```

## Answer Validation Export and Scoring

Tracked scripts:

```text
rag_eval/export_answer_validations.py
rag_eval/evals.py
```

Tracked dataset path:

```text
rag_eval/evals/datasets/litqa_answer_validations.jsonl
```

### 1. Export validation rows

LitQA:

```powershell
python rag_eval/export_answer_validations.py --dataset litqa --dataset-output rag_eval/evals/datasets/litqa_answer_validations.jsonl
```

SciFact:

```powershell
python rag_eval/export_answer_validations.py --dataset scifact --dataset-output rag_eval/evals/datasets/scifact_answer_validations.jsonl
```

Sample:

```powershell
python rag_eval/export_answer_validations.py --dataset litqa --max-samples 20 --dataset-output rag_eval/evals/datasets/litqa_answer_validations_sample.jsonl
```

### 2. Run RAGAS/project evaluation

```powershell
python rag_eval/evals.py --dataset-path rag_eval/evals/datasets/litqa_answer_validations.jsonl --max-samples 20 --max-concurrent 1 --output-file rag_eval/evals/experiments/ragas_eval_smoke.json
```

Resume an existing evaluation file:

```powershell
python rag_eval/evals.py --resume --output-file rag_eval/evals/experiments/ragas_eval_smoke.json --skip-db-update
```

Optional CSV/XLSX exports:

```powershell
python rag_eval/evals.py --dataset-path rag_eval/evals/datasets/litqa_answer_validations.jsonl --max-samples 20 --output-file rag_eval/evals/experiments/ragas_eval_smoke.json --export-csv "" --export-xlsx ""
```

## Retrieval Benchmark Workflow

Tracked benchmark scripts:

```text
rag_eval/ir_benchmark.py
run_e2e_benchmark.py
```

Tracked BEIR/SciFact files:

```text
data/beir_datasets/scifact/corpus.jsonl
data/beir_datasets/scifact/queries.jsonl
data/beir_datasets/scifact/qrels/test.tsv
data/beir_datasets/scifact/qrels/train.tsv
data/beir_datasets/scifact/queries_decomposed.jsonl
```

Run SciFact retrieval benchmark:

```powershell
python rag_eval/ir_benchmark.py --dataset scifact --max-queries 50 --decomposition both --compare-rerank
```

Run only original after-rerank:

```powershell
python rag_eval/ir_benchmark.py --dataset scifact --max-queries 50 --decomposition off
```

Interactive wrapper:

```powershell
python run_e2e_benchmark.py
```

Benchmark exports are written by `rag_eval/ir_benchmark.py` under:

```text
rag_eval/evals/experiments/<dataset>/ir_benchmark_<dataset>_<timestamp>/
```

The export contains:

- `results.json`
- `summary.csv`
- `components.csv`
- `<variant>_queries.csv`
- `comparison.txt`

## Optional SciFact Query Breakdown

Tracked script:

```text
scripts/cf_breakdown.py
```

This script reads tracked SciFact data from:

```text
data/beir_datasets/scifact/
```

It is intended for query-breakdown experiments using an Ollama-compatible local endpoint.

Run:

```powershell
python scripts/cf_breakdown.py
```

## Notes

- This README intentionally references only tracked files from this backend repository.
- Local `.env`, virtual environments, logs, and most generated outputs are ignored by git.
- BEIR retrieval metrics evaluate document retrieval. Answer correctness is evaluated separately through `rag_eval/export_answer_validations.py` and `rag_eval/evals.py`.
- Use `--limit`, `--max-questions`, `--max-samples`, and `--max-queries` for smoke tests before full runs.