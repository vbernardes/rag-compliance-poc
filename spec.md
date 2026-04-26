# RAG PoC — System Specification

## 1. Overview

A local-first Retrieval-Augmented Generation (RAG) proof-of-concept that lets a user ingest PDF documents, query them in natural language, and receive grounded answers with source citations.

**Scope:** Single-user, local machine. No cloud infrastructure required (LLM runs via Ollama; Claude is an optional swap-in via env flag).

```
┌─────────────┐     PDFs      ┌──────────────────────────────────────────┐
│  Streamlit  │──────────────▶│  Ingestion Pipeline                      │
│     UI      │               │  fitz (PyMuPDF) → chunk → NER → embed → store  │
│             │               └────────────────────┬─────────────────────┘
│  Chat input │                                    │
│             │               ┌────────────────────▼─────────────────────┐
│  Citations  │◀──────────────│  RAG Engine                              │
│  panel      │   answer +    │  query → retrieve (ChromaDB) → rerank    │
└─────────────┘   sources     │  → prompt → LLM (Mistral Nemo / Claude)  │
                              └──────────────────────────────────────────┘
```

---

## 2. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.9+ | tested on 3.9.18 |
| Orchestration | LangChain | chains, retrievers, prompt templates |
| LLM (default) | Mistral Nemo via Ollama | local; ~6–7 GB RAM (Q4), 128k context |
| LLM (option) | Mistral API (`open-mistral-nemo`) | `LLM_PROVIDER=mistral`; also switches embeddings to `mistral-embed` |
| LLM (option) | Claude `claude-sonnet-4-6` | `LLM_PROVIDER=anthropic` |
| Embeddings | `nomic-embed-text` (Ollama) / `mistral-embed` (API) | switches automatically with `LLM_PROVIDER` |
| Vector store | ChromaDB | file-backed, persisted to `./data/chroma/` |
| PDF parsing | PyMuPDF (`fitz`) | block-level text extraction with font size and bounding-box metadata |
| NER | spaCy `en_core_web_sm` (3.7.x) | entity mention extraction for metadata |
| UI | Streamlit | single-command startup |

**Prerequisites (one-time setup):**
```bash
brew install ollama
ollama pull mistral-nemo
ollama pull nomic-embed-text
```

---

## 3. Functional Requirements

### FR-1 — Document Ingestion

- Accept one or more PDF files via the Streamlit file uploader
- Parse each PDF with PyMuPDF, preserving page boundaries and spatial structure
- Deduplicate: skip re-indexing if a file with the same name is already in the ChromaDB collection
- Display ingestion progress (progress bar + chunk count on completion)

### FR-2 — Pre-processing Pipeline

- **Text cleaning:** strip repeated headers/footers, fix soft-hyphenation, normalise whitespace
- **Chunking:** `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `overlap=200` (both configurable via env)
- **Metadata per chunk:**
  - *Provenance:* `source` (filename), `page` (page number), `chunk_index`
  - *Structural:* `section_title` (nearest heading above chunk), `section_path` (e.g. `"3 > 3.2 Methods"`), `doc_title`
  - *Entity mentions:* `entities` — JSON list of `{text, label}` for persons, orgs, dates, locations (spaCy NER)

### FR-3 — RAG Engine

- Embed user query with `nomic-embed-text` and run similarity search over ChromaDB (default `k=5`)
- Apply **MMR re-ranking** (`fetch_k=20`) to reduce redundant chunks in the context window
- Construct prompt: system instruction + retrieved chunks (with metadata) + conversation history + user question
- Stream response token-by-token from the LLM

### FR-4 — Basic UI (Streamlit)

- **Sidebar:** file uploader, ingestion status, index stats (document count, chunk count), reset index button
- **Main panel:** chat interface with full conversation history (stored in `st.session_state`)
- **Citations expander:** below each answer, list retrieved chunks with filename, page, section path, and a snippet
- **Clear chat** button resets conversation history without wiping the index

---

## 4. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Query response (first token) under 10 s for a corpus up to 500 pages on a 16 GB RAM laptop |
| NFR-2 | Fully local by default — no network calls except optional Claude API |
| NFR-3 | ChromaDB index persists across restarts; no re-ingestion needed |
| NFR-4 | Single command startup: `streamlit run app.py` |
| NFR-5 | All secrets via `.env`; `.env` is gitignored |

---

## 5. User Journeys

### Journey 1 — First-time setup

1. Clone repo and run `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`; confirm Ollama models are pulled
3. Run `streamlit run app.py` — browser opens automatically
4. Upload one or more PDFs via the sidebar uploader
5. Wait for ingestion confirmation (chunk count displayed)
6. Type a question in the chat — receive a streamed answer with source citations

### Journey 2 — Iterative Q&A session

1. Open the app (index already populated from a prior session)
2. Type a question; system retrieves top-k chunks and streams an answer
3. Citations expander shows which pages and sections were used
4. User asks a follow-up; conversation history is included in the next prompt
5. User copies or screenshots the answer

### Journey 3 — Adding documents to an existing index

1. App is running with documents already indexed
2. User uploads a new PDF via the sidebar
3. System detects the file is not yet indexed, processes and embeds it
4. New document is immediately queryable alongside existing ones — no restart needed

---

## 6. Project Structure

```
poc/
├── app.py                  # Streamlit entry point
├── rag/
│   ├── __init__.py
│   ├── ingest.py           # PDF loading, chunking, NER, embedding, ChromaDB write
│   ├── retriever.py        # ChromaDB query, MMR re-ranking
│   └── chain.py            # RAG chain, prompt template, LLM selection, streaming
├── data/
│   └── chroma/             # Persisted ChromaDB index (gitignored)
├── uploads/                # Temp PDF storage (gitignored)
├── requirements.txt
├── .env.example
└── spec.md
```

**`.env.example`:**
```
# LLM selection: "ollama" (default) or "anthropic"
LLM_PROVIDER=ollama
OLLAMA_MODEL=mistral-nemo

# Only required when LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=

# Retrieval tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

---

## 7. Success Criteria

| # | Criterion | How to verify |
|---|---|---|
| SC-1 | PDF ingested and chunked correctly | ChromaDB collection count increases by expected number of chunks after upload |
| SC-2 | Retrieved chunks are topically relevant | Ask a question whose answer is on a known page; verify retrieved chunks include that page |
| SC-3 | Answer is grounded in retrieved context | Compare answer text to source chunks; confirm no facts from outside the corpus |
| SC-4 | Sources cited accurately | Citations panel shows correct filename, page, and section for each retrieved chunk |
| SC-5 | Multi-turn conversation works | Follow-up question references the prior answer coherently |
| SC-6 | Index survives restart | Stop and restart the app; prior documents remain queryable without re-upload |
| SC-7 | Single-command startup | `streamlit run app.py` launches cleanly on a fresh env with no manual steps beyond setup |

---

## 8. Out of Scope (PoC)

- Authentication or multi-user sessions
- Cloud deployment / containerisation
- Non-PDF formats (Word, HTML, web scraping)
- Fine-tuning or custom embedding models
- Async ingestion queue
- Automated evaluation metrics

---

## 9. Future Iterations

| Iteration | Feature | Notes |
|---|---|---|
| 2 | **RAGAS evaluation** | Automated metrics: faithfulness, answer relevancy, context recall |
| 2 | Hybrid search (BM25 + vector) | Better recall on keyword-heavy queries |
| 3 | Non-PDF format support | Word, HTML, Markdown |
| 3 | Cloud deployment | Docker + cloud-hosted Ollama or managed LLM |
