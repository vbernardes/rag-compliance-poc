# RAG PoC — System Specification

## 1. Overview

A local-first Retrieval-Augmented Generation (RAG) proof-of-concept with two modes:

1. **Q&A mode** — ingest PDF documents, query them in natural language, receive grounded answers with source citations.
2. **Compliance Check mode** — ingest a regulation as ground truth, upload a company policy PDF, and receive a structured compliance report indicating which regulatory requirements are met, partially met, not met, or not addressed.

**Scope:** Single-user, local machine. No cloud infrastructure required (LLM runs via Mistral API by default; Ollama and Claude are alternative swap-ins via env flag).

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI  (st.tabs)                                    │
│                                                             │
│  💬 Q&A tab                  ✅ Compliance Check tab        │
│  ─────────────────           ─────────────────────          │
│  Chat input/history          Run button + progress bar      │
│  Citations expander          Summary metrics                │
│                              Per-requirement expanders      │
│                              Markdown download              │
└───────────┬─────────────────────────┬───────────────────────┘
            │                         │
   ┌────────▼────────┐      ┌─────────▼──────────────────┐
   │  RAG Engine     │      │  Compliance Engine          │
   │  query → MMR    │      │  extract requirements (LLM) │
   │  → prompt → LLM │      │  → assess each req (LLM)   │
   └────────┬────────┘      │  → ComplianceReport         │
            │               └─────────┬──────────────────┘
   ┌────────▼────────┐      ┌─────────▼──────────────────┐
   │  ChromaDB       │      │  ChromaDB                   │
   │  ./data/chroma  │      │  ./data/chroma_compliance   │
   │  collection:    │      │  collection: "compliance"   │
   │  "rag_poc"      │      │  doc_role: regulation|policy│
   └─────────────────┘      └─────────────────────────────┘
```

---

## 2. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.9+ | tested on 3.9.18 |
| Orchestration | LangChain | chains, retrievers, prompt templates |
| LLM (default) | Mistral API (`open-mistral-nemo`) | `LLM_PROVIDER=mistral`; also switches embeddings to `mistral-embed` |
| LLM (option) | Mistral Nemo via Ollama | `LLM_PROVIDER=ollama`; local, ~6–7 GB RAM (Q4), 128k context |
| LLM (option) | Claude `claude-sonnet-4-6` | `LLM_PROVIDER=anthropic` |
| Embeddings | `mistral-embed` (API) / `nomic-embed-text` (Ollama) | switches automatically with `LLM_PROVIDER` |
| Vector store | ChromaDB | two file-backed collections: `./data/chroma/` (Q&A) and `./data/chroma_compliance/` (compliance) |
| PDF parsing | PyMuPDF (`fitz`) | block-level text extraction with font size and bounding-box metadata |
| NER | spaCy `en_core_web_sm` (3.7.x) | entity mention extraction for metadata |
| UI | Streamlit | single-command startup, dual-tab layout |

**Prerequisites (one-time setup):**
```bash
# For Mistral API (default):
# Set MISTRAL_API_KEY in .env

# For Ollama (optional local mode):
brew install ollama
ollama pull mistral-nemo
ollama pull nomic-embed-text
```

---

## 3. Functional Requirements

### FR-1 — Document Ingestion

- Accept one or more PDF files via the Streamlit file uploader
- Parse each PDF with PyMuPDF, preserving page boundaries and spatial structure
- Deduplicate: skip re-indexing if a file with the same name is already in the collection
- Display ingestion progress (progress bar + chunk count on completion)
- Tag each chunk with a `doc_role` metadata field: `"knowledge"` (Q&A), `"regulation"`, or `"policy"` (compliance)

### FR-2 — Pre-processing Pipeline

- **Text cleaning:** strip repeated headers/footers, fix soft-hyphenation, normalise whitespace
- **Chunking:** `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `overlap=200` (both configurable via env)
- **Metadata per chunk:**
  - *Provenance:* `source` (filename), `page` (page number), `chunk_index`
  - *Structural:* `section_title` (nearest heading above chunk), `section_path` (e.g. `"3 > 3.2 Methods"`), `doc_title`
  - *Role:* `doc_role` — `"knowledge"`, `"regulation"`, or `"policy"`
  - *Entity mentions:* `entities` — JSON list of `{text, label}` for persons, orgs, dates, locations (spaCy NER)

### FR-3 — RAG Engine (Q&A mode)

- Embed user query with the active embedding model and run similarity search over the Q&A ChromaDB collection (default `k=5`)
- Apply **MMR re-ranking** (`fetch_k=20`) to reduce redundant chunks in the context window
- Construct prompt: system instruction + retrieved chunks (with metadata) + conversation history + user question
- Stream response token-by-token from the LLM

### FR-4 — Basic UI (Streamlit)

- **Sidebar:** two collapsible sections — *Q&A Documents* and *Compliance Check*
- **Q&A tab:** chat interface with full conversation history (stored in `st.session_state`); citations expander; clear chat button
- **Compliance Check tab:** status metrics, Run button, per-requirement expandable cards, Markdown download button
- **Clear chat** button resets conversation history without wiping the index

### FR-5 — Compliance Verification Engine

**Document roles:**
- *Regulation* — the authoritative reference (e.g. GDPR, ISO 27001). Indexed to the `"compliance"` ChromaDB collection with `doc_role="regulation"`.
- *Policy* — the document under evaluation (e.g. company data policy). Indexed with `doc_role="policy"`.

**Algorithm (two-phase LLM pipeline):**

1. **Requirement extraction** — retrieve the regulation's top-k chunks using a broad query (`"requirements obligations shall must prohibited"`); ask the LLM to extract a JSON list of `{req_id, section, text}` objects.
2. **Per-requirement assessment** — for each extracted requirement, retrieve the top-k most relevant policy chunks; ask the LLM to return a JSON object with:
   - `status`: `COMPLIANT` | `PARTIAL` | `NON_COMPLIANT` | `NOT_ADDRESSED`
   - `evidence`: quote from the policy supporting the assessment
   - `gap`: description of what is missing (if status ≠ `COMPLIANT`)
   - `recommendation`: suggested policy change to achieve compliance

**ComplianceReport structure:**
- `regulation_docs`: list of source filenames
- `policy_doc`: policy filename
- `generated_at`: ISO-8601 timestamp
- `results`: list of `RequirementResult` objects (one per requirement)
- `summary`: count per status (`COMPLIANT`, `PARTIAL`, `NON_COMPLIANT`, `NOT_ADDRESSED`)

**Output:**
- Summary metrics (4 columns) in the Compliance tab
- Per-requirement expandable cards colour-coded by status
- Downloadable Markdown report (`compliance_report_YYYYMMDD_HHMMSS.md`)

---

## 4. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Q&A first token under 10 s for a corpus up to 500 pages on a 16 GB RAM laptop |
| NFR-2 | Fully local by default (Ollama) — no network calls except when using Mistral or Claude API |
| NFR-3 | Both ChromaDB indexes persist across restarts; no re-ingestion needed |
| NFR-4 | Single command startup: `streamlit run app.py` |
| NFR-5 | All secrets via `.env`; `.env` is gitignored |
| NFR-6 | Compliance check completes in under 5 minutes for a regulation with ≤ 30 requirements on Mistral API |

---

## 5. User Journeys

### Journey 1 — First-time setup

1. Clone repo and run `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`; set `MISTRAL_API_KEY` (or configure Ollama)
3. Run `streamlit run app.py` — browser opens automatically
4. Upload one or more PDFs via the sidebar Q&A uploader
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
2. User uploads a new PDF via the Q&A sidebar section
3. System detects the file is not yet indexed, processes and embeds it
4. New document is immediately queryable alongside existing ones — no restart needed

### Journey 4 — Compliance verification

1. User opens the **⚖️ Compliance Check** sidebar section
2. Uploads one or more regulation PDFs (e.g. GDPR, ISO standard) — indexed as `doc_role="regulation"`
3. Uploads the company policy PDF — indexed as `doc_role="policy"`
4. Switches to the **✅ Compliance Check** tab; a status row confirms regulation and policy are indexed
5. Clicks **Run Compliance Check** — progress bar shows extraction and per-requirement assessment steps
6. On completion, summary metrics appear (Compliant / Partial / Non-compliant / Not addressed)
7. User expands individual requirement cards to review evidence, gaps, and recommendations
8. Clicks **Download report (.md)** to save the full report

---

## 6. Project Structure

```
poc/
├── app.py                  # Streamlit entry point (dual-tab: Q&A + Compliance)
├── rag/
│   ├── __init__.py
│   ├── ingest.py           # PDF loading, chunking, NER, embedding, ChromaDB write
│   ├── retriever.py        # ChromaDB query, MMR re-ranking, optional doc_role filter
│   ├── chain.py            # RAG chain, prompt template, LLM selection, streaming
│   └── compliance.py       # Requirement extraction, per-req assessment, report assembly
├── data/
│   ├── chroma/             # Persisted Q&A ChromaDB index (gitignored)
│   └── chroma_compliance/  # Persisted compliance ChromaDB index (gitignored)
├── uploads/                # Temp PDF storage for Q&A (gitignored)
│   └── compliance/         # Temp PDF storage for compliance (gitignored)
├── requirements.txt
├── .env.example
└── spec.md
```

**`.env.example`:**
```
# LLM selection: "mistral" (default), "ollama", or "anthropic"
LLM_PROVIDER=mistral

# Mistral API (default provider)
MISTRAL_API_KEY=
MISTRAL_MODEL=open-mistral-nemo

# Ollama (local, no API key needed)
OLLAMA_MODEL=mistral-nemo

# Only required when LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=

# Retrieval tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5

# Compliance tuning (number of regulation chunks fed to requirement extraction)
COMPLIANCE_EXTRACTION_K=20

# Override compliance ChromaDB directory (default: ./data/chroma_compliance)
CHROMA_COMPLIANCE_DIR=./data/chroma_compliance

# Langfuse observability (optional — tracing is disabled when these are unset)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
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
| SC-8 | Regulation and policy indexed separately | After uploading regulation + policy, ChromaDB `compliance` collection contains chunks with both `doc_role="regulation"` and `doc_role="policy"` |
| SC-9 | Requirements extracted correctly | `extract_requirements()` returns a non-empty JSON list; each item has `req_id`, `section`, `text` |
| SC-10 | Compliance assessment is accurate | Upload a policy with one known-compliant and one known-non-compliant section; verify the report reflects this correctly |
| SC-11 | Markdown report downloads successfully | Downloaded `.md` file renders with summary table, per-requirement sections, and correct statuses |

---

## 8. Out of Scope (PoC)

- Authentication or multi-user sessions
- Cloud deployment / containerisation
- Non-PDF formats (Word, HTML, web scraping)
- Fine-tuning or custom embedding models
- Async ingestion queue
- Automated evaluation metrics
- Batch compliance checks across multiple policy documents simultaneously

---

## 9. Future Iterations

| Iteration | Feature | Notes |
|---|---|---|
| 2 | **RAGAS evaluation** | Automated metrics: faithfulness, answer relevancy, context recall |
| 2 | Hybrid search (BM25 + vector) | Better recall on keyword-heavy queries |
| 2 | Compliance report in PDF/DOCX | Export compliance report as a formatted document |
| 3 | Non-PDF format support | Word, HTML, Markdown |
| 3 | Cloud deployment | Docker + cloud-hosted Ollama or managed LLM |
| 3 | Batch policy comparison | Compare multiple policy documents against the same regulation simultaneously |
