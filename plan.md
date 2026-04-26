# RAG PoC — Technical Plan

## Technical Approach

### Architecture decisions

**Module boundaries** — the `rag/` package is split into three single-responsibility modules:
- `ingest.py` owns everything from raw PDF bytes → ChromaDB writes
- `retriever.py` owns everything from a query string → ranked `Document` list
- `chain.py` owns LLM selection, prompt assembly, and streaming

`app.py` only wires these three together; it holds no business logic.

**LLM abstraction** — both LLM backends (`ChatOllama` and `ChatAnthropic`) expose the same LangChain `BaseChatModel` interface. `chain.py` selects the implementation at startup based on `LLM_PROVIDER` env var, so the rest of the chain is provider-agnostic.

**Embeddings** — `OllamaEmbeddings(model="nomic-embed-text")` is used both at ingest time and at query time; a single `get_embeddings()` factory keeps them in sync.

**Structural metadata** — PyMuPDF exposes a block-level structure with bounding boxes and font sizes. The ingest pipeline infers section headings (blocks where font size > body average) and builds a `section_path` string before chunking. This metadata is stored as ChromaDB document metadata and surfaced in the citations panel.

**Deduplication** — ChromaDB collection is queried for existing `source` values before ingesting a new file. If a match is found, the file is skipped.

**Streaming** — `chain.stream()` yields token deltas. `app.py` uses `st.write_stream()` to render them incrementally. The full response is appended to `st.session_state.messages` once streaming completes.

---

## Dependency Graph

```
T0 scaffolding
├── T1 ingest.py  ─────────────────────────────────┐
└── T2 retriever.py ──── T3 chain.py ──────────────┴──── T4 app.py
```

- **T1 ∥ T2** — fully independent; develop in parallel
- **T3** — can start as soon as T2's public interface (`retrieve(query) → list[Document]`) is defined
- **T4** — needs T1 and T3 complete; T2 is exercised via T3

---

## Work Tasks

### T0 — Project scaffolding
**Depends on:** nothing  
**Files:** `requirements.txt`, `.env.example`, `.gitignore`, `rag/__init__.py`, `data/.gitkeep`, `uploads/.gitkeep`

- Create directory tree (`rag/`, `data/chroma/`, `uploads/`)
- Write `requirements.txt` with pinned major versions:
  ```
  langchain>=0.3
  langchain-community>=0.3
  langchain-ollama>=0.2
  langchain-anthropic>=0.3
  chromadb>=0.5
  pymupdf>=1.24
  spacy>=3.7,<3.8
  streamlit>=1.35
  python-dotenv>=1.0
  ```
- Write `.env.example` (as in spec §6)
- Write `.gitignore` (ignore `.env`, `data/chroma/`, `uploads/`)
- Write empty `rag/__init__.py`

**Review checklist:** `pip install -r requirements.txt` succeeds; `python -c "import langchain, chromadb, fitz, spacy, streamlit"` passes. (PyMuPDF imports as `fitz`.)

---

### T1 — Ingestion pipeline (`rag/ingest.py`)
**Depends on:** T0  
**Can run in parallel with:** T2

**Public interface:**
```python
def ingest_pdf(file_path: str, collection: chromadb.Collection) -> int:
    """Returns number of chunks added. Skips if already indexed."""
```

**Implementation steps:**

1. **Load** — call PyMuPDF to get a list of page blocks with text and bounding boxes
2. **Section extraction** — iterate blocks; tag any block whose font size exceeds the median by ≥20% as a heading; track a `section_path` stack as headings are encountered
3. **Text cleaning** — strip soft hyphens (`\xad`), collapse runs of whitespace, remove repeated header/footer lines (detected by appearing on ≥80% of pages)
4. **Chunking** — `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)` applied per page; each chunk inherits the page's current `section_path`
5. **NER** — run spaCy `en_core_web_sm` on each chunk text; serialize entities as `json.dumps([{"text": e.text, "label": e.label_} ...])` — stored as a metadata string (ChromaDB only accepts scalar metadata values)
6. **Embed + store** — build LangChain `Document` objects with full metadata dict; call `Chroma.from_documents()` with the shared embedding function

**Review checklist:**
- Upload a 10-page PDF; `collection.count()` increases
- Inspect a random chunk's metadata: all six fields present (`source`, `page`, `chunk_index`, `section_title`, `section_path`, `entities`)
- Upload the same PDF again; chunk count does not change

---

### T2 — Retriever (`rag/retriever.py`)
**Depends on:** T0  
**Can run in parallel with:** T1

**Public interface:**
```python
def get_retriever(collection: chromadb.Collection) -> BaseRetriever:
    """Returns a LangChain retriever with MMR re-ranking."""
```

**Implementation steps:**

1. Wrap the ChromaDB collection in a LangChain `Chroma` vectorstore using the shared `get_embeddings()` factory
2. Call `.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": 20})`
3. Return the retriever — no further logic needed here; MMR is handled inside LangChain

**Review checklist:**
- With an indexed collection, `retriever.invoke("test query")` returns `TOP_K` `Document` objects
- Each returned document has the expected metadata fields
- Two very similar chunks are not both returned (MMR effect)

---

### T3 — RAG chain (`rag/chain.py`)
**Depends on:** T2 interface defined (can start before T1 is complete)  
**Files:** `rag/chain.py`

**Public interface:**
```python
def build_chain(retriever: BaseRetriever) -> Runnable:
    """Returns a streamable LCEL chain: query → streamed answer string."""

def get_llm() -> BaseChatModel:
    """Reads LLM_PROVIDER from env and returns the appropriate model."""
```

**Implementation steps:**

1. **`get_llm()`** — reads `LLM_PROVIDER`:
   - `"ollama"` → `ChatOllama(model=OLLAMA_MODEL, temperature=0)`
   - `"anthropic"` → `ChatAnthropic(model="claude-sonnet-4-6", temperature=0)`

2. **Prompt template** — system message instructs the model to answer only from context and cite sources; human message template includes:
   ```
   Context:
   {context}

   Conversation history:
   {history}

   Question: {question}
   ```

3. **Context formatter** — a helper that takes `list[Document]` and formats each as:
   ```
   [source: {source}, page {page}, §{section_path}]
   {page_content}
   ```

4. **LCEL chain assembly:**
   ```python
   chain = (
       {"context": retriever | format_docs, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
       | prompt
       | llm
       | StrOutputParser()
   )
   ```
   History is injected by `app.py` via `chain.invoke({"question": q, "history": history_str})`.

**Review checklist:**
- `chain.invoke({"question": "...", "history": ""})` returns a non-empty string (not an error)
- Switching `LLM_PROVIDER=anthropic` in `.env` and restarting works without code changes
- `chain.stream(...)` yields string chunks (confirm streaming works before wiring into UI)

---

### T4 — Streamlit UI (`app.py`)
**Depends on:** T1 complete, T3 complete  
**Files:** `app.py`

**Implementation steps:**

1. **Startup** — load `.env`, initialise ChromaDB client + collection (`rag_poc`), build retriever (T2), build chain (T3)

2. **Session state init:**
   ```python
   st.session_state.setdefault("messages", [])   # [{role, content, sources}]
   ```

3. **Sidebar:**
   - `st.file_uploader` (accepts multiple PDFs)
   - On upload: save to `uploads/`, call `ingest_pdf()` (T1), show success toast with chunk count
   - Index stats: `collection.count()` displayed as "N chunks indexed"
   - "Reset index" button: deletes and recreates the ChromaDB collection

4. **Chat panel:**
   - Replay `st.session_state.messages` on each rerender (role-based bubbles)
   - For each assistant message, render a `st.expander("Sources")` with a table of retrieved chunk metadata
   - `st.chat_input` at the bottom

5. **On new user message:**
   ```python
   history_str = format_history(st.session_state.messages)
   sources = retriever.invoke(user_query)          # capture for citations
   with st.chat_message("assistant"):
       response = st.write_stream(
           chain.stream({"question": user_query, "history": history_str})
       )
   st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
   ```

6. **Clear chat** button: `st.session_state.messages = []` then `st.rerun()`

**Review checklist (maps directly to success criteria):**
- SC-1: upload PDF → sidebar shows increased chunk count
- SC-2: ask question about known page → sources expander lists that page
- SC-3: answer text references only information present in source chunks
- SC-4: sources expander shows correct filename, page, `section_path`
- SC-5: ask follow-up → answer coherently references prior exchange
- SC-6: stop app, restart, ask question → still works without re-upload
- SC-7: `streamlit run app.py` on a clean env with `.env` set → no errors

---

## Parallelization Summary

| Phase | Tasks | Can run in parallel |
|---|---|---|
| 1 | T0 | — (foundation) |
| 2 | T1, T2 | Yes — fully independent |
| 3 | T3 | After T2 interface is known |
| 4 | T4 | After T1 + T3 |

Estimated complexity: T0 (small) · T1 (large) · T2 (small) · T3 (medium) · T4 (medium)
