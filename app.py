# Streamlit Cloud ships an old SQLite; swap in pysqlite3 when available.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv

from rag.chain import build_chain, get_langfuse_handler, get_llm
from rag.compliance import (
    ComplianceReport,
    render_report_markdown,
    run_compliance_check,
)
from rag.embeddings import get_model_name
from rag.ingest import ingest_pdf
from rag.retriever import get_retriever

load_dotenv()

CHROMA_DIR = "./data/chroma"
COLLECTION = "rag_poc"
UPLOADS_DIR = "./uploads"

COMPLIANCE_CHROMA_DIR = os.getenv("CHROMA_COMPLIANCE_DIR", "./data/chroma_compliance")
COMPLIANCE_COLLECTION = "compliance"
COMPLIANCE_UPLOADS_DIR = "./uploads/compliance"

# Bundled sample PDFs (relative to this file)
_HERE = Path(__file__).parent
DEMO_REGULATION_PDF = _HERE / "EU_AI_Act_sample.pdf"
DEMO_POLICY_PDF = _HERE / "sample_ai_policy.pdf"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(COMPLIANCE_UPLOADS_DIR, exist_ok=True)

st.set_page_config(page_title="RAG PoC", layout="wide")

# --- Password gate (active only when APP_PASSWORD secret is set) ---
_app_password = os.getenv("APP_PASSWORD")
if _app_password:
    if not st.session_state.get("authenticated"):
        st.title("RAG PoC — Demo Access")
        _pwd = st.text_input("Enter demo password:", type="password")
        if st.button("Continue"):
            if _pwd == _app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()


@st.cache_resource
def init_retriever():
    return get_retriever(CHROMA_DIR, COLLECTION)


@st.cache_resource
def init_chain():
    return build_chain(init_retriever())


# --- Session state ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "compliance_regulation_names" not in st.session_state:
    st.session_state.compliance_regulation_names = set()

if "compliance_policy_name" not in st.session_state:
    st.session_state.compliance_policy_name = None

if "compliance_report" not in st.session_state:
    st.session_state.compliance_report = None

# --- Sidebar ---

st.sidebar.title("RAG PoC")

with st.sidebar.expander("💬 Q&A Documents", expanded=True):
    _qa_demo_col, _qa_dl_col = st.columns(2)
    if _qa_demo_col.button("⚡ Load demo", key="qa_demo", help="Ingests the bundled EU AI Act sample PDF"):
        if DEMO_REGULATION_PDF.exists() and DEMO_REGULATION_PDF.name not in st.session_state.uploaded_names:
            dest = os.path.join(UPLOADS_DIR, DEMO_REGULATION_PDF.name)
            shutil.copy(DEMO_REGULATION_PDF, dest)
            n = ingest_pdf(dest, CHROMA_DIR, COLLECTION)
            if n > 0:
                st.success(f"{DEMO_REGULATION_PDF.name}: {n} chunks indexed")
            else:
                st.info(f"{DEMO_REGULATION_PDF.name}: already indexed")
            st.session_state.uploaded_names.add(DEMO_REGULATION_PDF.name)
        else:
            st.info("Demo document already loaded.")
    if DEMO_REGULATION_PDF.exists():
        _qa_dl_col.download_button(
            "⬇ Sample PDF",
            data=DEMO_REGULATION_PDF.read_bytes(),
            file_name=DEMO_REGULATION_PDF.name,
            mime="application/pdf",
            key="qa_dl",
        )

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True, key="qa_uploader"
    )
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_names:
                dest = os.path.join(UPLOADS_DIR, file.name)
                with open(dest, "wb") as f:
                    f.write(file.read())
                n = ingest_pdf(dest, CHROMA_DIR, COLLECTION)
                if n > 0:
                    st.success(f"{file.name}: {n} chunks indexed")
                else:
                    st.info(f"{file.name}: already indexed")
                st.session_state.uploaded_names.add(file.name)

    qa_client = chromadb.PersistentClient(path=CHROMA_DIR)
    qa_col = qa_client.get_or_create_collection(COLLECTION)

    _sidecar = os.path.join(CHROMA_DIR, ".embedding_model")
    if os.path.exists(_sidecar):
        with open(_sidecar) as _f:
            _stored = _f.read().strip()
        if _stored != get_model_name():
            qa_client.delete_collection(COLLECTION)
            qa_client.get_or_create_collection(COLLECTION)
            os.remove(_sidecar)
            st.session_state.uploaded_names = set()
            st.cache_resource.clear()
            st.warning(
                f"Embedding model changed ({_stored} → {get_model_name()}). "
                "Q&A index reset — please re-upload documents."
            )

    st.metric("Chunks indexed", qa_col.count())

    col_r, col_c = st.columns(2)
    if col_r.button("Reset index", key="qa_reset"):
        qa_client.delete_collection(COLLECTION)
        qa_client.get_or_create_collection(COLLECTION)
        st.session_state.uploaded_names = set()
        st.cache_resource.clear()
        st.rerun()
    if col_c.button("Clear chat", key="qa_clear"):
        st.session_state.messages = []
        st.rerun()

with st.sidebar.expander("⚖️ Compliance Check", expanded=False):
    _comp_demo_col, _comp_dl_col = st.columns(2)
    if _comp_demo_col.button("⚡ Load demo", key="comp_demo", help="Ingests EU AI Act (regulation) + sample AI policy"):
        loaded = []
        if DEMO_REGULATION_PDF.exists() and DEMO_REGULATION_PDF.name not in st.session_state.compliance_regulation_names:
            dest = os.path.join(COMPLIANCE_UPLOADS_DIR, DEMO_REGULATION_PDF.name)
            shutil.copy(DEMO_REGULATION_PDF, dest)
            n = ingest_pdf(dest, COMPLIANCE_CHROMA_DIR, COMPLIANCE_COLLECTION, doc_role="regulation")
            loaded.append(f"{DEMO_REGULATION_PDF.name}: {n} chunks" if n > 0 else f"{DEMO_REGULATION_PDF.name}: already indexed")
            st.session_state.compliance_regulation_names.add(DEMO_REGULATION_PDF.name)
        if DEMO_POLICY_PDF.exists() and DEMO_POLICY_PDF.name != st.session_state.compliance_policy_name:
            dest = os.path.join(COMPLIANCE_UPLOADS_DIR, DEMO_POLICY_PDF.name)
            shutil.copy(DEMO_POLICY_PDF, dest)
            n = ingest_pdf(dest, COMPLIANCE_CHROMA_DIR, COMPLIANCE_COLLECTION, doc_role="policy")
            loaded.append(f"{DEMO_POLICY_PDF.name}: {n} chunks" if n > 0 else f"{DEMO_POLICY_PDF.name}: already indexed")
            st.session_state.compliance_policy_name = DEMO_POLICY_PDF.name
            st.session_state.compliance_report = None
        if loaded:
            st.success("\n".join(loaded))
        else:
            st.info("Demo documents already loaded.")
    with _comp_dl_col:
        if DEMO_REGULATION_PDF.exists():
            st.download_button(
                "⬇ Regulation",
                data=DEMO_REGULATION_PDF.read_bytes(),
                file_name=DEMO_REGULATION_PDF.name,
                mime="application/pdf",
                key="comp_dl_reg",
            )
        if DEMO_POLICY_PDF.exists():
            st.download_button(
                "⬇ Policy",
                data=DEMO_POLICY_PDF.read_bytes(),
                file_name=DEMO_POLICY_PDF.name,
                mime="application/pdf",
                key="comp_dl_pol",
            )

    st.markdown("**Regulation documents**")
    reg_files = st.file_uploader(
        "Upload regulation PDFs",
        type="pdf",
        accept_multiple_files=True,
        key="reg_uploader",
    )
    if reg_files:
        for file in reg_files:
            if file.name not in st.session_state.compliance_regulation_names:
                dest = os.path.join(COMPLIANCE_UPLOADS_DIR, file.name)
                with open(dest, "wb") as f:
                    f.write(file.read())
                n = ingest_pdf(dest, COMPLIANCE_CHROMA_DIR, COMPLIANCE_COLLECTION, doc_role="regulation")
                if n > 0:
                    st.success(f"{file.name}: {n} chunks indexed")
                else:
                    st.info(f"{file.name}: already indexed")
                st.session_state.compliance_regulation_names.add(file.name)

    st.divider()
    st.markdown("**Policy document**")
    pol_file = st.file_uploader(
        "Upload policy PDF",
        type="pdf",
        accept_multiple_files=False,
        key="pol_uploader",
    )
    if pol_file and pol_file.name != st.session_state.compliance_policy_name:
        dest = os.path.join(COMPLIANCE_UPLOADS_DIR, pol_file.name)
        with open(dest, "wb") as f:
            f.write(pol_file.read())
        n = ingest_pdf(dest, COMPLIANCE_CHROMA_DIR, COMPLIANCE_COLLECTION, doc_role="policy")
        if n > 0:
            st.success(f"{pol_file.name}: {n} chunks indexed")
        else:
            st.info(f"{pol_file.name}: already indexed")
        st.session_state.compliance_policy_name = pol_file.name
        st.session_state.compliance_report = None

    comp_client = chromadb.PersistentClient(path=COMPLIANCE_CHROMA_DIR)
    comp_col = comp_client.get_or_create_collection(COMPLIANCE_COLLECTION)

    reg_count = len(
        (comp_col.get(where={"doc_role": "regulation"}, limit=1) or {}).get("ids") or []
    )
    pol_count = len(
        (comp_col.get(where={"doc_role": "policy"}, limit=1) or {}).get("ids") or []
    )
    st.caption(f"Regulation chunks: {comp_col.count()} total  |  Policy: {'✅' if pol_count else '—'}")

    if st.button("Reset compliance index", key="comp_reset"):
        comp_client.delete_collection(COMPLIANCE_COLLECTION)
        comp_client.get_or_create_collection(COMPLIANCE_COLLECTION)
        _comp_sidecar = os.path.join(COMPLIANCE_CHROMA_DIR, ".embedding_model")
        if os.path.exists(_comp_sidecar):
            os.remove(_comp_sidecar)
        st.session_state.compliance_regulation_names = set()
        st.session_state.compliance_policy_name = None
        st.session_state.compliance_report = None
        st.rerun()

# --- Main tabs ---

st.title("RAG PoC")
tab_qa, tab_compliance = st.tabs(["💬 Q&A", "✅ Compliance Check"])

# ── Q&A tab ──────────────────────────────────────────────────────────────────

with tab_qa:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg["sources"]:
                with st.expander("Sources"):
                    rows = [
                        {
                            "source": doc.metadata.get("source", ""),
                            "page": doc.metadata.get("page", ""),
                            "section_path": doc.metadata.get("section_path", ""),
                            "snippet": doc.page_content[:200] + "...",
                        }
                        for doc in msg["sources"]
                    ]
                    st.dataframe(rows)

    user_query = st.chat_input("Ask a question...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_query)

        history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages[:-1]
        )

        langfuse_handler = get_langfuse_handler()
        trace_config = {}
        if langfuse_handler:
            trace_config = {
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_session_id": st.session_state.session_id,
                    "langfuse_tags": ["rag", "streamlit"],
                },
            }

        sources = init_retriever().invoke(user_query, config=trace_config)

        with st.chat_message("assistant"):
            response = st.write_stream(
                init_chain().stream(
                    {"question": user_query, "history": history}, config=trace_config
                )
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )

        if langfuse_handler:
            from langfuse import get_client
            get_client().flush()

# ── Compliance tab ────────────────────────────────────────────────────────────

with tab_compliance:
    comp_client_tab = chromadb.PersistentClient(path=COMPLIANCE_CHROMA_DIR)
    comp_col_tab = comp_client_tab.get_or_create_collection(COMPLIANCE_COLLECTION)

    reg_meta = comp_col_tab.get(where={"doc_role": "regulation"}, limit=10000).get("metadatas") or []
    reg_sources_tab = sorted({m.get("source", "") for m in reg_meta if m})

    pol_meta = comp_col_tab.get(where={"doc_role": "policy"}, limit=10000).get("metadatas") or []
    pol_sources_tab = sorted({m.get("source", "") for m in pol_meta if m})

    c1, c2, c3 = st.columns(3)
    c1.metric("Regulation docs", len(reg_sources_tab))
    c2.metric("Regulation chunks", len(reg_meta))
    c3.metric("Policy doc", pol_sources_tab[0] if pol_sources_tab else "—")

    has_regulation = bool(reg_sources_tab)
    has_policy = bool(pol_sources_tab)

    if not has_regulation:
        st.info("Upload regulation documents in the sidebar to get started.")
    elif not has_policy:
        st.info("Upload a policy document in the sidebar to run a compliance check.")
    else:
        if st.button("Run Compliance Check", type="primary"):
            progress_bar = st.progress(0, text="Starting...")

            def _on_progress(cur: int, tot: int, msg: str) -> None:
                frac = cur / max(tot, 1)
                progress_bar.progress(frac, text=msg)

            try:
                report = run_compliance_check(
                    COMPLIANCE_CHROMA_DIR,
                    COMPLIANCE_COLLECTION,
                    get_llm(),
                    progress_callback=_on_progress,
                )
                st.session_state.compliance_report = report
            except Exception as exc:
                st.error(f"Compliance check failed: {exc}")
            finally:
                progress_bar.empty()

    report: ComplianceReport = st.session_state.compliance_report
    if report:
        st.subheader("Results")

        _icons = {"COMPLIANT": "✅", "PARTIAL": "⚠️", "NON_COMPLIANT": "❌", "NOT_ADDRESSED": "➖"}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ Compliant", report.summary.get("COMPLIANT", 0))
        m2.metric("⚠️ Partial", report.summary.get("PARTIAL", 0))
        m3.metric("❌ Non-compliant", report.summary.get("NON_COMPLIANT", 0))
        m4.metric("➖ Not addressed", report.summary.get("NOT_ADDRESSED", 0))

        md = render_report_markdown(report)
        st.download_button(
            label="Download report (.md)",
            data=md,
            file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

        st.divider()

        for r in report.results:
            icon = _icons.get(r.status, "❓")
            label = f"{icon} {r.req_id}"
            if r.section:
                label += f" — {r.section}"
            label += f": {r.requirement_text[:90]}{'…' if len(r.requirement_text) > 90 else ''}"
            with st.expander(label):
                st.markdown(f"**Status:** {icon} `{r.status}`")
                st.markdown(f"**Requirement:** {r.requirement_text}")
                if r.evidence:
                    st.markdown(f"**Policy evidence:** _{r.evidence}_")
                if r.gap:
                    st.markdown(f"**Gap:** {r.gap}")
                if r.recommendation:
                    st.markdown(f"**Recommendation:** {r.recommendation}")
                src_col1, src_col2 = st.columns(2)
                if r.regulation_source:
                    src_col1.caption(f"Regulation: {r.regulation_source}")
                if r.policy_source:
                    src_col2.caption(f"Policy: {r.policy_source}")
