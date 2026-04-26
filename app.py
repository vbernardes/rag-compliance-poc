import os
import uuid
import chromadb
import streamlit as st
from dotenv import load_dotenv
from rag.ingest import ingest_pdf
from rag.retriever import get_retriever
from rag.chain import build_chain, get_langfuse_handler
from rag.embeddings import get_model_name

load_dotenv()

CHROMA_DIR = "./data/chroma"
COLLECTION = "rag_poc"
UPLOADS_DIR = "./uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)

st.set_page_config(page_title="RAG PoC", layout="wide")


@st.cache_resource
def init_retriever():
    return get_retriever(CHROMA_DIR, COLLECTION)


@st.cache_resource
def init_chain():
    retriever = init_retriever()
    return build_chain(retriever)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.sidebar.title("Documents")

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_names:
            dest = os.path.join(UPLOADS_DIR, file.name)
            with open(dest, "wb") as f:
                f.write(file.read())
            n = ingest_pdf(dest, CHROMA_DIR, COLLECTION)
            if n > 0:
                st.sidebar.success(f"{file.name}: {n} chunks indexed")
            else:
                st.sidebar.info(f"{file.name}: already indexed")
            st.session_state.uploaded_names.add(file.name)

client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_or_create_collection(COLLECTION)

_sidecar = os.path.join(CHROMA_DIR, ".embedding_model")
if os.path.exists(_sidecar):
    with open(_sidecar) as _f:
        _stored_model = _f.read().strip()
    if _stored_model != get_model_name():
        client.delete_collection(COLLECTION)
        client.get_or_create_collection(COLLECTION)
        os.remove(_sidecar)
        st.session_state.uploaded_names = set()
        st.cache_resource.clear()
        st.warning(
            f"Embedding model changed ({_stored_model} → {get_model_name()}). "
            "Index was reset — please re-upload your documents."
        )

st.sidebar.metric("Chunks indexed", col.count())

if st.sidebar.button("Reset index"):
    client.delete_collection(COLLECTION)
    client.get_or_create_collection(COLLECTION)
    st.session_state.uploaded_names = set()
    st.cache_resource.clear()
    st.rerun()

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

st.title("RAG PoC")

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
            init_chain().stream({"question": user_query, "history": history}, config=trace_config)
        )

    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})

    if langfuse_handler:
        from langfuse import get_client
        get_client().flush()
