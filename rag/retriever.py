import os

from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from rag.embeddings import get_embeddings


def get_retriever(chroma_persist_dir: str, collection_name: str) -> VectorStoreRetriever:
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=chroma_persist_dir,
    )
    top_k = int(os.getenv("TOP_K", "5"))
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": 20},
    )
