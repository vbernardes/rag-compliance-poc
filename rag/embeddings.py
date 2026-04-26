import os

from langchain_core.embeddings import Embeddings
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_ollama import OllamaEmbeddings


def get_model_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "mistral":
        return "mistral-embed"
    return "nomic-embed-text"


def get_embeddings() -> Embeddings:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "mistral":
        return MistralAIEmbeddings(model="mistral-embed")
    return OllamaEmbeddings(model="nomic-embed-text")
