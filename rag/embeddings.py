import os

from langchain_core.embeddings import Embeddings
from langchain_mistralai.embeddings import MistralAIEmbeddings


def get_model_name() -> str:
    return "mistral-embed"


def get_embeddings() -> Embeddings:
    return MistralAIEmbeddings(model="mistral-embed")
