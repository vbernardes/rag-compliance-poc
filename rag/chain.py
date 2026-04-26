import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

load_dotenv()


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "mistral-nemo"), temperature=0)
    elif provider == "anthropic":
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    elif provider == "mistral":
        return ChatMistralAI(model=os.getenv("MISTRAL_MODEL", "open-mistral-nemo"), temperature=0)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _format_docs(docs):
    parts = []
    for doc in docs:
        parts.append(
            f"[source: {doc.metadata['source']}, page {doc.metadata['page']}, section: {doc.metadata['section_path']}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def build_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say 'I don't have enough information to answer that.' Always cite the source file and page number when you use information from the context."),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nConversation history:\n{history}\n\nQuestion: {question}"
        ),
    ])

    llm = get_llm()

    retrieval = RunnableParallel(
        context=(lambda x: x["question"]) | retriever | _format_docs,
        question=RunnablePassthrough() | (lambda x: x["question"]),
        history=RunnablePassthrough() | (lambda x: x["history"]),
    )

    return retrieval | prompt | llm | StrOutputParser()
