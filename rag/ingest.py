import json
import os
import re
import statistics
from collections import Counter

import chromadb
import fitz
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import get_embeddings, get_model_name

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _ENTITY_LABELS = {"PERSON", "ORG", "DATE", "GPE", "LOC"}
except Exception:
    _nlp = None


def ingest_pdf(
    file_path: str,
    chroma_persist_dir: str,
    collection_name: str,
    doc_role: str = "knowledge",
) -> int:
    """
    Ingest a PDF into ChromaDB. Returns number of chunks added.
    Returns 0 if the file is already indexed (deduplication by filename).
    doc_role tags each chunk: "knowledge" (Q&A default), "regulation", or "policy".
    """
    client = chromadb.PersistentClient(path=chroma_persist_dir)
    collection = client.get_or_create_collection(collection_name)

    basename = os.path.basename(file_path)
    existing = collection.get(where={"source": basename}, limit=1)
    if existing and existing["ids"]:
        return 0

    doc_fitz = fitz.open(file_path)
    doc_title = doc_fitz.metadata.get("title", basename) or basename

    page_data = []
    for page_num, page in enumerate(doc_fitz, start=1):
        blocks = page.get_text("dict")["blocks"]

        all_sizes = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    if size > 0:
                        all_sizes.append(size)

        median_size = statistics.median(all_sizes) if all_sizes else 0
        heading_threshold = median_size * 1.2

        section_stack = []
        heading_sizes = []

        text_parts = []
        raw_lines = []
        section_title = ""
        section_path = ""

        for block in blocks:
            if block.get("type") != 0:
                continue

            lines = block.get("lines", [])
            block_text = " ".join(
                span.get("text", "")
                for line in lines
                for span in line.get("spans", [])
            ).strip()

            if not block_text:
                continue

            max_span_size = max(
                (span.get("size", 0) for line in lines for span in line.get("spans", [])),
                default=0,
            )

            if max_span_size >= heading_threshold:
                rank = 0
                for i, sz in enumerate(sorted(set(heading_sizes), reverse=True)):
                    if max_span_size >= sz:
                        rank = i
                        break
                heading_sizes.append(max_span_size)

                while len(section_stack) > rank:
                    section_stack.pop()
                if len(section_stack) == rank:
                    if section_stack:
                        section_stack.pop()
                section_stack.append(block_text)

                section_title = section_stack[-1] if section_stack else ""
                section_path = " > ".join(section_stack)
            else:
                text_parts.append(block_text)
                for line in lines:
                    line_text = " ".join(span.get("text", "") for span in line.get("spans", []))
                    if line_text.strip():
                        raw_lines.append(line_text.strip())

        raw_text = "\n".join(raw_lines)
        page_data.append({
            "page": page_num,
            "text_parts": text_parts,
            "raw_text": raw_text,
            "section_title": section_title,
            "section_path": section_path,
        })

    first_lines = [p["raw_text"].split("\n")[0] for p in page_data if p["raw_text"]]
    last_lines = [p["raw_text"].split("\n")[-1] for p in page_data if p["raw_text"]]
    total_pages = len(page_data)
    repeated = set()
    if total_pages > 0:
        threshold = 0.8 * total_pages
        all_boundary_lines = first_lines + last_lines
        counts = Counter(all_boundary_lines)
        for line, count in counts.items():
            if line and count >= threshold:
                repeated.add(line)

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_docs = []
    chunk_index = 0

    for page in page_data:
        raw = " ".join(page["text_parts"])
        raw = raw.replace("\xad", "")
        raw = re.sub(r"\s+", " ", raw).strip()

        for bad in repeated:
            raw = raw.replace(bad, "")
        raw = re.sub(r"\s+", " ", raw).strip()

        if not raw:
            continue

        chunks = splitter.split_text(raw)
        for chunk_text in chunks:
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": basename,
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "section_title": page["section_title"] or "",
                    "section_path": page["section_path"] or "",
                    "doc_title": doc_title,
                    "doc_role": doc_role,
                    "entities": "[]",
                },
            )
            all_docs.append(doc)
            chunk_index += 1

    if _nlp is not None:
        for doc in all_docs:
            spacy_doc = _nlp(doc.page_content)
            ents = [e for e in spacy_doc.ents if e.label_ in _ENTITY_LABELS]
            doc.metadata["entities"] = json.dumps(
                [{"text": e.text, "label": e.label_} for e in ents]
            )

    if not all_docs:
        return 0

    embedding = get_embeddings()
    Chroma.from_documents(
        all_docs,
        embedding,
        persist_directory=chroma_persist_dir,
        collection_name=collection_name,
    )

    sidecar = os.path.join(chroma_persist_dir, ".embedding_model")
    with open(sidecar, "w") as f:
        f.write(get_model_name())

    return len(all_docs)
