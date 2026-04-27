import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage

from rag.retriever import get_retriever

_EXTRACTION_SYSTEM = (
    "You are a regulatory compliance expert. Extract every individual, verifiable requirement "
    "from the provided regulation text. Focus on language like \"shall\", \"must\", \"must not\", "
    "\"is required to\", \"is prohibited from\". Ignore definitions, preamble, and interpretive "
    "notes — only enforceable obligations.\n\n"
    "Return ONLY valid JSON — a list of objects with no other text:\n"
    "[{\"req_id\": \"REQ-001\", \"section\": \"<heading or empty string>\", "
    "\"text\": \"<requirement text>\"}]"
)

_ASSESSMENT_SYSTEM = (
    "You are a compliance auditor. Given a regulatory requirement and policy text, assess whether "
    "the policy fulfills the requirement.\n\n"
    "Respond ONLY with valid JSON and no other text:\n"
    "{\"status\": \"COMPLIANT\"|\"PARTIAL\"|\"NON_COMPLIANT\"|\"NOT_ADDRESSED\", "
    "\"evidence\": \"<quote from policy, or empty string>\", "
    "\"gap\": \"<what is missing, or empty string>\", "
    "\"recommendation\": \"<suggested fix, or empty string>\"}\n\n"
    "Status definitions:\n"
    "- COMPLIANT: policy fully satisfies the requirement\n"
    "- PARTIAL: policy addresses the requirement but incompletely\n"
    "- NON_COMPLIANT: policy contradicts or fails to meet the requirement\n"
    "- NOT_ADDRESSED: the policy makes no mention of this requirement"
)

_STATUS_ICONS = {
    "COMPLIANT": "✅",
    "PARTIAL": "⚠️",
    "NON_COMPLIANT": "❌",
    "NOT_ADDRESSED": "➖",
}


@dataclass
class RequirementResult:
    req_id: str
    section: str
    requirement_text: str
    status: str
    evidence: str
    regulation_source: str
    policy_source: str
    gap: str
    recommendation: str


@dataclass
class ComplianceReport:
    regulation_docs: List[str]
    policy_doc: str
    generated_at: str
    results: List[RequirementResult]
    summary: dict = field(default_factory=dict)


def _parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}")


def _regulation_source(chroma_persist_dir: str, collection_name: str, req_text: str) -> str:
    try:
        retriever = get_retriever(
            chroma_persist_dir, collection_name, where={"doc_role": "regulation"}
        )
        retriever.search_kwargs["k"] = 1
        retriever.search_kwargs["fetch_k"] = 5
        docs = retriever.invoke(req_text)
        if docs:
            m = docs[0].metadata
            return f"{m.get('source', '')}, p.{m.get('page', '')}"
    except Exception:
        pass
    return ""


def extract_requirements(
    chroma_persist_dir: str, collection_name: str, llm
) -> List[dict]:
    retriever = get_retriever(
        chroma_persist_dir, collection_name, where={"doc_role": "regulation"}
    )
    extraction_k = int(os.getenv("COMPLIANCE_EXTRACTION_K", "20"))
    retriever.search_kwargs["k"] = extraction_k
    retriever.search_kwargs["fetch_k"] = extraction_k * 3

    docs = retriever.invoke("requirements obligations shall must prohibited")
    if not docs:
        return []

    context = "\n\n".join(
        f"[{doc.metadata.get('source', '')}, p.{doc.metadata.get('page', '')}]\n{doc.page_content}"
        for doc in docs
    )

    response = llm.invoke([
        SystemMessage(content=_EXTRACTION_SYSTEM),
        HumanMessage(content=f"Regulation text:\n\n{context}"),
    ])

    requirements = _parse_json(response.content)
    if not isinstance(requirements, list):
        raise ValueError("Expected a JSON list of requirements from the LLM")

    for i, req in enumerate(requirements):
        req["req_id"] = f"REQ-{i + 1:03d}"
        req.setdefault("section", "")
        req.setdefault("text", "")

    return requirements


def assess_requirement(
    req: dict,
    chroma_persist_dir: str,
    collection_name: str,
    llm,
) -> RequirementResult:
    policy_retriever = get_retriever(
        chroma_persist_dir, collection_name, where={"doc_role": "policy"}
    )
    policy_retriever.search_kwargs["k"] = int(os.getenv("TOP_K", "5"))
    policy_retriever.search_kwargs["fetch_k"] = 20

    policy_docs = policy_retriever.invoke(req["text"])

    if not policy_docs:
        return RequirementResult(
            req_id=req["req_id"],
            section=req.get("section", ""),
            requirement_text=req["text"],
            status="NOT_ADDRESSED",
            evidence="",
            regulation_source=_regulation_source(chroma_persist_dir, collection_name, req["text"]),
            policy_source="",
            gap="No relevant content found in the policy document.",
            recommendation="Add a section explicitly addressing this requirement.",
        )

    policy_context = "\n\n".join(
        f"[{doc.metadata.get('source', '')}, p.{doc.metadata.get('page', '')}]\n{doc.page_content}"
        for doc in policy_docs
    )
    policy_source = "; ".join(
        f"{doc.metadata.get('source', '')}, p.{doc.metadata.get('page', '')}"
        for doc in policy_docs[:2]
    )

    response = llm.invoke([
        SystemMessage(content=_ASSESSMENT_SYSTEM),
        HumanMessage(
            content=f"Requirement ({req['req_id']}): {req['text']}\n\nPolicy text:\n\n{policy_context}"
        ),
    ])

    try:
        result = _parse_json(response.content)
    except ValueError:
        result = {
            "status": "NOT_ADDRESSED",
            "evidence": "",
            "gap": "Could not parse LLM assessment.",
            "recommendation": "",
        }

    return RequirementResult(
        req_id=req["req_id"],
        section=req.get("section", ""),
        requirement_text=req["text"],
        status=result.get("status", "NOT_ADDRESSED"),
        evidence=result.get("evidence", ""),
        regulation_source=_regulation_source(chroma_persist_dir, collection_name, req["text"]),
        policy_source=policy_source,
        gap=result.get("gap", ""),
        recommendation=result.get("recommendation", ""),
    )


def run_compliance_check(
    chroma_persist_dir: str,
    collection_name: str,
    llm,
    progress_callback: Optional[Callable] = None,
) -> ComplianceReport:
    client = chromadb.PersistentClient(path=chroma_persist_dir)
    col = client.get_or_create_collection(collection_name)

    reg_meta = col.get(where={"doc_role": "regulation"}, limit=10000).get("metadatas") or []
    reg_sources = sorted({m.get("source", "") for m in reg_meta if m})

    pol_meta = col.get(where={"doc_role": "policy"}, limit=10000).get("metadatas") or []
    pol_sources = sorted({m.get("source", "") for m in pol_meta if m})

    if progress_callback:
        progress_callback(0, 1, "Extracting requirements from regulation...")

    requirements = extract_requirements(chroma_persist_dir, collection_name, llm)

    results = []
    total = len(requirements)
    for i, req in enumerate(requirements):
        if progress_callback:
            progress_callback(i, total, f"Assessing {req['req_id']} ({i + 1}/{total})...")
        results.append(assess_requirement(req, chroma_persist_dir, collection_name, llm))

    if progress_callback:
        progress_callback(total, total, "Done.")

    summary = {k: 0 for k in _STATUS_ICONS}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1

    return ComplianceReport(
        regulation_docs=reg_sources,
        policy_doc=", ".join(pol_sources),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        results=results,
        summary=summary,
    )


def render_report_markdown(report: ComplianceReport) -> str:
    lines = [
        "# Compliance Report",
        "",
        f"**Regulation:** {', '.join(report.regulation_docs) or 'N/A'}",
        f"**Policy:** {report.policy_doc or 'N/A'}",
        f"**Generated:** {report.generated_at}",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for status, icon in _STATUS_ICONS.items():
        lines.append(f"| {icon} {status} | {report.summary.get(status, 0)} |")

    lines += ["", "---", "", "## Requirements", ""]

    for r in report.results:
        icon = _STATUS_ICONS.get(r.status, "❓")
        section_str = f" — {r.section}" if r.section else ""
        lines += [
            f"### {r.req_id}{section_str}",
            "",
            f"**Requirement:** {r.requirement_text}",
            "",
            f"**Status:** {icon} {r.status}",
            "",
        ]
        if r.evidence:
            lines += [f"**Policy evidence:** _{r.evidence}_", ""]
        if r.regulation_source:
            lines += [f"**Regulation source:** {r.regulation_source}", ""]
        if r.policy_source:
            lines += [f"**Policy source:** {r.policy_source}", ""]
        if r.gap:
            lines += [f"**Gap:** {r.gap}", ""]
        if r.recommendation:
            lines += [f"**Recommendation:** {r.recommendation}", ""]
        lines += ["---", ""]

    return "\n".join(lines)
