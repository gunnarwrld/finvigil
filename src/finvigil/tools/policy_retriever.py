"""AML and PSD2 regulatory policy RAG retriever with official CELEX/FATF references."""

from __future__ import annotations

import json
import math

from data.seed_ledger import compute_text_embedding
from pydantic import BaseModel
from sqlalchemy import select

from finvigil.db.models import PolicyChunk
from finvigil.db.session import session_scope


class PolicyCitation(BaseModel):
    """Retrieved regulatory guidance chunk with CELEX/FATF reference and cosine similarity."""

    chunk_id: str
    regulation: str
    celex_or_source_ref: str
    section_title: str
    content: str
    similarity_score: float


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return round(dot / (norm_a * norm_b), 4)


def retrieve_aml_policies(
    query: str,
    *,
    top_k: int = 3,
    db_url: str | None = None,
) -> list[PolicyCitation]:
    """Retrieve the top-k relevant EU AMLR, PSD2, FATF, and EU AI Act policy chunks."""
    query_vec = compute_text_embedding(query)

    with session_scope(db_url, readonly=True) as session:
        chunks = list(session.scalars(select(PolicyChunk)).all())

    scored: list[PolicyCitation] = []
    for chunk in chunks:
        doc_vec = [float(x) for x in json.loads(chunk.embedding_json)]
        sim = _cosine_similarity(query_vec, doc_vec)
        scored.append(
            PolicyCitation(
                chunk_id=chunk.chunk_id,
                regulation=chunk.regulation,
                celex_or_source_ref=chunk.celex_or_source_ref,
                section_title=chunk.section_title,
                content=chunk.content,
                similarity_score=sim,
            )
        )

    scored.sort(key=lambda item: item.similarity_score, reverse=True)
    return scored[:top_k]
