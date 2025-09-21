"""Utility functions for querying the Infini-gram API.

Lightweight helper around the public HTTPS endpoint documented at
https://infini-gram.readthedocs.io/en/latest/pkg.html

Dependencies: requests (already in pyproject)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.infini-gram.io/"


@dataclass
class InfiniDoc:
    rank: int
    text: str
    score: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "meta": self.meta or {},
        }


def _post_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = requests.post(BASE_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Infini-gram request failed: {e}") from e


def get_documents(
    query: str,
    *,
    index: str = "v4_rpj_llama_s4",
    max_docs: int = 5,
    max_disp_len: int = 300,
    query_type: str = "get_doc_by_rank",
    start_rank: int = 0,
) -> List[InfiniDoc]:
    """Retrieve documents related to a query/prompt.

    Parameters
    ----------
    query: str
        Natural language query or prompt.
    index: str
        Index name (default matches documentation examples).
    max_docs: int
        Max number of sequential ranks to attempt fetching (early stop if none).
    max_disp_len: int
        Maximum length of displayed document text.
    query_type: str
        Infini-gram query_type (default: get_doc_by_rank).
    start_rank: int
        Rank offset to begin from.

    Returns
    -------
    list[InfiniDoc]
    """
    docs: List[InfiniDoc] = []

    # First, request the rank segment boundaries (s=0 gets just bounds per docs)
    segment_payload = {
        "index": index,
        "query_type": "get_rank_segment",
        "query": query,
        "s": 0,
    }
    segment = _post_json(segment_payload)
    try:
        start, end = segment["segment_by_shard"][0]
    except Exception as e:
        logger.warning("Unexpected segment structure: %s", segment)
        raise RuntimeError("Failed to parse segment boundaries") from e

    for r in range(start_rank + start, min(end, start_rank + start + max_docs)):
        payload = {
            "index": index,
            "query_type": query_type,
            "query": query,
            "s": 0,
            "rank": r,
            "max_disp_len": max_disp_len,
        }
        data = _post_json(payload)
        # Expected keys: 'doc', 'score', maybe others (robust fallback)
        doc_text = data.get("doc") or data.get("text") or ""
        score = data.get("score")
        docs.append(InfiniDoc(rank=r, text=doc_text, score=score, meta=data))

    return docs


def get_documents_texts(query: str, **kwargs) -> List[str]:
    """Convenience wrapper returning only document texts."""
    return [d.text for d in get_documents(query, **kwargs)]


if __name__ == "__main__":  # Simple quick manual test
    for d in get_documents("natural language processing", max_docs=3):
        print(d.rank, (d.text[:120] + "...") if len(d.text) > 120 else d.text)
