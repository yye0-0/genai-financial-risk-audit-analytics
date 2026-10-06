from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PATHS


class PolicyRetriever:
    """Small auditable TF-IDF retriever without an external vector service."""

    def __init__(self, path: Path = PATHS.policies):
        self.documents = json.loads(path.read_text(encoding="utf-8"))
        corpus = [doc["title"] + " " + doc["text"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(-scores)[:top_k]
        return [{**self.documents[index], "retrieval_score": float(scores[index])}
                for index in order if scores[index] > 0]

    def by_ids(self, policy_ids: list[str]) -> list[dict]:
        ids = set(policy_ids)
        return [doc for doc in self.documents if doc["id"] in ids]


def retrieve_for_flags(policy_ids: list[str]) -> list[dict]:
    # Exact control-to-policy mapping is safer than semantic retrieval when a
    # deterministic rule already carries the policy identifier.
    return PolicyRetriever().by_ids(policy_ids)


def evaluate_retrieval(cases: list[dict], top_k: int = 3) -> dict:
    retriever = PolicyRetriever()
    hits_at_1 = hits_at_k = reciprocal_rank = 0.0
    details = []
    for case in cases:
        results = retriever.retrieve(case["query"], top_k=top_k)
        ids = [item["id"] for item in results]
        expected = case["policy_id"]
        rank = ids.index(expected) + 1 if expected in ids else None
        hits_at_1 += int(rank == 1)
        hits_at_k += int(rank is not None)
        reciprocal_rank += 1 / rank if rank else 0
        details.append({"query": case["query"], "expected": expected,
                        "retrieved": ids, "rank": rank})
    n = max(1, len(cases))
    return {"cases": len(cases), "hit_at_1": hits_at_1 / n,
            f"hit_at_{top_k}": hits_at_k / n, "mrr": reciprocal_rank / n,
            "details": details}

