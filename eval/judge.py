"""
Deterministic Retrieval/Generation Metrics & LLM-as-a-Judge (`eval/judge.py`).

Implements:
1. `recall_at_k(retrieved_ids, relevant_ids, k)`
2. `mrr_at_k(retrieved_ids, relevant_ids, k)`
3. `ndcg_at_k(retrieved_ids, relevant_ids, k)`
4. `numerical_accuracy(answer, expected_facts, should_abstain)`
5. `citation_precision(answer, relevant_ids, retrieved_ids, should_abstain)`
6. `abstention_accuracy(abstained, answer, should_abstain)`
7. `score_faithfulness(question, context, answer)` using `gemma-4-E4B-it` at `temperature=0`
   with Pydantic v2 schema validation (`FaithfulnessVerdict`).
"""

from __future__ import annotations

import math
import os
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from app.llm import LLMClient, llm_client


class FaithfulnessVerdict(BaseModel):
    """Pydantic v2 schema for LLM-as-a-Judge faithfulness evaluation."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Faithfulness score between 0.0 (hallucinated/unsupported) and 1.0 (100% grounded in context)",
    )
    supported_claims_count: int = Field(default=0, ge=0, description="Number of factual claims supported by context")
    unsupported_claims_count: int = Field(default=0, ge=0, description="Number of factual claims contradicted or unsupported")
    rationale: str = Field(..., description="Concise explanation of grounding assessment")


def _normalize_doc_tag(tag: str) -> str:
    """Normalize `[ADVANC_56-1_2025:c2]` -> `ADVANC_56-1_2025:c2`."""
    return tag.strip().lstrip("[").rstrip("]")


def _doc_prefix(tag: str) -> str:
    """Extract document stem before `:cN`, e.g., `ADVANC_56-1_2025:c2` -> `ADVANC_56-1_2025`."""
    norm = _normalize_doc_tag(tag)
    return norm.split(":")[0]


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 3,
    match_doc_level_for_naive: bool = False,
) -> float:
    """
    Compute Recall@K.
    - For `markdown_section`, exact `doc_id:chunk_id` matching is used.
    - When `match_doc_level_for_naive=True`, checks whether the chunk text/doc covers the gold section.
    """
    if not relevant_ids:
        return 1.0
    top_k = [_normalize_doc_tag(x) for x in retrieved_ids[:k]]
    gold = [_normalize_doc_tag(x) for x in relevant_ids]

    if not top_k:
        return 0.0

    hits = 0
    for g in gold:
        if g in top_k:
            hits += 1
        elif match_doc_level_for_naive and any(_doc_prefix(r) == _doc_prefix(g) for r in top_k):
            hits += 1
    return round(hits / len(gold), 4)


def mrr_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 3,
) -> float:
    """Compute Mean Reciprocal Rank (MRR@K) on exact `doc_id:chunk_id` matches."""
    if not relevant_ids:
        return 1.0
    top_k = [_normalize_doc_tag(x) for x in retrieved_ids[:k]]
    gold_set = {_normalize_doc_tag(x) for x in relevant_ids}

    for idx, r_id in enumerate(top_k, start=1):
        if r_id in gold_set:
            return round(1.0 / idx, 4)
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 3,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K)."""
    if not relevant_ids:
        return 1.0
    top_k = [_normalize_doc_tag(x) for x in retrieved_ids[:k]]
    gold_set = {_normalize_doc_tag(x) for x in relevant_ids}

    dcg = 0.0
    for idx, r_id in enumerate(top_k, start=1):
        if r_id in gold_set:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def _fact_present_in_answer(fact: str, answer: str) -> bool:
    """
    Check whether an expected numerical fact or key entity (e.g. '48,750', '85.0%', '34.84%')
    is present in the generated answer, handling comma formatting and trailing zero variations
    (e.g., '85.0%' vs '85%', '30,132.35' vs '30132.35').
    """
    ans_lower = answer.lower()
    fact_lower = fact.strip().lower()

    if fact_lower in ans_lower:
        return True

    # Strip commas for numeric comparison
    ans_no_comma = ans_lower.replace(",", "")
    fact_no_comma = fact_lower.replace(",", "")
    if fact_no_comma in ans_no_comma:
        return True

    # Handle trailing .0 or .00 in percentages or decimals (e.g., '85.0%' -> '85%', '525.0' -> '525')
    if re.match(r"^\d+\.0+%?$", fact_no_comma):
        stripped = re.sub(r"\.0+(%?)$", r"\1", fact_no_comma)
        if stripped in ans_no_comma:
            return True

    # Handle 2-decimal vs 1-decimal rounding (e.g. '85.2%' vs '85.20%')
    if fact_no_comma.endswith("%"):
        num_part = fact_no_comma[:-1]
        try:
            val = float(num_part)
            if f"{val:.1f}%" in ans_no_comma or f"{val:.2f}%" in ans_no_comma:
                return True
        except ValueError:
            pass
    else:
        try:
            val = float(fact_no_comma)
            if f"{val:.1f}" in ans_no_comma or f"{val:.2f}" in ans_no_comma or f"{int(round(val))}" == fact_no_comma:
                return True
        except ValueError:
            pass

    return False


def numerical_accuracy(
    answer: str,
    expected_facts: List[str],
    should_abstain: bool = False,
) -> float:
    """
    Compute the fraction of gold expected numbers/facts present in `answer`.
    For unanswerable trap queries (`should_abstain=True`), returns 1.0 if the model abstained cleanly
    without fabricating numbers, or 0.0 if it hallucinated a response.
    """
    is_abstain = answer.strip().upper().startswith("ABSTAIN")
    if should_abstain:
        return 1.0 if is_abstain else 0.0

    if is_abstain:
        return 0.0

    if not expected_facts:
        return 1.0

    matched = sum(1 for f in expected_facts if _fact_present_in_answer(f, answer))
    return round(matched / len(expected_facts), 4)


def citation_precision(
    answer: str,
    relevant_ids: List[str],
    retrieved_ids: List[str],
    should_abstain: bool = False,
) -> float:
    """
    Compute inline `[doc_id:chunk_id]` citation precision:
    - For `should_abstain=True`: 1.0 if no fabricated citations are emitted on abstention, else 0.0.
    - For answerable queries: fraction of cited `[doc_id:chunk_id]` tags in `answer` that belong
      to the verified retrieved/gold document set.
    """
    cited_matches = re.findall(r"\[([A-Za-z0-9_\-]+:c\d+)\]", answer)
    is_abstain = answer.strip().upper().startswith("ABSTAIN")

    if should_abstain:
        return 1.0 if (is_abstain and len(cited_matches) == 0) else 0.0

    if not cited_matches:
        return 0.0

    valid_pool = {_normalize_doc_tag(x) for x in (relevant_ids + retrieved_ids)}
    valid_doc_prefixes = {_doc_prefix(x) for x in relevant_ids}

    valid_count = 0
    for c_tag in cited_matches:
        norm_c = _normalize_doc_tag(c_tag)
        if norm_c in valid_pool or _doc_prefix(norm_c) in valid_doc_prefixes:
            valid_count += 1

    return round(valid_count / len(cited_matches), 4)


def abstention_accuracy(
    abstained: bool,
    answer: str,
    should_abstain: bool,
) -> bool:
    """Return True if the system's abstention decision matches `should_abstain`."""
    detected_abstain = bool(abstained or answer.strip().upper().startswith("ABSTAIN"))
    return detected_abstain == should_abstain


def score_faithfulness(
    question: str,
    context: str,
    answer: str,
    should_abstain: bool = False,
    client: Optional[LLMClient] = None,
) -> FaithfulnessVerdict:
    """
    LLM-as-a-Judge faithfulness evaluator using `gemma-4-E4B-it` at `temperature=0`
    with Pydantic v2 `FaithfulnessVerdict` validation and a deterministic groundedness fallback.
    """
    llm = client or llm_client
    is_abstain = answer.strip().upper().startswith("ABSTAIN")

    if should_abstain:
        if is_abstain:
            return FaithfulnessVerdict(
                score=1.0,
                supported_claims_count=1,
                unsupported_claims_count=0,
                rationale="System correctly abstained on an out-of-corpus trap query without hallucinating.",
            )
        return FaithfulnessVerdict(
            score=0.0,
            supported_claims_count=0,
            unsupported_claims_count=1,
            rationale="System failed to abstain on an unanswerable out-of-corpus query.",
        )

    if is_abstain:
        return FaithfulnessVerdict(
            score=0.0,
            supported_claims_count=0,
            unsupported_claims_count=1,
            rationale="System abstained on an answerable in-corpus query.",
        )

    if not context.strip():
        return FaithfulnessVerdict(
            score=0.0,
            supported_claims_count=0,
            unsupported_claims_count=1,
            rationale="No grounding context or tool output was provided to support the answer.",
        )

    # Deterministic overlap & numeric verification baseline for fallback
    ans_numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", answer)
    ctx_clean = context.replace(",", "").lower()
    supported_nums = 0
    for num in ans_numbers:
        n_clean = num.replace(",", "").lower()
        if n_clean in ctx_clean or n_clean.rstrip("%") in ctx_clean:
            supported_nums += 1

    det_ratio = (supported_nums / len(ans_numbers)) if ans_numbers else 0.85
    has_citation = bool(re.search(r"\[[A-Za-z0-9_\-]+:c\d+\]", answer))
    det_score = round(min(1.0, (0.75 * det_ratio) + (0.25 if has_citation else 0.0)), 4)

    fallback_verdict = FaithfulnessVerdict(
        score=det_score,
        supported_claims_count=supported_nums,
        unsupported_claims_count=max(0, len(ans_numbers) - supported_nums),
        rationale=f"Verified {supported_nums}/{len(ans_numbers)} numerical claims and inline citations against retrieved context.",
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if os.getenv("FORCE_OFFLINE_LLM", "0") == "1" or not api_key or api_key.startswith(("sk-local", "sk-your")):
        return fallback_verdict

    messages = [
        {
            "role": "system",
            "content": (
                "You are an institutional Financial Audit Judge. Evaluate whether the ASSISTANT ANSWER "
                "is strictly faithful to and supported by the GROUNDING CONTEXT (Retrieval + SQL/Quant outputs). "
                "Penalize any fabricated numbers or unsupported claims."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\n"
                f"GROUNDING CONTEXT:\n{context[:5000]}\n\n"
                f"ASSISTANT ANSWER:\n{answer}\n"
            ),
        },
    ]
    return llm.structured_completion(
        messages=messages,
        response_model=FaithfulnessVerdict,
        max_retries=1,
        fallback_instance=fallback_verdict,
    )