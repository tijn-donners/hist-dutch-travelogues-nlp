"""Stage 2: Candidate Reranking for Entity Linking.

Reranks entity candidates from Stage 1 by scoring how well each candidate
description matches the entity's surrounding context in the travelogue text.

Uses Ollama LLM pointwise scoring (gemma4) — no heuristic fallback.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ollama_utils import stream_ollama_chat

TEMPERATURE = 0.0

_RERANK_PROMPT = """Je beoordeelt kandidaat-matches voor een toponiem uit een 19e-eeuws Nederlands reisverslag. De auteur is een Groningse student die rond 1816 door Duitsland reist. Alle toponiemen zijn locaties (steden, dorpen, rivieren, bergen, gebouwen, pleinen) in Duitsland of aangrenzende gebieden.

Toponiem: "{entity_text}"
Context: "{context}"

Kandidaten:
{candidates_text}

Geef voor elke kandidaat een score van 0 tot 10 voor hoe goed deze past bij het toponiem in deze context.
Antwoord ALLEEN met ZONDER MARKDOWN ELEMENTEN en ZONDER TEKST DAARVOOR OF ER NAAR, maar ALLEEN HET ONDERSTAANDE FORMAT:
1: <score>
2: <score>
..."""


def _ollama_rerank(
    entity_text: str,
    context: str,
    candidates: list[dict],
    top_k: int = 3,
    ollama_url: str = "http://localhost:11434",
    model_name: str = "gemma4:31b-cloud",
    ollama_headers: dict | None = None,
    think: bool | str | None = None,
) -> list[dict]:
    """Score all candidates in a single Ollama call and keep the top_k.

    Sends a batch scoring prompt asking the LLM to rate each candidate 0-10
    for fit with the entity and context. Parses numbered scores from the
    response with multi-format fallback logic.

    Args:
        entity_text: The entity surface form.
        context: Surrounding text from the travelogue.
        candidates: List of candidate dicts from Stage 1.
        top_k: Number of top candidates to retain.
        ollama_url: Ollama server base URL.
        model_name: Ollama model name.
        ollama_headers: Optional auth headers for cloud API.
        think: Thinking mode for the LLM (True, False, "low", "medium", "high").
               None (default) uses the model's default thinking behavior.
               Set to False or "low" for thinking models that over-think on
               simple scoring tasks (e.g. kimi-k2.7-code).

    Returns:
        Top-k candidates sorted by rerank_score (descending).

    Raises:
        Exception: Propagated up on failure (no fallback).
    """
    lines = []
    for i, cand in enumerate(candidates, 1):
        lines.append(
            f"{i}. [{cand['id']}] {cand['label']}\n"
            f"   Beschrijving: {cand['description']}"
        )
    candidates_text = "\n".join(lines)

    prompt = _RERANK_PROMPT.format(
        entity_text=entity_text,
        context=context,
        candidates_text=candidates_text,
    )

    api_key = None
    if ollama_headers:
        auth = ollama_headers.get("Authorization", "")
        api_key = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else None
    try:
        answer = stream_ollama_chat(
            model=model_name,
            prompt=prompt,
            host=ollama_url,
            api_key=api_key,
            timeout=120.0,
            temperature=TEMPERATURE,
            think=think,
        ).strip()
    except Exception as e:
        print(f"    [Stage 2] Ollama batch rerank failed: {e}")
        raise

    # Parse numbered scores: "1: 8.5", "1) 8", "1. 8.5", "1 - 8", etc.
    for line in answer.split("\n"):
        line = line.strip()
        match = re.match(r"(\d+)\s*[):.\-_]\s*(\d+(?:[.,]\d+)?)", line)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidates):
                score_str = match.group(2).replace(",", ".")
                candidates[idx]["rerank_score"] = float(score_str) / 10.0

    # Fallback: bare numbers on separate lines, one per candidate
    if all(c.get("rerank_score", 0) == 0 for c in candidates):
        bare_scores = []
        for line in answer.split("\n"):
            bare_match = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*$", line.strip())
            if bare_match:
                bare_scores.append(float(bare_match.group(1).replace(",", ".")))
        for i, score in enumerate(bare_scores):
            if i < len(candidates):
                candidates[i]["rerank_score"] = score / 10.0

    # Default 0 for any candidate the LLM didn't score
    for cand in candidates:
        cand.setdefault("rerank_score", 0.0)

    # Warn if most candidates weren't scored — likely a truncated response
    # (thinking models can exhaust num_predict/context before producing output)
    scored = sum(1 for c in candidates if c.get("rerank_score", 0) > 0)
    if scored < len(candidates) / 2 and len(candidates) > 2:
        print(f"    [Stage 2] ⚠ Only {scored}/{len(candidates)} candidates scored "
              f"— response may be truncated (thinking model?). "
              f"Consider --think false or --think low.")

    candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]


def rerank_candidates(
    entity_text: str,
    context: str,
    candidates: list[dict],
    top_k: int = 3,
    ollama_url: str = "http://localhost:11434",
    model_name: str = "gemma4:31b-cloud",
    ollama_headers: dict | None = None,
    think: bool | str | None = None,
) -> list[dict]:
    """Rerank candidates for an entity mention.

    Args:
        entity_text: The entity surface form.
        context: Surrounding text (e.g. 100 chars each side).
        candidates: List of candidate dicts from generate_candidates().
        top_k: Number of top candidates to retain for LLM selection.
        ollama_url: Ollama server base URL.
        model_name: Ollama model name for reranking.
        think: Thinking mode (True, False, "low", "medium", "high").
               None uses the model's default. Set to False or "low" for
               thinking models that over-think on simple scoring tasks.

    Returns:
        Reranked list of at most top_k candidates, each with a "rerank_score".
    """
    if len(candidates) <= 1:
        for c in candidates:
            c["rerank_score"] = 1.0
        return candidates

    # Ollama reranker (required — no fallback, to keep LLM-only evaluation clean)
    result = _ollama_rerank(entity_text, context, candidates, top_k,
                            ollama_url=ollama_url, model_name=model_name,
                            ollama_headers=ollama_headers, think=think)
    print(f"  [Stage 2] {model_name} reranked {len(candidates)} -> {len(result)}")
    for i, c in enumerate(result):
        print(f"    {i+1}. {c['id']} {c['label']} (score={c['rerank_score']:.3f})")
    return result
