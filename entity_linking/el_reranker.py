"""Stage 2: Candidate Reranking for Entity Linking.

Reranks entity candidates from Stage 1 by scoring how well each candidate
description matches the entity's surrounding context in the travelogue text.

Primary: Ollama LLM pointwise scoring (gemma4)
Fallback: Levenshtein + description overlap heuristic (no dependencies)
"""

import re
import requests

_RERANK_PROMPT = """Je beoordeelt kandidaat-matches voor een toponiem uit een 19e-eeuws Nederlands reisverslag. De auteur is een Groningse student die rond 1816 door Duitsland reist. Alle toponiemen zijn locaties (steden, dorpen, rivieren, bergen, gebouwen, pleinen) in Duitsland of aangrenzende gebieden.

Toponiem: "{entity_text}"
Context: "{context}"

Kandidaten:
{candidates_text}

Geef voor elke kandidaat een score van 0 tot 10 voor hoe goed deze past bij het toponiem in deze context.
Antwoord ALLEEN met:
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
) -> list[dict]:
    """Score all candidates in a single Ollama call and keep the top_k."""
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

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": len(candidates) * 8},
            },
            timeout=120,
        )
        data = resp.json()
        answer = data.get("response", "").strip()
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

    candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]


def _heuristic_rerank(
    entity_text: str,
    context: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Fallback reranker using string similarity and context overlap.

    Designed for historical toponyms where the surface form is often an
    archaic spelling that differs from the modern KB label (e.g. "Parys"
    → "Paris", "Cassel" → "Kassel"). Weights context-description overlap
    and geographic domain signals over exact label matching.
    """
    import re

    # Countries/regions in the travelogue's geographic domain
    domain_terms = {
        # Countries
        "germany", "duitsland", "deutschland", "nederland", "netherlands",
        "france", "frankrijk", "belgium", "belgie", "belgië", "luxembourg",
        "luxemburg", "switzerland", "zwitserland", "austria", "oostenrijk",
        "italy", "italie", "italië",
        # German regions/cities (travelogue area)
        "hesse", "hessen", "beieren", "bavaria", "north rhine", "noordrijn",
        "saxony", "saksen", "prussia", "pruisen", "westphalia", "westfalen",
        "berlin", "kassel", "cassel", "aachen", "aken", "cologne", "keulen",
        "munich", "münchen", "frankfurt", "hamburg", "dresden", "leipzig",
        "stuttgart", "nürnberg", "hannover", "bremen",
        # Dutch regions/cities
        "hague", "den haag", "amsterdam", "rotterdam", "utrecht", "groningen",
        "leiden", "haarlem", "dordrecht", "maastricht", "arnhem", "nijmegen",
        # Other signals
        "europe", "europa", "rijn", "rhine", "elbe", "weser", "main",
    }

    context_tokens = set(re.findall(r'\w+', context.lower()))

    def token_overlap(text: str) -> float:
        tokens = set(re.findall(r'\w+', text.lower()))
        if not tokens:
            return 0.0
        return len(context_tokens & tokens) / len(tokens)

    def domain_bonus(description: str) -> float:
        """Bonus for candidates in the travelogue's geographic domain."""
        desc_lower = description.lower()
        for term in domain_terms:
            if term in desc_lower:
                return 1.0
        return 0.0

    for cand in candidates:
        desc = cand.get("description", "")
        label = cand.get("label", "")

        # 1. Name similarity (low weight — archaic spellings differ from modern)
        name_score = _levenshtein_ratio(entity_text, label)
        # Also check if entity_text appears in the description
        if entity_text.lower() in desc.lower():
            name_score = max(name_score, 0.6)
        if label.lower() in entity_text.lower() or entity_text.lower() in label.lower():
            name_score += 0.2

        # 2. Description-context overlap
        desc_overlap = token_overlap(desc)

        # 3. Geographic domain signal (key differentiator for this task)
        #    e.g. "France" → strong signal for a European travelogue
        geo_score = domain_bonus(desc)

        # 4. Penalize candidates with no description (they're uninformative)
        empty_penalty = -0.1 if not desc.strip() else 0.0

        # 5. Source bonus
        source = cand.get("source", "")
        source_bonus = 0.05 if source.startswith("wikidata") else 0.0

        cand["rerank_score"] = float(
            name_score * 0.15
            + desc_overlap * 0.35
            + geo_score * 0.30
            + empty_penalty
            + source_bonus
        )

    candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Character-level similarity based on shared prefix and containment."""
    if not s1 or not s2:
        return 0.0
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    if s1 in s2 or s2 in s1:
        return 0.85
    shorter = min(len(s1), len(s2))
    match = 0
    for i in range(shorter):
        if s1[i] == s2[i]:
            match += 1
        else:
            break
    return match / max(len(s1), len(s2))


def rerank_candidates(
    entity_text: str,
    context: str,
    candidates: list[dict],
    top_k: int = 3,
    ollama_url: str = "http://localhost:11434",
    model_name: str = "gemma4:31b-cloud",
) -> list[dict]:
    """Rerank candidates for an entity mention.

    Args:
        entity_text: The entity surface form.
        context: Surrounding text (e.g. 100 chars each side).
        candidates: List of candidate dicts from generate_candidates().
        top_k: Number of top candidates to retain for LLM selection.
        ollama_url: Ollama server base URL.
        model_name: Ollama model name for reranking.

    Returns:
        Reranked list of at most top_k candidates, each with a "rerank_score".
    """
    if len(candidates) <= 1:
        for c in candidates:
            c["rerank_score"] = 1.0
        return candidates

    # Try Ollama reranker first
    try:
        result = _ollama_rerank(entity_text, context, candidates, top_k,
                                ollama_url=ollama_url, model_name=model_name)
        print(f"  [Stage 2] Ollama reranked {len(candidates)} -> {len(result)}")
        for i, c in enumerate(result):
            print(f"    {i+1}. {c['id']} {c['label']} (score={c['rerank_score']:.3f})")
        return result
    except Exception as e:
        print(f"  [Stage 2] Ollama rerank failed ({e}), falling back to heuristic")

    result = _heuristic_rerank(entity_text, context, candidates, top_k)
    print(f"  [Stage 2] Heuristic reranked {len(candidates)} -> {len(result)}")
    for i, c in enumerate(result):
        print(f"    {i+1}. {c['id']} {c['label']} (score={c['rerank_score']:.3f})")
    return result
