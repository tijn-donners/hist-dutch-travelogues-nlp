"""Stage 3: LLM Candidate Selection for Entity Linking.

Uses an Ollama LLM to select the best KB candidate for each entity mention.
Domain-aware Dutch prompt that describes the travelogue context.
Enriches top candidates with Wikipedia extracts for better disambiguation.
"""

import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ollama_utils import stream_ollama_chat

HEADERS = {
    "User-Agent": "DutchTravelogueNLP/1.0 (https://github.com/tijn-do/hist-dutch-travelogues-nlp; research project)"
}

TEMPERATURE = 0.0


_SELECTION_PROMPT = """Je beoordeelt kandidaat-matches voor een toponiem uit een 19e-eeuws Nederlands reisverslag. De auteur is een Groningse student die rond 1816 door Duitsland reist voor zijn Bildung. Alle entiteiten zijn locaties (steden, dorpen, rivieren, bergen, gebouwen, pleinen) voornamelijk in Duitsland, Nederland of aangrenzende gebieden.

Toponiem: "{entity_text}"
Tekst rondom het toponiem: "{context}"

Kandidaten:
{candidates_text}

Kies de kandidaat die het beste overeenkomt met het toponiem in de gegeven context.
Antwoord ALLEEN met het ID van de gekozen kandidaat.
De kandidaat-match mag GEEN generieke entiteit zijn, maar naar een SPECIFIEKE plaats verwijzen om zo het toponiem te disambigueren.
Als geen enkele kandidaat past of het een generieke kandidaat-match is, antwoord dan: NIL"""


def _enrich_candidates(candidates: list[dict]) -> None:
    """Fetch Wikipedia intro extracts for Wikidata candidates and attach them in-place.

    Resolves Wikidata Q-IDs to English Wikipedia sitelinks via wbgetentities,
    then fetches the summary extract for each matched page. Already-enriched
    candidates and non-Wikidata entries (e.g. gn:*) are skipped. Extracts are
    capped at ~300 words to keep LLM prompts manageable.

    Args:
        candidates: List of candidate dicts. Enrichment is added as a
                    "wikipedia_extract" key on matching dicts in-place.
    """
    # Skip if already enriched or no candidates with Wikidata Q-IDs
    qids = [
        c["id"] for c in candidates
        if c["id"].startswith("Q") and "wikipedia_extract" not in c
    ]
    if not qids:
        return

    # Step 1: Resolve Q-IDs to enwiki titles
    title_by_qid: dict[str, str] = {}
    try:
        resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": "|".join(qids),
                "props": "sitelinks",
                "sitefilter": "enwiki",
                "format": "json",
            },
            headers=HEADERS,
            timeout=15,
        )
        entities = resp.json().get("entities", {})
        for qid, entity in entities.items():
            sitelinks = entity.get("sitelinks", {})
            if "enwiki" in sitelinks:
                title_by_qid[qid] = sitelinks["enwiki"]["title"]
    except Exception as e:
        print(f"  [Stage 3] Wikipedia title resolution failed: {e}")
        return

    if not title_by_qid:
        return

    # Step 2: Fetch extracts for each title
    for qid, title in title_by_qid.items():
        try:
            resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                headers=HEADERS,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                # Cap at ~300 words to keep prompt manageable
                words = extract.split()
                if len(words) > 300:
                    extract = " ".join(words[:300]) + "..."
                # Attach to the matching candidate
                for cand in candidates:
                    if cand["id"] == qid:
                        cand["wikipedia_extract"] = extract
                        break
        except Exception as e:
            print(f"  [Stage 3] Wikipedia extract failed for {title}: {e}")


def _format_candidates(candidates: list[dict]) -> str:
    """Format a candidate list into a readable text block for the LLM prompt.

    Each candidate is numbered and shows its ID, source tag, label, and
    description. Wikipedia extracts are appended when available.

    Args:
        candidates: List of candidate dicts from Stage 2 reranking.

    Returns:
        Formatted multi-line string ready for interpolation into the prompt.
    """
    lines = []
    for i, cand in enumerate(candidates, 1):
        source_tag = {
            "wikidata_text": "[Wikidata]",
            "wikidata_hybrid_epg": "[Wikidata]",
            "wikipedia_epg": "[Wikipedia]",
            "geonames_epg": "[GeoNames]",
        }.get(cand.get("source", ""), "")
        lines.append(
            f"{i}. {cand['id']} {source_tag}\n"
            f"   Naam: {cand['label']}\n"
            f"   Beschrijving: {cand['description']}"
        )
        if cand.get("wikipedia_extract"):
            lines.append(f"   Wikipedia: {cand['wikipedia_extract']}")
    return "\n".join(lines)


def select_candidate(
    entity_text: str,
    context: str,
    candidates: list[dict],
    model_name: str = "gemma4:31b-cloud",
    ollama_url: str = "http://localhost:11434",
    ollama_headers: dict | None = None,
    think: bool | str | None = None,
) -> str:
    """Ask the LLM to select the best candidate for an entity.

    Args:
        entity_text: The entity surface form (e.g. "Cassel").
        context: Surrounding text from the travelogue.
        candidates: Reranked candidate list from rerank_candidates().
        model_name: Ollama model name.
        ollama_url: Ollama server URL.
        ollama_headers: Optional auth headers for cloud API.
        think: Thinking mode (True, False, "low", "medium", "high").
               None uses the model's default.

    Returns:
        The selected candidate ID (e.g. "Q2861") or "NIL" if no match.
    """
    if not candidates:
        print(f"  [Stage 3] No candidates to select from -> NIL")
        return "NIL"

    # Enrich with Wikipedia extracts for better disambiguation
    _enrich_candidates(candidates)

    # Even for a single candidate, verify via LLM — lone Wikipedia results
    # are often false positives (e.g. "Bellevuestraße" → Roland Freisler).
    if len(candidates) == 1:
        solo = candidates[0]
        print(f"  [Stage 3] Single candidate: {solo['id']} {solo['label']} (verifying)")

    print(f"  [Stage 3] Asking LLM to pick from {len(candidates)} candidates...")
    candidates_text = _format_candidates(candidates)
    prompt = _SELECTION_PROMPT.format(
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
            timeout=180.0,
            temperature=TEMPERATURE,
            think=think,
        ).strip()
        print(f"  [Stage 3] LLM raw response: \"{answer[:200]}\"")
    except Exception as e:
        print(f"  [Stage 3] LLM call failed: {e}")
        return "NIL"

    # Extract the ID from the response — look for Q-IDs, geoname IDs, or NIL

    # Match Wikidata Q-ID
    q_match = re.search(r'Q\d+', answer)
    if q_match:
        return q_match.group(0)

    # Match GeoNames ID
    gn_match = re.search(r'gn:\d+', answer)
    if gn_match:
        return gn_match.group(0)

    # Match NIL
    if "NIL" in answer.upper():
        return "NIL"

    # Fallback: look for any candidate ID in the response
    for cand in candidates:
        if cand["id"] in answer:
            return cand["id"]

    # If the response is just a number, map to candidate index
    try:
        idx = int(answer.strip()) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]["id"]
    except ValueError:
        pass

    return "NIL"
