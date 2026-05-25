"""Stage 3: LLM Candidate Selection for Entity Linking.

Uses an Ollama LLM to select the best KB candidate for each entity mention.
Domain-aware Dutch prompt that describes the travelogue context.
Enriches top candidates with Wikipedia extracts for better disambiguation.
"""

import re
import requests

HEADERS = {
    "User-Agent": "DutchTravelogueNLP/1.0 (https://github.com/tijn-do/hist-dutch-travelogues-nlp; research project)"
}


_SELECTION_PROMPT = """Je beoordeelt kandidaat-matches voor een toponiem uit een 19e-eeuws Nederlands reisverslag. De auteur is een Groningse student die rond 1816 door Duitsland reist voor zijn Bildung. Alle entiteiten zijn locaties (steden, dorpen, rivieren, bergen, gebouwen, pleinen) voornamelijk in Duitsland, Nederland of aangrenzende gebieden.

Toponiem: "{entity_text}"
Tekst rondom het toponiem: "{context}"

Kandidaten:
{candidates_text}

Kies de kandidaat die het beste overeenkomt met het toponiem in de gegeven context. Antwoord ALLEEN met het ID van de gekozen kandidaat. Als geen enkele kandidaat past, antwoord dan: NIL"""


def _enrich_candidates(candidates: list[dict]) -> None:
    """Fetch Wikipedia extracts for candidates and attach them in-place.

    Resolves Wikidata Q-IDs to English Wikipedia sitelinks, then fetches
    the intro extract for each. Already-enriched candidates and non-Wikidata
    candidates are skipped.
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
    """Format candidates list for the LLM prompt."""
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
    ollama_base_url: str = "http://localhost:11434",
) -> str:
    """Ask the LLM to select the best candidate for an entity.

    Args:
        entity_text: The entity surface form (e.g. "Cassel").
        context: Surrounding text from the travelogue.
        candidates: Reranked candidate list from rerank_candidates().
        model_name: Ollama model name.
        ollama_base_url: Ollama server URL.

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

    try:
        resp = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=180,
        )
        data = resp.json()
        answer = data.get("response", "").strip()
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
