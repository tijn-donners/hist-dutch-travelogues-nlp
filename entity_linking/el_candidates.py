"""Stage 1: Candidate Generation for Entity Linking.

Queries Wikidata and GeoNames APIs using an Entity Profile Generation (EPG)
approach. Instead of blind text matching, an LLM acts as a dense encoder,
predicting modern target properties (modern name, coordinates, region) from 
archaic Dutch context to unlock highly accurate semantic candidate matching.
"""

import json
import re
import requests

_cache: dict[str, list[dict]] = {}
_llm_cache: dict[str, dict] = {}

HEADERS = {
    "User-Agent": "DutchTravelogueNLP/1.0 (https://github.com/tijn-do/hist-dutch-travelogues-nlp; research project)"
}

# Prompting the LLM to act as our dense structural profile generator
_EPG_PROMPT = """Je bent een expert in historische geografie en 19e-eeuwse Nederlandse reisliteratuur.
Analyseer de historische entiteit en context, en voorspel het profiel van de moderne entiteit.

Entiteit (historische spelling): "{entity_text}"
Type: "{entity_label}"
Context uit reisverhaal: "{context}"

Geef het resultaat STRICT terug als een geldig JSON object met de volgende structuur:
{{
  "modern_name": "De huidige gestandaardiseerde internationale of lokale naam",
  "country_or_region": "Het huidige land of de regio waar dit ligt",
  "type_keywords": ["maximaal", "3", "type", "keywords" (bijv: "castle", "city", "mountain")]
}}

Antwoord uitsluitend met de valide JSON. Geen inleiding, geen markdown blocks, geen extra tekst."""


def _wikidata_search(entity_text: str, lang: str = "nl") -> list[dict]:
    """Search Wikidata for an entity string and return candidates."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_text,
        "language": lang,
        "format": "json",
        "limit": 15,
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = resp.json()
    except Exception:
        return []

    candidates = []
    for result in data.get("search", []):
        candidates.append({
            "id": result["id"],
            "label": result.get("label", ""),
            "description": result.get("description", ""),
            "source": "wikidata_text",
        })
    return candidates


def _predict_entity_profile(
    entity_text: str,
    entity_label: str,
    context: str,
    ollama_url: str,
    model_name: str,
) -> dict | None:
    """LLM-based Entity Profile Generation (EPG).
    
    Acts as our conceptual dense vector encoder, turning obscure 19th-century context
    into a structured modern identity block for targeting.
    """
    cache_key = f"{entity_text}|{entity_label}|{context[:300]}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    prompt = _EPG_PROMPT.format(entity_text=entity_text, entity_label=entity_label, context=context)
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150},
            },
            timeout=30,
        )
        raw_response = resp.json().get("response", "").strip()
        
        # Strip markdown code blocks if the LLM accidentally added them
        raw_response = re.sub(r"```(?:json)?\s*|```", "", raw_response).strip()
        profile = json.loads(raw_response)
        
        _llm_cache[cache_key] = profile
        return profile
    except Exception as e:
        print(f"  [Stage 1] EPG Profile Generation failed for '{entity_text}': {e}")
        return None


def _wikidata_hybrid_sparql_search(profile: dict) -> list[dict]:
    """Targeted Wikidata search using the LLM-generated structural profile.

    Searches by modern_name and filters results by location terms in the
    description, avoiding the combined-string trap where "Leipziger Platz
    Kassel" matches no label.
    """
    modern_name = profile.get("modern_name", "")
    location_hint = profile.get("country_or_region", "")

    if not modern_name:
        return []

    # Build SPARQL FILTERs for location terms (exclude broad country names
    # and short tokens that would match too many results).
    _broad_terms = {
        "germany", "deutschland", "duitsland", "france", "frankrijk",
        "netherlands", "nederland", "belgium", "belgie", "belgië",
        "europe", "europa", "austria", "oostenrijk", "switzerland",
        "zwitserland", "italy", "italie", "italië", "luxembourg",
        "luxemburg",
    }
    location_filters = ""
    for term in re.split(r"[,;]\s*", location_hint):
        term = term.strip().lower()
        if term and len(term) > 3 and term not in _broad_terms:
            # Escape single quotes for SPARQL safety
            safe_term = term.replace("'", "\\'")
            location_filters += (
                f'  FILTER(CONTAINS(LCASE(?desc), "{safe_term}")) .\n'
            )

    sparql_query = f"""
    SELECT DISTINCT ?item ?itemLabel ?itemDescription WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch" .
        bd:serviceParam wikibase:endpoint "www.wikidata.org" .
        bd:serviceParam mwapi:search "{modern_name}" .
        bd:serviceParam mwapi:language "en" .
        ?item wikibase:apiOutputItem mwapi:item .
      }}
      ?item schema:description ?desc .
{location_filters}  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "nl,en,de,fr". }}
    }} LIMIT 15
    """

    url = "https://query.wikidata.org/sparql"
    try:
        resp = requests.get(
            url,
            params={"query": sparql_query, "format": "json"},
            headers=HEADERS,
            timeout=15,
        )
        data = resp.json()

        candidates = []
        for row in data.get("results", {}).get("bindings", []):
            qid = row["item"]["value"].split("/")[-1]
            candidates.append({
                "id": qid,
                "label": row.get("itemLabel", {}).get("value", ""),
                "description": row.get("itemDescription", {}).get("value", "No description available"),
                "source": "wikidata_hybrid_epg",
            })
        return candidates
    except Exception as e:
        print(f"  [Stage 1] SPARQL Hybrid lane failed: {e}")
        return []


def _wikipedia_search(profile: dict) -> list[dict]:
    """Search Wikipedia full-text and resolve results to Wikidata IDs.

    Wikipedia's full-text search handles multi-word queries better than
    Wikidata's label-only EntitySearch, making it useful for EPG profiles
    where the modern name + location form a natural search phrase.
    """
    modern_name = profile.get("modern_name", "")
    location_hint = profile.get("country_or_region", "")

    if not modern_name:
        return []

    search_query = f"{modern_name} {location_hint}".strip()

    # Step 1: Wikipedia full-text search
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srlimit": 8,
                "format": "json",
            },
            headers=HEADERS,
            timeout=10,
        )
        results = resp.json().get("query", {}).get("search", [])
    except Exception as e:
        print(f"  [Stage 1] Wikipedia search failed: {e}")
        return []

    if not results:
        return []

    # Step 2: Batch resolve page titles to Wikidata IDs via pageprops
    titles = "|".join(r["title"] for r in results)
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "pageprops",
                "titles": titles,
                "format": "json",
            },
            headers=HEADERS,
            timeout=10,
        )
        pages = resp.json().get("query", {}).get("pages", {})
    except Exception as e:
        print(f"  [Stage 1] Wikipedia pageprops failed: {e}")
        return []

    # Build a lookup of title → wikidata ID
    title_to_qid: dict[str, str] = {}
    for page_id, page_data in pages.items():
        if page_id == "-1":
            continue
        props = page_data.get("pageprops", {})
        qid = props.get("wikibase_item", "")
        if qid:
            title_to_qid[page_data["title"]] = qid

    candidates = []
    for result in results:
        title = result["title"]
        qid = title_to_qid.get(title, "")
        if not qid:
            continue
        snippet = result.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', '')
        candidates.append({
            "id": qid,
            "label": title,
            "description": f"Wikipedia: {snippet}",
            "source": "wikipedia_epg",
        })

    return candidates


def _geonames_profile_search(profile: dict, username: str | None) -> list[dict]:
    """Search GeoNames using modernized target profiles rather than archaic spellings."""
    if not username or not profile.get("modern_name"):
        return []

    url = "http://api.geonames.org/search"
    search_term = f"{profile['modern_name']} {profile.get('country_or_region', '')}".strip()
    
    params = {
        "q": search_term,
        "maxRows": 10,
        "username": username,
        "type": "json",
        "fuzzy": 0.8,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        candidates = []
        for result in data.get("geonames", []):
            candidates.append({
                "id": f"gn:{result['geonameId']}",
                "label": result.get("name", ""),
                "description": (
                    f"{result.get('name', '')}, "
                    f"{result.get('adminName1', '')}, "
                    f"{result.get('countryName', '')}"
                ),
                "source": "geonames_epg",
            })
        return candidates
    except Exception:
        return []


def generate_candidates(
    entity_text: str,
    entity_label: str,
    context: str = "",
    geonames_username: str | None = None,
    ollama_url: str = "http://localhost:11434",
    model_name: str = "gemma4:31b-cloud",
    use_cache: bool = True,
) -> list[dict]:
    """Generate candidate KB entries utilizing Hybrid EPG Generation."""
    cache_key = f"{entity_text}|{entity_label}"
    if use_cache and cache_key in _cache:
        print(f"  [Stage 1] '{entity_text}' -> {len(_cache[cache_key])} candidates (cached)")
        return _cache[cache_key]

    candidates: list[dict] = []
    seen_ids: set[str] = set()

    # 1. Sparse Lane: Try matching the surface form directly against native API first
    for lang in ["nl", "en", "de"]:
        for cand in _wikidata_search(entity_text, lang=lang):
            if cand["id"] not in seen_ids:
                seen_ids.add(cand["id"])
                candidates.append(cand)
                
    print(f"  [Stage 1] Baseline text search for '{entity_text}': {len(candidates)} results")

    # 2. Dense/EPG Lane: Generate profile using surrounding context to unlock structural matches
    if context:
        profile = _predict_entity_profile(
            entity_text, entity_label, context, ollama_url, model_name
        )
        if profile:
            print(f"  [Stage 1] EPG Predicted Profile: {profile}")
            
            # Hybrid search with the predicted anchor targets
            epg_candidates = _wikidata_hybrid_sparql_search(profile)
            new_epg_count = 0
            for cand in epg_candidates:
                if cand["id"] not in seen_ids:
                    seen_ids.add(cand["id"])
                    candidates.append(cand)
                    new_epg_count += 1
            print(f"  [Stage 1] EPG Hybrid lane added {new_epg_count} new structural candidates.")

            # Wikipedia full-text search lane
            wiki_candidates = _wikipedia_search(profile)
            new_wiki_count = 0
            for cand in wiki_candidates:
                if cand["id"] not in seen_ids:
                    seen_ids.add(cand["id"])
                    candidates.append(cand)
                    new_wiki_count += 1
            if new_wiki_count:
                print(f"  [Stage 1] EPG Wikipedia lane added {new_wiki_count} candidates.")

            # GeoNames targeted lane
            gn_candidates = _geonames_profile_search(profile, username=geonames_username)
            new_gn_count = 0
            for cand in gn_candidates:
                if cand["id"] not in seen_ids:
                    seen_ids.add(cand["id"])
                    candidates.append(cand)
                    new_gn_count += 1
            if new_gn_count:
                print(f"  [Stage 1] EPG GeoNames lane added {new_gn_count} candidates.")

    print(f"  [Stage 1] '{entity_text}' -> {len(candidates)} candidates total")
    _cache[cache_key] = candidates
    return candidates


def clear_cache() -> None:
    """Clear all caches."""
    _cache.clear()
    _llm_cache.clear()