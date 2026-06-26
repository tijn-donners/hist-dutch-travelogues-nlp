"""Stage 1: Candidate Generation for Entity Linking.

Queries Wikidata and GeoNames APIs using an Entity Profile Generation (EPG)
approach. Instead of blind text matching, an LLM acts as a dense encoder,
predicting modern target properties (modern name, coordinates, region) from 
archaic Dutch context to unlock highly accurate semantic candidate matching.
"""

import json
import re
import sys
from pathlib import Path

import requests
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ollama_utils import stream_ollama_chat

_cache: dict[str, list[dict]] = {}
_llm_cache: dict[str, dict] = {}

# Load GeoNames gazetteer for Germany at module startup
_GAZETTEER_PATH = Path(__file__).resolve().parent.parent / "entity_linking" / "GeoNames_DE_gazetteer.txt"
_GAZETTEER_DATA = []  # List of (geonameid, name, asciiname, alternatenames_list, lat, lon, feature_class, feature_code, country_code, admin1_code, population, elevation)

if _GAZETTEER_PATH.exists():
    with open(_GAZETTEER_PATH, encoding="utf8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 11:
                geonameid = parts[0]
                name = parts[1]
                asciiname = parts[2]
                alternatenames = parts[3].split(",") if parts[3] else []
                # Parse numeric fields, handling empty values
                try:
                    lat = float(parts[4]) if parts[4] else 0.0
                except ValueError:
                    lat = 0.0
                try:
                    lon = float(parts[5]) if parts[5] else 0.0
                except ValueError:
                    lon = 0.0
                feature_class = parts[6] if len(parts) > 6 else ""
                feature_code = parts[7] if len(parts) > 7 else ""
                country_code = parts[8] if len(parts) > 8 else ""
                admin1_code = parts[10] if len(parts) > 10 and parts[10] else ""
                try:
                    population = int(parts[14]) if len(parts) > 14 and parts[14] else 0
                except ValueError:
                    population = 0
                try:
                    elevation = int(parts[15]) if len(parts) > 15 and parts[15] else 0
                except ValueError:
                    elevation = 0

                _GAZETTEER_DATA.append((
                    geonameid, name, asciiname, alternatenames, lat, lon,
                    feature_class, feature_code, country_code, admin1_code,
                    population, elevation
                ))
else:
    print(f"Warning: GeoNames gazetteer not found at {_GAZETTEER_PATH}")

# Human-readable labels for GeoNames feature class codes
_FEATURE_CLASS_LABELS = {
    "P": "populated place",
    "A": "administrative region",
    "H": "water feature",
    "L": "lake",
    "T": "mountain/hill",
    "S": "spot/building",
    "R": "road/railroad",
    "V": "forest/vegetation",
    "U": "undersea",
}

# Human-readable labels for common GeoNames feature codes (DE-relevant subset)
_FEATURE_CODE_LABELS = {
    "PPL": "city/town/village",
    "PPLA": "administrative seat",
    "PPLA2": "district seat",
    "PPLA3": "municipality seat",
    "PPLA4": "borough seat",
    "PPLC": "capital city",
    "PPLCH": "historical capital",
    "PPLF": "farm/village",
    "PPLG": "seat of government",
    "PPLH": "historical populated place",
    "PPLQ": "abandoned place",
    "PPLR": "religious populated place",
    "PPLS": "populated locality",
    "PPLW": "destroyed place",
    "PPLX": "section of populated place",
    "ST": "stream",
    "STM": "stream",
    "STMI": "intermittent stream",
    "STMR": "meandering stream",
    "STMSB": "stream bend",
    "LK": "lake",
    "LKI": "intermittent lake",
    "LKN": "salt lake",
    "RSV": "reservoir",
    "RSVT": "water tank",
    "CNL": "canal",
    "DTCH": "ditch",
    "MT": "mountain",
    "MTT": "mountain range",
    "HLL": "hill",
    "HLLS": "hills",
    "PK": "peak",
    "RK": "rock",
    "VLC": "volcano",
    "VAL": "valley",
    "PLN": "plain",
    "FLD": "field",
    "FRM": "farm",
    "CH": "church",
    "CHS": "church (historical)",
    "MNA": "monastery",
    "MN": "mine",
    "MUS": "museum",
    "HTL": "hotel",
    "INN": "inn",
    "CAST": "castle",
    "PAL": "palace",
    "FT": "fort",
    "BTL": "battlefield",
    "PARK": "park",
    "GDN": "garden",
    "SQ": "square",
    "BRDG": "bridge",
    "RSTN": "railroad station",
    "RSTP": "railroad stop",
    "BD": "border post",
    "PRK": "parking area",
    "PYR": "pyramid",
    "TOWR": "tower",
    "MNMT": "monument",
    "FCL": "facility",
    "HSP": "hospital",
    "SCH": "school",
    "UNIV": "university",
    "THTR": "theater",
    "LCTY": "locality",
    "AREA": "area",
    "RG": "region",
    "RGN": "region",
    "ISL": "island",
    "PEN": "peninsula",
    "PT": "point",
    "BAY": "bay",
    "GULF": "gulf",
    "STRT": "strait",
    "CHN": "channel",
    "HBR": "harbor",
    "BCH": "beach",
    "CLF": "cliff",
    "CAPE": "cape",
    "FRST": "forest",
    "PRT": "port",
    "RUIN": "ruin",
    "SHRN": "shrine",
    "STM": "stream",
    "WLL": "well",
    "SPNG": "spring",
    "FLLS": "waterfall",
    "GLCR": "glacier",
    "DAM": "dam",
    "PIER": "pier",
    "WHF": "wharf",
    "LTHSE": "lighthouse",
    "AIRP": "airport",
    "CMP": "camp",
    "CSTL": "castle",
    "PAL": "palace",
}

# German admin1 (state) code mapping (as used in this GeoNames export)
_ADMIN1_DE = {
    "01": "Baden-Württemberg",
    "02": "Bayern",
    "03": "Bremen",
    "04": "Hamburg",
    "05": "Hessen",
    "06": "Niedersachsen",
    "07": "Nordrhein-Westfalen",
    "08": "Rheinland-Pfalz",
    "09": "Saarland",
    "10": "Schleswig-Holstein",
    "11": "Brandenburg",
    "12": "Mecklenburg-Vorpommern",
    "13": "Sachsen",
    "14": "Sachsen-Anhalt",
    "15": "Thüringen",
    "16": "Berlin",
}

HEADERS = {
    "User-Agent": "DutchTravelogueNLP/1.0 (https://github.com/tijn-do/hist-dutch-travelogues-nlp; research project)"
}

# Default LLM temperature for EPG profile generation. The EL pipeline threads
# the --temperature CLI value (default 0.0) through to this stage; this 0.0
# default is used only when generate_candidates/_predict_entity_profile are
# called standalone without an explicit temperature.
TEMPERATURE = 0.0

# Prompting the LLM to act as our dense structural profile generator
_EPG_PROMPT = """Je bent een expert in historische geografie en 19e-eeuwse Nederlandse reisliteratuur.
De topnoniemen zijn afkomstig uit 19e-eeuws Nederlands reisverslag. De auteur is een Groningse student die rond 1816 door Duitsland reist voor zijn Bildung. Alle entiteiten zijn locaties (steden, dorpen, rivieren, bergen, gebouwen, pleinen) voornamelijk in Duitsland, Nederland of aangrenzende gebieden.
Analyseer de historische entiteit en context, en voorspel het profiel van de moderne entiteit.
Het doel is om het toponiem vindbaar te maken in Knowledge Bases zoals Wikidata en Geonames, daarom is het van belang dat je bij generieke termen identificeert welke plaats/gebouw/structuur er bedoeld wordt uit de context.


Entiteit (historische spelling): "{entity_text}"
Type: "{entity_label}"
Context uit reisverhaal: "{context}"

Geef het resultaat STRICT terug als een geldig JSON object met de volgende structuur:
{{
  "modern_name": "De DUITSE naam voor toponiemen in Duitsland, anders de gangbare Nederlandse of Engelse naam",
  "country_or_region": "Het huidige land of de regio waar dit ligt (in het Engels, bv. 'Germany', 'Netherlands')",
  "type_keywords": ["maximaal", "3", "type", "keywords" (bijv: "castle", "city", "mountain")]
}}

Antwoord uitsluitend met de valide JSON. Geen inleiding, geen markdown blocks, geen extra tekst."""


def _wikidata_search(entity_text: str, lang: str = "nl") -> list[dict]:
    """Search Wikidata EntitySearch API for a raw entity string.

    Args:
        entity_text: The surface form to search for.
        lang: Language code for the search (nl, en, or de).

    Returns:
        List of candidate dicts with id, label, description, source.
    """
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
    ollama_headers: dict | None = None,
    think: bool | str | None = None,
    temperature: float = TEMPERATURE,
) -> dict | None:
    """Use an LLM to predict a modern identity profile for an archaic entity mention.

    Entity Profile Generation (EPG) turns a 19th-century surface form and its
    surrounding context into a structured profile (modern_name, country_or_region,
    type_keywords) that enables high-recall semantic search against KB APIs.

    Args:
        entity_text: The archaic entity surface form (e.g. "Cassel").
        entity_label: NER label (E53_Place or E18_Physical_Thing).
        context: Surrounding text from the travelogue.
        ollama_url: Ollama server base URL.
        model_name: Ollama model name.
        ollama_headers: Optional auth headers for cloud API.
        think: Thinking mode (True, False, "low", "medium", "high").
               None uses the model's default.

    Returns:
        Dict with modern_name, country_or_region, type_keywords, or None on failure.
    """
    cache_key = f"{entity_text}|{entity_label}|{context[:300]}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    prompt = _EPG_PROMPT.format(entity_text=entity_text, entity_label=entity_label, context=context)
    api_key = None
    if ollama_headers:
        auth = ollama_headers.get("Authorization", "")
        api_key = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else None
    try:
        raw_response = stream_ollama_chat(
            model=model_name,
            prompt=prompt,
            host=ollama_url,
            api_key=api_key,
            timeout=30.0,
            temperature=temperature,
            think=think,
        ).strip()

        # Strip markdown code blocks if the LLM accidentally added them
        raw_response = re.sub(r"```(?:json)?\s*|```", "", raw_response).strip()
        profile = json.loads(raw_response)

        _llm_cache[cache_key] = profile
        return profile
    except Exception as e:
        print(f"  [Stage 1] EPG Profile Generation failed for '{entity_text}': {e}")
        return None


def _wikidata_hybrid_sparql_search(profile: dict) -> list[dict]:
    """Search Wikidata via SPARQL using the EPG-predicted modern name and location.

    Uses Wikidata's mwapi:EntitySearch service inside a SPARQL query, filtered
    by non-broad location terms from the profile's country_or_region field.
    Broad terms like "Germany" are excluded to avoid over-matching.

    Args:
        profile: EPG profile dict with modern_name and country_or_region keys.

    Returns:
        List of candidate dicts with id, label, description, source.
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
    """Search English Wikipedia full-text and resolve results to Wikidata Q-IDs.

    Combines the EPG modern_name and country_or_region into a search phrase,
    runs Wikipedia's full-text search, then batch-resolves page titles to
    Wikidata IDs via the pageprops API.

    Args:
        profile: EPG profile dict with modern_name and country_or_region keys.

    Returns:
        List of candidate dicts with id (Q-ID), label (page title), description
        (snippet), source="wikipedia_epg".
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



def _geonames_gazetteer_search(profile: dict) -> list[dict]:
    """Search local GeoNames gazetteer using EPG-predicted modern name and location.

    Args:
        profile: EPG profile dict with modern_name and country_or_region keys.

    Returns:
        List of candidate dicts with id (gn:NNN), label, description, source="geonames_gazetteer".
    """
    if not _GAZETTEER_DATA or not profile.get("modern_name"):
        return []

    modern_name = profile.get("modern_name", "").strip()
    country_or_region = profile.get("country_or_region", "").strip()

    # Create search string from modern_name only.
    # country_or_region is NOT included because it's a Dutch label (e.g. "Duitsland")
    # that doesn't appear in the German/English gazetteer text and would only
    # dilute the fuzzy match score, causing correct candidates to fall below threshold.
    search_string = modern_name.lower()

    # Scan through gazetteer data and score matches
    scored_matches = []

    for geonameid, name, asciiname, alternatenames, lat, lon, feature_class, feature_code, country_code, admin1_code, population, elevation in _GAZETTEER_DATA:
        # Skip if country doesn't match (if we have a strong location hint)
        # For now, we'll do text matching and let ranking handle it

        # Create text to search against: name + asciiname + alternatenames
        search_text = f"{name} {asciiname} {' '.join(alternatenames)}".lower()

        # Use rapidfuzz to get similarity score
        # We'll try matching against the full search string
        score = fuzz.WRatio(search_string, search_text)

        # Also try partial matching for cases where the entity name is part of a longer name
        partial_score = fuzz.partial_ratio(search_string, search_text)

        # Take the maximum of the two scores
        final_score = max(score, partial_score)

        # Only consider matches above a threshold
        if final_score >= 85:  # Similarity threshold
            # Boost score for exact matches
            if search_string in search_text or search_text in search_string:
                final_score = min(100, final_score + 10)

            # Boost score for population (more important places)
            # Normalize population to 0-20 bonus points
            pop_bonus = min(20, population / 10000)  # 200k population = max bonus
            final_score = min(100, final_score + pop_bonus)

            # Boost score for certain feature types (cities, towns, etc.)
            if feature_class == "P":  # Populated place
                final_score = min(100, final_score + 5)
            elif feature_class in ["H", "L"]:  # Hydrographic or Lake
                final_score = min(100, final_score + 3)

            # Build a rich description with feature type, admin region, country, and population
            # so the reranker and selector have enough context to disambiguate
            desc_parts = [name]

            # Feature type: class + code in human-readable form
            feat_class_label = _FEATURE_CLASS_LABELS.get(feature_class, "")
            feat_code_label = _FEATURE_CODE_LABELS.get(feature_code, "")
            type_str = " — ".join(filter(None, [feat_class_label, feat_code_label]))
            if type_str:
                desc_parts.append(f"({type_str})")

            # Admin region (state) for German entries
            if country_code == "DE" and admin1_code:
                state = _ADMIN1_DE.get(admin1_code, "")
                if state:
                    desc_parts.append(state)

            # Country (always, not just when non-DE)
            if country_code:
                desc_parts.append(country_code)

            # Population for populated places
            if population > 0:
                desc_parts.append(f"pop. {population}")

            description = ", ".join(desc_parts)

            scored_matches.append((
                final_score,
                {
                    "id": f"gn:{geonameid}",
                    "label": name,
                    "description": description,
                    "source": "geonames_gazetteer"
                }
            ))

    # Sort by score descending and take top matches
    scored_matches.sort(key=lambda x: x[0], reverse=True)

    # Return top 10 matches (similar to API limit)
    candidates = [match[1] for match in scored_matches[:10]]

    return candidates


def generate_candidates(
    entity_text: str,
    entity_label: str,
    context: str = "",
    ollama_url: str = "http://localhost:11434",
    model_name: str = "gemma4:31b-cloud",
    ollama_headers: dict | None = None,
    use_cache: bool = True,
    think: bool | str | None = None,
    temperature: float = TEMPERATURE,
) -> dict[str, list[dict]]:
    """Generate KB candidate entries for an entity mention using a multi-lane approach.

    Stage 1 of the EL pipeline. Runs three lanes:
      - Sparse lane: wikidata text search in nl/en/de against the raw surface form.
      - Dense/EPG lane: LLM predicts a modern profile, then queries Wikidata SPARQL,
        Wikipedia full-text, and local GeoNames gazetteer with the predicted modern name and location.

    Args:
        entity_text: The entity surface form (e.g. "Cassel").
        entity_label: NER label (E53_Place or E18_Physical_Thing).
        context: Surrounding text from the travelogue (enables EPG lane).
        ollama_url: Ollama server base URL.
        model_name: Ollama model name for EPG profile prediction.
        ollama_headers: Optional auth headers for cloud API.
        use_cache: Whether to return cached results for duplicate queries.
        think: Thinking mode (True, False, "low", "medium", "high").
               None uses the model's default.

    Returns:
        Dict with separate candidate lists for each KB type:
        {"wikidata": [...], "geonames": [...]} where each list contains candidate dicts.
    """
    cache_key = f"{entity_text}|{entity_label}"
    if use_cache and cache_key in _cache:
        wikidata_count = len(_cache[cache_key]["wikidata"])
        geonames_count = len(_cache[cache_key]["geonames"])
        print(f"  [Stage 1] '{entity_text}' -> {wikidata_count} Wikidata + {geonames_count} GeoNames candidates (cached)")
        return _cache[cache_key]

    # Initialize result structure
    candidates = {
        "wikidata": [],
        "geonames": []
    }
    seen_wikidata_ids: set[str] = set()
    seen_geonames_ids: set[str] = set()

    # 1. Sparse Lane: Try matching the surface form directly against native API first
    for lang in ["nl", "en", "de"]:
        for cand in _wikidata_search(entity_text, lang=lang):
            if cand["id"] not in seen_wikidata_ids:
                seen_wikidata_ids.add(cand["id"])
                candidates["wikidata"].append(cand)

    print(f"  [Stage 1] Baseline text search for '{entity_text}': {len(candidates['wikidata'])} Wikidata results")

    # 2. Dense/EPG Lane: Generate profile using surrounding context to unlock structural matches
    if context:
        profile = _predict_entity_profile(
            entity_text, entity_label, context, ollama_url, model_name, ollama_headers, think=think,
            temperature=temperature,
        )
        if profile:
            print(f"  [Stage 1] EPG Predicted Profile: {profile}")

            # Hybrid search with the predicted anchor targets
            epg_candidates = _wikidata_hybrid_sparql_search(profile)
            new_epg_count = 0
            for cand in epg_candidates:
                if cand["id"] not in seen_wikidata_ids:
                    seen_wikidata_ids.add(cand["id"])
                    candidates["wikidata"].append(cand)
                    new_epg_count += 1
            print(f"  [Stage 1] EPG Hybrid lane added {new_epg_count} new structural candidates.")

            # Wikipedia full-text search lane
            wiki_candidates = _wikipedia_search(profile)
            new_wiki_count = 0
            for cand in wiki_candidates:
                if cand["id"] not in seen_wikidata_ids:
                    seen_wikidata_ids.add(cand["id"])
                    candidates["wikidata"].append(cand)
                    new_wiki_count += 1
            if new_wiki_count:
                print(f"  [Stage 1] EPG Wikipedia lane added {new_wiki_count} candidates.")

            # GeoNames gazetteer lane (replaces API-based GeoNames search)
            gn_candidates = _geonames_gazetteer_search(profile)
            new_gn_count = 0
            for cand in gn_candidates:
                if cand["id"] not in seen_geonames_ids:
                    seen_geonames_ids.add(cand["id"])
                    candidates["geonames"].append(cand)
                    new_gn_count += 1
            if new_gn_count:
                print(f"  [Stage 1] EPG GeoNames gazetteer lane added {new_gn_count} candidates.")

    wikidata_total = len(candidates["wikidata"])
    geonames_total = len(candidates["geonames"])
    print(f"  [Stage 1] '{entity_text}' -> {wikidata_total} Wikidata + {geonames_total} GeoNames candidates total")
    _cache[cache_key] = candidates
    return candidates


def clear_cache() -> None:
    """Clear the candidate and EPG profile caches (useful between runs on different letters)."""
    _cache.clear()
    _llm_cache.clear()