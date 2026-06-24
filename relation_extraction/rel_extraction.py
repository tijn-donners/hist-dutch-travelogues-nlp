"""ATO Relation Extraction & RDF Generation.

Loads NER+EL output (_el.spacy), extracts travel events (E9_Move),
activities (E7_Activity), and their relationships to places according
to the Academic Tourism Ontology. Generates RDF output.

Usage:
    python relation_extraction/rel_extraction.py
    python relation_extraction/rel_extraction.py --model gemma4:31b --temperature 0.3
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path

import spacy
from dotenv import load_dotenv
from jinja2 import Template
import httpx
import ollama
from spacy.tokens import DocBin
from spacy.util import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ollama_utils import stream_ollama_chat
from rdflib import RDF, RDFS, Literal, Namespace, URIRef
from rdflib.graph import Graph

# Register custom extension attributes for EL data (must happen before DocBin loading)
from spacy.tokens import Span, Token
if not Span.has_extension("kb_id_wikidata_"):
    Span.set_extension("kb_id_wikidata_", default=None)
if not Token.has_extension("ent_kb_id_wikidata_"):
    Token.set_extension("ent_kb_id_wikidata_", default=None)
if not Span.has_extension("kb_id_geonames_"):
    Span.set_extension("kb_id_geonames_", default=None)
if not Token.has_extension("ent_kb_id_geonames_"):
    Token.set_extension("ent_kb_id_geonames_", default=None)

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NER_RESULTS_DIR = ROOT_DIR / "ner" / "ner-output"
EL_RESULTS_DIR = ROOT_DIR / "entity_linking" / "el-results"
EXAMPLES_FILE = str(SCRIPT_DIR / "re_examples.yaml")
PROMPT_TEMPLATE = str(SCRIPT_DIR / "re_prompt_dutch.jinja")
OUTPUT_DIR_RE = ROOT_DIR / "output" / "re"
OUTPUT_DIR_RDF = ROOT_DIR / "output" / "rdf"

OLLAMA_MODEL = "deepseek-v4-flash"
OLLAMA_HOST = "https://ollama.com"
OLLAMA_KEY = os.environ.get("OLLAMA_API_KEY")
TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
def select_el_spacy_file():
    """Scan el-results/ for *_el.spacy files and let the user pick one.

    Returns:
        Path to the selected *_el.spacy file.
    """
    spacy_files = sorted(EL_RESULTS_DIR.glob("*_el.spacy"))
    if not spacy_files:
        print(f"No *_el.spacy files found in {EL_RESULTS_DIR}")
        raise SystemExit(1)

    if len(spacy_files) == 1:
        print(f"Auto-selected: {spacy_files[0].name}")
        return str(spacy_files[0])

    print("Available _el.spacy files:")
    for i, f in enumerate(spacy_files, 1):
        print(f"  [{i}] {f.name}")
    choice = input("Select number: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(spacy_files):
            return str(spacy_files[idx])
    except ValueError:
        pass
    print(f"Invalid selection: {choice}")
    raise SystemExit(1)


def find_offset_map(el_spacy_path):
    """Find the matching offset map for an _el.spacy file.

    Derives the model_run portion from the spacy filename and looks for the
    corresponding offset map in ner-output/ (checked first) or el-results/.

    Args:
        el_spacy_path: Path to an *_el.spacy file.

    Returns:
        Path to the matching offset map JSON, or None if not found.
    """
    stem = Path(el_spacy_path).stem  # e.g. 1816_third_letter_gemma4:31b_t0.1_fewshot_el
    # Strip _el suffix to get the base spacy stem
    if stem.endswith("_el"):
        base_stem = stem[:-3]  # 1816_third_letter_gemma4:31b_t0.1_fewshot
    else:
        base_stem = stem

    # Derive offset map name: new convention appends _offset_map
    offset_stem = base_stem + "_offset_map"

    # Check el-results/ first (sibling of the _el.spacy, copied by el.py)
    candidate = Path(el_spacy_path).parent / f"{offset_stem}.json"
    if candidate.exists():
        return str(candidate)

    # Also check el-results/ flat (legacy)
    candidate = EL_RESULTS_DIR / f"{offset_stem}.json"
    if candidate.exists():
        return str(candidate)

    # Fallback: search ner-output/ recursively
    for candidate in sorted(NER_RESULTS_DIR.rglob(f"{offset_stem}.json")):
        return str(candidate)

    print(f"Warning: no offset map found for {Path(el_spacy_path).name}")
    return None


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
ATO = Namespace("http://academictourism.com/entity/")
ACADEMICTOURISM = Namespace("http://academictourism.com/academictourism#")
CIDOC = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
WD = Namespace("https://www.wikidata.org/wiki/")
GN = Namespace("https://www.geonames.org/")
LRMOO = Namespace("http://iflastandards.info/ns/lrm/lrmoo/")


# ---------------------------------------------------------------------------
# Page sort
# ---------------------------------------------------------------------------
_ROMAN_VALS = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
               'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13,
               'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18,
               'XIX': 19, 'XX': 20}


def _page_sort_key(pn):
    s = str(pn)
    if s.isdigit():
        return int(s)
    if s.upper() in _ROMAN_VALS:
        return _ROMAN_VALS[s.upper()]
    return 0


# ---------------------------------------------------------------------------
# Build annotated text
# ---------------------------------------------------------------------------
_MARKER_RE = re.compile(r'^\[[IVXLCDM\d]+\]\s*')


def build_annotated_text(spacy_file, offset_map_file):
    """Build full letter text with NER entities marked inline with unique mention IDs.

    Returns:
        annotated_text: full text with [entity](LABEL, QID, eN) markers
        all_entities: list of {text, label, kb_id, start, end, mention_id}
        mention_map: dict of mention_id -> {text, label, kb_id, start, end, page}
    """
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(spacy_file)
    docs = list(db.get_docs(nlp.vocab))

    with open(offset_map_file) as f:
        offset_map = json.load(f)

    sorted_pages = sorted(offset_map.items(), key=lambda x: _page_sort_key(x[0]))

    if len(docs) != len(sorted_pages):
        print(f"Warning: {len(docs)} docs but {len(sorted_pages)} pages")

    # Detect [N] markers
    has_markers = any(doc.text.lstrip().startswith('[') for doc in docs[:3])

    # Build full text and page positions
    full_text_parts = []
    page_positions = {}
    pos = 0
    for i in range(min(len(docs), len(sorted_pages))):
        doc = docs[i]
        page_num, _ = sorted_pages[i]
        text = doc.text
        if has_markers:
            text = _MARKER_RE.sub('', text, count=1)
        page_positions[page_num] = pos
        full_text_parts.append(text)
        pos += len(text)

    full_text = "".join(full_text_parts)

    # Build global entity list with mention IDs
    all_entities = []
    mention_counter = 0
    for i in range(min(len(docs), len(sorted_pages))):
        doc = docs[i]
        page_num, _ = sorted_pages[i]
        base = page_positions[page_num]

        doc_text = doc.text
        marker_len = 0
        if has_markers:
            stripped = _MARKER_RE.sub('', doc_text, count=1)
            marker_len = len(doc_text) - len(stripped)

        for ent in doc.ents:
            g_start = base + ent.start_char - marker_len
            g_end = base + ent.end_char - marker_len
            # Extract both KB IDs from the entity (using extension attributes)
            try:
                wikidata_id = ent._.kb_id_wikidata_
            except (AttributeError, KeyError):
                wikidata_id = None
            try:
                geonames_id = ent._.kb_id_geonames_
            except (AttributeError, KeyError):
                geonames_id = None
            # For backward compatibility, define a primary kb_id (prefer Wikidata if available)
            kb_id = wikidata_id if wikidata_id is not None else geonames_id
            mention_counter += 1
            all_entities.append({
                "text": ent.text,
                "label": ent.label_,
                "kb_id": kb_id,  # Primary ID for backward compatibility
                "kb_id_wikidata": wikidata_id,
                "kb_id_geonames": geonames_id,
                "start": g_start,
                "end": g_end,
                "mention_id": f"e{mention_counter}",
            })

    all_entities.sort(key=lambda x: (x["start"], -x["end"]))

    # Remove overlapping (keep longer)
    filtered = []
    for ent in all_entities:
        if filtered and ent["start"] >= filtered[-1]["start"] \
           and ent["end"] <= filtered[-1]["end"]:
            continue
        filtered.append(ent)

    # Build annotated text by inserting markers from end to start
    annotated = full_text
    for ent in reversed(filtered):
        marker = f"[{ent['text']}]({ent['label']}, {ent['mention_id']})"
        annotated = annotated[:ent["start"]] + marker + annotated[ent["end"]:]

    # Build mention_map for downstream TEI export and RDF citation generation
    mention_map = {}
    for ent in filtered:
        # Note: 'ent' is a dict at this point (built from doc.ents above),
        # not a spaCy Span, so use dict keys directly.
        mention_map[ent["mention_id"]] = {
            "text": ent["text"],
            # Normalize old E22_Human-made_Object label to Mode_of_Transportation
            # for backward compatibility with pre-rename spacy files
            "label": "Mode_of_Transportation" if ent["label"] == "E22_Human-made_Object" else ent["label"],
            "kb_id": ent["kb_id"],  # Primary ID for backward compatibility
            "kb_id_wikidata": ent.get("kb_id_wikidata"),
            "kb_id_geonames": ent.get("kb_id_geonames"),
            "start": ent["start"],
            "end": ent["end"],
        }

    return annotated, filtered, mention_map


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def load_examples(examples_file):
    """Load few-shot examples from YAML."""
    import yaml
    with open(examples_file) as f:
        return yaml.safe_load(f)


def build_prompt(annotated_text, examples=None):
    """Render the Jinja2 prompt template."""
    with open(PROMPT_TEMPLATE) as f:
        template = Template(f.read())

    return template.render(
        text=annotated_text,
        prompt_examples=examples,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def query_llm(prompt, model=OLLAMA_MODEL, think=None, think_log_path=None):
    """Send prompt to Ollama and return response text.

    Args:
        think_log_path: Optional path to persist the model's thinking trace,
            so expensive reasoning-model runs can be salvaged even when the
            content channel is empty or truncated.

    Raises:
        ollama.ResponseError, ollama.RequestError, httpx.TimeoutException
    """
    return stream_ollama_chat(
        model=model,
        prompt=prompt,
        host=OLLAMA_HOST,
        api_key=OLLAMA_KEY,
        timeout=600.0,
        temperature=TEMPERATURE,
        think=think,
        think_log_path=think_log_path,
    )


# ---------------------------------------------------------------------------
# JSON parsing & validation
# ---------------------------------------------------------------------------
def extract_json(text):
    """Extract JSON object from LLM response (may contain markdown fences)."""
    # Try to find JSON between ```json ... ``` fences
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1)

    # Try to find outermost { ... }
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)

    return json.loads(text)


def validate_events(data, mention_map):
    """Validate that all referenced mention_ids exist in the mention_map.

    With the mention_id-based approach, events reference entities via
    mention_id strings (e.g. "e7") instead of {text, kb_id} dicts.
    """
    issues = []

    def check_mention_id(mid, event_id, role):
        if mid is None:
            return
        if mid not in mention_map:
            issues.append(f"{event_id}: {role} mention_id '{mid}' "
                          f"not found in mention_map")

    def check_event(event):
        eid = event.get("id", "?")
        check_mention_id(event.get("moved_from"), eid, "moved_from")
        check_mention_id(event.get("moved_to"), eid, "moved_to")
        check_mention_id(event.get("took_place_at"), eid, "took_place_at")
        check_mention_id(event.get("on_or_within"), eid, "on_or_within")
        check_mention_id(event.get("mode_of_transportation"), eid, "mode_of_transportation")

        # New fields
        for mp in event.get("mentions_places", []):
            check_mention_id(mp, eid, "mentions_places")
        for vp in event.get("via_points", []):
            check_mention_id(vp, eid, "via_points")
        for va in event.get("viewed_artifacts", []):
            check_mention_id(va, eid, "viewed_artifacts")
        for np in event.get("near_places", []):
            check_mention_id(np, eid, "near_places")
        for ts in (event.get("time_span", []) or []):
            check_mention_id(ts, eid, "time_span")
        for sec in event.get("sections", []):
            if isinstance(sec, dict):
                check_mention_id(sec.get("room"), eid, "sections.room")
                check_mention_id(sec.get("building"), eid, "sections.building")

        for sub in event.get("sub_events", []):
            if isinstance(sub, dict):
                check_event(sub)

    for event in data.get("events", []):
        check_event(event)

    for mp in data.get("mentioned_only_places", []):
        if isinstance(mp, dict):
            check_mention_id(mp.get("mention_id"), "mentioned_only", "place")
        else:
            check_mention_id(mp, "mentioned_only", "place")

    # --- Validate toponym_relations (Soni et al. spatial categories) ---
    VALID_CATEGORIES = {"IN", "NEAR", "THRU", "TO", "FROM", "NO_REL"}
    TOPONYM_LABELS = {"E53_Place", "E18_Physical_Thing"}
    toponym_relations = data.get("toponym_relations", {})

    if toponym_relations:
        # Check every key exists in mention_map and value is valid
        for mid, category in toponym_relations.items():
            if mid not in mention_map:
                issues.append(f"toponym_relations: mention_id '{mid}' "
                              f"not found in mention_map")
            if category not in VALID_CATEGORIES:
                issues.append(f"toponym_relations: '{mid}' has invalid "
                              f"category '{category}' (expected one of "
                              f"{sorted(VALID_CATEGORIES)})")

        # Check completeness: every E53/E18 entity should be in toponym_relations
        for mid, info in mention_map.items():
            if info.get("label") in TOPONYM_LABELS:
                if mid not in toponym_relations:
                    issues.append(f"toponym_relations: E53/E18 entity '{mid}' "
                                  f"('{info['text']}') missing from "
                                  f"toponym_relations")

        # Check consistency with event roles (warnings, not errors)
        # Build a map of mention_id -> set of roles from events
        role_map = {}  # mid -> set of roles
        def _collect_roles(event):
            eid = event.get("id", "?")
            for role, field in [("moved_from", "moved_from"),
                                ("moved_to", "moved_to"),
                                ("took_place_at", "took_place_at"),
                                ("on_or_within", "on_or_within")]:
                mid = event.get(field)
                if mid:
                    role_map.setdefault(mid, set()).add(role)
            for vp in event.get("via_points", []):
                role_map.setdefault(vp, set()).add("via_points")
            for np in event.get("near_places", []):
                role_map.setdefault(np, set()).add("near_places")
            for sub in event.get("sub_events", []):
                if isinstance(sub, dict):
                    _collect_roles(sub)
        for event in data.get("events", []):
            _collect_roles(event)

        # Expected category per role
        ROLE_TO_CATEGORY = {
            "moved_from": "FROM", "moved_to": "TO",
            "took_place_at": "IN", "on_or_within": "IN",
            "via_points": "THRU",
            "near_places": "NEAR",
        }
        for mid, roles in role_map.items():
            if mid not in toponym_relations:
                continue
            cat = toponym_relations[mid]
            for role in roles:
                expected = ROLE_TO_CATEGORY.get(role)
                if expected and cat != expected:
                    issues.append(f"toponym_relations: '{mid}' category '{cat}' "
                                  f"inconsistent with event role '{role}' "
                                  f"(expected '{expected}')")
    else:
        issues.append("toponym_relations: missing from output "
                      "(required for E53/E18 entity classification)")

    return issues


# ---------------------------------------------------------------------------
# RDF Generation
# ---------------------------------------------------------------------------
def event_uri(event_id):
    return ATO[event_id]


def loc_uri(text, kb_id):
    """Generate a location URI. Use text-based slug for consistent identification."""
    # Generate a slug from text for consistent URI regardless of KB ID
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:30]
    return ATO[f"LOC.{slug}"]


# ---------------------------------------------------------------------------
# Letter metadata (Gap 12)
# ---------------------------------------------------------------------------
LETTER_META = {
    "BRF0003": {
        "title": "Brief 3 — Cassel, July 1816",
        "creator": "Reneke de Marees van Swinderen",
        "date": "1816-07-09",
    },
}


def generate_rdf(data, output_path, mention_map=None):
    """Generate RDF from extracted events JSON following ATO patterns.

    Args:
        data: Parsed events JSON dict.
        output_path: Path for the output .ttl file.
        mention_map: Optional dict of mention_id -> {text, label, kb_id, start, end}.
                     When provided, CT.* citation entities are generated.
    """
    g = Graph()
    g.bind("ato", ATO)
    g.bind("academictourism", ACADEMICTOURISM)
    g.bind("cidoc-crm", CIDOC)
    g.bind("skos", SKOS)
    g.bind("lrmoo", LRMOO)

    letter_id = data.get("letter_id", "BRF0003")
    letter_meta = LETTER_META.get(letter_id, {})

    # --- Letter as F2_Expression (enriched per Gap 12) ---
    letter_uri = ATO[letter_id]
    g.add((letter_uri, RDF.type, LRMOO.F2_Expression))
    g.add((letter_uri, RDFS.label, Literal(letter_meta.get("title", f"Letter {letter_id}"))))

    # --- Tour Group ---
    tg_info = data.get("tour_group", {})
    tg_id = tg_info.get("id", "TG")
    tg_uri = ATO[tg_id]
    g.add((tg_uri, RDF.type, CIDOC.E74_Group))
    g.add((tg_uri, RDFS.label, Literal(tg_info.get("label", f"Tour Group {letter_id}"))))

    # --- Overall journey event ---
    journey_id = f"{letter_id}.RS"
    journey_uri = ATO[journey_id]
    _add_event_types(g, journey_uri, is_move=True)
    g.add((journey_uri, RDFS.label, Literal(f"Tour in letter {letter_id[-1]}")))
    g.add((journey_uri, CIDOC["P11_had_participant"], tg_uri))
    g.add((journey_uri, CIDOC["P12_occurred_in_the_presence_of"], tg_uri))
    g.add((journey_uri, CIDOC["P67i_is_referred_to_by"], letter_uri))

    # --- Letter → journey (P129_is_about) ---
    g.add((letter_uri, CIDOC["P129_is_about"], journey_uri))
    g.add((journey_uri, CIDOC["P129i_is_subject_of"], letter_uri))

    # --- Letter metadata (creator/date stored in LETTER_META for documentation) ---

    # --- Collect all event IDs ---
    all_event_ids = []
    processed_mention_ids = set()

    def _resolve_event_id(raw_id, letter_id):
        """Resolve an event ID to a full ATO URI suffix.

        Supports dot-notation for deep nesting: RS0002.RS0001 → BRF0003.RS0002.RS0001
        """
        # Replace underscores with dots for nested IDs
        resolved = raw_id.replace("_", ".")
        if not resolved.startswith(letter_id):
            resolved = f"{letter_id}.{resolved}"
        return resolved

    def collect_event_ids(events):
        for ev in events:
            if isinstance(ev, str):
                # Bare string ID reference
                full_id = _resolve_event_id(ev, letter_id)
                all_event_ids.append(full_id)
                continue
            eid = ev.get("id")
            if eid:
                full_id = _resolve_event_id(eid, letter_id)
                all_event_ids.append(full_id)
                if "sub_events" in ev:
                    collect_event_ids(ev["sub_events"])

    collect_event_ids(data.get("events", []))

    # --- Process each event ---
    prev_event_uri = None

    def _resolve(mid):
        """Resolve a mention_id to (text, wikidata_id, geonames_id, label)."""
        if mid is None:
            return None
        info = mention_map.get(mid)
        if info is None:
            return None
        return (info["text"], info.get("kb_id_wikidata"),
                info.get("kb_id_geonames"), info.get("label"))

    def process_events(events, parent_uri=None):
        nonlocal prev_event_uri

        for ev in events:
            eid = ev.get("id")
            if not eid:
                continue

            full_id = _resolve_event_id(eid, letter_id)
            ev_uri = ATO[full_id]
            ev_type = ev.get("type", "translocation")
            is_move = ev_type == "translocation"

            _add_event_types(g, ev_uri, is_move=is_move)
            g.add((ev_uri, RDFS.label, Literal(ev.get("label", ""))))

            # Participant — Tour Group always participates
            g.add((ev_uri, CIDOC["P11_had_participant"], tg_uri))
            g.add((ev_uri, CIDOC["P12_occurred_in_the_presence_of"], tg_uri))

            # Mereology — check falls_within from JSON first
            parent_from_json = ev.get("falls_within")
            if parent_from_json is not None and parent_from_json != "null":
                parent_id = _resolve_event_id(str(parent_from_json), letter_id)
                resolved_parent = ATO[parent_id]
                g.add((ev_uri, CIDOC["P10_falls_within"], resolved_parent))
                g.add((resolved_parent, CIDOC["P10_contains"], ev_uri))
            elif parent_uri:
                g.add((ev_uri, CIDOC["P10_falls_within"], parent_uri))
                g.add((parent_uri, CIDOC["P10_contains"], ev_uri))
            else:
                g.add((ev_uri, CIDOC["P10_falls_within"], journey_uri))
                g.add((journey_uri, CIDOC["P10_contains"], ev_uri))

            # Spatiotemporal overlap with journey AND immediate parent (Gap 10)
            g.add((ev_uri, CIDOC["P132_spatiotemporally_overlaps_with"], journey_uri))
            if parent_uri:
                g.add((ev_uri, CIDOC["P132_spatiotemporally_overlaps_with"], parent_uri))

            # --- Translocation properties ---
            if is_move:
                mfrom = _resolve(ev.get("moved_from"))
                if mfrom:
                    from_uri = _ensure_place(g, mfrom[0], mfrom[1], mfrom[2])
                    g.add((ev_uri, CIDOC["P27_moved_from"], from_uri))
                    mfrom_mid = ev.get("moved_from")
                    if mfrom_mid and mention_map and mfrom_mid in mention_map:
                        _ensure_citation(g, mfrom_mid, mention_map[mfrom_mid], ev_uri, letter_id)
                        processed_mention_ids.add(mfrom_mid)

                mto = _resolve(ev.get("moved_to"))
                if mto:
                    to_uri = _ensure_place(g, mto[0], mto[1], mto[2])
                    g.add((ev_uri, CIDOC["P26_moved_to"], to_uri))
                    mto_mid = ev.get("moved_to")
                    if mto_mid and mention_map and mto_mid in mention_map:
                        _ensure_citation(g, mto_mid, mention_map[mto_mid], ev_uri, letter_id)
                        processed_mention_ids.add(mto_mid)

            # --- Tour/Stay properties ---
            if ev_type in ("indoor_tour", "outdoor_tour", "stay"):
                tpa = _resolve(ev.get("took_place_at"))
                if tpa:
                    place_uri = _ensure_place(g, tpa[0], tpa[1], tpa[2])
                    g.add((ev_uri, CIDOC["P7_took_place_at"], place_uri))
                    tpa_mid = ev.get("took_place_at")
                    if tpa_mid and mention_map and tpa_mid in mention_map:
                        _ensure_citation(g, tpa_mid, mention_map[tpa_mid], ev_uri, letter_id)
                        processed_mention_ids.add(tpa_mid)

                onwi = _resolve(ev.get("on_or_within"))
                if onwi:
                    thing_uri = _ensure_physical_thing(g, onwi[0], onwi[1], onwi[2])
                    g.add((ev_uri, CIDOC["P8_took_place_on_or_within"], thing_uri))
                    onwi_mid = ev.get("on_or_within")
                    if onwi_mid and mention_map and onwi_mid in mention_map:
                        _ensure_citation(g, onwi_mid, mention_map[onwi_mid], ev_uri, letter_id)
                        processed_mention_ids.add(onwi_mid)

            # --- P59_has_section for building→room relationships (Gap 2) ---
            for sec in ev.get("sections", []):
                if isinstance(sec, dict):
                    room_mid = sec.get("room")
                    building_mid = sec.get("building")
                    room_info = mention_map.get(room_mid) if room_mid else None
                    building_info = mention_map.get(building_mid) if building_mid else None
                    if room_info and building_info:
                        building_uri = _ensure_physical_thing(
                            g, building_info["text"],
                            building_info.get("kb_id_wikidata"),
                            building_info.get("kb_id_geonames"))
                        room_uri = _ensure_place(
                            g, room_info["text"],
                            room_info.get("kb_id_wikidata"),
                            room_info.get("kb_id_geonames"))
                        # Guard against self-reference (e.g. when EL assigns the
                        # same KB ID to a room and its containing building)
                        if building_uri != room_uri:
                            g.add((building_uri, CIDOC["P59_has_section"], room_uri))
                            g.add((room_uri, CIDOC["P59i_is_section_of"], building_uri))

            # --- Citations — link event to CT.* entities ---
            for cit in ev.get("citations", []):
                mention_id = cit.get("mention_id", "")
                if mention_id and mention_map and mention_id in mention_map:
                    _ensure_citation(g, mention_id, mention_map[mention_id], ev_uri,
                                     letter_id)
                    processed_mention_ids.add(mention_id)

                # Mode of transportation: use P101_had_as_general_use for generic
                # modes (E55_Type) and P16_used_specific_object for specific
                # vehicles (E22_Human-made_Object) — Gap 4
                if (cit.get("role") == "mode_of_transportation"
                        and mention_id and mention_map and mention_id in mention_map):
                    minfo = mention_map[mention_id]
                    if minfo.get("label") == "Mode_of_Transportation":
                        if _is_specific_vehicle(minfo["text"]):
                            veh_uri = _ensure_specific_vehicle(g, minfo["text"])
                            g.add((ev_uri, CIDOC["P16_used_specific_object"], veh_uri))
                        else:
                            mode_uri = _ensure_mode_of_transportation(g, minfo["text"])
                            g.add((ev_uri, CIDOC["P101_had_as_general_use"], mode_uri))

                # Time-span citations → P4_has_time-span
                if (cit.get("role") == "time_span"
                        and mention_id and mention_map and mention_id in mention_map):
                    minfo = mention_map[mention_id]
                    if minfo.get("label") == "E52_Time_Span":
                        ts_uri = _ensure_time_span(g, minfo["text"])
                        g.add((ev_uri, CIDOC["P4_has_time-span"], ts_uri))

            # --- time_span field (list of mention_ids) ---
            for ts_mid in (ev.get("time_span", []) or []):
                if ts_mid and ts_mid not in processed_mention_ids:
                    ts_info = mention_map.get(ts_mid)
                    if ts_info and ts_info.get("label") == "E52_Time_Span":
                        ts_uri = _ensure_time_span(g, ts_info["text"])
                        g.add((ev_uri, CIDOC["P4_has_time-span"], ts_uri))
                        processed_mention_ids.add(ts_mid)

            # --- Temporal ordering for top-level events ---
            if ev.get("falls_within") in (None, "null"):
                if prev_event_uri:
                    _add_ordering(g, prev_event_uri, ev_uri)
                prev_event_uri = ev_uri

            # --- via_points: THRU sub-moves (P9_consists_of sub-moves) ---
            # Per CIDOC CRM E9_Move scope note, intermediate places on a
            # trajectory are modelled as P9_consists_of sub-moves, where each
            # via-point is the P26_moved_to of one sub-move and the
            # P27_moved_from of the next.  This replaces the old P67_refers_to
            # approach which incorrectly treated THRU places as merely mentioned.
            via_points = ev.get("via_points", [])
            if via_points and is_move:
                # Determine the previous location: start with moved_from,
                # then each via_point becomes the previous for the next.
                prev_loc_mid = ev.get("moved_from")
                prev_sub_uri = None
                vp_sub_uris = []  # collect for P183 chain integration
                for i, vp_mid in enumerate(via_points):
                    vp_info = mention_map.get(vp_mid)
                    if not vp_info:
                        continue
                    # Create sub-move for this via-point
                    sub_id = f"{full_id}.VP{i + 1:04d}"
                    sub_uri = ATO[sub_id]
                    vp_sub_uris.append(sub_uri)
                    _add_event_types(g, sub_uri, is_move=True)
                    g.add((sub_uri, RDFS.label, Literal(
                        f"Pass through {vp_info['text']}")))
                    g.add((sub_uri, CIDOC["P10_falls_within"], ev_uri))
                    g.add((ev_uri, CIDOC["P10_contains"], sub_uri))
                    g.add((sub_uri, CIDOC["P11_had_participant"], tg_uri))
                    g.add((sub_uri, CIDOC["P12_occurred_in_the_presence_of"],
                           tg_uri))
                    g.add((sub_uri, CIDOC["P132_spatiotemporally_overlaps_with"],
                           ev_uri))
                    # P27 = previous location
                    if prev_loc_mid:
                        prev_info = mention_map.get(prev_loc_mid)
                        if prev_info:
                            from_uri = _ensure_by_label(g, prev_info)
                            g.add((sub_uri, CIDOC["P27_moved_from"], from_uri))
                    # P26 = via-point
                    to_uri = _ensure_by_label(g, vp_info)
                    g.add((sub_uri, CIDOC["P26_moved_to"], to_uri))
                    # CT citation for the via-point sub-move
                    _ensure_citation(g, vp_mid, vp_info, sub_uri, letter_id)
                    processed_mention_ids.add(vp_mid)
                    # Advance
                    prev_loc_mid = vp_mid
                    prev_sub_uri = sub_uri
            elif via_points and not is_move:
                # Non-translocation events shouldn't have via_points,
                # but if they do, treat them as P67_refers_to (fallback)
                for vp_mid in via_points:
                    vp_info = mention_map.get(vp_mid)
                    if vp_info:
                        place_uri = _ensure_by_label(g, vp_info)
                        g.add((letter_uri, CIDOC["P67_refers_to"], place_uri))
                        processed_mention_ids.add(vp_mid)

            # --- viewed_artifacts: objects/artworks viewed during tours (Gap 13, 14, 16) ---
            for va_mid in ev.get("viewed_artifacts", []):
                va_info = mention_map.get(va_mid)
                if va_info:
                    art_uri = _ensure_by_label(g, va_info)
                    g.add((ev_uri, CIDOC["P12_occurred_in_the_presence_of"], art_uri))
                    g.add((letter_uri, CIDOC["P67_refers_to"], art_uri))
                    processed_mention_ids.add(va_mid)

            # --- near_places: NEAR entities linked via happened_in_proximity_of ---
            for np_mid in ev.get("near_places", []):
                np_info = mention_map.get(np_mid)
                if np_info:
                    place_uri = _ensure_by_label(g, np_info)
                    g.add((ev_uri, ACADEMICTOURISM["happened_in_proximity_of"], place_uri))
                    _ensure_citation(g, np_mid, np_info, ev_uri, letter_id)
                    processed_mention_ids.add(np_mid)

            # --- Legacy mentions_places (backward compatibility) ---
            for mp_mid in ev.get("mentions_places", []):
                mp_info = mention_map.get(mp_mid)
                if mp_info:
                    place_uri = _ensure_by_label(g, mp_info)
                    g.add((letter_uri, CIDOC["P67_refers_to"], place_uri))
                    # F2_Expression artworks → P12_occurred_in_the_presence_of
                    if mp_info.get("label") == "F2_Expression":
                        g.add((ev_uri, CIDOC["P12_occurred_in_the_presence_of"], place_uri))
                    processed_mention_ids.add(mp_mid)

            # --- Sub-events: support deep nesting (Gap 21) ---
            if "sub_events" in ev:
                dict_subs = [s for s in ev["sub_events"] if isinstance(s, dict)]
                if dict_subs:
                    process_events(dict_subs, parent_uri=ev_uri)

    process_events(data.get("events", []))

    # --- Mentioned-only places ---
    for mp in data.get("mentioned_only_places", []):
        if isinstance(mp, dict):
            mp_mid = mp.get("mention_id")
        else:
            mp_mid = mp
        mp_info = mention_map.get(mp_mid) if mp_mid else None
        if mp_info:
            place_uri = _ensure_by_label(g, mp_info)
            g.add((letter_uri, CIDOC["P67_refers_to"], place_uri))
            # Create CT citation linked to journey (so the entity is findable
            # via P129i_is_subject_of in downstream evaluation)
            _ensure_citation(g, mp_mid, mp_info, journey_uri, letter_id)
            processed_mention_ids.add(mp_mid)

    # --- All remaining non-visited mentions ---
    for mid, minfo in mention_map.items():
        label = minfo.get("label", "")
        if label not in ("E53_Place", "E18_Physical_Thing", "Mode_of_Transportation",
                     "F2_Expression", "E52_Time_Span",
                     "E19_Physical_Object", "E20_Biological_Object", "E31_Document"):
            continue
        if mid in processed_mention_ids:
            continue
        # CT citation linked to journey
        _ensure_citation(g, mid, minfo, journey_uri, letter_id)
        # Letter-level P67_refers_to
        place_uri = _ensure_by_label(g, minfo)
        g.add((letter_uri, CIDOC["P67_refers_to"], place_uri))

    # --- Serialize ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=output_path, format="turtle")
    print(f"RDF saved to: {output_path}")
    print(f"  Triples: {len(g)}")
    return g


def _add_event_types(graph, uri, is_move=False):
    """Add full CIDOC CRM type hierarchy as used in ATO.rdf.

    The ATO.rdf reference implementation includes the complete chain:
    E1_CRM_Entity → E2_Temporal_Entity → E4_Period → E5_Event →
    E7_Activity → E92_Spacetime_Volume → (E9_Move for translocations).
    """
    # Base hierarchy (all events)
    graph.add((uri, RDF.type, CIDOC.E1_CRM_Entity))
    graph.add((uri, RDF.type, CIDOC.E2_Temporal_Entity))
    graph.add((uri, RDF.type, CIDOC.E4_Period))
    graph.add((uri, RDF.type, CIDOC.E5_Event))
    graph.add((uri, RDF.type, CIDOC.E7_Activity))
    graph.add((uri, RDF.type, CIDOC.E92_Spacetime_Volume))

    if is_move:
        graph.add((uri, RDF.type, CIDOC.E9_Move))


def _add_ordering(graph, earlier_uri, later_uri):
    """Add Allen interval temporal ordering.

    Uses P183_ends_before_the_start_of as the primary ordering property
    (the earlier event definitively ends before the later one starts).
    Also adds P182_ends_before_or_with_the_start_of as a conservative
    alternative. The full Allen interval algebra can be inferred by a
    reasoner from these two assertions.
    """
    graph.add((earlier_uri, CIDOC["P183_ends_before_the_start_of"], later_uri))
    graph.add((earlier_uri, CIDOC["P182_ends_before_or_with_the_start_of"], later_uri))


# ---------------------------------------------------------------------------
# KB-ID based URI cache for deduplication (Gap 11)
# ---------------------------------------------------------------------------
# Two mentions of the same real-world place (e.g. "Ossendorf" / "Ocksendorf")
# that share a Wikidata or GeoNames ID should map to the same ATO-local URI.
_kb_id_cache = {}  # {(wikidata_id, geonames_id): uri}


def _cached_uri(wikidata_id, geonames_id):
    """Return a cached URI if either KB ID is known, else None."""
    if wikidata_id and (wikidata_id, None) in _kb_id_cache:
        return _kb_id_cache[(wikidata_id, None)]
    if geonames_id and (None, geonames_id) in _kb_id_cache:
        return _kb_id_cache[(None, geonames_id)]
    if wikidata_id and geonames_id and (wikidata_id, geonames_id) in _kb_id_cache:
        return _kb_id_cache[(wikidata_id, geonames_id)]
    return None


def _cache_uri(wikidata_id, geonames_id, uri):
    """Store a URI in the KB-ID cache."""
    if wikidata_id:
        _kb_id_cache[(wikidata_id, None)] = uri
    if geonames_id:
        _kb_id_cache[(None, geonames_id)] = uri
    if wikidata_id and geonames_id:
        _kb_id_cache[(wikidata_id, geonames_id)] = uri


def _add_kb_links(graph, uri, wikidata_id=None, geonames_id=None):
    """Add skos:closeMatch triples for Wikidata and/or GeoNames IDs.

    Unlike the old single-kb_id approach, this helper adds the external links
    unconditionally (no guard) so both IDs can coexist on the same ATO-local entity.
    """
    if wikidata_id is not None:
        graph.add((uri, SKOS.closeMatch, WD[wikidata_id]))
    if geonames_id is not None:
        # Strip "gn:" prefix if present; GeoNames IDs are numeric
        geonameid = geonames_id[3:] if geonames_id.startswith("gn:") else geonames_id
        graph.add((uri, SKOS.closeMatch, GN[geonameid]))


def _ensure_place(graph, text, wikidata_id=None, geonames_id=None):
    """Get or create an E53_Place instance. Type triples are guarded (one-shot),
    but skos:closeMatch links are added on every call so both IDs accumulate.

    Uses KB-ID cache to deduplicate: if two mentions share a Wikidata or
    GeoNames ID, they map to the same ATO-local URI regardless of spelling
    variations (e.g. "Ossendorf" / "Ocksendorf").
    """
    # Check KB-ID cache first
    cached = _cached_uri(wikidata_id, geonames_id)
    if cached is not None:
        uri = cached
        # Still add any new KB links that weren't added before
        _add_kb_links(graph, uri, wikidata_id, geonames_id)
        return uri

    uri = loc_uri(text, wikidata_id or geonames_id)
    # Type triples only on first creation
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E1_CRM_Entity))
        graph.add((uri, RDF.type, CIDOC.E53_Place))
        graph.add((uri, RDFS.label, Literal(text)))
    # External links always — may add the second ID on a later call
    _add_kb_links(graph, uri, wikidata_id, geonames_id)
    # Cache for future deduplication
    _cache_uri(wikidata_id, geonames_id, uri)
    return uri


def _ensure_physical_thing(graph, text, wikidata_id=None, geonames_id=None):
    """Get or create an E18_Physical_Thing + E53_Place instance.

    Uses KB-ID cache for deduplication (same place, different spelling).
    """
    # Check KB-ID cache first
    cached = _cached_uri(wikidata_id, geonames_id)
    if cached is not None:
        uri = cached
        _add_kb_links(graph, uri, wikidata_id, geonames_id)
        return uri

    uri = loc_uri(text, wikidata_id or geonames_id)
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E1_CRM_Entity))
        graph.add((uri, RDF.type, CIDOC.E18_Physical_Thing))
        graph.add((uri, RDF.type, CIDOC.E53_Place))
        graph.add((uri, RDFS.label, Literal(text)))
    _add_kb_links(graph, uri, wikidata_id, geonames_id)
    _cache_uri(wikidata_id, geonames_id, uri)
    return uri


def _ensure_mode_of_transportation(graph, text):
    """Get or create a Mode_of_Transportation instance (E55_Type) for transport modes.

    Transportation modes are generic types (walking, carriage, coach, etc.)
    modeled as E55_Type / ato:Mode_of_Transportation.
    Uses a MOD. prefix instead of LOC. to avoid conflating modes with places.

    For specific vehicles ("de Munstersche wagen", "een rijtuig"), use
    _ensure_specific_vehicle instead, which models them as E22_Human-made_Object.
    """
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:30]
    uri = ATO[f"MOD.{slug}"]
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E1_CRM_Entity))
        graph.add((uri, RDF.type, CIDOC.E55_Type))
        graph.add((uri, RDF.type, ACADEMICTOURISM["Mode_of_Transportation"]))
        graph.add((uri, RDFS.label, Literal(text)))
    return uri


# Heuristic: words that suggest a specific vehicle (not a generic mode)
_SPECIFIC_VEHICLE_WORDS = {
    "wagen", "rytuig", "rijtuig", "koets", "kar", "karos", "chaise",
    "postwagen", "extrapost", "diligence", "sjees", "calèche", "barouchette",
    "Munstersche", "Munstersche wagen",
}


def _is_specific_vehicle(text):
    """Heuristic: determine if a Mode_of_Transportation mention is a specific
    vehicle (E22_Human-made_Object) rather than a generic mode (E55_Type).

    Generic modes: "walking", "wandelen", "te voet", "per schip", "te paard"
    Specific vehicles: "de wagen", "een rijtuig", "de Munstersche wagen"
    """
    text_lower = text.lower().strip()
    # Generic modes (verbs, abstract)
    generic_patterns = [
        "walking", "wandelen", "wandeling", "te voet", "te paard",
        "per schip", "per boot", "varen", "vliegen", "rijden",
    ]
    for pat in generic_patterns:
        if pat in text_lower:
            return False
    # Specific vehicle indicators
    for word in _SPECIFIC_VEHICLE_WORDS:
        if word.lower() in text_lower:
            return True
    # If it contains an article or "een", likely a specific vehicle
    if any(w in text_lower.split() for w in ("de", "het", "een", "onze", "mijn")):
        return True
    return False


def _ensure_specific_vehicle(graph, text, wikidata_id=None, geonames_id=None):
    """Get or create an E22_Human-made_Object instance for a specific vehicle.

    Specific vehicles (e.g. "de Munstersche wagen", "een rijtuig") are
    modeled as E22_Human-made_Object, distinct from generic transport modes
    (E55_Type). Linked to translocation events via P16_used_specific_object.
    Uses a VEH. prefix to distinguish from modes and places.
    """
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:30]
    uri = ATO[f"VEH.{slug}"]
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E1_CRM_Entity))
        graph.add((uri, RDF.type, CIDOC.E18_Physical_Thing))
        graph.add((uri, RDF.type, CIDOC.E22_Human_Made_Object))
        graph.add((uri, RDFS.label, Literal(text)))
    _add_kb_links(graph, uri, wikidata_id, geonames_id)
    return uri


def _ensure_time_span(graph, text):
    """Get or create an E52_Time-Span instance for a temporal expression.

    Temporal expressions (dates, times, durations) are modeled as E52_Time-Span
    linked to events via P4_has_time-span. The URI is derived from the text
    for consistency across pages.
    """
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:40]
    uri = ATO[f"TS.{slug}"]
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC["E52_Time-Span"]))
        graph.add((uri, RDFS.label, Literal(text)))
    return uri


def _ensure_f2_expression(graph, text):
    """Get or create an F2_Expression instance for an artwork mention.

    Artworks mentioned in the text (mosaics, paintings, sculptures) are
    modeled as F2_Expression per LRMoo, linked to the letter via P67_refers_to
    and to viewing events via P12_occurred_in_the_presence_of.
    """
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:40]
    uri = ATO[f"ART.{slug}"]
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, LRMOO.F2_Expression))
        graph.add((uri, RDFS.label, Literal(text)))
    return uri


def _ensure_physical_object(graph, text, wikidata_id=None, geonames_id=None):
    """Get or create an E19_Physical_Object instance for a movable museum object.

    Movable non-biological natural objects (fossils, minerals, shells, amber)
    are modeled as E19_Physical_Object / E18_Physical_Thing.
    Uses a PHO. prefix to distinguish from LO C. place entities.
    """
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:30]
    uri = ATO[f"PHO.{slug}"]
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E1_CRM_Entity))
        graph.add((uri, RDF.type, CIDOC.E18_Physical_Thing))
        graph.add((uri, RDF.type, CIDOC.E19_Physical_Object))
        graph.add((uri, RDFS.label, Literal(text)))
    _add_kb_links(graph, uri, wikidata_id, geonames_id)
    return uri


def _ensure_biological_object(graph, text, wikidata_id=None, geonames_id=None):
    """Get or create an E20_Biological_Object instance for a biological specimen.

    Biological specimens (eggs, taxidermy, skeletons) are modeled as
    E20_Biological_Object / E19_Physical_Object / E18_Physical_Thing.
    Uses a BIO. prefix to distinguish from other entity types.
    """
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:30]
    uri = ATO[f"BIO.{slug}"]
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E1_CRM_Entity))
        graph.add((uri, RDF.type, CIDOC.E18_Physical_Thing))
        graph.add((uri, RDF.type, CIDOC.E19_Physical_Object))
        graph.add((uri, RDF.type, CIDOC.E20_Biological_Object))
        graph.add((uri, RDFS.label, Literal(text)))
    _add_kb_links(graph, uri, wikidata_id, geonames_id)
    return uri


def _ensure_document(graph, text, wikidata_id=None, geonames_id=None):
    """Get or create an E31_Document instance for a document mention.

    Documents (books, journals, catalogs) are modeled as
    E31_Document / E73_Information_Object.
    Uses a DOC. prefix to distinguish from other entity types.
    """
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:30]
    uri = ATO[f"DOC.{slug}"]
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E1_CRM_Entity))
        graph.add((uri, RDF.type, CIDOC.E73_Information_Object))
        graph.add((uri, RDF.type, CIDOC.E31_Document))
        graph.add((uri, RDFS.label, Literal(text)))
    _add_kb_links(graph, uri, wikidata_id, geonames_id)
    return uri


def _ensure_by_label(graph, mention_info):
    """Dispatch to the correct _ensure_* function based on label in mention_info.

    Args:
        graph: RDF graph to add triples to.
        mention_info: Dict with keys 'text', 'label', 'kb_id_wikidata', 'kb_id_geonames'.

    Returns:
        URI of the ATO-local entity.
    """
    text = mention_info.get("text", "")
    label = mention_info.get("label", "")
    wikidata_id = mention_info.get("kb_id_wikidata")
    geonames_id = mention_info.get("kb_id_geonames")
    if label == "E18_Physical_Thing":
        return _ensure_physical_thing(graph, text, wikidata_id, geonames_id)
    elif label == "Mode_of_Transportation":
        # Distinguish generic modes from specific vehicles
        if _is_specific_vehicle(text):
            return _ensure_specific_vehicle(graph, text, wikidata_id, geonames_id)
        return _ensure_mode_of_transportation(graph, text)
    elif label == "F2_Expression":
        return _ensure_f2_expression(graph, text)
    elif label == "E52_Time_Span":
        return _ensure_time_span(graph, text)
    elif label == "E19_Physical_Object":
        return _ensure_physical_object(graph, text, wikidata_id, geonames_id)
    elif label == "E20_Biological_Object":
        return _ensure_biological_object(graph, text, wikidata_id, geonames_id)
    elif label == "E31_Document":
        return _ensure_document(graph, text, wikidata_id, geonames_id)
    else:
        return _ensure_place(graph, text, wikidata_id, geonames_id)


def _ensure_citation(graph, mention_id, mention_info, event_uri, letter_id):
    """Create a CT.* citation (E89_Propositional_Object) linking an event to a text mention.

    Creates a citation entity of the form CT.<letter_id>.<mention_id> with
    P129_is_about → ATO-local entity (typed via _ensure_place/_ensure_physical_thing),
    P67i_is_referred_to_by back-link, and academictourism:conveys / is_conveyed_by
    to the event. The ATO entity receives both Wikidata and GeoNames skos:closeMatch
    links in a single call.
    """
    ct_uri = ATO[f"CT.{letter_id}.{mention_id}"]

    # Avoid duplicate CT creation
    if (ct_uri, RDF.type, None) in graph:
        return ct_uri

    graph.add((ct_uri, RDF.type, CIDOC.E89_Propositional_Object))
    graph.add((ct_uri, RDFS.label, Literal(
        f"Citation: '{mention_info['text']}' ({mention_info['label']})"
    )))

    label = mention_info.get("label", "")
    text = mention_info.get("text", "")
    wikidata_id = mention_info.get("kb_id_wikidata")
    geonames_id = mention_info.get("kb_id_geonames")

    # Choose the right ATO-local entity type based on the label
    if label == "E18_Physical_Thing":
        target_uri = _ensure_physical_thing(graph, text, wikidata_id, geonames_id)
    elif label == "Mode_of_Transportation":
        if _is_specific_vehicle(text):
            target_uri = _ensure_specific_vehicle(graph, text, wikidata_id, geonames_id)
        else:
            target_uri = _ensure_mode_of_transportation(graph, text)
    elif label == "F2_Expression":
        target_uri = _ensure_f2_expression(graph, text)
    elif label == "E52_Time_Span":
        target_uri = _ensure_time_span(graph, text)
    elif label == "E19_Physical_Object":
        target_uri = _ensure_physical_object(graph, text, wikidata_id, geonames_id)
    elif label == "E20_Biological_Object":
        target_uri = _ensure_biological_object(graph, text, wikidata_id, geonames_id)
    elif label == "E31_Document":
        target_uri = _ensure_document(graph, text, wikidata_id, geonames_id)
    else:
        target_uri = _ensure_place(graph, text, wikidata_id, geonames_id)

    # CT → ato:LOC.* (via tussenlaag, behouden)
    graph.add((ct_uri, CIDOC["P129_is_about"], target_uri))

    # CT → Wikidata/GeoNames (directe link, zoals ATO.rdf)
    if wikidata_id is not None:
        graph.add((ct_uri, CIDOC["P67_refers_to"], WD[wikidata_id]))
    if geonames_id is not None:
        # Strip "gn:" prefix if present; GeoNames IDs are numeric
        geonameid = geonames_id[3:] if geonames_id.startswith("gn:") else geonames_id
        graph.add((ct_uri, CIDOC["P67_refers_to"], GN[geonameid]))

    # ato:LOC.* → CT (beide inverse properties)
    graph.add((target_uri, CIDOC["P129i_is_subject_of"], ct_uri))
    graph.add((target_uri, CIDOC["P67i_is_referred_to_by"], ct_uri))

    # Link citation to the event via custom ATO property
    graph.add((event_uri, ACADEMICTOURISM.conveys, ct_uri))
    graph.add((ct_uri, ACADEMICTOURISM.is_conveyed_by, event_uri))

    return ct_uri


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ATO Relation Extraction")
    parser.add_argument("--model", default=None,
                        help="Ollama model (overrides OLLAMA_MODEL constant)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature (overrides TEMPERATURE constant)")
    parser.add_argument("--think", default=None,
                        choices=["true", "false", "low", "medium", "high"],
                        help="Thinking mode (overrides model default)")
    args = parser.parse_args()

    # Override constants if CLI args provided
    global OLLAMA_MODEL, TEMPERATURE
    if args.model:
        OLLAMA_MODEL = args.model
    if args.temperature is not None:
        TEMPERATURE = args.temperature

    # Resolve think parameter
    think_param = args.think
    if think_param == "true":
        think_param = True
    elif think_param == "false":
        think_param = False

    # Store for query_llm to use
    _THINK = think_param

    print("=" * 60)
    print("ATO Relation Extraction & RDF Generation")
    print("=" * 60)

    # Select EL spacy file and find matching offset map
    spacy_file = select_el_spacy_file()
    offset_map_file = find_offset_map(spacy_file)
    if offset_map_file is None:
        print("Cannot proceed without offset map. Run NER pipeline first.")
        return

    # Derive output filenames from the input spacy file
    spacy_stem = Path(spacy_file).stem  # e.g. 1816_third_letter_gemma4-31b_t0.1_fewshot_el
    if spacy_stem.endswith("_el"):
        spacy_stem = spacy_stem[:-3]  # 1816_third_letter_gemma4-31b_t0.1_fewshot
    # Append RE model info so the output filename reflects which RE model was used
    re_model_slug = re.sub(r'[^a-zA-Z0-9_.-]', '_', OLLAMA_MODEL)
    if _THINK is True:
        think_slug = "_thinkTrue"
    elif _THINK is False:
        think_slug = "_thinkFalse"
    elif _THINK in ("low", "medium", "high"):
        think_slug = f"_think{_THINK.capitalize()}"
    else:
        think_slug = "_thinkDefault"
    spacy_stem = f"{spacy_stem}__{re_model_slug}_t{TEMPERATURE}{think_slug}"
    output_json = str(OUTPUT_DIR_RE / f"{spacy_stem}_events.json")
    output_rdf = str(OUTPUT_DIR_RDF / f"{spacy_stem}_events.ttl")

    # Load and annotate
    print("\n1. Loading NER+EL entities...")
    annotated_text, entities, mention_map = build_annotated_text(spacy_file, offset_map_file)
    print(f"   Entities: {len(entities)} ({len(mention_map)} mention IDs)")
    print(f"   Text length: {len(annotated_text)} chars")

    # Load examples
    examples = None
    if Path(EXAMPLES_FILE).exists():
        examples = load_examples(EXAMPLES_FILE)
        print(f"\n2. Loaded {len(examples)} few-shot examples")
    else:
        print("\n2. No examples file found, running zero-shot")

    # Build prompt
    print("\n3. Building prompt...")
    prompt = build_prompt(annotated_text, examples)
    print(f"   Prompt length: {len(prompt)} chars")

    # Truncate if needed (model has context limit ~256K but prompt can be large)
    max_prompt_chars = 200000
    if len(prompt) > max_prompt_chars:
        print(f"   Truncating prompt to {max_prompt_chars} chars...")
        prompt = prompt[:max_prompt_chars]

    # Query LLM with retry
    max_retries = 3
    data = None
    response = None
    t_start = time.perf_counter()
    num_api_failures = 0

    for attempt in range(1, max_retries + 1):
        print(f"\n4. Querying {OLLAMA_MODEL} (attempt {attempt}/{max_retries})...")
        think_log_path = str(Path(output_json).with_suffix(".think.txt"))
        try:
            response = query_llm(prompt, model=OLLAMA_MODEL, think=_THINK,
                                 think_log_path=think_log_path)
        except (ollama.RequestError, ollama.ResponseError,
                httpx.TimeoutException, httpx.ConnectError) as e:
            num_api_failures += 1
            print(f"   API error: {e}")
            if attempt < max_retries:
                print(f"   Retrying ({attempt}/{max_retries})...")
                continue
            else:
                print(f"   All {max_retries} attempts failed.")
                # Save raw error info for debugging
                debug_path = str(Path(output_json).with_suffix(".error.txt"))
                Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, 'w') as f:
                    f.write(f"API error after {max_retries} attempts: {e}\n")
                print(f"   Error info saved to: {debug_path}")
                return

        print(f"   Response length: {len(response)} chars")

        # Parse JSON
        print(f"\n5. Parsing response (attempt {attempt}/{max_retries})...")
        try:
            data = extract_json(response)
            print("   JSON parsed successfully.")
            break  # success — exit retry loop
        except json.JSONDecodeError as e:
            print(f"   JSON parse error: {e}")
            # Save raw response for debugging
            debug_path = str(Path(output_json).with_suffix(f".raw_attempt{attempt}.txt"))
            Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
            with open(debug_path, 'w') as f:
                f.write(response)
            print(f"   Raw response saved to: {debug_path}")
            if attempt < max_retries:
                print(f"   Retrying ({attempt}/{max_retries})...")
            else:
                print(f"   All {max_retries} attempts exhausted.")
                return

    duration_seconds = round(time.perf_counter() - t_start, 1)
    print(f"\n   Inference time: {duration_seconds}s")

    # Validate
    print("\n6. Validating against mention_map...")
    issues = validate_events(data, mention_map)
    if issues:
        print(f"   WARNING: {len(issues)} validation issues:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print("   All entity references valid")

    # Post-process: derive mentioned_only_places from toponym_relations
    toponym_relations = data.get("toponym_relations", {})
    if toponym_relations:
        derived_mentioned = [
            {"mention_id": mid, "reason": f"category={category}"}
            for mid, category in toponym_relations.items()
            if category == "NO_REL"
        ]
        # If LLM also returned a mentioned_only_places, warn on discrepancy
        llm_mentioned = data.get("mentioned_only_places", [])
        if llm_mentioned:
            llm_mids = {mp.get("mention_id") if isinstance(mp, dict) else mp
                        for mp in llm_mentioned}
            derived_mids = {mp["mention_id"] for mp in derived_mentioned}
            if llm_mids != derived_mids:
                extra = llm_mids - derived_mids
                missing = derived_mids - llm_mids
                if extra:
                    print(f"   NOTE: {len(extra)} mention_ids in LLM's "
                          f"mentioned_only_places but not NO_REL in "
                          f"toponym_relations (using derived list)")
                if missing:
                    print(f"   NOTE: {len(missing)} NO_REL toponyms missing "
                          f"from LLM's mentioned_only_places (using derived list)")
        # Override with derived list for consistency
        data["mentioned_only_places"] = derived_mentioned
        n_toponyms = len(toponym_relations)
        n_visited = n_toponyms - len(derived_mentioned)
        print(f"   Toponym relations: {n_toponyms} total "
              f"({n_visited} visited, {len(derived_mentioned)} NO_REL)")

        # Warn about NEAR entities not assigned to any event's near_places
        near_mids = {mid for mid, cat in toponym_relations.items() if cat == "NEAR"}
        if near_mids:
            assigned_near = set()
            for ev in data.get("events", []):
                for np_mid in ev.get("near_places", []):
                    assigned_near.add(np_mid)
                for sub in ev.get("sub_events", []):
                    if isinstance(sub, dict):
                        for np_mid in sub.get("near_places", []):
                            assigned_near.add(np_mid)
            unassigned = near_mids - assigned_near
            if unassigned:
                print(f"   WARNING: {len(unassigned)} NEAR toponyms not assigned "
                      f"to any event's near_places: {sorted(unassigned)}")
    elif not data.get("mentioned_only_places"):
        print("   WARNING: no toponym_relations and no mentioned_only_places "
              "in output")

    # Save JSON
    json_dir = Path(output_json).parent
    json_dir.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n7. JSON saved to: {output_json}")

    # Save mention map (for TEI export later)
    mention_map_path = str(Path(output_json).with_name(
        Path(output_json).stem.replace("events", "mention_map") + ".json"
    ))
    with open(mention_map_path, 'w') as f:
        json.dump(mention_map, f, indent=2, ensure_ascii=False)
    print(f"   Mention map saved to: {mention_map_path}")

    # Generate RDF
    print("\n8. Generating RDF...")
    rdf_graph = generate_rdf(data, output_rdf, mention_map)

    # Save metadata (timing, config) alongside the events JSON
    meta_path = str(Path(output_json).with_suffix(".meta.json"))
    think_mode_str = _THINK
    if think_mode_str is True:
        think_mode_str = "true"
    elif think_mode_str is False:
        think_mode_str = "false"
    with open(meta_path, 'w') as f:
        json.dump({
            "model": OLLAMA_MODEL,
            "temperature": TEMPERATURE,
            "think_mode": think_mode_str,
            "source_text": Path(spacy_file).stem,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
            "num_api_failures": num_api_failures,
            "num_retries": attempt - 1,
            "prompt_length_chars": len(prompt),
            "response_length_chars": len(response),
            "num_events": len(data.get("events", [])),
            "num_toponym_relations": len(data.get("toponym_relations", {})),
            "num_entities": len(mention_map),
            "rdf_triples": len(rdf_graph),
        }, f, indent=2)
    print(f"   Metadata saved to: {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
