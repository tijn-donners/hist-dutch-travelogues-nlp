"""ATO Relation Extraction & RDF Generation.

Loads NER+EL output (_el.spacy), extracts travel events (E9_Move),
activities (E7_Activity), and their relationships to places according
to the Academic Tourism Ontology. Generates RDF output.

Usage:
    python relation_extraction/re.py
    (edit paths under Configuration)
"""

import json
import os
import re
from pathlib import Path

import spacy
from dotenv import load_dotenv
from jinja2 import Template
from ollama import Client
from spacy.tokens import DocBin
from spacy.util import load_config

from rdflib import RDF, RDFS, Literal, Namespace, URIRef
from rdflib.graph import Graph

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPACY_FILE = str(ROOT_DIR / "ner" / "ner-results"
                 / "1816_all_pages_gemma4:31b-cloud_el.spacy")
OFFSET_MAP_FILE = str(ROOT_DIR / "ner" / "ner-results"
                      / "1816_offset_map_gemma4:31b-cloud.json")
TXT_FILE = str(ROOT_DIR / "data" / "1816_third_letter.txt")
EXAMPLES_FILE = str(SCRIPT_DIR / "re_examples.yaml")
PROMPT_TEMPLATE = str(SCRIPT_DIR / "re_prompt_dutch.jinja")
OUTPUT_JSON = str(ROOT_DIR / "output" / "re" / "1816_third_letter_events.json")
OUTPUT_RDF = str(ROOT_DIR / "output" / "rdf" / "1816_third_letter_events.ttl")

OLLAMA_MODEL = "deepseek-v4-pro"
OLLAMA_HOST = "https://ollama.com"
OLLAMA_KEY = os.environ.get("OLLAMA_API_KEY")

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
ATO = Namespace("http://academictourism.com/entity/")
CIDOC = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
AAT = Namespace("http://vocab.getty.edu/page/aat/")
WD = Namespace("https://www.wikidata.org/wiki/")
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
    """Build full letter text with NER entities marked inline.

    Returns:
        annotated_text: full text with [entity](LABEL, QID) markers
        all_entities: list of {text, label, kb_id, start, end}
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

    # Build global entity list
    all_entities = []
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
            kb_id = ent.kb_id_ if ent.kb_id_ else None
            all_entities.append({
                "text": ent.text,
                "label": ent.label_,
                "kb_id": kb_id,
                "start": g_start,
                "end": g_end,
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
        kb_str = ent["kb_id"] if ent["kb_id"] else "null"
        marker = f"[{ent['text']}]({ent['label']}, {kb_str})"
        annotated = annotated[:ent["start"]] + marker + annotated[ent["end"]:]

    return annotated, filtered


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
def query_llm(prompt, model=OLLAMA_MODEL):
    """Send prompt to Ollama and return response text."""
    client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_KEY}"} if OLLAMA_KEY else None,
    )

    response = client.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.0},
    )
    return response["response"]


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


def validate_events(data, entities):
    """Validate that all referenced entities exist in the NER output."""
    entity_map = {}
    for e in entities:
        key = (e["text"].lower(), e["kb_id"])
        entity_map[key] = e

    issues = []

    def check_place(ref, event_id, role):
        if ref is None:
            return
        key = (ref["text"].lower(), ref["kb_id"])
        if key not in entity_map:
            issues.append(f"{event_id}: {role} '{ref['text']}' (kb_id={ref['kb_id']}) "
                          f"not found in NER entities")

    def check_event(event):
        eid = event.get("id", "?")
        check_place(event.get("moved_from"), eid, "moved_from")
        check_place(event.get("moved_to"), eid, "moved_to")
        check_place(event.get("took_place_at"), eid, "took_place_at")
        check_place(event.get("on_or_within"), eid, "on_or_within")

        for mp in event.get("mentions_places", []):
            check_place(mp, eid, "mentions_places")

        for sub in event.get("sub_events", []):
            if isinstance(sub, dict):
                check_event(sub)

    for event in data.get("events", []):
        check_event(event)

    for mp in data.get("mentioned_only_places", []):
        check_place(mp, "mentioned_only", "place")

    return issues


# ---------------------------------------------------------------------------
# RDF Generation
# ---------------------------------------------------------------------------
def event_uri(event_id):
    return ATO[event_id]


def loc_uri(text, kb_id):
    """Generate a location URI. Use Wikidata Q-ID if available, otherwise slug."""
    if kb_id:
        # Check if it's a Wikidata ID already
        return ATO[f"LOC.{kb_id}"]
    # Generate a slug from text
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text)[:30]
    return ATO[f"LOC.{slug}"]


def generate_rdf(data, output_path):
    """Generate RDF from extracted events JSON following ATO patterns."""
    g = Graph()
    g.bind("ato", ATO)
    g.bind("cidoc-crm", CIDOC)
    g.bind("skos", SKOS)
    g.bind("aat", AAT)
    g.bind("lrmoo", LRMOO)

    letter_id = data.get("letter_id", "BRF0003")

    # --- Letter as F2_Expression ---
    letter_uri = ATO[letter_id]
    g.add((letter_uri, RDF.type, LRMOO.F2_Expression))
    g.add((letter_uri, RDFS.label, Literal(f"Brief 3 — Cassel, July 1816")))

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

    # --- Collect all event IDs ---
    all_event_ids = []

    def collect_event_ids(events):
        for ev in events:
            if isinstance(ev, str):
                # Bare string ID reference
                full_id = ev.replace("_", ".RS")
                if not full_id.startswith(letter_id):
                    full_id = f"{letter_id}.{full_id}"
                all_event_ids.append(full_id)
                continue
            eid = ev.get("id")
            if eid:
                full_id = eid.replace("_", ".RS")
                if not full_id.startswith(letter_id):
                    full_id = f"{letter_id}.{full_id}"
                all_event_ids.append(full_id)
                if "sub_events" in ev:
                    collect_event_ids(ev["sub_events"])

    collect_event_ids(data.get("events", []))

    # --- Process each event ---
    prev_event_uri = None

    def process_events(events, parent_uri=None):
        nonlocal prev_event_uri

        for ev in events:
            eid = ev.get("id")
            if not eid:
                continue

            full_id = eid.replace("_", ".RS")
            if not full_id.startswith(letter_id):
                full_id = f"{letter_id}.{full_id}"

            ev_uri = ATO[full_id]
            ev_type = ev.get("type", "translocation")
            is_move = ev_type == "translocation"

            _add_event_types(g, ev_uri, is_move=is_move)
            g.add((ev_uri, RDFS.label, Literal(ev.get("label", ""))))

            # Participant
            g.add((ev_uri, CIDOC["P11_had_participant"], tg_uri))
            g.add((ev_uri, CIDOC["P12_occurred_in_the_presence_of"], tg_uri))

            # Mereology
            if parent_uri:
                g.add((ev_uri, CIDOC["P10_falls_within"], parent_uri))
                g.add((parent_uri, CIDOC["P10_contains"], ev_uri))
            else:
                g.add((ev_uri, CIDOC["P10_falls_within"], journey_uri))
                g.add((journey_uri, CIDOC["P10_contains"], ev_uri))

            # Spatiotemporal overlap with journey
            g.add((ev_uri, CIDOC["P132_spatiotemporally_overlaps_with"], journey_uri))

            # Translocation properties
            if is_move:
                mfrom = ev.get("moved_from")
                if mfrom:
                    from_uri = _ensure_place(g, mfrom["text"], mfrom.get("kb_id"))
                    g.add((ev_uri, CIDOC["P27_moved_from"], from_uri))

                mto = ev.get("moved_to")
                if mto:
                    to_uri = _ensure_place(g, mto["text"], mto.get("kb_id"))
                    g.add((ev_uri, CIDOC["P26_moved_to"], to_uri))

                mode = ev.get("mode_of_transportation")
                if mode:
                    g.add((ev_uri, CIDOC["P2_has_type"],
                           AAT["300248181"]))  # carriage/wheeled vehicle

            # Tour/Stay properties
            if ev_type in ("indoor_tour", "outdoor_tour", "stay"):
                tpa = ev.get("took_place_at")
                if tpa:
                    place_uri = _ensure_place(g, tpa["text"], tpa.get("kb_id"))
                    g.add((ev_uri, CIDOC["P7_took_place_at"], place_uri))

                onwi = ev.get("on_or_within")
                if onwi:
                    thing_uri = _ensure_physical_thing(g, onwi["text"], onwi.get("kb_id"))
                    g.add((ev_uri, CIDOC["P8_took_place_on_or_within"], thing_uri))

            # Temporal ordering
            if prev_event_uri:
                _add_ordering(g, prev_event_uri, ev_uri)

            # Mentions
            for mp in ev.get("mentions_places", []):
                place_uri = _ensure_place(g, mp["text"], mp.get("kb_id"))
                g.add((letter_uri, CIDOC["P67_refers_to"], place_uri))

            prev_event_uri = ev_uri

            # Sub-events referenced by string ID are top-level events
            # linked via falls_within — they're processed independently.
            # Only recurse into inline dict sub-events.
            if "sub_events" in ev:
                dict_subs = [s for s in ev["sub_events"] if isinstance(s, dict)]
                if dict_subs:
                    process_events(dict_subs, parent_uri=ev_uri)

    process_events(data.get("events", []))

    # --- Mentioned-only places ---
    for mp in data.get("mentioned_only_places", []):
        place_uri = _ensure_place(g, mp["text"], mp.get("kb_id"))
        g.add((letter_uri, CIDOC["P67_refers_to"], place_uri))

    # --- Serialize ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=output_path, format="turtle")
    print(f"RDF saved to: {output_path}")
    print(f"  Triples: {len(g)}")
    return g


def _add_event_types(graph, uri, is_move=False):
    """Add leaf CIDOC CRM types — superclasses are inferred by a reasoner."""
    # E7_Activity and E92_Spacetime_Volume are on separate branches
    # (E7 specializes E5_Event, E92 specializes E4_Period), so both needed.
    graph.add((uri, RDF.type, CIDOC.E7_Activity))
    graph.add((uri, RDF.type, CIDOC.E92_Spacetime_Volume))
    if is_move:
        graph.add((uri, RDF.type, CIDOC.E9_Move))


def _add_ordering(graph, earlier_uri, later_uri):
    """Add Allen interval ordering — forward direction only (inverses are implied)."""
    graph.add((earlier_uri, CIDOC["P183_ends_before_the_start_of"], later_uri))


def _ensure_place(graph, text, kb_id):
    """Get or create an E53_Place instance."""
    uri = loc_uri(text, kb_id)
    # Check if already exists
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E53_Place))
        graph.add((uri, RDFS.label, Literal(text)))
        if kb_id:
            graph.add((uri, SKOS.closeMatch, WD[kb_id]))
    return uri


def _ensure_physical_thing(graph, text, kb_id):
    """Get or create an E18_Physical_Thing + E53_Place instance."""
    uri = loc_uri(text, kb_id)
    if (uri, RDF.type, None) not in graph:
        graph.add((uri, RDF.type, CIDOC.E18_Physical_Thing))
        graph.add((uri, RDF.type, CIDOC.E53_Place))
        graph.add((uri, RDFS.label, Literal(text)))
        if kb_id:
            graph.add((uri, SKOS.closeMatch, WD[kb_id]))
    return uri


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ATO Relation Extraction & RDF Generation")
    print("=" * 60)

    # Find EL spacy file
    spacy_file = SPACY_FILE
    if not Path(spacy_file).exists():
        # Try to find any _el.spacy file
        candidates = list(Path(SPACY_FILE).parent.glob("*_el.spacy"))
        if candidates:
            spacy_file = str(sorted(candidates)[-1])
            print(f"Using latest EL file: {spacy_file}")
        else:
            print("No _el.spacy file found. Run EL pipeline first.")
            return

    # Load and annotate
    print("\n1. Loading NER+EL entities...")
    annotated_text, entities = build_annotated_text(spacy_file, OFFSET_MAP_FILE)
    print(f"   Entities: {len(entities)}")
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

    # Query LLM
    print(f"\n4. Querying {OLLAMA_MODEL}...")
    response = query_llm(prompt)
    print(f"   Response length: {len(response)} chars")

    # Parse JSON
    print("\n5. Parsing response...")
    try:
        data = extract_json(response)
    except json.JSONDecodeError as e:
        print(f"   JSON parse error: {e}")
        # Save raw response for debugging
        debug_path = str(Path(OUTPUT_JSON).with_suffix(".raw.txt"))
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, 'w') as f:
            f.write(response)
        print(f"   Raw response saved to: {debug_path}")
        return

    # Validate
    print("\n6. Validating against NER entities...")
    issues = validate_events(data, entities)
    if issues:
        print(f"   WARNING: {len(issues)} validation issues:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print("   All entity references valid")

    # Save JSON
    json_dir = Path(OUTPUT_JSON).parent
    json_dir.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n7. JSON saved to: {OUTPUT_JSON}")

    # Generate RDF
    print("\n8. Generating RDF...")
    generate_rdf(data, OUTPUT_RDF)

    print("\nDone.")


if __name__ == "__main__":
    main()
