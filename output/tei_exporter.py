"""Export spaCy DocBin (with NER + EL annotations) to TEI/XML.

Reads an _el.spacy DocBin, offset maps, and optionally RE pipeline outputs
(mention_map.json + events.ttl) to produce a self-contained TEI document with:
- <pb n="page"/> page milestones
- <lb/> linebreak milestones (when text has newlines)
- <placeName xml:id="e5" type="E53_Place" ref="..." key="..."> for places
- <rs xml:id="e12" type="E18_Physical_Thing" ref="..." key="..."> for physical things
- <standOff> with embedded RDF/XML containing ATO triples and CT.* citations

Usage:
    python output/tei_exporter.py
    (interactively selects _el.spacy file)
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import spacy
from spacy.tokens import Span, Token, DocBin

# Register custom extension attributes for EL data (must happen before DocBin loading)
if not Span.has_extension("kb_id_wikidata_"):
    Span.set_extension("kb_id_wikidata_", default=None)
if not Token.has_extension("ent_kb_id_wikidata_"):
    Token.set_extension("ent_kb_id_wikidata_", default=None)
if not Span.has_extension("kb_id_geonames_"):
    Span.set_extension("kb_id_geonames_", default=None)
if not Token.has_extension("ent_kb_id_geonames_"):
    Token.set_extension("ent_kb_id_geonames_", default=None)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EL_RESULTS_DIR = ROOT_DIR / "entity_linking" / "el-results"
NER_RESULTS_DIR = ROOT_DIR / "ner" / "ner-output"
RE_OUTPUT_DIR = ROOT_DIR / "output" / "re"
RDF_OUTPUT_DIR = ROOT_DIR / "output" / "rdf"
OUTPUT_FILE = str(SCRIPT_DIR / "tei" / "1816_third_letter.tei.xml")


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
def select_el_spacy_file():
    """Scan el-results/ for *_el.spacy files and let the user pick one."""
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
    """Find the matching offset map for an _el.spacy file."""
    stem = Path(el_spacy_path).stem
    if stem.endswith("_el"):
        base_stem = stem[:-3]
    else:
        base_stem = stem
    offset_stem = base_stem + "_offset_map"

    # First try sibling of the _el.spacy file (copied by el.py to el-results/)
    candidate = Path(el_spacy_path).parent / f"{offset_stem}.json"
    if candidate.exists():
        return str(candidate)

    # Then try flat EL_RESULTS_DIR (legacy)
    candidate = EL_RESULTS_DIR / f"{offset_stem}.json"
    if candidate.exists():
        return str(candidate)

    # Fallback: search NER output recursively
    for candidate in sorted(NER_RESULTS_DIR.rglob(f"{offset_stem}.json")):
        return str(candidate)

    print(f"Warning: no offset map found for {Path(el_spacy_path).name}")
    return None


def find_re_outputs(el_spacy_path):
    """Auto-discover mention_map.json and events.ttl from RE pipeline output.

    Args:
        el_spacy_path: Path to the _el.spacy file.

    Returns:
        Tuple of (mention_map_path, rdf_path) — either may be None.
    """
    stem = Path(el_spacy_path).stem
    if stem.endswith("_el"):
        base_stem = stem[:-3]
    else:
        base_stem = stem

    # mention_map: the RE output JSON is always named 1816_third_letter_mention_map.json
    mention_map_path = RE_OUTPUT_DIR / "1816_third_letter_mention_map.json"
    if not mention_map_path.exists():
        mention_map_path = None

    # rdf: look for 1816_third_letter_events.ttl
    rdf_path = RDF_OUTPUT_DIR / "1816_third_letter_events.ttl"
    if not rdf_path.exists():
        rdf_path = None

    if mention_map_path:
        print(f"Found mention map: {mention_map_path}")
    if rdf_path:
        print(f"Found RDF: {rdf_path}")

    return str(mention_map_path) if mention_map_path else None, \
        str(rdf_path) if rdf_path else None

# ---------------------------------------------------------------------------
# Page number sorting
# ---------------------------------------------------------------------------
_ROMAN_VALUES = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
                 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13,
                 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18,
                 'XIX': 19, 'XX': 20}


def _page_sort_key(pn):
    s = str(pn)
    if s.isdigit():
        return int(s)
    if s.upper() in _ROMAN_VALUES:
        return _ROMAN_VALUES[s.upper()]
    return 0


_MARKER_RE = re.compile(r'^\[[IVXLCDM\d]+\]\s*')


def _strip_marker(text):
    """Strip leading [N] or [XIV] marker from text (present in .txt mode)."""
    return _MARKER_RE.sub('', text, count=1)


def build_tei(spacy_path, offset_map_path, output_path,
              line_map_path=None, mention_map_path=None, rdf_path=None):
    # --- Load data ---
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(spacy_path)
    docs = list(db.get_docs(nlp.vocab))

    with open(offset_map_path) as f:
        offset_map = json.load(f)

    line_map = None
    if line_map_path and Path(line_map_path).exists():
        with open(line_map_path) as f:
            line_map = json.load(f)

    # --- Load mention map for xml:id assignment ---
    mention_lookup = {}  # (start, end, text_lower) -> mention_id
    if mention_map_path and Path(mention_map_path).exists():
        with open(mention_map_path) as f:
            mention_map = json.load(f)
        for mid, info in mention_map.items():
            key = (info["start"], info["end"], info["text"].lower())
            mention_lookup[key] = mid
        print(f"Loaded {len(mention_map)} mention IDs from {mention_map_path}")

    sorted_pages = sorted(offset_map.items(), key=lambda x: _page_sort_key(x[0]))

    if len(docs) != len(sorted_pages):
        print(f"Warning: {len(docs)} docs but {len(sorted_pages)} pages in offset map")

    # --- Detect [N] markers (.txt mode) ---
    has_markers = any(doc.text.lstrip().startswith('[') for doc in docs[:3])
    if has_markers:
        print("Detected [N] page markers in text — stripping for TEI output")

    # --- Build full text and page-start positions ---
    full_text_parts = []
    page_positions = {}  # page_num -> position in full_text
    pos = 0
    for i in range(min(len(docs), len(sorted_pages))):
        doc = docs[i]
        page_num, _ = sorted_pages[i]
        text = doc.text
        if has_markers:
            text = _strip_marker(text)
        page_positions[page_num] = pos
        full_text_parts.append(text)
        pos += len(text)

    full_text = "".join(full_text_parts)

    # --- Build global entity list ---
    all_entities = []  # (global_start, global_end, label, kb_id, text, mention_id, wikidata_id, geonames_id)
    for i in range(min(len(docs), len(sorted_pages))):
        doc = docs[i]
        page_num, _ = sorted_pages[i]
        base_offset = page_positions[page_num]

        doc_text = doc.text
        marker_len = 0
        if has_markers:
            stripped = _strip_marker(doc_text)
            marker_len = len(doc_text) - len(stripped)

        for ent in doc.ents:
            global_start = base_offset + ent.start_char - marker_len
            global_end = base_offset + ent.end_char - marker_len
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
            mention_key = (global_start, global_end, ent.text.lower())
            mention_id = mention_lookup.get(mention_key)
            # Normalize old E22_Human-made_Object label for backward compatibility
            label = "Mode_of_Transportation" if ent.label_ == "E22_Human-made_Object" else ent.label_
            all_entities.append((global_start, global_end, label, kb_id, ent.text, mention_id, wikidata_id, geonames_id))

    all_entities.sort(key=lambda x: (x[0], -x[1]))

    # --- Build open/close event maps ---
    open_events = defaultdict(list)
    close_events = defaultdict(list)

    for idx, (start, end, label, kb_id, text, mention_id, wikidata_id, geonames_id) in enumerate(all_entities):
        if 0 <= start < end <= len(full_text):
            open_events[start].append(idx)
            close_events[end].append(idx)

    # --- Build page boundary positions ---
    page_boundaries = {page_positions[pn]: pn for pn, _ in sorted_pages
                       if pn in page_positions}

    # --- Walk through text, emit TEI in <ab> paragraphs ---
    raw_paragraphs = full_text.split('\n\n')
    body_parts = []
    para_start_pos = 0

    for para_text in raw_paragraphs:
        para_len = len(para_text)

        if not para_text.strip():
            para_start_pos += para_len + 2  # +2 for the \n\n separator
            continue

        para_end_pos = para_start_pos + para_len

        # Build open/close events for entities within this paragraph
        para_open = defaultdict(list)
        para_close = defaultdict(list)

        for idx, (start, end, label, kb_id, text, mention_id, wikidata_id, geonames_id) in enumerate(all_entities):
            if start >= para_start_pos and end <= para_end_pos:
                para_open[start - para_start_pos].append(
                    (idx, end - para_start_pos, label, kb_id, text, mention_id, wikidata_id, geonames_id))
                para_close[end - para_start_pos].append(idx)
            elif start < para_start_pos < end and end <= para_end_pos:
                # Entity starts before paragraph — truncate start
                para_open[0].append((idx, end - para_start_pos, label, kb_id, text, mention_id, wikidata_id, geonames_id))
                para_close[end - para_start_pos].append(idx)
            elif start >= para_start_pos and start < para_end_pos < end:
                # Entity ends after paragraph — truncate end
                para_open[start - para_start_pos].append(
                    (idx, para_len, label, kb_id, text, mention_id, wikidata_id, geonames_id))
                para_close[para_len].append(idx)

        # Page boundaries within this paragraph
        para_boundaries = {}
        for pb_pos, pn in page_boundaries.items():
            if para_start_pos <= pb_pos < para_end_pos:
                para_boundaries[pb_pos - para_start_pos] = pn

        # Process paragraph character by character
        ab_parts = []
        open_tags = []  # stack of (end_pos, tag)

        for pos, char in enumerate(para_text):
            # Close entities ending at this position
            while open_tags and open_tags[-1][0] <= pos:
                _, tag = open_tags.pop()
                ab_parts.append(f'</{tag}>')

            # Page boundary
            if pos in para_boundaries:
                pn = para_boundaries[pos]
                ab_parts.append(f'<pb n="{pn}"/>')

            # Open entities starting at this position (longest first for nesting)
            for ent_idx, end, label, kb_id, text, mention_id, wikidata_id, geonames_id in sorted(
                    para_open.get(pos, []), key=lambda x: -x[1]):
                if label == 'E53_Place':
                    tag = 'placeName'
                elif label == 'E52_Time_Span':
                    tag = 'date'
                elif label == 'F2_Expression':
                    tag = 'rs'
                else:
                    tag = 'rs'
                attrs = f'type="{label}"'
                if mention_id:
                    attrs += f' xml:id="{mention_id}"'
                # Handle KB IDs (not for generic types like Mode_of_Transportation and F2_Expression)
                refs = []
                key_val = None
                if wikidata_id is not None and label not in ('Mode_of_Transportation', 'F2_Expression'):
                    refs.append(f'http://www.wikidata.org/entity/{wikidata_id}')
                    key_val = wikidata_id
                if geonames_id is not None and label not in ('Mode_of_Transportation', 'F2_Expression'):
                    refs.append(f'http://www.geonames.org/{geonames_id}')
                    if key_val is None:
                        key_val = geonames_id
                if refs:
                    attrs += f' ref="{" ".join(refs)}" key="{key_val}"'
                ab_parts.append(f'<{tag} {attrs}>')
                open_tags.append((end, tag))

            # Emit character
            if char == '\n':
                ab_parts.append('<lb/>')
            else:
                ab_parts.append(xml_escape(char))

        # Close any remaining open entities
        while open_tags:
            _, tag = open_tags.pop()
            ab_parts.append(f'</{tag}>')

        ab_inner = ''.join(ab_parts)
        body_parts.append(f'<ab>{ab_inner}</ab>')

        para_start_pos = para_end_pos + 2  # +2 for the \n\n separator

    body_inner = '\n'.join(body_parts)

    # --- Build full TEI document ---
    TEI_NS = "http://www.tei-c.org/ns/1.0"
    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    CIDOC_NS = "http://www.cidoc-crm.org/cidoc-crm/"
    ATO_NS = "http://academictourism.com/entity/"
    ACADT_NS = "http://academictourism.com/academictourism#"

    # --- Load RDF and serialize to RDF/XML for standOff ---
    standoff_xml = ""
    if rdf_path and Path(rdf_path).exists():
        from rdflib import Graph
        g = Graph()
        g.parse(rdf_path, format="turtle")
        rdf_xml = g.serialize(format="xml")
        if isinstance(rdf_xml, bytes):
            rdf_xml = rdf_xml.decode("utf-8")
        # Strip the <?xml ...?> declaration — we embed inside TEI's <xenoData>
        rdf_xml = re.sub(r'<\?xml[^?]*\?>\s*', '', rdf_xml)
        standoff_xml = f'\n  <standOff>\n    <xenoData>\n    {rdf_xml}\n    </xenoData>\n  </standOff>'
        print(f"  Embedded {len(g)} RDF triples in <standOff>")

    encoding_note = ""
    if mention_lookup:
        encoding_note += ('<p>Entity mentions carry xml:id attributes linking to '
                          'CT.* citations in the standOff RDF.</p>')
    if standoff_xml:
        encoding_note += ('<p>Relation extraction triples (ATO/CIDOC CRM) are '
                          'embedded as RDF/XML in <gi>standOff</gi>.</p>')

    tei = f'''<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="{TEI_NS}"
     xmlns:rdf="{RDF_NS}"
     xmlns:crm="{CIDOC_NS}"
     xmlns:ato="{ATO_NS}"
     xmlns:acadt="{ACADT_NS}">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Derde Brief — Cassel, July 1816</title>
        <author>Groningen Student (Academic Tourism Project)</author>
      </titleStmt>
      <publicationStmt>
        <p>Generated by hist-dutch-travelogues-nlp NER + Entity Linking pipeline</p>
      </publicationStmt>
      <sourceDesc>
        <p>Transcription of a letter from a Groningen student traveling through Germany, ~1816.</p>
      </sourceDesc>
    </fileDesc>
    <encodingDesc>
      <editorialDecl>
        <p>Diplomatic transcription of the manuscript.</p>
        {encoding_note}
      </editorialDecl>
    </encodingDesc>
  </teiHeader>
  <text>
    <body>
      {body_inner}
    </body>
  </text>{standoff_xml}
</TEI>'''

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(tei)

    print(f"TEI exported to: {output_path}")
    print(f"  Entities: {len(all_entities)}")
    print(f"  Pages: {len(sorted_pages)}")
    print(f"  Text length: {len(full_text)} chars")
    if mention_lookup:
        print(f"  Entities with xml:id: {sum(1 for e in all_entities if e[5])}")

    return tei


if __name__ == "__main__":
    spacy_file = select_el_spacy_file()
    offset_map = find_offset_map(spacy_file)
    if offset_map is None:
        print("Cannot proceed without offset map.")
        raise SystemExit(1)
    mention_map_path, rdf_path = find_re_outputs(spacy_file)
    build_tei(spacy_file, offset_map, OUTPUT_FILE,
              mention_map_path=mention_map_path, rdf_path=rdf_path)
