"""Export spaCy DocBin (with NER + EL annotations) to TEI/XML.

Reads an _el.spacy DocBin and offset maps, produces a single TEI document with:
- <pb n="page"/> page milestones
- <lb/> linebreak milestones (when text has newlines)
- <placeName type="E53_Place" ref="..." key="..."> for places
- <rs type="E19_Physical_Thing" ref="..." key="..."> for physical things

Usage:
    python output/tei_exporter.py
    (edit SPACY_FILE, OFFSET_MAP, and LINE_MAP constants at top of file)
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import spacy
from spacy.tokens import DocBin

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPACY_FILE = str(ROOT_DIR / "ner" / "ner-results" / "1816_all_pages_gemma4:31b-cloud_el.spacy")
OFFSET_MAP = str(ROOT_DIR / "ner" / "ner-results" / "1816_offset_map_gemma4:31b-cloud.json")
LINE_MAP = str(ROOT_DIR / "ner" / "ner-results" / "1816_line_map_gemma4:31b-cloud.json")
OUTPUT_FILE = str(SCRIPT_DIR / "tei" / "1816_third_letter.tei.xml")

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


def build_tei(spacy_path, offset_map_path, output_path, line_map_path=None):
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
    all_entities = []  # (global_start, global_end, label, kb_id, text)
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
            kb_id = ent.kb_id_ if ent.kb_id_ else None
            all_entities.append((global_start, global_end, ent.label_, kb_id, ent.text))

    all_entities.sort(key=lambda x: (x[0], -x[1]))

    # --- Build open/close event maps ---
    open_events = defaultdict(list)
    close_events = defaultdict(list)

    for idx, (start, end, label, kb_id, text) in enumerate(all_entities):
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

        for idx, (start, end, label, kb_id, text) in enumerate(all_entities):
            if start >= para_start_pos and end <= para_end_pos:
                para_open[start - para_start_pos].append(
                    (idx, end - para_start_pos, label, kb_id))
                para_close[end - para_start_pos].append(idx)
            elif start < para_start_pos < end and end <= para_end_pos:
                # Entity starts before paragraph — truncate start
                para_open[0].append((idx, end - para_start_pos, label, kb_id))
                para_close[end - para_start_pos].append(idx)
            elif start >= para_start_pos and start < para_end_pos < end:
                # Entity ends after paragraph — truncate end
                para_open[start - para_start_pos].append(
                    (idx, para_len, label, kb_id))
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
            for ent_idx, end, label, kb_id in sorted(
                    para_open.get(pos, []), key=lambda x: -x[1]):
                tag = 'placeName' if label == 'E53_Place' else 'rs'
                attrs = f'type="{label}"'
                if kb_id:
                    attrs += f' ref="http://www.wikidata.org/entity/{kb_id}" key="{kb_id}"'
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
    tei = f'''<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="{TEI_NS}">
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
      </editorialDecl>
    </encodingDesc>
  </teiHeader>
  <text>
    <body>
      {body_inner}
    </body>
  </text>
</TEI>'''

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(tei)

    print(f"TEI exported to: {output_path}")
    print(f"  Entities: {len(all_entities)}")
    print(f"  Pages: {len(sorted_pages)}")
    print(f"  Text length: {len(full_text)} chars")

    return tei


if __name__ == "__main__":
    build_tei(SPACY_FILE, OFFSET_MAP, OUTPUT_FILE, LINE_MAP)
