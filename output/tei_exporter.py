"""TEI/XML canonical edition — zero entity loss, alongside the ALTO viewer edition.

Exports the spaCy DocBin (NER + EL) to a single self-contained TEI document for the
whole letter. This is the **canonical edition**: it carries no coordinates and performs
no HTR-line alignment, so it tags *every* entity directly from the gold text (313/313),
including the handful the ALTO edition cannot attach (the title `Derde Brief`, editorial
illustration captions, HTR/gold numeric conflicts). The per-page ALTO files remain the
coordinate-bearing viewer edition.

The body is built from the **verbatim** gold ``full_text`` (reusing
``alto_exporter.build_full_text_and_entities``), so entity offsets from
``mention_map.json`` line up by construction — no marker-stripping arithmetic. A single
character-walk emits:

* ``<pb n="page"/>`` milestones at each page boundary (the residual ``NN]`` page markers
  are suppressed from the canonical text);
* ``<lb/>`` for line breaks, ``<ab>`` blocks for paragraph breaks;
* inline entity tags — ``placeName`` / ``persName`` / ``orgName`` / ``date`` / ``rs`` —
  with ``@xml:id`` (the mention id ``eNN``), ``@type``, ``@ref`` (Wikidata + GeoNames)
  and ``@key``;
* ``<note type="editorial">`` wrapping editorial illustration/marginalia markers such as
  ``[illustratie linker blad: Wilhelms Brücke in Cassel]``, entities tagged inside;
* the full CIDOC-CRM / ATO relation-extraction RDF embedded as RDF/XML in
  ``<standOff><xenoData>``, linked back to the inline mentions via the ``ato:CT.*``
  citation URIs.

Usage:
    python3 output/tei_exporter.py                 # uses working defaults
    python3 output/tei_exporter.py --spacy FILE ... # override inputs
"""

import argparse
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import spacy
from spacy.tokens import Span, Token, DocBin

# Register custom extension attributes for EL data (must happen before DocBin loading).
# Also registered by alto_exporter on import, but the guards make this idempotent.
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
sys.path.insert(0, str(SCRIPT_DIR))
import alto_exporter as ae  # reuse the gold-text + RDF pipeline

EL_RESULTS_DIR = ROOT_DIR / "entity_linking" / "el-results"
NER_RESULTS_DIR = ROOT_DIR / "ner" / "ner-output"
RE_OUTPUT_DIR = ROOT_DIR / "output" / "re"
RDF_OUTPUT_DIR = ROOT_DIR / "output" / "rdf"
TEI_NS = "http://www.tei-c.org/ns/1.0"

# Defaults that exist on a current checkout (mirrors alto_exporter.main).
DEFAULT_SPACY = str(EL_RESULTS_DIR / "1816_el_gs_el.spacy")
DEFAULT_OFFSET_MAP = str(EL_RESULTS_DIR / "1816_el_gs_offset_map.json")
DEFAULT_MENTION_MAP = str(
    RE_OUTPUT_DIR / "1816_el_gs__deepseek-v4-pro_t0.0_thinkDefault_mention_map.json")
DEFAULT_RDF = str(
    RDF_OUTPUT_DIR / "1816_el_gs__deepseek-v4-pro_t0.0_thinkDefault_events.ttl")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "tei" / "1816_third_letter.tei.xml")
DEFAULT_LETTER_ID = "BRF0003"

# Namespaces declared on the root <TEI> for readability; the embedded <rdf:RDF>
# re-declares the ones it needs.
NS = {
    "": TEI_NS,
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "crm": "http://www.cidoc-crm.org/cidoc-crm/",
    "ato": "http://academictourism.com/entity/",
    "acadt": "http://academictourism.com/academictourism#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "lrmoo": "http://iflastandards.info/ns/lrm/lrmoo/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}


# ---------------------------------------------------------------------------
# Tag selection
# ---------------------------------------------------------------------------
def _tei_tag(label: str) -> str:
    """CIDOC-CRM label -> TEI element name for an inline entity mention."""
    if label == "E53_Place":
        return "placeName"
    if label in ("E21_Person", "E39_Actor"):
        return "persName"
    if label == "E74_Group":
        return "orgName"
    if label == "E52_Time-Span":
        return "date"
    return "rs"


def _entity_attrs(label, mid, wd, gn):
    """Build the attribute string for an inline entity tag."""
    attrs = f'type="{xml_escape(label)}"'
    if mid:
        attrs += f' xml:id="{xml_escape(str(mid))}"'
    refs = []
    key_val = None
    if wd:
        refs.append(f"http://www.wikidata.org/entity/{wd}")
        key_val = wd
    if gn:
        refs.append(f"https://sws.geonames.org/{gn}/")
        if key_val is None:
            key_val = gn
    if refs:
        attrs += f' ref="{" ".join(refs)}" key="{xml_escape(str(key_val))}"'
    return attrs


# ---------------------------------------------------------------------------
# Page-marker + editorial-marker span computation
# ---------------------------------------------------------------------------
def _page_marker_spans(full_text, page_positions, sorted_pages):
    """Return (pb_at, skip) where:
    * ``pb_at[start] = page_num`` — emit <pb n="page"/> at this verbatim position;
    * ``skip[start] = end`` — verbatim char interval [start, end) to suppress
      (the page marker plus its trailing whitespace separator).

    The marker is detected by **inspecting the actual text** at each page
    boundary (not assumed), so this works for any page-number width — single-digit
    ``[1]``, multi-digit ``[111]``, roman ``[XIV]`` — and is robust to a future
    gold rebuild that changes the marker shape. Three forms are recognised:

    * full ``[page]`` beginning exactly at the page start (a future cleaned gold);
    * residual ``page]`` at the page start with the ``[`` straddling the previous
      slice (the current gold, e.g. ``16]`` for page 16, ``17]`` for page 17 with
      ``[`` at the end of page 16's text);
    * no recognisable marker — emit ``<pb/>`` anyway, skip nothing.

    If no marker is found, ``skip[start] == start`` (the walker still advances).
    """
    pb_at, skip = {}, {}
    n = len(full_text)
    for page, _ in sorted_pages:
        p = page_positions[page]
        s = str(page)
        L = len(s)
        if full_text[p:p + L + 2] == f"[{s}]":          # full "[page]" at p
            start, end = p, p + L + 2
        elif full_text[p:p + L + 1] == f"{s}]":          # residual "page]" at p
            start = p - 1 if (p > 0 and full_text[p - 1] == "[") else p
            end = p + L + 1
        else:                                            # no marker found
            start, end = p, p
        # swallow the trailing whitespace separator, but never a '[' (an editorial
        # marker) or other non-whitespace.
        while end < n and full_text[end] in " \t\n":
            end += 1
        pb_at[start] = page
        skip[start] = end
    return pb_at, skip


_PAGE_LIKE_RE = re.compile(r"\[(?:\d+|[IVXLCDM]+)\]$")


def _editorial_note_spans(full_text):
    """Detect editorial illustration/marginalia markers ``[... ...]``.

    Returns {start: end} for the bracketed span [start, end). The ``[``, ``]`` are
    dropped at emit time; the content is wrapped in <note type="editorial">.

    Page-number markers (``[16]``, ``[111]``, roman ``[XIV]``) are excluded — they
    are page breaks, not editorial notes. (Roman markers only appear as literal
    text when the gold builder — which slices on ``\\[\\d+\\]`` only — did not
    split them into the offset map; they pass through as plain text.)
    """
    spans = {}
    for m in re.finditer(r"\[[^\]]+?\]", full_text):
        if _PAGE_LIKE_RE.match(m.group()):
            continue
        spans[m.start()] = m.end()
    return spans


# ---------------------------------------------------------------------------
# Core build
# ---------------------------------------------------------------------------
def build_tei(spacy_path, offset_map_path, output_path,
              mention_map_path=None, rdf_path=None, letter_id=DEFAULT_LETTER_ID):
    # --- load data ---
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(spacy_path)
    docs = list(db.get_docs(nlp.vocab))
    offset_map = json.load(open(offset_map_path))
    sorted_pages = sorted(offset_map.items(), key=lambda x: ae._page_sort_key(x[0]))
    mention_map = {}
    if mention_map_path and Path(mention_map_path).exists():
        mention_map = json.load(open(mention_map_path))
    mention_lookup = ae.build_mention_lookup(mention_map)

    # verbatim full_text + entities (offsets match mention_map by construction)
    full_text, page_positions, all_entities = ae.build_full_text_and_entities(
        docs, sorted_pages, mention_lookup)
    if len(docs) != len(sorted_pages):
        print(f"  Warning: {len(docs)} docs but {len(sorted_pages)} pages")

    pb_at, skip = _page_marker_spans(full_text, page_positions, sorted_pages)
    note_span = _editorial_note_spans(full_text)

    # entity open map: position -> list of (end, label, mid, wd, gn) longest-first
    opens = {}
    for (gs, ge, label, kb, text, mid, wd, gn) in all_entities:
        if not (0 <= gs < ge <= len(full_text)):
            continue
        opens.setdefault(gs, []).append((ge, label, mid, wd, gn))
    for gs in opens:
        opens[gs].sort(key=lambda t: -t[0])  # longest first for safe nesting

    # --- single character-walk over the verbatim text ---
    parts = ["<ab>"]
    ab_has_content = False
    open_stack = []   # (end, tag) of currently-open entity tags
    note_end = None   # verbatim position of the closing ']' of the active note
    i, n = 0, len(full_text)
    while i < n:
        # close entities ending at/before this position
        while open_stack and open_stack[-1][0] <= i:
            parts.append(f"</{open_stack.pop()[1]}>")

        # page marker -> emit <pb/> and skip the marker span (advancing at least
        # one char so a missing-marker boundary, skip[i]==i, cannot loop)
        if i in skip:
            # @ed ("print") marks the page break as belonging to the original
            # printed source's pagination (TEI att.edition); required by stricter
            # ODD profiles that reject a bare <pb n="..."/>.
            parts.append(f'<pb ed="print" n="{pb_at[i]}"/>')
            ab_has_content = True
            i = max(skip[i], i + 1)
            continue

        # editorial note start -> open <note>, skip the '['
        if i in note_span:
            parts.append('<note type="editorial">')
            note_end = note_span[i] - 1
            ab_has_content = True
            i += 1
            continue

        # editorial note end (the ']') -> close <note>, skip the ']'
        if note_end is not None and i == note_end:
            parts.append("</note>")
            note_end = None
            i += 1
            continue

        # open entities starting here
        for (ge, label, mid, wd, gn) in opens.get(i, []):
            tag = _tei_tag(label)
            parts.append(f"<{tag} {_entity_attrs(label, mid, wd, gn)}>")
            open_stack.append((ge, tag))
            ab_has_content = True

        ch = full_text[i]
        if ch == "\n":
            j = i
            while j < n and full_text[j] == "\n":
                j += 1
            run = j - i
            if run >= 2 and not open_stack and ab_has_content:
                # paragraph break: only when no entity is open and the block has text
                parts.append("</ab>\n<ab>")
                ab_has_content = False
            else:
                parts.append("<lb/>")
                ab_has_content = True
            i = j
            continue

        parts.append(xml_escape(ch))
        ab_has_content = True
        i += 1

    while open_stack:
        parts.append(f"</{open_stack.pop()[1]}>")
    # drop a trailing empty <ab></ab>
    if parts[-1] == "<ab>":
        parts.pop()
    else:
        parts.append("</ab>")
    body_inner = "".join(parts)

    # --- standOff RDF (canonical, child-sorted RDF/XML) ---
    standoff_xml = ""
    n_triples = 0
    if rdf_path and Path(rdf_path).exists():
        rdf_xml = ae._rdf_to_xml(rdf_path)
        standoff_xml = (f'\n  <standOff>\n    <xenoData>\n    {rdf_xml}\n    '
                        f'</xenoData>\n  </standOff>')
        try:
            import rdflib
            n_triples = len(rdflib.Graph().parse(rdf_path, format="turtle"))
        except Exception:
            pass

    ns_decl = " ".join(f'xmlns{":" + k if k else ""}="{v}"' for k, v in NS.items())
    tei = f'''<?xml version="1.0" encoding="UTF-8"?>
<TEI {ns_decl}>
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Derde Brief — Cassel, July 1816</title>
        <author>Groningen student (Academic Tourism Project)</author>
      </titleStmt>
      <publicationStmt>
        <p>Generated by the hist-dutch-travelogues-nlp NER + Entity Linking +
          Relation Extraction pipeline.</p>
      </publicationStmt>
      <sourceDesc>
        <p>Diplomatic transcription of a letter from a Groningen student
          travelling through Germany, ca. 1816. Source:
          <ptr target="data/1816_third_letter.txt"/> (printed pages 16–25).</p>
      </sourceDesc>
    </fileDesc>
    <encodingDesc>
      <editorialDecl>
        <p>This is the canonical edition: the body is the verbatim manual
          transcription, with no coordinate/line alignment (the coordinate
          edition is the per-page ALTO XML). Every named entity is tagged inline,
          so no entity is lost.</p>
        <p>Entity mentions carry <att>xml:id</att> attributes
          (<val>e1</val>…<val>e{len(all_entities)}</val>) that link to the
          <gi>standOff</gi> RDF: the citation <val>ato:CT.{letter_id}.eNN</val>
          (an LRMoo F2_Expression) <att>P67_refers_to</att> the entity it denotes.
          <att>ref</att> points to Wikidata / GeoNames authority records.</p>
        <p>Page breaks are encoded as <gi>pb</gi> milestones
          (<tag>pb n="16"</tag>…<tag>pb n="25"</tag>); the residual
          <val>NN]</val> page markers present in the source transcription are
          suppressed.</p>
        <p>Editorial illustration/marginalia markers (e.g.
          <val>[illustratie linker blad: …]</val>) are wrapped in
          <tag>note type="editorial"</tag>; entities inside them are tagged.</p>
        <p>Relation-extraction triples (CIDOC-CRM / ATO) are embedded as RDF/XML
          in <gi>standOff</gi>/<gi>xenoData</gi>.</p>
      </editorialDecl>
    </encodingDesc>
  </teiHeader>{standoff_xml}
  <text>
    <body>
      {body_inner}
    </body>
  </text>
</TEI>'''

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(tei, encoding="utf-8")

    n_tagged = sum(1 for e in all_entities if e[5])
    print(f"TEI canonical edition exported to: {output_path}")
    print(f"  entities: {len(all_entities)}  (with xml:id: {n_tagged})")
    print(f"  pages: {len(sorted_pages)}  ({sorted_pages[0][0]}–{sorted_pages[-1][0]})")
    print(f"  editorial notes: {len(note_span)}")
    if n_triples:
        print(f"  standOff RDF triples: {n_triples}")
    print(f"  body length: {len(full_text)} chars")
    return tei


# ---------------------------------------------------------------------------
# Interactive fallbacks (only used when --spacy is not given)
# ---------------------------------------------------------------------------
def select_el_spacy_file():
    spacy_files = sorted(EL_RESULTS_DIR.glob("*_el.spacy"))
    if not spacy_files:
        print(f"No *_el.spacy files in {EL_RESULTS_DIR}"); raise SystemExit(1)
    if len(spacy_files) == 1:
        return str(spacy_files[0])
    print("Available _el.spacy files:")
    for i, f in enumerate(spacy_files, 1):
        print(f"  [{i}] {f.name}")
    idx = int(input("Select number: ").strip()) - 1
    if 0 <= idx < len(spacy_files):
        return str(spacy_files[idx])
    raise SystemExit(1)


def find_offset_map(el_spacy_path):
    stem = Path(el_spacy_path).stem
    base = stem[:-3] if stem.endswith("_el") else stem
    cand = Path(el_spacy_path).parent / f"{base}_offset_map.json"
    if cand.exists():
        return str(cand)
    cand = EL_RESULTS_DIR / f"{base}_offset_map.json"
    if cand.exists():
        return str(cand)
    for c in sorted(NER_RESULTS_DIR.rglob(f"{base}_offset_map.json")):
        return str(c)
    return None


def main():
    ap = argparse.ArgumentParser(description="Export the gold spacy DocBin to canonical TEI/XML.")
    ap.add_argument("--spacy", default=DEFAULT_SPACY, help=f"_el.spacy path (default: {DEFAULT_SPACY})")
    ap.add_argument("--offset-map", default=None, help="offset map JSON (auto from --spacy if omitted)")
    ap.add_argument("--mention-map", default=DEFAULT_MENTION_MAP)
    ap.add_argument("--rdf", default=DEFAULT_RDF)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--letter-id", default=DEFAULT_LETTER_ID)
    args = ap.parse_args()

    spacy_path = args.spacy or select_el_spacy_file()
    offset_map = args.offset_map or find_offset_map(spacy_path)
    if offset_map is None:
        print("Cannot proceed without an offset map."); raise SystemExit(1)
    build_tei(spacy_path, offset_map, args.output,
              mention_map_path=args.mention_map, rdf_path=args.rdf,
              letter_id=args.letter_id)


if __name__ == "__main__":
    main()