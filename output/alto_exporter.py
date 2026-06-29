"""ALTO-XML exporter for the 1816 third letter (BRF0003, printed pages 16-25).

The pipeline runs `.txt -> .spacy` (NER -> EL -> RE). The source `.txt` is the
*manual* transcription, reflowed (only `\n\n` paragraph breaks + inline `[N]`
page markers; no manuscript linebreaks). Loghi produced `data/page/*.xml`
(PRImA PAGE-XML, line-level `<TextLine>` with `<Coords>` + `<Baseline>`, no
`<Word>`) whose HTR text is low quality and not character-identical to the
manual transcription.

This exporter emits one coordinate-bearing ALTO 4 document **per page scan**,
each carrying:
  * Loghi baseline/coords as that page's `<Layout>` geometry,
  * the **manual transcription** (from the enriched `.spacy`) as `<String>` text,
  * NER + EL inline as `<OtherTag>` tags (page-scoped) applied via `TAGREFS`
    (linked to the `ato:CT.BRF0003.eNN` mention ids), and
  * no embedded RDF — the full RE/RDF (deepseek-v4-pro run) is kept in a single
    shared sidecar `.ttl` next to the per-page files; each `<OtherTag>` carries
    its `ato:CT.*` URI + KB id inline, so the viewer resolves provenance via the
    sidecar without per-page RDF duplication.

It is split into two stages:

  * **Stage A (alignment, expensive, ~10 LLM calls, cached):** per page, the LLM
    line-aligner splits the manual page text across Loghi's `<TextLine>`s and
    assigns each line a global char-offset range. Cached to
    `output/alto/line_alignment.json`; Stage B then makes 0 LLM calls.
  * **Stage B (ALTO build, fast, deterministic):** consumes the `.spacy`,
    `mention_map.json`, the cached alignment, and the `.ttl` (as sidecar) -> one
    ALTO file per page (`{stem}_scan{NNNN}.alto.xml`) + a shared `{stem}.ttl`.

Coordinate space: the ALTO uses the `.spacy` `doc.text` **verbatim** (the
residual `16]` page marker is kept) so line offsets match `mention_map.json`
exactly. Do **not** strip markers or consult the gold CSV `offset_map` values
(theirs is a different, BOM-retained space); the offset map is used for page
*ordering* only. The `16]` artifact is inherited from the gold `.spacy` builder
(sliced off only the leading `[`); a future gold rebuild would clean it.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

import spacy
from spacy.tokens import Span, DocBin

# ollama_utils lives at the repo root.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from ollama_utils import resolve_ollama_host, stream_ollama_chat  # noqa: E402

# Register the custom EL extension attributes so the gold `.spacy` loads.
if not Span.has_extension("kb_id_wikidata_"):
    Span.set_extension("kb_id_wikidata_", default=None)
if not Span.has_extension("kb_id_geonames_"):
    Span.set_extension("kb_id_geonames_", default=None)

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    spacy="entity_linking/el-results/1816_el_gs_el.spacy",
    offset_map="entity_linking/el-results/1816_el_gs_offset_map.json",
    mention_map="output/re/1816_el_gs__deepseek-v4-pro_t0.0_thinkDefault_mention_map.json",
    rdf="output/rdf/1816_el_gs__deepseek-v4-pro_t0.0_thinkDefault_events.ttl",
    pagexml_dir="data/page",
    scan_page_csv="data/1816-scannumber-to-pagenumber.csv",
    line_alignment="output/alto/line_alignment.json",
    output="output/alto/1816_third_letter",
    letter_id="BRF0003",
    model="gemma4:31b-cloud",
    host=None,
)

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
STANDOFF_NS = "http://academictourism.com/standoff/"
ATO_NS = "http://academictourism.com/entity/"

A = "{" + ALTO_NS + "}"
P = "{" + PAGE_NS + "}"

# ── Page-ordering helpers (copied from tei_exporter.py) ─────────────────────

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _page_sort_key(page_num: str) -> tuple:
    """Sort page labels: pure integers numerically, Roman numerals after."""
    s = str(page_num).strip()
    if s.isdigit():
        return (0, int(s), "")
    roman = s.upper()
    if roman and all(ch in _ROMAN_VALUES for ch in roman):
        total = 0
        for i, ch in enumerate(roman):
            v = _ROMAN_VALUES[ch]
            if i + 1 < len(roman) and _ROMAN_VALUES[roman[i + 1]] > v:
                total -= v
            else:
                total += v
        return (1, total, s)
    return (2, 0, s)


# ── Scan <-> page helpers (copied from ner/ner.py) ──────────────────────────

def extract_scan_number(xml_filename: str) -> int:
    """`0552_0179_0049.xml` -> 49 (last `_`-separated stem segment)."""
    return int(Path(xml_filename).stem.split("_")[-1])


def load_scan_page_mapping(csv_path: str) -> dict:
    """Load the scan->page CSV; returns {scan_int: page_str}."""
    import csv as _csv
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        return {int(row["Scan Number"]): str(row["Page Number"]) for row in reader}


# ── Shared coordinate-space helpers (verbatim doc.text) ─────────────────────

def _ext(span: Span, name: str):
    """Read a Span extension safely (None if unset)."""
    try:
        return span._.get(name)
    except Exception:
        return None


def build_mention_lookup(mention_map: dict) -> dict:
    """`(global_start, global_end, text.lower()) -> mention_id`."""
    lookup = {}
    for mid, info in mention_map.items():
        key = (info["start"], info["end"], info["text"].lower())
        lookup[key] = mid
    return lookup


def build_full_text_and_entities(docs, sorted_pages, mention_lookup):
    """Concatenate the `doc.text`s verbatim (in page order) and collect entities.

    Returns ``(full_text, page_positions, all_entities)`` where
    `page_positions[page]` is the global char offset of that page's `doc.text`,
    and `all_entities` is a list of tuples:
    ``(gstart, gend, label, kb_id, text, mention_id, wikidata, geonames)``.

    The verbatim `doc.text` (residual `16]` marker kept) is what `mention_map`
    offsets are relative to, so global offsets here line up exactly.
    """
    full_text_parts = []
    page_positions = {}
    pos = 0
    n = min(len(docs), len(sorted_pages))
    for i in range(n):
        page_num = sorted_pages[i][0]
        text = docs[i].text  # VERBATIM — do NOT strip markers
        page_positions[page_num] = pos
        full_text_parts.append(text)
        pos += len(text)
    full_text = "".join(full_text_parts)

    all_entities = []
    for i in range(n):
        page_num = sorted_pages[i][0]
        base = page_positions[page_num]
        for ent in docs[i].ents:
            gs = base + ent.start_char
            ge = base + ent.end_char
            wd = _ext(ent, "kb_id_wikidata_")
            gn = _ext(ent, "kb_id_geonames_")
            kb = wd if wd is not None else gn
            # Normalise the legacy E22 transport label to the EL-skip name.
            label = ("Mode_of_Transportation"
                     if ent.label_ == "E22_Human-made_Object" else ent.label_)
            mid = mention_lookup.get((gs, ge, ent.text.lower()))
            all_entities.append((gs, ge, label, kb, ent.text, mid, wd, gn))
    all_entities.sort(key=lambda x: (x[0], -x[1]))
    return full_text, page_positions, all_entities


# ── Stage A: Loghi PAGE-XML parsing ─────────────────────────────────────────

def _parse_points(points_str: str) -> list:
    """`"400,1 427,1"` -> [(400,1), (427,1)]."""
    pts = []
    for tok in points_str.split():
        xs, ys = tok.split(",")
        pts.append((int(xs), int(ys)))
    return pts


def parse_loghi_page(xml_path: str) -> dict:
    """Parse a Loghi PAGE-XML into a geometry-bearing structure.

    Walks the reading order (`OrderedGroup/RegionRefIndexed` by `@index`), then
    each `TextRegion`'s `TextLine`s in document order, keeping Coords,
    Baseline, region/line ids, HTR text, image filename and size.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    page = root.find(f"{P}Page")
    if page is None:
        raise ValueError(f"No <Page> in {xml_path}")

    # Reading order -> ordered list of region ids.
    region_order = []
    ro = page.find(f"{P}ReadingOrder")
    if ro is not None:
        og = ro.find(f"{P}OrderedGroup")
        if og is not None:
            refs = og.findall(f"{P}RegionRefIndexed")
            refs.sort(key=lambda r: int(r.get("index", "0")))
            region_order = [r.get("regionRef") for r in refs]

    regions_by_id = {r.get("id"): r for r in page.findall(f"{P}TextRegion")}

    regions_out = []
    # Regions in reading order, then any leftover regions in document order.
    seen = set()
    ordered_regions = []
    for rid in region_order:
        if rid in regions_by_id:
            ordered_regions.append(regions_by_id[rid])
            seen.add(rid)
    for rid, r in regions_by_id.items():
        if rid not in seen:
            ordered_regions.append(r)

    for ri, region in enumerate(ordered_regions):
        rid = region.get("id") or f"region_{ri}"
        ccoords = region.find(f"{P}Coords")
        coords_points = _parse_points(ccoords.get("points")) if ccoords is not None else []

        lines_out = []
        for li, line in enumerate(region.findall(f"{P}TextLine")):
            lid = line.get("id") or f"line_{ri}_{li}"
            lc = line.find(f"{P}Coords")
            lb = line.find(f"{P}Baseline")
            te = line.find(f"{P}TextEquiv")
            htr_text = ""
            if te is not None:
                u = te.find(f"{P}Unicode")
                if u is not None and u.text is not None:
                    htr_text = u.text
            lines_out.append({
                "line_id": lid,
                "idx": li,
                "coords_points": _parse_points(lc.get("points")) if lc is not None else [],
                "baseline_points": _parse_points(lb.get("points")) if lb is not None else [],
                "htr_text": htr_text,
            })
        regions_out.append({
            "region_id": rid,
            "region_idx": ri,
            "coords_points": coords_points,
            "lines": lines_out,
        })

    return {
        "image_filename": page.get("imageFilename"),
        "image_width": int(page.get("imageWidth", "0")),
        "image_height": int(page.get("imageHeight", "0")),
        "regions": regions_out,
    }


def select_loghi_pages(pagexml_dir: str, scan_page_csv: str,
                       pages=range(16, 26)) -> list:
    """Return `[(scan, page, xml_path), ...]` for the third-letter pages,
    sorted by scan number. Only scans present in the CSV and in range."""
    scan2page = load_scan_page_mapping(scan_page_csv)
    page_set = {str(p) for p in pages}
    out = []
    pdir = Path(pagexml_dir)
    for xml in sorted(pdir.glob("*.xml")):
        scan = extract_scan_number(xml.name)
        page = scan2page.get(scan)
        if page is None or page not in page_set:
            continue
        out.append((scan, page, str(xml)))
    out.sort(key=lambda t: t[0])
    return out


# ── Stage A: LLM line alignment ─────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


def _parse_json_response(text: str):
    """Strip markdown code fences and json.loads; None on failure."""
    if text is None:
        return None
    s = text.strip()
    s = _JSON_FENCE_RE.sub("", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Fall back to extracting the first {...} block.
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def align_page_lines(htr_lines, gt_page_text, page_label, model, host, api_key):
    """Send the Loghi HTR lines + manual page text to the LLM; get back a
    `{line_idx: substring}` mapping (+ confidence).

    Reuses the gt_to_pagexml prompt verbatim. Returns `{}` on failure so the
    caller can fall back to HTR text.
    """
    htr_block = "\n".join(f"{i}: {ln}" for i, ln in enumerate(htr_lines))
    prompt = (
        "You are aligning an HTR (handwritten-text-recognition) transcription "
        "to a gold-standard manual transcription of the SAME page.\n\n"
        f"PAGE: {page_label}\n\n"
        "HTR lines (index: text):\n"
        f"{htr_block}\n\n"
        "GOLD (manual) page text:\n"
        f"\"\"\"\n{gt_page_text}\n\"\"\"\n\n"
        "For EACH HTR line index, return the substring of the GOLD text that "
        "corresponds to that line. Use ONLY text that appears verbatim in the "
        "GOLD text; concatenate the words exactly as written. If an HTR line "
        "has no gold equivalent (garbage, page number, ornament), omit its "
        "index. Respond as compact JSON: "
        '{"lines": {"0": "<gold substring>", "2": "...", ...}, '
        '"confidence": 0.0-1.0}. No prose.'
    )
    try:
        resp = stream_ollama_chat(
            model=model, prompt=prompt, host=host, api_key=api_key,
            timeout=900.0, temperature=0.0,
        )
    except Exception as e:
        print(f"  [align {page_label}] LLM call failed: {e}")
        return {}
    parsed = _parse_json_response(resp)
    if not parsed or "lines" not in parsed:
        print(f"  [align {page_label}] no JSON returned; falling back to HTR")
        return {}
    lines = parsed["lines"]
    mapping = {int(k): v for k, v in lines.items()}
    conf = parsed.get("confidence")
    return {"mapping": mapping, "confidence": conf}


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def tile_lines_to_offsets(subs: list, gt_page_text: str) -> dict:
    """Tile line substrings into local char offsets over the verbatim page text.

    `subs` = list of `(line_idx, substring)`. Returns
    `{line_idx: (local_start, local_end, matched)}`.

    The LLM returns substrings in *HTR reading order*, which for header regions
    (page number, title, date) need not match the gold text's linear order. So
    we do NOT assume a monotonic cursor: each line is placed at its earliest
    position in the verbatim text that is not already consumed by another line
    (an unconsumed-interval search). This is non-monotonic-safe and lets the
    page marker `16]` (which the LLM may return third) still claim offset 0.

    The text emitted for a matched line is the **verbatim span**
    `gt_page_text[start:end]` (NOT the LLM's substring), so the offset invariant
    `full_text[start:end] == manual_substring` holds by construction and NER
    local offsets line up. Try exact `str.find`, then a whitespace-tolerant
    token-run match; if neither finds an unconsumed span, the line is marked
    `matched=False` (caller falls back to Loghi HTR text, no NER tags).
    """
    result = {}
    consumed = []  # sorted list of [start, end) intervals already claimed
    vtoks = list(re.finditer(r"\S+", gt_page_text))  # verbatim non-space tokens
    # Punctuation-stripped token forms, kept aligned with `vtoks` (same index).
    # Used by the normalized fallback so a stray comma/period in the LLM output
    # (e.g. "bergagtig," vs gold "bergagtig") doesn't drop a whole correct line.
    vtoks_norm = [(m, _norm_tok(m.group())) for m in vtoks]
    for line_idx, sub in subs:
        if not sub:
            continue
        span = _find_unconsumed_exact(sub, gt_page_text, consumed)
        if span is None:
            span = _find_unconsumed_tokenrun(sub, vtoks, consumed)
        if span is None:
            span = _find_unconsumed_normrun(sub, vtoks_norm, consumed)
        if span is None:
            span = _find_unconsumed_fuzzyrun(sub, vtoks_norm, consumed)
        if span is None:
            # Not found verbatim (LLM hallucination / wording drift): emit with
            # HTR fallback text, no global offset, no NER tagging.
            result[line_idx] = (None, None, False)
            continue
        result[line_idx] = (span[0], span[1], True)
        _insert_interval(consumed, span)
    return result


def _overlaps(intervals, s, e) -> bool:
    for (a, b) in intervals:
        if a < e and s < b:
            return True
    return False


def _insert_interval(intervals, span):
    intervals.append(span)
    intervals.sort()


def _find_unconsumed_exact(sub, gt, consumed):
    """First occurrence of `sub` in `gt` whose span is unconsumed, else None."""
    start = 0
    while True:
        pos = gt.find(sub, start)
        if pos < 0:
            return None
        if not _overlaps(consumed, pos, pos + len(sub)):
            return (pos, pos + len(sub))
        start = pos + 1


def _find_unconsumed_tokenrun(sub, vtoks, consumed):
    """First contiguous run of verbatim tokens equal to `sub.split()` whose
    span is unconsumed. Whitespace-tolerant (ignores spacing between tokens)."""
    tokens = sub.split()
    if not tokens:
        return None
    for i, mt in enumerate(vtoks):
        if mt.group() != tokens[0]:
            continue
        j = i
        ok = True
        for t in tokens:
            if j >= len(vtoks) or vtoks[j].group() != t:
                ok = False
                break
            j += 1
        if ok:
            span = (vtoks[i].start(), vtoks[j - 1].end())
            if not _overlaps(consumed, span[0], span[1]):
                return span
    return None


# Punctuation stripped when matching the LLM output to the gold text. Kept
# narrow on purpose: ampersand (&), equals (=), ½, slashes and hyphens are all
# meaningful in this corpus and are NOT stripped.
_NORM_PUNCT = ".,;:!?\"'()[]…‘’“”·"


def _norm_tok(t: str) -> str:
    """A token with clause punctuation stripped from both ends."""
    return t.strip(_NORM_PUNCT)


def _find_unconsumed_normrun(sub, vtoks_norm, consumed):
    """Punctuation-tolerant fallback: first contiguous run of gold tokens whose
    punctuation-stripped forms equal `sub`'s stripped tokens, unconsumed.

    This recovers lines the LLM aligned correctly but whose returned substring
    differs from the gold only by stray commas/periods (e.g. ``bergagtig,`` vs
    ``bergagtig``). Returns the *verbatim* gold span so the offset invariant
    ``full_text[start:end] == manual_substring`` still holds. Requires >= 3
    tokens so short ornaments/page-numbers don't spuriously match.
    """
    tokens = [t for t in (_norm_tok(x) for x in sub.split()) if t]
    if len(tokens) < 3:
        return None
    n = len(tokens)
    for i in range(len(vtoks_norm) - n + 1):
        if all(vtoks_norm[i + k][1] == tokens[k] for k in range(n)):
            span = (vtoks_norm[i][0].start(), vtoks_norm[i + n - 1][0].end())
            if not _overlaps(consumed, span[0], span[1]):
                return span
    return None


def _find_unconsumed_fuzzyrun(sub, vtoks_norm, consumed):
    """Fuzzy-prefix fallback: recover lines the LLM aligned correctly but
    truncated or invented at the tail (e.g. ``…uit Parys zyn`` vs gold
    ``…uit Parys terug gekomen``).

    Aligns the LLM's punctuation-stripped tokens to the gold tokens via
    ``difflib.SequenceMatcher`` and, if the matched tokens (a) start at the
    line's beginning (first block at LLM-token index <= 1 — the line is
    anchored where the LLM says it starts), (b) cover >= 50% of the LLM tokens
    and >= 4 tokens, and (c) span a gold region no larger than 1.5x the LLM
    text (so a stray far-apart phrase can't stitch together an over-long span),
    returns the *verbatim* gold span from the first to the last matched token.

    Uses the LLM's returned tokens to LOCATE the line in gold (response-level)
    but tolerates a wrong/truncated trailing word; because we emit the gold
    span, ``full_text[start:end] == manual_substring`` still holds and the
    entity offsets that sit in the matched prefix land correctly.
    """
    import difflib
    ltoks = [t for t in (_norm_tok(x) for x in sub.split()) if t]
    if len(ltoks) < 4:
        return None
    gtoks = [nm for (_, nm) in vtoks_norm]
    sm = difflib.SequenceMatcher(None, ltoks, gtoks, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    # anchor at the line start: the LLM's first (or second, if it dropped a
    # leading word like "Kastelein" -> "lein") token must match gold, and the
    # anchor block must be substantial (>= 3 tokens) — a lone common word like
    # "de" matching at token 0 is not a real anchor.
    if blocks[0].a > 1 or blocks[0].size < 3:
        return None
    first_g = blocks[0].b
    last_g = blocks[0].b + blocks[0].size - 1
    matched = blocks[0].size
    # Extend the span across later blocks ONLY while they stay close in gold
    # (within 1.5x the LLM token count). A trailing block that matches an
    # unrelated occurrence elsewhere on the page (e.g. a repeated phrase like
    # "schilderachtig was het") would stretch the span across a large gap — we
    # stop at it instead of stretching, so the valid prefix is kept.
    for b in blocks[1:]:
        cand = b.b + b.size - 1
        if (cand - first_g + 1) > len(ltoks) * 1.5:
            break
        last_g = cand
        matched += b.size
    if matched < 4 or matched / len(ltoks) < 0.5:
        return None
    gstart = vtoks_norm[first_g][0].start()
    gend = vtoks_norm[last_g][0].end()
    if _overlaps(consumed, gstart, gend):
        return None
    return (gstart, gend)


def calculate_fuzzy_accuracy(original_gt: str, mapping: dict) -> float:
    """Reconstruct the mapped GT and compare to the original (SequenceMatcher)."""
    if not mapping:
        return 0.0
    reconstructed = " ".join(mapping[k] for k in sorted(mapping))
    return SequenceMatcher(None, _norm_ws(original_gt), _norm_ws(reconstructed)).ratio()


def build_line_alignment(docs, sorted_pages, page_positions, pagexml_dir,
                         scan_page_csv, model, host, api_key,
                         cache_path, force=False):
    """Stage A driver (cached). Returns the alignment dict.

    On a cache hit (and not `force`), loads and returns. Otherwise runs one LLM
    call per page, tiles the substrings to offsets, and writes the cache.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        print(f"  [align] using cached {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    selected = select_loghi_pages(pagexml_dir, scan_page_csv)
    if len(selected) != len(sorted_pages):
        print(f"  [align] WARNING: {len(selected)} Loghi pages vs "
              f"{len(sorted_pages)} gold pages")
    # Match Loghi pages to gold pages by printed page number.
    loghi_by_page = {page: (scan, xml) for scan, page, xml in selected}

    alignment = {"pages": []}
    full_text_len = sum(len(d.text) for d in docs)
    alignment["full_text_len"] = full_text_len

    n = min(len(docs), len(sorted_pages))
    for i in range(n):
        page_num = sorted_pages[i][0]
        gt_page_text = docs[i].text  # verbatim
        page_text_start = page_positions[page_num]
        page_text_end = page_text_start + len(gt_page_text)

        if page_num not in loghi_by_page:
            print(f"  [align] no Loghi page for printed page {page_num}; skipping")
            alignment["pages"].append({
                "page": page_num, "scan": None, "xml": None,
                "image_filename": None, "image_width": 0, "image_height": 0,
                "page_text_start": page_text_start, "page_text_end": page_text_end,
                "confidence": None, "fuzzy_accuracy": 0.0,
                "regions": [],
            })
            continue

        scan, xml_path = loghi_by_page[page_num]
        parsed = parse_loghi_page(xml_path)
        htr_lines = []
        for region in parsed["regions"]:
            for line in region["lines"]:
                htr_lines.append(line["htr_text"])

        aligned = align_page_lines(htr_lines, gt_page_text, page_num,
                                   model, host, api_key)
        mapping = aligned.get("mapping", {}) if aligned else {}
        conf = aligned.get("confidence") if aligned else None
        fuzzy = calculate_fuzzy_accuracy(gt_page_text, mapping)
        print(f"  [align page {page_num}] {len(mapping)}/{len(htr_lines)} lines, "
              f"conf={conf}, fuzzy={fuzzy:.3f}")

        # Tile matched substrings to local offsets, in line order.
        subs = [(idx, mapping[idx]) for idx in sorted(mapping)]
        offsets = tile_lines_to_offsets(subs, gt_page_text)

        # Build region/line structure, attaching offsets to matched lines.
        # For matched lines the emitted text is the VERBATIM span
        # gt_page_text[ls:le] (not the LLM substring), so offsets and text agree
        # by construction and NER local offsets are consistent.
        regions_out = []
        line_global_idx = 0
        for region in parsed["regions"]:
            lines_out = []
            for line in region["lines"]:
                local = offsets.get(line_global_idx)
                if local is not None and local[2]:  # matched
                    ls, le, _ = local
                    lines_out.append({
                        "line_id": line["line_id"],
                        "idx": line_global_idx,
                        "coords_points": line["coords_points"],
                        "baseline_points": line["baseline_points"],
                        "htr_text": line["htr_text"],
                        "manual_substring": gt_page_text[ls:le],
                        "full_text_start": page_text_start + ls,
                        "full_text_end": page_text_start + le,
                        "matched": True,
                    })
                else:
                    # Unmatched: keep HTR text, no global offset, geometry only.
                    lines_out.append({
                        "line_id": line["line_id"],
                        "idx": line_global_idx,
                        "coords_points": line["coords_points"],
                        "baseline_points": line["baseline_points"],
                        "htr_text": line["htr_text"],
                        "manual_substring": None,
                        "full_text_start": None,
                        "full_text_end": None,
                        "matched": False,
                    })
                line_global_idx += 1
            regions_out.append({
                "region_id": region["region_id"],
                "region_idx": region["region_idx"],
                "coords_points": region["coords_points"],
                "lines": lines_out,
            })

        alignment["pages"].append({
            "page": page_num, "scan": scan, "xml": xml_path,
            "image_filename": parsed["image_filename"],
            "image_width": parsed["image_width"],
            "image_height": parsed["image_height"],
            "page_text_start": page_text_start, "page_text_end": page_text_end,
            "confidence": conf, "fuzzy_accuracy": fuzzy,
            # Raw LLM mapping (idx -> substring) kept so the alignment can be
            # re-tiled deterministically with an improved matcher (--retile)
            # without re-calling the LLM, and so responses are inspectable.
            "raw_mapping": {str(k): v for k, v in mapping.items()},
            "regions": regions_out,
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(alignment, f, ensure_ascii=False, indent=2)
    print(f"  [align] cached -> {cache_path}")
    return alignment


def retile_alignment(alignment, docs, sorted_pages, cache_path=None):
    """Re-apply `tile_lines_to_offsets` to each page's stored `raw_mapping`
    without calling the LLM. Updates matched/manual_substring/full_text_start/
    end in place so matcher improvements (e.g. punctuation tolerance) take
    effect deterministically. Returns the number of newly matched lines.

    Used by `--retile`. Requires the cache to carry `raw_mapping` (written by
    `build_line_alignment` since the tolerant matcher was added).
    """
    newly = 0
    for page in alignment["pages"]:
        rm = page.get("raw_mapping")
        if rm is None:
            continue
        pn = str(page["page"])
        di = next((k for k, (key, _) in enumerate(sorted_pages)
                   if key.split("_")[0] == pn), None)
        if di is None:
            continue
        gt = docs[di].text
        pstart = page["page_text_start"]
        mapping = {int(k): v for k, v in rm.items()}
        subs = [(idx, mapping[idx]) for idx in sorted(mapping)]
        offsets = tile_lines_to_offsets(subs, gt)
        for region in page["regions"]:
            for line in region["lines"]:
                local = offsets.get(line["idx"])
                was = line["matched"]
                if local is not None and local[2]:
                    ls, le, _ = local
                    line["manual_substring"] = gt[ls:le]
                    line["full_text_start"] = pstart + ls
                    line["full_text_end"] = pstart + le
                    line["matched"] = True
                else:
                    line["manual_substring"] = None
                    line["full_text_start"] = None
                    line["full_text_end"] = None
                    line["matched"] = False
                if line["matched"] and not was:
                    newly += 1
    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(alignment, f, ensure_ascii=False, indent=2)
        print(f"  [retile] re-tiled cache -> {cache_path}")
    return newly


# ── Stage B: ALTO geometry helpers ──────────────────────────────────────────

def coords_to_bbox(points) -> tuple:
    """List of (x,y) -> (hpos, vpos, width, height)."""
    if not points:
        return (0, 0, 0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def baseline_str(points) -> str:
    """List of (x,y) -> `"x,y x,y ..."`."""
    return " ".join(f"{x},{y}" for x, y in points)


# ── Stage B: baseline-anchored line boxes ────────────────────────────────────
# Loghi/laypa Coords polygons span the full ascender-to-descender extent of a
# line and are inflated (~2× the baseline pitch), so the Coords AABB used
# directly as the ALTO box crosses neighbouring baselines. We instead derive
# each line's vertical extent from its BASELINE: a band that rises at most
# ASC_FRAC*pitch above the baseline and DESC_FRAC*pitch below, clamped to the
# midpoints to the neighbouring baselines so a box can never reach an adjacent
# baseline. Horizontal extent is still the Coords AABB (the line's left/right
# reach is reliable). The result is a refinement of the Coords AABB: the new
# box is always ⊆ the old one and ⊆ the no-crossing band.
ASC_FRAC = 0.5      # upward extension above the baseline, in units of pitch
DESC_FRAC = 0.2     # descender depth below the baseline, in units of pitch
DEFAULT_PITCH = 72   # fallback line pitch (px) when a page has <2 baselines


def baseline_median_y(points) -> float:
    """Median y of a baseline polyline (robust to slope / outlier points)."""
    ys = sorted(p[1] for p in points)
    n = len(ys)
    if not n:
        return 0.0
    mid = n // 2
    return float(ys[mid]) if n % 2 else float((ys[mid - 1] + ys[mid]) / 2)


def page_pitch(regions) -> float:
    """Median within-region baseline-to-baseline gap across a page's regions.

    Lines are sorted by baseline-y *within each region* (regions are separate
    columns, so a global y-sort would mix unrelated lines), consecutive gaps
    are pooled, and the median is taken — robust to the small/large outliers
    (adjacent columns, isolated headings) seen in the data.
    """
    gaps = []
    for region in regions:
        bs = []
        for line in region.get("lines", []):
            pts = line.get("baseline_points")
            if pts:
                bs.append(baseline_median_y(pts))
        bs.sort()
        gaps.extend(bs[i + 1] - bs[i] for i in range(len(bs) - 1))
    if not gaps:
        return DEFAULT_PITCH
    import statistics
    return float(statistics.median(gaps))


def line_box_from_baseline(coords_points, baseline_points, b, b_prev, b_next,
                           pitch) -> tuple:
    """Baseline-anchored (hpos, vpos, width, height) for one TextLine.

    ``b`` is this line's baseline y; ``b_prev``/``b_next`` the neighbouring
    baselines' y (or None). Falls back to the Coords AABB when there is no
    baseline. The box is clamped to lie within the Coords AABB and within the
    [prev_mid, next_mid] band, so it can never cross a neighbouring baseline.
    """
    if not baseline_points or not coords_points:
        return coords_to_bbox(coords_points)
    xs = [p[0] for p in coords_points]
    ys = [p[1] for p in coords_points]
    cx_min, cx_max = min(xs), max(xs)
    cy_min, cy_max = min(ys), max(ys)
    prev_mid = (b_prev + b) / 2 if b_prev is not None else float("-inf")
    next_mid = (b + b_next) / 2 if b_next is not None else float("inf")
    top = max(cy_min, prev_mid, b - ASC_FRAC * pitch)
    bottom = min(cy_max, next_mid, b + DESC_FRAC * pitch)
    if bottom < top:                      # degenerate (overlapping baselines): hug baseline
        top, bottom = b - ASC_FRAC * pitch, b + DESC_FRAC * pitch
    hpos = cx_min
    width = cx_max - cx_min
    return (round(hpos), round(top), round(width), round(bottom - top))


_TOKEN_RE = re.compile(r"\S+|\s+")


def tokenize_line(text: str) -> list:
    """Split into a sequence of tokens (word runs and whitespace runs)."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def distribute_word_geometry(line_bbox, tokens):
    """Proportional pixel geometry for each token within a line box.

    Returns a list of `(token_text, is_space, hpos, vpos, width, height,
    tok_start, tok_end)`. Width is split by char fraction across ALL tokens
    (words + spaces) so the line text fills the line box. The `tok_start`/
    `tok_end` are char offsets within the line text (for NER mapping).
    """
    hpos, vpos, width, height = line_bbox
    total_chars = sum(len(t) for t, _, _ in tokens) or 1
    out = []
    x = hpos
    for tok_text, ts, te in tokens:
        is_space = tok_text.isspace()
        w = round(width * len(tok_text) / total_chars)
        out.append((tok_text, is_space, x, vpos, w, height, ts, te))
        x += w
    return out


# ── Stage B: NER -> String mapping ──────────────────────────────────────────

def map_entity_to_strings(entity, line_alignment) -> list:
    """Return `[{"page":..,"region_idx":..,"line_idx":..,"string_indices":[...],
    "covered_text":...}, ...]` for the lines a `[gstart,gend)` entity covers.

    Only matched lines are considered (their `manual_substring` carries the
    manual text the entity offsets are relative to). For each covered line, the
    covered local char range is intersected with token char ranges to find the
    `<String>` indices to tag.
    """
    gstart, gend, _, _, text, mid, _, _ = entity
    out = []
    for page in line_alignment["pages"]:
        for region in page["regions"]:
            for line in region["lines"]:
                if not line["matched"]:
                    continue
                ls = line["full_text_start"]
                le = line["full_text_end"]
                # Overlap of [gstart,gend) with [ls,le).
                cov_start = max(gstart, ls)
                cov_end = min(gend, le)
                if cov_end <= cov_start:
                    continue
                local_start = cov_start - ls
                local_end = cov_end - ls
                tokens = tokenize_line(line["manual_substring"])
                # Word indices (non-space) whose range intersects the covered range.
                word_idx = 0
                string_indices = []
                for tok_text, ts, te in tokens:
                    if tok_text.isspace():
                        continue
                    if te > local_start and ts < local_end:
                        string_indices.append(word_idx)
                    word_idx += 1
                if not string_indices:
                    continue
                covered = line["manual_substring"][local_start:local_end]
                out.append({
                    "page": page["page"],
                    "region_idx": region["region_idx"],
                    "line_idx": line["idx"],
                    "string_indices": string_indices,
                    "covered_text": covered,
                })
    return out


# ── Stage B: RDF embedding ──────────────────────────────────────────────────

def _sort_xml_children(el):
    """Recursively sort an ElementTree element's children for byte-determinism.

    Used to canonicalise rdflib's RDF/XML, whose element order follows the
    graph's triple iteration order — which is process-dependent via Python's
    string-hash randomisation. RDF is unordered, so reordering children is
    semantically a no-op.

    Whitespace-only ``.text``/``.tail`` (rdflib's indentation) is stripped
    first, so it cannot travel with the (hash-ordered) children and perturb the
    canonical order. The RDF is then emitted compact, which is reproducible.
    """
    if el.text is not None:
        el.text = el.text.strip() or None
    if el.tail is not None:
        el.tail = el.tail.strip() or None
    for child in el:
        _sort_xml_children(child)
    el[:] = sorted(
        el,
        key=lambda e: (e.tag,
                       tuple(sorted((k, v) for k, v in e.attrib.items())),
                       (e.text or "")),
    )


def _rdf_to_xml(rdf_path: str) -> str:
    """Parse the Turtle and serialise to canonical RDF/XML (no XML decl).

    The element order is canonicalised (sorted) so the embedded RDF is
    byte-reproducible across runs despite rdflib's hash-order-dependent
    serialisation.
    """
    import rdflib
    g = rdflib.Graph()
    g.parse(rdf_path, format="turtle")
    rdf_xml = g.serialize(format="xml")
    rdf_xml = re.sub(r"<\?xml[^?]*\?>\s*", "", rdf_xml).strip()
    try:
        root = ET.fromstring(rdf_xml)
        _sort_xml_children(root)
        rdf_xml = ET.tostring(root, encoding="unicode")
    except ET.ParseError:
        pass  # keep rdflib's output if canonicalisation parse fails
    return rdf_xml


# ── Stage B: ALTO assembly ──────────────────────────────────────────────────

XENODATA_PLACEHOLDER = "<!--XENODATA_RDF-->"


def prepare_tags(all_entities, line_alignment, letter_id):
    """Stage B precompute: build the document-wide tag table and the
    page-keyed String→tag lookup.

    Returns (tags, tag_ids, entity_string_hits, string_tags) where:
      * tags / tag_ids are parallel lists over all_entities (one OtherTag per
        entity; tag_id = ne_<mid> or ne_g<gs> for mention-less spans).
      * entity_string_hits is [(entity_index, hits)] for entities that land on
        at least one matched line (via map_entity_to_strings).
      * string_tags maps (page, region_idx, line_idx) -> {string_index: [tag_ids]}.
    """
    # Build the <Tags> table: one <OtherTag> per entity.
    # OtherTag DESCRIPTION carries "CT=<uri>; KB=<id>" (schema-friendly; the
    # CT link is also implicit in the ID convention ne_<mention_id>).
    tags = []
    tag_ids = []  # parallel list of tag ids for string assignment
    for (gs, ge, label, kb, text, mid, wd, gn) in all_entities:
        if mid is not None:
            tag_id = f"ne_{mid}"
            ct_uri = f"ato:CT.{letter_id}.{mid}"
        else:
            tag_id = f"ne_g{gs}"
            ct_uri = ""
        desc_parts = [f"CT={ct_uri}"]
        kb_str = wd if wd is not None else gn
        if kb_str:
            desc_parts.append(f"KB={kb_str}")
        tags.append({
            "id": tag_id, "label": label,
            "description": "; ".join(desc_parts),
            "uri": ct_uri,
        })
        tag_ids.append(tag_id)

    # Precompute, per entity, the String positions to tag.
    entity_string_hits = []
    for ei, entity in enumerate(all_entities):
        hits = map_entity_to_strings(entity, line_alignment)
        if hits:
            entity_string_hits.append((ei, hits))

    # Build a lookup: (page, region_idx, line_idx) -> {string_index: [tag_ids]}.
    string_tags = {}
    for ei, hits in entity_string_hits:
        tid = tag_ids[ei]
        for h in hits:
            key = (h["page"], h["region_idx"], h["line_idx"])
            d = string_tags.setdefault(key, {})
            for si in h["string_indices"]:
                d.setdefault(si, []).append(tid)

    return tags, tag_ids, entity_string_hits, string_tags


def page_entity_indices(entity_string_hits, page_num):
    """Return the set of entity indices whose hits land on ``page_num``."""
    idxs = set()
    for ei, hits in entity_string_hits:
        if any(h["page"] == page_num for h in hits):
            idxs.add(ei)
    return idxs


def build_page_root(page_num, pinfo, scan, page_tags, string_tags,
                    letter_id, pagexml_dir):
    """Stage B: assemble a single-page ALTO 4 ElementTree (one <Page>).

    ``page_tags`` is the page-scoped <OtherTag> list (only entities on this
    page); ``string_tags`` is the document-wide (page, region, line) lookup.
    No <xenoData> — RDF provenance lives in the shared sidecar .ttl.
    """
    ET.register_namespace("", ALTO_NS)
    root = ET.Element(f"{A}alto")

    # <Description> — the page's own scan image; no embedded RDF.
    img_fn = pinfo.get("image_filename") or ""
    # Swap .jpg -> .png for the viewer (Loghi stores .jpg; scans are .png).
    img_png = re.sub(r"\.jpe?g$", ".png", img_fn, flags=re.I)
    if img_fn and not img_fn.lower().endswith(".png"):
        png_path = Path(pagexml_dir) / img_png
        if not png_path.exists():
            print(f"  [alto] page {page_num}: {img_png} not on disk (kept anyway)")
    desc = ET.SubElement(root, f"{A}Description")
    sii = ET.SubElement(desc, f"{A}sourceImageInformation")
    fn = ET.SubElement(sii, f"{A}fileName")
    fn.text = img_png
    did = ET.SubElement(desc, f"{A}documentID")
    did.text = letter_id

    # <Styles>
    styles = ET.SubElement(root, f"{A}Styles")
    ET.SubElement(styles, f"{A}TextStyle", attrib={
        "ID": "default", "FONTFAMILY": "serif", "FONTSIZE": "10",
    })

    # <Tags> — page-scoped (only entities on this page).
    tags_el = ET.SubElement(root, f"{A}Tags")
    for t in page_tags:
        attrib = {"ID": t["id"], "LABEL": t["label"],
                  "DESCRIPTION": t["description"]}
        if t["uri"]:
            attrib["URI"] = t["uri"]
        ET.SubElement(tags_el, f"{A}OtherTag", attrib=attrib)

    # <Layout> / single <Page>
    layout = ET.SubElement(root, f"{A}Layout")
    page_el = ET.SubElement(layout, f"{A}Page", attrib={
        "ID": f"P{page_num}",
        "PHYSICAL_IMG_NR": str(page_num),
        "IMAGE": img_png,
        "WIDTH": str(pinfo.get("image_width", 0)),
        "HEIGHT": str(pinfo.get("image_height", 0)),
    })
    ps = ET.SubElement(page_el, f"{A}PrintSpace")

    pitch = page_pitch(pinfo["regions"])

    for region in pinfo["regions"]:
        rbbox = coords_to_bbox(region["coords_points"])
        tb = ET.SubElement(ps, f"{A}TextBlock", attrib={
            "ID": f"tb_{scan}_{region['region_idx']}",
            "HPOS": str(rbbox[0]), "VPOS": str(rbbox[1]),
            "WIDTH": str(rbbox[2]), "HEIGHT": str(rbbox[3]),
        })
        # Neighbouring baselines within this region (sorted by baseline y), so
        # each line's box can be clamped to the midpoints above/below it and
        # never cross an adjacent baseline.
        blines = sorted(
            (baseline_median_y(l["baseline_points"]), l["idx"])
            for l in region["lines"] if l.get("baseline_points"))
        neigh = {}
        for i, (b, idx) in enumerate(blines):
            neigh[idx] = (b,
                          blines[i - 1][0] if i > 0 else None,
                          blines[i + 1][0] if i < len(blines) - 1 else None)
        for line in region["lines"]:
            if line.get("baseline_points") and line["idx"] in neigh:
                b, b_prev, b_next = neigh[line["idx"]]
                lbbox = line_box_from_baseline(line["coords_points"],
                                                line["baseline_points"],
                                                b, b_prev, b_next, pitch)
            else:
                lbbox = coords_to_bbox(line["coords_points"])
            attrib = {
                "ID": f"tl_{scan}_{region['region_idx']}_{line['idx']}",
                "HPOS": str(lbbox[0]), "VPOS": str(lbbox[1]),
                "WIDTH": str(lbbox[2]), "HEIGHT": str(lbbox[3]),
            }
            if line["baseline_points"]:
                attrib["BASELINE"] = baseline_str(line["baseline_points"])
            tl = ET.SubElement(tb, f"{A}TextLine", attrib=attrib)

            line_text = line["manual_substring"] if line["matched"] \
                else line["htr_text"]
            if line_text is None:
                line_text = ""
            # Normalise newlines to spaces for single-line geometry.
            line_text = line_text.replace("\n", " ").replace("\r", " ")

            tokens = tokenize_line(line_text)
            geom = distribute_word_geometry(lbbox, tokens)

            line_tags = string_tags.get(
                (page_num, region["region_idx"], line["idx"]), {})

            word_idx = 0
            for (tok_text, is_space, x, vy, w, h, ts, te) in geom:
                if is_space:
                    ET.SubElement(tl, f"{A}SP", attrib={
                        "HPOS": str(x), "VPOS": str(vy),
                        "WIDTH": str(w), "HEIGHT": str(h),
                    })
                    continue
                s_attrib = {
                    "ID": f"st_{scan}_{region['region_idx']}"
                          f"_{line['idx']}_{word_idx}",
                    "HPOS": str(x), "VPOS": str(vy),
                    "WIDTH": str(w), "HEIGHT": str(h),
                    "CONTENT": tok_text,
                    "WC": "0.95" if line["matched"] else "0.5",
                }
                tids = line_tags.get(word_idx)
                if tids:
                    s_attrib["TAGREFS"] = " ".join(tids)
                ET.SubElement(tl, f"{A}String", attrib=s_attrib)
                word_idx += 1

    return root


def serialize_alto(root, rdf_path, output_path):
    """Serialize the ALTO tree, splicing in the RDF/XML at the xenoData marker."""
    xml_str = ET.tostring(root, encoding="unicode")
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str
    if rdf_path and Path(rdf_path).exists():
        rdf_xml = _rdf_to_xml(rdf_path)
        xeno_inner = (f'<xenoData xmlns="{STANDOFF_NS}">\n'
                      f'{rdf_xml}\n'
                      f'    </xenoData>')
        xml_str = xml_str.replace(XENODATA_PLACEHOLDER, xeno_inner)
    else:
        xml_str = xml_str.replace(XENODATA_PLACEHOLDER, "")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return xml_str


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export a coordinate-bearing ALTO-XML for the 1816 third letter.")
    parser.add_argument("--spacy", default=DEFAULTS["spacy"])
    parser.add_argument("--offset-map", default=DEFAULTS["offset_map"])
    parser.add_argument("--mention-map", default=DEFAULTS["mention_map"])
    parser.add_argument("--rdf", default=DEFAULTS["rdf"])
    parser.add_argument("--pagexml-dir", default=DEFAULTS["pagexml_dir"])
    parser.add_argument("--scan-page-csv", default=DEFAULTS["scan_page_csv"])
    parser.add_argument("--line-alignment", default=DEFAULTS["line_alignment"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--letter-id", default=DEFAULTS["letter_id"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--host", default=DEFAULTS["host"],
                        help="Ollama host (None/cloud/localhost/URL). Default: auto.")
    parser.add_argument("--force-alignment", action="store_true",
                        help="Re-run Stage A (LLM alignment) even if cached.")
    parser.add_argument("--retile", action="store_true",
                        help="Re-apply the matcher to the cached raw LLM mappings "
                             "without calling the LLM; updates matched/offsets. "
                             "Requires a cache with raw_mapping (built since the "
                             "tolerant matcher was added).")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    def _r(p): return str(Path(p) if Path(p).is_absolute() else root_dir / p)

    spacy_path = _r(args.spacy)
    offset_map_path = _r(args.offset_map)
    mention_map_path = _r(args.mention_map)
    rdf_path = _r(args.rdf)
    pagexml_dir = _r(args.pagexml_dir)
    scan_page_csv = _r(args.scan_page_csv)
    cache_path = _r(args.line_alignment)
    output_path = _r(args.output)

    host_url, api_key = resolve_ollama_host(args.host)
    print(f"ALTO export — letter {args.letter_id}")
    print(f"  model: {args.model}  host: {host_url}")
    print(f"  spacy:   {spacy_path}")
    print(f"  rdf:     {rdf_path}")
    print(f"  output:  {output_path}")

    # Load spaCy docs once: page order + positions for Stage A, and
    # all_entities for Stage B (no reload in Stage B).
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(spacy_path)
    docs = list(db.get_docs(nlp.vocab))
    offset_map = json.load(open(offset_map_path))
    sorted_pages = sorted(offset_map.items(), key=lambda kv: _page_sort_key(kv[0]))
    full_text, page_positions, all_entities = build_full_text_and_entities(
        docs, sorted_pages, build_mention_lookup(json.load(open(mention_map_path))))

    # Stage A.
    print("\nStage A — line alignment:")
    if args.retile:
        if not Path(cache_path).exists():
            print(f"  [retile] no cache at {cache_path}; nothing to re-tile")
            sys.exit(1)
        line_alignment = json.load(open(cache_path))
        newly = retile_alignment(line_alignment, docs, sorted_pages, cache_path)
        print(f"  [retile] {newly} newly matched line(s)")
    else:
        line_alignment = build_line_alignment(
            docs, sorted_pages, page_positions, pagexml_dir, scan_page_csv,
            args.model, host_url, api_key, cache_path, force=args.force_alignment)
        # If the cache predates raw_mapping, a tolerant re-tile isn't possible
        # without re-running the LLM; surface that so the user can decide.
        if all("raw_mapping" not in p for p in line_alignment.get("pages", [])):
            print("  [align] cache has no raw_mapping; use --force-alignment to "
                  "rebuild with the tolerant matcher (stores raw_mapping).")

    if len(full_text) and len(full_text) != line_alignment.get("full_text_len"):
        print(f"  [alto] WARNING: full_text length {len(full_text)} vs "
              f"alignment {line_alignment.get('full_text_len')}")

    # Stage B — one ALTO file per page scan.
    print("\nStage B — ALTO build (per page):")
    output_stem = output_path  # --output is now a stem, not a file.
    tags, tag_ids, entity_string_hits, string_tags = prepare_tags(
        all_entities, line_alignment, args.letter_id)

    import shutil
    written = []
    for p in line_alignment["pages"]:
        page_num = p["page"]
        scan = p.get("scan")
        if scan is None:
            print(f"  [alto] page {page_num}: no scan, skipped")
            continue
        page_tags = [tags[ei] for ei in sorted(
            page_entity_indices(entity_string_hits, page_num))]
        root = build_page_root(
            page_num, p, scan, page_tags, string_tags,
            args.letter_id, pagexml_dir)
        out = f"{output_stem}_scan{scan:04d}.alto.xml"
        xml_str = serialize_alto(root, None, out)
        # Well-formedness per file.
        try:
            ET.fromstring(xml_str)
        except ET.ParseError as e:
            print(f"  page {page_num} (scan {scan:04d}): well-formedness FAILED — {e}")
            sys.exit(1)
        written.append((page_num, scan, out, len(xml_str)))

    # One shared sidecar .ttl (full letter RDF) next to the per-page files.
    sidecar = f"{output_stem}.ttl"
    if Path(rdf_path).exists():
        shutil.copy2(rdf_path, sidecar)
    else:
        print(f"  [alto] WARNING: rdf not found ({rdf_path}); sidecar not written")
    print(f"  wrote {len(written)} page files -> {output_stem}_scan*.alto.xml")
    for page_num, scan, out, n in written:
        print(f"    page {page_num} (scan {scan:04d}): {n} chars -> {Path(out).name}")
    print(f"  sidecar RDF -> {sidecar}")

    # Brief summary.
    n_lines = sum(len(r["lines"]) for p in line_alignment["pages"] for r in p["regions"])
    n_matched = sum(1 for p in line_alignment["pages"] for r in p["regions"]
                    for l in r["lines"] if l["matched"])
    print(f"  pages={len(written)} lines={n_lines} matched={n_matched} entities={len(all_entities)}")


if __name__ == "__main__":
    main()