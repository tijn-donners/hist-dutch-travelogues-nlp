#!/usr/bin/env python3
"""Build a gold-standard .spacy file from entity annotations CSV.

Reads a gold-standard CSV (with text, label, character offsets, KB IDs),
builds page-structured spaCy Doc objects with doc.ents populated from the
gold annotations, and saves the result to ner/ner-output/ so the EL pipeline
can read it like any other NER output file.

Usage:
    python entity_linking/el_from_gs.py
    python entity_linking/el_from_gs.py --gold entity_linking/1816_el_gs.csv
    python entity_linking/el_from_gs.py --gold entity_linking/1816_el_gs.csv \\
                                        --offset-map ner/ner-output/PATH_offset_map.json

Output:
    ner/ner-output/{stem}/{stem}.spacy
    ner/ner-output/{stem}/{stem}_offset_map.json

Then run EL with:
    python entity_linking/el.py
    # and select the gold .spacy file from the interactive picker
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import spacy
from spacy.tokens import DocBin, Span

# ── Register custom extensions (must match el.py) ──────────────────────────
for attr in ("kb_id_wikidata_", "kb_id_geonames_"):
    if not Span.has_extension(attr):
        Span.set_extension(attr, default=None)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
NER_OUTPUT_DIR = ROOT_DIR / "ner" / "ner-output"
DEFAULT_GOLD_CSV = SCRIPT_DIR / "1816_el_gs.csv"
DEFAULT_SOURCE_TEXT = ROOT_DIR / "data" / "1816_third_letter.txt"


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalise_id(raw: str | None) -> str | None:
    """Normalise a KB ID (same logic as el_evaluate.py)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() in ("NIL", "NONE", ""):
        return None
    if s.startswith("gn:"):
        return s[3:]
    return s


def load_gold_csv(path: Path) -> list[dict]:
    """Load gold-standard CSV with entity annotations and KB IDs."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(row["start_char"])
                end = int(row["end_char"])
            except (ValueError, KeyError) as e:
                print(f"  Skipping row with invalid offsets: {e}")
                continue
            wd = _normalise_id(row.get("wikidata_qid", ""))
            gn = _normalise_id(row.get("geonames_id", ""))
            rows.append({
                "text": row.get("text", ""),
                "label": row.get("label", ""),
                "start_char": start,
                "end_char": end,
                "wikidata_qid": wd,
                "geonames_id": gn,
                "note": row.get("note", "").strip(),
            })
    return rows


def load_source_text(path: Path) -> str:
    """Load source text with utf-8-sig to strip BOM."""
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def find_offset_map() -> Path | None:
    """Scan ner-output/ for an existing offset map file.

    Returns the first *offset_map*.json found, or None.
    """
    candidates = sorted(NER_OUTPUT_DIR.rglob("*offset_map*.json"))
    if candidates:
        return candidates[0]
    return None


def generate_offset_map(text: str) -> dict[str, int]:
    """Generate an offset map from source text by parsing [N] markers.

    The offsets are in utf-8-sig coordinates (BOM already stripped).
    To match the convention used by the NER pipeline (utf-8 coordinates
    where BOM is at position 0), we add 1 to each offset.

    Returns:
        dict mapping page number (str) to global character offset (int).
    """
    pattern = re.compile(r'\[(\d+)\]')
    offset_map = {}
    for m in pattern.finditer(text):
        page_num = m.group(1)
        # Add 1 to match utf-8 coordinate convention (BOM at position 0)
        offset_map[page_num] = m.start() + 1
    return offset_map


def build_page_boundaries(offset_map: dict) -> list[tuple[int, int, int]]:
    """Return sorted list of (page_number, start_offset, end_offset).

    Offsets are in utf-8 coordinates (BOM at position 0).
    """
    sorted_pages = sorted(offset_map.items(), key=lambda x: int(x[0]))
    boundaries = []
    for i, (page_str, start) in enumerate(sorted_pages):
        page = int(page_str)
        end = (int(sorted_pages[i + 1][1])
               if i + 1 < len(sorted_pages) else float("inf"))
        boundaries.append((page, int(start), end))
    return boundaries


def assign_gold_to_pages(
    gold_rows: list[dict],
    boundaries: list[tuple[int, int, int]],
) -> dict[int, list[dict]]:
    """Map gold entities to pages, converting to page-local offsets.

    Gold entity offsets are in utf-8 coordinates (BOM at position 0).
    Page boundaries are also in utf-8 coordinates.
    Page-local offsets are computed as: local = global - page_start.

    The page text is loaded with utf-8-sig (BOM stripped), so page_start-1
    in utf-8-sig corresponds to page_start in utf-8. The subtraction
    global - page_start correctly converts to utf-8-sig page-local offsets.
    """
    gold_by_page: dict[int, list[dict]] = {b[0]: [] for b in boundaries}
    for row in gold_rows:
        assigned = False
        for page, page_start, page_end in boundaries:
            if page_start <= row["start_char"] < page_end:
                local = {
                    **row,
                    "local_start": row["start_char"] - page_start,
                    "local_end": row["end_char"] - page_start,
                }
                gold_by_page[page].append(local)
                assigned = True
                break
        if not assigned:
            print(f"  Warning: gold entity '{row['text']}' (offset "
                  f"{row['start_char']}) not within any page boundary")
    return gold_by_page


def make_span(doc, local_start: int, local_end: int, label: str) -> Span | None:
    """Create a spaCy Span from character offsets, with token-boundary fallback.

    Tries doc.char_span() first. If that fails (offsets don't align with
    token boundaries), finds the nearest token boundaries and creates the
    span from those.
    """
    span = doc.char_span(local_start, local_end, label=label)
    if span is not None:
        return span

    # Fallback: find nearest token boundaries
    start_token = None
    end_token = None
    for token in doc:
        tok_start = token.idx
        tok_end = token.idx + len(token.text)
        if tok_start <= local_start < tok_end:
            start_token = token
        if tok_start < local_end <= tok_end:
            end_token = token
        if start_token is not None and end_token is not None:
            break

    if start_token is not None and end_token is not None:
        span = Span(doc, start_token.i, end_token.i + 1, label=label)
        return span

    # Last resort: find any token that overlaps the range
    for token in doc:
        tok_start = token.idx
        tok_end = token.idx + len(token.text)
        if tok_start < local_end and tok_end > local_start:
            if start_token is None:
                start_token = token
            end_token = token

    if start_token is not None and end_token is not None:
        span = Span(doc, start_token.i, end_token.i + 1, label=label)
        return span

    return None


def build_gold_docs(
    gold_by_page: dict[int, list[dict]],
    source_text: str,
    boundaries: list[tuple[int, int, int]],
) -> tuple[list, list[int]]:
    """Build spaCy Docs with gold-standard entities as doc.ents.

    Args:
        gold_by_page: Gold entities grouped by page (with page-local offsets).
        source_text: Full source text loaded with utf-8-sig (BOM stripped).
        boundaries: List of (page, start_offset, end_offset) in utf-8 coordinates.

    Returns:
        Tuple of (list of spaCy Doc objects, list of page numbers in order).
    """
    nlp = spacy.blank("nl")
    docs = []
    page_order = []

    for page, page_start, page_end in boundaries:
        golds = gold_by_page.get(page, [])
        if not golds:
            continue

        # Extract page text: convert utf-8 offsets to utf-8-sig indices
        # utf-8 offset N → utf-8-sig index N-1 (since BOM is at utf-8 position 0)
        sig_start = max(0, page_start - 1)
        if page_end == float("inf"):
            sig_end = len(source_text)
        else:
            sig_end = page_end - 1

        page_text = source_text[sig_start:sig_end]
        if not page_text.strip():
            continue

        doc = nlp.make_doc(page_text)
        spans = []

        for g in golds:
            span = make_span(doc, g["local_start"], g["local_end"], g["label"])
            if span is not None:
                spans.append(span)
            else:
                print(f"  Warning: could not create span for '{g['text']}' "
                      f"on page {page} (local offsets {g['local_start']}-"
                      f"{g['local_end']})")

        try:
            doc.ents = spans
        except Exception as e:
            print(f"  Warning: could not set doc.ents for page {page}: {e}")
            # Try setting with filter of overlapping spans
            filtered = []
            for s in spans:
                if not any(s.start < o.end and o.start < s.end
                          for o in filtered):
                    filtered.append(s)
                else:
                    print(f"    Dropping overlapping span '{s.text}'")
            doc.ents = filtered

        docs.append(doc)
        page_order.append(page)

    return docs, page_order


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build gold-standard .spacy file from entity annotations CSV."
    )
    parser.add_argument("--gold", default=None,
                        help="Path to gold-standard CSV")
    parser.add_argument("--offset-map", default=None,
                        help="Path to offset map JSON (auto-detected if not given)")
    parser.add_argument("--source-text", default=None,
                        help="Path to source text file")
    args = parser.parse_args()

    # ── 1. Load gold CSV ──────────────────────────────────────────────────
    gold_path = Path(args.gold) if args.gold else DEFAULT_GOLD_CSV
    if not gold_path.exists():
        print(f"Gold CSV not found: {gold_path}")
        sys.exit(1)
    print(f"Gold CSV:  {gold_path.name}")

    gold_rows = load_gold_csv(gold_path)
    print(f"Gold entities loaded: {len(gold_rows)}")

    # ── 2. Load source text ───────────────────────────────────────────────
    source_path = Path(args.source_text) if args.source_text else DEFAULT_SOURCE_TEXT
    if not source_path.exists():
        print(f"Source text not found: {source_path}")
        sys.exit(1)
    source_text = load_source_text(source_path)
    print(f"Source text: {source_path.name} ({len(source_text)} chars)")

    # ── 3. Load / generate offset map ─────────────────────────────────────
    offset_map_path = None
    if args.offset_map:
        offset_map_path = Path(args.offset_map)
        if not offset_map_path.exists():
            print(f"Offset map not found: {offset_map_path}")
            sys.exit(1)
    else:
        found = find_offset_map()
        if found:
            offset_map_path = found
            print(f"Offset map auto-detected: {offset_map_path.name}")
        else:
            print("No offset map found in ner-output/, generating from source text...")
            with open(source_path, encoding="utf-8") as f:
                text_utf8 = f.read()
            offset_map = generate_offset_map(text_utf8)
            # Save to a temp path
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp())
            offset_map_path = tmp_dir / "generated_offset_map.json"
            with open(offset_map_path, "w") as f:
                json.dump(offset_map, f)
            print(f"Offset map generated with {len(offset_map)} pages")

    with open(offset_map_path, encoding="utf-8") as f:
        offset_map = json.load(f)
    boundaries = build_page_boundaries(offset_map)
    print(f"Page boundaries: {len(boundaries)} pages "
          f"({min(b[0] for b in boundaries)}-{max(b[0] for b in boundaries)})")

    # ── 4. Assign gold entities to pages ──────────────────────────────────
    gold_by_page = assign_gold_to_pages(gold_rows, boundaries)
    total_assigned = sum(len(v) for v in gold_by_page.values())
    print(f"Gold entities assigned to pages: {total_assigned}")
    if total_assigned < len(gold_rows):
        print(f"  Warning: {len(gold_rows) - total_assigned} entities not assigned")

    # ── 5. Build spaCy Docs with gold entities ────────────────────────────
    print("\nBuilding spaCy Docs with gold-standard entities...")
    docs, page_order = build_gold_docs(gold_by_page, source_text, boundaries)
    print(f"Docs created: {len(docs)} (pages {page_order[0]}-{page_order[-1]})")

    total_ents = sum(len(doc.ents) for doc in docs)
    print(f"Total gold entities in docs: {total_ents}")

    # ── 6. Save gold-standard .spacy + offset map to ner-output/ ──────────
    output_dir = NER_OUTPUT_DIR / gold_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    spacy_path = output_dir / f"{gold_path.stem}.spacy"
    docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(str(spacy_path))
    print(f"\nGold .spacy saved to: {spacy_path}")

    # Copy offset map alongside
    import shutil
    offset_dest = output_dir / f"{gold_path.stem}_offset_map.json"
    shutil.copy2(str(offset_map_path), str(offset_dest))
    print(f"Offset map saved to: {offset_dest}")

    # Clean up temp offset map if it was generated
    if args.offset_map is None and found is None:
        import tempfile
        try:
            import shutil
            shutil.rmtree(offset_map_path.parent)
        except OSError:
            pass

    print("\nDone. Now run the EL pipeline and select this file:")
    print(f"  python entity_linking/el.py --model <model> --temperature <temp>")


if __name__ == "__main__":
    main()
