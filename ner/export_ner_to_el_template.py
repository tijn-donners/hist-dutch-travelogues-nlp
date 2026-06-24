"""Create a CSV template for manual Entity Linking gold standard annotation.

Reads a spaCy NER DocBin (`*.spacy`) and its offset map (`*offset_map*.json`),
extracts every entity, and writes a CSV with empty Wikidata QID and GeoNames ID
columns for manual filling.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Span, Token, DocBin

# ── Register custom extensions (must match tei_exporter.py) ──────────────────
for attr in ("kb_id_wikidata_", "kb_id_geonames_"):
    if not Span.has_extension(attr):
        Span.set_extension(attr, default=None)
    if not Token.has_extension(attr):
        Token.set_extension(attr, default=None)

# ── Helpers ──────────────────────────────────────────────────────────────────
NER_RESULTS_DIR = Path(__file__).resolve().parent / "ner-output"

ROMAN_VALUES = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}


def page_sort_key(pn):
    s = str(pn)
    if s.isdigit():
        return int(s)
    return ROMAN_VALUES.get(s.upper(), 0)


def select_spacy_file():
    """Interactive file picker for .spacy files in ner-output/."""
    files = sorted(NER_RESULTS_DIR.rglob("*.spacy"))
    if len(files) == 0:
        print(f"No .spacy files found in {NER_RESULTS_DIR}")
        sys.exit(1)
    if len(files) == 1:
        return files[0]

    print("Available NER result files:")
    for i, f in enumerate(files, 1):
        rel = f.relative_to(NER_RESULTS_DIR)
        print(f"  [{i}] {rel}")
    choice = input("Select number: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return files[idx]
    except ValueError:
        pass
    print(f"Invalid selection: {choice}")
    sys.exit(1)


def find_offset_map(spacy_path):
    """Find the matching offset map for a .spacy file."""
    stem = spacy_path.stem
    for d in (spacy_path.parent,):
        for candidate in sorted(d.glob(f"*offset_map*")):
            if candidate.exists():
                return candidate
    # Broader search in ner-output/
    for candidate in sorted(NER_RESULTS_DIR.rglob(f"*offset_map*")):
        if candidate.exists():
            return candidate
    return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Select input file
    if len(sys.argv) > 1:
        spacy_path = Path(sys.argv[1])
        if not spacy_path.exists():
            print(f"File not found: {spacy_path}")
            sys.exit(1)
    else:
        spacy_path = select_spacy_file()

    print(f"Reading: {spacy_path.name}")

    # 2. Find offset map
    offset_map_path = find_offset_map(spacy_path)
    offset_map = {}
    if offset_map_path and offset_map_path.exists():
        with open(offset_map_path) as f:
            offset_map = json.load(f)
        print(f"Offset map: {offset_map_path.name} ({len(offset_map)} pages)")
    else:
        print("Warning: no offset map found — page numbers will be unknown")

    # 3. Build reverse offset lookup: sorted list of (offset, page_number)
    sorted_offsets = sorted(
        (int(pos), pn) for pn, pos in offset_map.items()
        if str(pos).isdigit()
    )

    def lookup_page(char_offset):
        """Find the page number for a given character offset."""
        best = None
        for off, pn in sorted_offsets:
            if off <= char_offset:
                best = pn
            else:
                break
        return best

    # 4. Load DocBin
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(str(spacy_path))
    docs = list(db.get_docs(nlp.vocab))
    print(f"Docs loaded: {len(docs)}")

    # 5. Build entity rows
    rows = []
    global_offset = 0
    seen_spans = set()  # dedup identical spans across pages

    for doc_idx, doc in enumerate(docs):
        # Detect [N] marker in first doc
        marker_len = 0
        if doc_idx == 0:
            stripped = doc.text.lstrip()
            marker_len = len(doc.text) - len(stripped)

        for ent in doc.ents:
            start = global_offset + ent.start_char - marker_len
            end = global_offset + ent.end_char - marker_len
            page = lookup_page(start) or f"doc_{doc_idx}"

            # Dedup: same entity, same page, same text
            key = (page, ent.text.lower(), ent.label_)
            if key in seen_spans:
                continue
            seen_spans.add(key)

            try:
                wd = ent._.kb_id_wikidata_ or ""
            except (AttributeError, KeyError):
                wd = ""
            try:
                gn = ent._.kb_id_geonames_ or ""
            except (AttributeError, KeyError):
                gn = ""

            rows.append({
                "page": page,
                "text": ent.text,
                "label": ent.label_,
                "start_char": start,
                "end_char": end,
                "wikidata_qid": wd,
                "geonames_id": gn,
            })

        global_offset += len(doc.text) - marker_len

    # 6. Write CSV
    df = pd.DataFrame(rows, columns=[
        "page", "text", "label", "start_char", "end_char",
        "wikidata_qid", "geonames_id",
    ])

    out_path = spacy_path.with_name(spacy_path.stem + "_el_gold_template.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nWritten: {out_path}")
    print(f"  Entities: {len(df)}")
    print(f"  Labels:   {df['label'].value_counts().to_dict()}")
    print(f"  Pages:    {df['page'].nunique()}")
    print("\nFill in 'wikidata_qid' and 'geonames_id' columns manually in a spreadsheet.")
    print("Leave blank for entities that should not be linked (e.g. generic types).")


if __name__ == "__main__":
    main()