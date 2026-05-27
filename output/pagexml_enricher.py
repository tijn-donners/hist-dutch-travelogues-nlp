"""Write NER + Entity Linking annotations back into PageXML <Metadata>.

Reads an _el.spacy DocBin and offset/line maps, adds structured entity
annotations to the <Metadata> section of each PageXML file.

Usage:
    python output/pagexml_enricher.py
    (edit the paths under Configuration)
"""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
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
PAGEXML_IN = str(ROOT_DIR / "data" / "page_updated/")  # corrected PageXML input
PAGEXML_OUT = str(SCRIPT_DIR / "page/")  # enriched output
SCAN_CSV = str(ROOT_DIR / "data" / "1816-scannumber-to-pagenumber.csv")

# Fallback to uncorrected PageXML if no corrected files exist
FALLBACK_PAGEXML = str(ROOT_DIR / "data" / "page/")

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

# ---------------------------------------------------------------------------
# Helpers
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


def _ns(root):
    """Retrieve namespace from root tag."""
    import re
    match = re.match(r'\{([^}]+)\}', root.tag)
    return match.group(1) if match else ''


def extract_scan_number(xml_filename):
    stem = Path(xml_filename).stem
    parts = stem.split('_')
    return int(parts[-1])


def build_page_to_scan_map(scan_csv):
    """Reverse mapping: page_number (str) -> scan_number (int)."""
    df = pd.read_csv(scan_csv)
    return dict(zip(df['Page Number'].astype(str), df['Scan Number']))


def find_pagexml_file(pagexml_dir, scan_number):
    """Find XML file matching a scan number (last underscore-separated component)."""
    for f in sorted(Path(pagexml_dir).glob("*.xml")):
        if extract_scan_number(f.name) == scan_number:
            return f
    return None


def add_annotation_metadata(tree, root, ns, entities, page_num):
    """Add NER annotation <MetadataItem> to PageXML <Metadata> section."""
    metadata = root.find(f"{{{ns}}}Metadata")
    if metadata is None:
        metadata = ET.SubElement(root, f"{{{ns}}}Metadata")

    # Remove previous NER annotations if present
    for item in metadata.findall(f"{{{ns}}}MetadataItem"):
        if item.get("type") == "ner-annotations":
            metadata.remove(item)

    # Create new annotation block
    item = ET.SubElement(metadata, f"{{{ns}}}MetadataItem")
    item.set("type", "ner-annotations")
    item.set("page", str(page_num))

    for ent in entities:
        ann = ET.SubElement(item, f"{{{ns}}}Annotation")
        ann.set("label", ent["label"])
        ann.set("start", str(ent["start"]))
        ann.set("end", str(ent["end"]))
        ann.set("line", str(ent["line"]))
        if ent.get("kb_id"):
            ann.set("kb_id", ent["kb_id"])
        ann.text = ent["text"]

    return metadata


def main():
    # --- Load data ---
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(SPACY_FILE)
    docs = list(db.get_docs(nlp.vocab))

    with open(OFFSET_MAP) as f:
        offset_map = json.load(f)

    line_map = None
    if Path(LINE_MAP).exists():
        with open(LINE_MAP) as f:
            line_map = json.load(f)
        print(f"Loaded line map with {len(line_map)} pages")
    else:
        print("No line map found — line indices will be computed from text")

    sorted_pages = sorted(offset_map.items(), key=lambda x: _page_sort_key(x[0]))

    if len(docs) != len(sorted_pages):
        print(f"Warning: {len(docs)} docs but {len(sorted_pages)} pages in offset map")

    # --- Determine PageXML input directory ---
    pagexml_dir = PAGEXML_IN
    if not Path(pagexml_dir).exists() or not list(Path(pagexml_dir).glob("*.xml")):
        print(f"No PageXML files found in {pagexml_dir}, falling back to {FALLBACK_PAGEXML}")
        pagexml_dir = FALLBACK_PAGEXML

    # --- Build page-to-scan mapping ---
    page_to_scan = build_page_to_scan_map(SCAN_CSV)
    print(f"Loaded {len(page_to_scan)} page→scan mappings")

    # --- Ensure output directory ---
    out_dir = Path(PAGEXML_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Enrich each page ---
    enriched = 0
    skipped = 0
    for i in range(min(len(docs), len(sorted_pages))):
        doc = docs[i]
        page_num, _ = sorted_pages[i]

        # Find scan number and XML file
        scan_num = page_to_scan.get(str(page_num))
        if scan_num is None:
            print(f"  Page {page_num}: no scan mapping found — skipping")
            skipped += 1
            continue

        xml_path = find_pagexml_file(pagexml_dir, scan_num)
        if xml_path is None:
            print(f"  Page {page_num} (scan {scan_num}): no XML file found — skipping")
            skipped += 1
            continue

        # Build entity annotations for this page
        page_line_map = None
        if line_map:
            raw = line_map.get(str(page_num), {})
            page_line_map = {}
            for k, v in raw.items():
                page_line_map[int(k)] = v

        entities = []
        for ent in doc.ents:
            start_line = 0
            if page_line_map:
                start_line = page_line_map.get(ent.start_char, 0)
            else:
                start_line = doc.text[:ent.start_char].count('\n')

            kb_id = ent.kb_id_ if ent.kb_id_ else None
            entities.append({
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "line": start_line,
                "kb_id": kb_id,
                "text": ent.text,
            })

        if not entities:
            print(f"  Page {page_num} (scan {scan_num}): no entities — copying as-is")
            shutil.copy2(str(xml_path), str(out_dir / xml_path.name))
            skipped += 1
            continue

        # Parse and update PageXML
        ET.register_namespace('', PAGE_NS)
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        ns = _ns(root)

        add_annotation_metadata(tree, root, ns, entities, page_num)

        out_path = out_dir / xml_path.name
        tree.write(str(out_path), encoding='UTF-8', xml_declaration=True)

        linked = sum(1 for e in entities if e["kb_id"])
        print(f"  Page {page_num} (scan {scan_num}): {len(entities)} entities"
              f" ({linked} linked) → {out_path.name}")
        enriched += 1

    print(f"\nDone. Enriched: {enriched}, skipped: {skipped}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
