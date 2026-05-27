import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import re as regex
from dotenv import load_dotenv
from spacy_llm.util import assemble_from_config
from spacy.tokens import DocBin
from spacy.util import load_config
import logging

# ---------------------------------------------------------------------------
# Page number sorting helper
# ---------------------------------------------------------------------------
_ROMAN_VALUES = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
    'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13,
    'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18,
    'XIX': 19, 'XX': 20, 'XXI': 21, 'XXII': 22, 'XXIII': 23,
    'XXIV': 24, 'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28,
    'XXIX': 29, 'XXX': 30, 'XXXI': 31, 'XXXII': 32, 'XXXIII': 33,
    'XXXIV': 34, 'XXXV': 35, 'XXXVI': 36, 'XXXVII': 37, 'XXXVIII': 38,
    'XXXIX': 39, 'XL': 40,
}

def _page_sort_key(pn):
    """Sort key: Roman numerals -> numeric, Arabic -> numeric, fallback -> 0."""
    s = str(pn)
    if s.isdigit():
        return int(s)
    if s.upper() in _ROMAN_VALUES:
        return _ROMAN_VALUES[s.upper()]
    return 0

# make sure ollama is serving to http://localhost:11434
# in order to use ollama's cloud models you have to create an account
# login via the terminal with `ollama signin`
# local models can be run without logging in

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
# Configuration
INPUT_MODE = "txt"  # "txt" or "pagexml" (pagexml requires corrected files in page_updated/)
TXT_FILE = str(ROOT_DIR / "data/1816_third_letter.txt")
PAGEXML_DIR = str(ROOT_DIR / "data/page_updated/")  # corrected PageXML
SCAN_CSV = str(ROOT_DIR / "data/1816-scannumber-to-pagenumber.csv")
OUTPUT_DIR = SCRIPT_DIR / "ner-results"
ner_config = "ner_config_fewshot.cfg"
_run_label = ner_config.replace("ner_config_", "").replace(".cfg", "")


# ---------------------------------------------------------------------------
# PageXML input helpers
# ---------------------------------------------------------------------------
def _ns(root):
    """Retrieve namespace from root element."""
    match = regex.match(r'\{([^}]+)\}', root.tag)
    return match.group(1) if match else ''


def get_pagexml_text_lines(xml_path):
    """Return list of text lines (in reading order) from a PageXML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = _ns(root)

    ordered_group = root.find(f".//{{{ns}}}OrderedGroup")
    if ordered_group is None:
        return []

    lines = []
    for ref in ordered_group.findall(f"{{{ns}}}RegionRefIndexed"):
        region_id = ref.attrib['regionRef']
        region = root.find(f".//{{{ns}}}TextRegion[@id='{region_id}']")
        if region is not None:
            for line_el in region.findall(f"{{{ns}}}TextLine"):
                text_el = line_el.find(f"{{{ns}}}TextEquiv/{{{ns}}}PlainText")
                text = text_el.text if text_el is not None and text_el.text else ""
                lines.append(text)
    return lines


def extract_scan_number(xml_filename):
    """Extract scan number from '0552_0179_0001.xml' -> 1."""
    stem = Path(xml_filename).stem
    parts = stem.split('_')
    return int(parts[-1])


def load_scan_page_mapping(csv_path):
    """Return dict: scan_number -> page_number (str because of roman numerals)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return dict(zip(df['Scan Number'], df['Page Number'].astype(str)))


# ---------------------------------------------------------------------------
# Load text depending on input mode
# ---------------------------------------------------------------------------
def load_pages_from_txt(txt_path):
    """Parse .txt with [N] markers, return (pages dict, None)."""
    with open(txt_path, 'r') as f:
        letter = f.read()

    marker_matches = list(regex.finditer(r'\[(\d+)\]', letter))
    pages = {}
    for i, m in enumerate(marker_matches):
        page_num = int(m.group(1))
        page_start = m.start()
        next_start = marker_matches[i + 1].start() if i + 1 < len(marker_matches) else len(letter)
        page_text = letter[page_start:next_start]
        if page_text.strip():
            pages[page_num] = (page_start, page_text)
    return pages, None


def load_pages_from_pagexml(pagexml_dir, scan_csv):
    """Parse corrected PageXML files, return (pages dict, line_offset_map).

    pages: {pagenumber: (full_text_offset, page_text)}
        page_text has lines joined with \\n.
        full_text_offset is a virtual offset (concatenation of all pages).

    line_offset_map: {pagenumber: {char_offset_within_page: line_index}}
        maps each character position within a page to the line it belongs to.
    """
    scan_to_page = load_scan_page_mapping(scan_csv)
    xml_files = sorted(Path(pagexml_dir).glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found in {pagexml_dir}")

    # Build per-page data
    page_data = {}  # scan_number -> (page_number, line_texts)
    for xml_path in xml_files:
        scan_num = extract_scan_number(xml_path.name)
        page_num = scan_to_page.get(scan_num)
        if page_num is None:
            print(f"  Skipping {xml_path.name}: no scan→page mapping for scan {scan_num}")
            continue
        lines = get_pagexml_text_lines(str(xml_path))
        if not lines:
            print(f"  Skipping {xml_path.name}: no text lines found")
            continue
        page_data[scan_num] = (page_num, lines)
        print(f"  {xml_path.name}: scan {scan_num} → page {page_num} ({len(lines)} lines)")

    # Sort by scan number, build pages dict with virtual full-text offsets
    pages = {}
    line_offset_map = {}
    full_text_offset = 0
    for scan_num in sorted(page_data):
        page_num, lines = page_data[scan_num]
        page_text = "\n".join(lines)
        pages[int(page_num) if page_num.isdigit() else page_num] = (full_text_offset, page_text)

        # Build line offset map for this page
        char_pos = 0
        line_map = {}
        for li, line_text in enumerate(lines):
            for _ in line_text:
                line_map[char_pos] = li
                char_pos += 1
            # the newline belongs to the current line
            if li < len(lines) - 1:
                line_map[char_pos] = li
                char_pos += 1  # the \n character
        line_offset_map[page_num] = line_map

        full_text_offset += len(page_text)

    return pages, line_offset_map


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    """Run the NER pipeline on the travelogue text.

    Loads pages from .txt or corrected PageXML, assembles a spaCy-LLM pipeline
    from the few-shot config, and runs NER per page. Saves merged DocBin,
    full-text offset map, and (in PageXML mode) a line offset map.
    """
    load_dotenv()
    os.chdir(SCRIPT_DIR)  # config relative paths resolve against script dir
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("spacy_llm").setLevel(logging.DEBUG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get the model name from the config file
    config_path = str(SCRIPT_DIR / ner_config)
    config = load_config(config_path)
    model_name = config["components"]["llm"]["model"]["name"]

    if os.environ.get("OLLAMA_API_KEY"):
        config["components"]["llm"]["model"]["config"]["base_url"] = "https://ollama.com"
        config["components"]["llm"]["model"]["config"]["headers"] = {
            "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"
        }
        print("Using Ollama cloud API with API key from environment")
    else:
        print("Using Ollama at localhost (no OLLAMA_API_KEY found)")

    # Load pages
    if INPUT_MODE == "pagexml":
        print(f"Loading corrected PageXML from: {PAGEXML_DIR}")
        pages, line_offset_map = load_pages_from_pagexml(PAGEXML_DIR, SCAN_CSV)
    else:
        print(f"Loading text from: {TXT_FILE}")
        pages, line_offset_map = load_pages_from_txt(TXT_FILE)

    print(f"Loaded {len(pages)} pages")
    print(f"Starting LLM NER in SpaCy framework with: {model_name}\n")

    # Assemble the Ollama LLM config as spacy nlp object
    nlp = assemble_from_config(config)
    print('Pipeline loaded:', nlp.pipe_names)

    # Collect all docs for merging later
    all_docs = []
    offset_map = {}  # pagenumber -> full-text character offset

    # Run NER per page
    for pagenumber, (page_offset, page_text) in sorted(pages.items(), key=lambda x: _page_sort_key(x[0])):
        doc = nlp(page_text)
        offset_map[pagenumber] = page_offset
        print(f"SpaCy DocBin created for page {pagenumber} (full-text offset: {page_offset})")

        # Print the entities that were found
        if doc.ents:
            for ent in doc.ents:
                print(f"  {ent.text}  ({ent.label_})")
        else:
            print("  There were no entities recognised")

        all_docs.append(doc)

    # Merge all docs into one DocBin and save
    merged_docbin = DocBin(docs=all_docs)
    spacy_path = OUTPUT_DIR / f"1816_all_pages_{model_name}_{_run_label}.spacy"
    merged_docbin.to_disk(spacy_path)
    print(f"\nMerged DocBin saved to: {spacy_path}")

    # Save offset map for aligning with full-text annotations
    offset_map_path = OUTPUT_DIR / f"1816_offset_map_{model_name}_{_run_label}.json"
    with open(offset_map_path, 'w') as f:
        json.dump(offset_map, f)
    print(f"Offset map saved to: {offset_map_path}")

    # Save line offset map if available (PageXML mode)
    if line_offset_map is not None:
        line_map_path = OUTPUT_DIR / f"1816_line_map_{model_name}_{_run_label}.json"
        with open(line_map_path, 'w') as f:
            json.dump(line_offset_map, f)
        print(f"Line offset map saved to: {line_map_path}")


if __name__ == "__main__":
    main()
